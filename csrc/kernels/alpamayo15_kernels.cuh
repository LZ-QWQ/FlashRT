// SPDX-License-Identifier: Apache-2.0
//
// Alpamayo 1.5 (Thor sm_110) model-local kernels.
//
// Each kernel reproduces the rounding sequence of the official PyTorch module
// graph under torch.autocast(bfloat16), so the fused path stays numerically
// interchangeable with the reference:
//
//   vision RoPE      q/k BF16 -> FP32 rotate-half RoPE with FP32 tables ->
//                    one BF16 rounding (apply_rotary_pos_emb_vision).
//   expert RMSNorm   FP32 residual (+ optional BF16 delta) -> FP32 RMSNorm ->
//                    FP32 weight product -> one BF16 rounding (the autocast
//                    cast at the next linear).
//   expert QK prep   BF16 q/k -> Qwen3 head RMSNorm (BF16 rounding of the
//                    normalized value, then of the weight product) -> FP32
//                    RoPE with FP32 tables -> one BF16 rounding (autocast cast
//                    at SDPA); V is copied unchanged.
//   VLM RMSNorm      optional BF16 residual add (rounded), FP32 statistics,
//                    BF16 rounding of the normalized value, BF16 weight
//                    product (Qwen3VLTextRMSNorm on BF16 activations).
//   VLM QK prep      Qwen3 head RMSNorm as above -> BF16 RoPE with BF16
//                    tables (each product and the sum rounded to BF16) -> K/V
//                    written to the cache row selected by an optional device
//                    slot (CUDA-graph friendly decode).
//
//   W8A16 GEMM       opt-in FP8 decode-weight tier (not part of the BF16
//                    contract): y = bf16(scale[n] * sum_k e4m3(w[n,k]) * x[m,k])
//                    with exact e4m3 -> bf16 widening and FP32 tensor-core
//                    accumulation.
//
// All FP32 products and sums use explicit round-to-nearest intrinsics so no
// FMA contraction changes the result relative to PyTorch's elementwise ops.
// The module is built with --ftz=true, so FP32 denormals flush to zero (PyTorch
// eager keeps them); reductions use a different order than torch.mean.
// Vectorized 8-byte accesses: tensor pointers must be 8-byte aligned (checked).

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

// qkv  : [S, 3, H, D] bf16 (q, k rotated in place; v untouched)
// cos/sin: [S, D] fp32
// Returns 0 on success, nonzero on invalid arguments.
int alpamayo15_vision_rope_qk_bf16(void* qkv, const float* cos, const float* sin,
                                   int S, int H, int D, cudaStream_t stream);

// residual: [rows, dim] fp32 (updated in place when delta != nullptr)
// delta   : [rows, dim] bf16 or nullptr
// weight  : [dim] bf16
// out     : [rows, dim] bf16
int alpamayo15_res_rms_f32_bf16(float* residual, const void* delta, const void* weight,
                                void* out, int rows, int dim, float eps, cudaStream_t stream);

// qkv  : [rows, (Hq + 2*Hk) * 128] bf16 (q | k | v per row)
// q_norm_w, k_norm_w : [128] bf16
// cos/sin : [rows, 128] fp32
// q_out : [rows, Hq, 128] bf16
// k_out / v_out : K/V of row r written at cache row
//   (r / rows_per_batch) * batch_stride + r % rows_per_batch + slot[r / rows_per_batch]
//   (slot = 0 when slot_ptr is null), [Hk, 128] bf16 per row, rows k_row_stride elements apart
int alpamayo15_expert_qk_prep_bf16(const void* qkv, const void* q_norm_w, const void* k_norm_w,
                                   const float* cos, const float* sin,
                                   void* q_out, void* k_out, void* v_out, const long long* slot_ptr,
                                   int rows, int Hq, int Hk, int k_row_stride,
                                   int rows_per_batch, long long batch_stride, float eps,
                                   cudaStream_t stream);

// residual: [rows, dim] bf16, updated in place (residual += delta) when delta != nullptr
// out = RMSNorm(residual) * weight, [rows, dim] bf16
int alpamayo15_rms_bf16(void* residual, const void* delta, const void* weight, void* out,
                        int rows, int dim, float eps, cudaStream_t stream);

// The W8A8 prefill tier's producers with the per-row FP8 E4M3 quantization fused (same codes and scales as the BF16
// kernel followed by alpamayo15_quantize_fp8_rows_bf16; the BF16 row is never written):
//   rms_fp8_rows      residual (+= delta) -> Qwen3 RMSNorm * weight -> q [rows, dim] e4m3 bytes, scale [rows] fp32
//   silu_mul_fp8_rows bf16(silu(gate)) * up over rows ld elements apart -> q [rows, F], scale [rows]
// dim / F <= 16384 and a multiple of 4; 8-byte aligned bf16 pointers, 4-byte aligned q.
int alpamayo15_rms_fp8_rows_bf16(void* residual, const void* delta, const void* weight, void* q, float* scale,
                                 int rows, int dim, float eps, cudaStream_t stream);
int alpamayo15_silu_mul_fp8_rows_bf16(const void* gate, const void* up, int ld, void* q, float* scale, int rows, int F,
                                      cudaStream_t stream);
//   res_rms_f32_fp8_rows  the FP32-residual expert RMSNorm (alpamayo15_res_rms_f32_bf16) -> q [rows, dim], scale [rows]
int alpamayo15_res_rms_f32_fp8_rows_bf16(float* residual, const void* delta, const void* weight, void* q, float* scale,
                                         int rows, int dim, float eps, cudaStream_t stream);

// qkv : [rows, (Hq + 2*Hk) * 128] bf16; cos/sin : [rows, 128] bf16
// q_out : [rows, Hq, 128] bf16
// K/V of row r go to cache row (r / rows_per_batch) * batch_stride + r % rows_per_batch
// + slot, rows kv_row_stride elements apart, where slot = *slot_ptr (int64 device
// scalar) when slot_ptr != nullptr, else 0.
int alpamayo15_vlm_qk_prep_bf16(const void* qkv, const void* q_norm_w, const void* k_norm_w,
                                const void* cos, const void* sin, void* q_out,
                                void* k_cache, void* v_cache, const long long* slot_ptr,
                                int rows, int Hq, int Hk, int kv_row_stride,
                                int rows_per_batch, long long batch_stride, float eps,
                                cudaStream_t stream);


// Skinny GEMM y[M, N] = x[M, K] @ dequant(w)^T for M <= 16 (decode projections).
// w_frag : e4m3 codes in mma.m16n8k16 A-fragment order, 16-byte aligned:
//          T = ceil(N / 16) row tiles (rows >= N zero), K / 32 column chunks,
//          32 lanes x 16 bytes each; lane L = 4g + c, byte j = 8kt + 2ri + e holds
//          w[16t + g + 8(ri & 1), 32kc + 16kt + 8(ri >> 1) + 2c + e]
// scale  : [N] fp32 per output row (fp8_fragment_layout pads it to T * 16 with zeros)
// x      : [M, K] bf16 row-major (4-byte aligned); y : [M, N] bf16 row-major
// K must be a multiple of 32.
// fmt: 0 = FP8 E4M3 codes, 1 = INT8 codes (both per-output-row scales, same fragment order).
int alpamayo15_w8a16_gemm_bf16(const void* w_frag, const float* scale, const void* x, void* y,
                               int M, int N, int K, int fmt, cudaStream_t stream);

// Grouped few-row W4A8 GEMM (decode tier; Marlin's integer arithmetic, see the .cu): n_problems (1-3) linears that
// share one input width K and the row count M (<= 16), one launch.
//   xq[i]      [>= ceil(M/8)*8, K] int8 row-major per problem (its own rotated / quantized activations)
//   a_scale[i] [M] fp32 per problem
//   w_pack     one buffer [tiles][K/128][1088 B] (pack_w4a8_grouped): per tile and group 16 zero points (u8), 16
//              integer group scales (i16, 0..4096), 16 B padding, then the tile's 1 KB of nibbles for the group
//              (pack_w4a8_small layout); the problems' tiles in order
//   out[i]     bf16, rows ld_out elements apart (one row-major [M, sum N] block when the pointers are its column blocks)
//   col_scale  optional per problem: [N_i] fp32 multiplied into the output (own quantization with per-channel
//              normalized integer group scales); nullptr (or a null array) = Marlin's epilogue
//   n_valid    optional per problem: output channels written (<= N_i; the rest of the last tile is padding)
// prefetch (2 or 4 weight groups in flight per warp), warps per CTA (4 or 8), ksplit (0 = auto, 1, 2 or 4 warps per tile).
// N_i % 16 == 0; ld_out >= n_valid_i (any parity); K % 128 == 0 and (K / 128 / ksplit) % prefetch == 0.
//   pf_ptr / pf_bytes  optional 128-byte aligned region prefetched into the L2 by the CTAs during the weight stream
//              (the next group's rotation tables: on Thor a cold read of a few hundred KB right before a streaming
//              kernel slows that kernel by about 30%; prefetched inside the stream it costs nothing)
int alpamayo15_w4a8_gemm_grouped(int n_problems, const void* const* xq, const float* const* a_scale, const void* w_pack,
                                 void* const* out, int M, const int* N, int K, int ld_out, int prefetch, int warps,
                                 int ksplit, cudaStream_t stream, const float* const* col_scale = nullptr,
                                 const int* n_valid = nullptr, const void* pf_ptr = nullptr, size_t pf_bytes = 0);

// The same GEMM with the per-token INT8 quantization of the FlashDrive activation preparation folded into the
// prologue (bit-exact with flashdrive_w4a8_act_prep's quantization pass): xr[i] [rows, K] fp16 rotated activations,
// part[i] [rows, K/128] fp32 per-group row absmax (both from the rotation pass), igs[i] = input_global_scale (fp32
// device scalar). M <= 4 and M <= warps; a CTA's tiles belong to one problem (every problem's first tile index must be
// a multiple of warps / ksplit, as for the engine's q|k|v and gate|up groups). Dynamic shared memory M * K + 32 bytes.
int alpamayo15_w4a8_gemm_grouped_q(int n_problems, const void* const* xr, const float* const* part, const float* const* igs,
                                   const void* w_pack, void* const* out, int M, const int* N, int K, int ld_out, int prefetch,
                                   int warps, int ksplit, cudaStream_t stream, const float* const* col_scale = nullptr,
                                   const int* n_valid = nullptr, const void* pf_ptr = nullptr, size_t pf_bytes = 0);

// Per-token INT8 quantization of BF16 rows: q [rows, K] int8 = round-to-nearest-even(x / scale[row]) in [-127, 127],
// scale[row] = absmax / 127 (1 for an all-zero row). x rows ld_x elements apart (8-byte aligned); K % 4 == 0.
int alpamayo15_quantize_int8_rows_bf16(const void* x, int ld_x, void* q, float* scale, int rows, int K, cudaStream_t stream);

// cuBLASLt BF16 GEMM with a fused epilogue (csrc/gemm/alpamayo15_cublaslt_bf16.cu):
//   D[M, N] = act(A[M, K] @ W[N, K]^T + bias[N]) + C[M, N]     (act 0 identity, 1 GELU tanh; bias and C optional)
// row-major operands (16-byte aligned), D and C rows ldd / ldc elements apart (multiples of 8), FP32 accumulation
// and one BF16 rounding. alpamayo15_cublaslt_init() creates the handle and the workspace (not inside graph capture).
int alpamayo15_cublaslt_init();
int alpamayo15_cublaslt_gemm_bf16(const void* A, const void* W, const void* bias, const void* C, void* D, int M, int N,
                                  int K, int ldc, int ldd, int act, cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
