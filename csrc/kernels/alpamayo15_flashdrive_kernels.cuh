// SPDX-License-Identifier: Apache-2.0
//
// FlashDrive (Alpamayo 1.5) backend kernels for Thor (sm_110).
//
// FlashDrive keeps keys *un-rotated* in its static cache (layout [1, Hk, S, 128]
// per layer) and rotates the whole cache on every forward with a fixed position
// table. The backend keeps that cache as the source of truth and maintains a
// rotated attention copy, so every attention reads keys equal to FlashDrive's
// per-forward rotation:
//
//   qk prep       per projection row: Qwen3 head RMSNorm (BF16 rounding of the
//                 normalized value, then of the weight product); q rotated with
//                 BF16 tables (each product and the sum rounded to BF16, as
//                 FlashDrive's apply_rotary_pos_emb_single on BF16 tables); the
//                 normalized un-rotated k written to the FlashDrive cache, its
//                 rotation to the attention copy, v to the FlashDrive cache.
//   rope rows     rotate cached rows into an attention copy:
//                 kBf16Rule  BF16 tables, BF16 rounding of each product and of the
//                            sum (the VLM's view of the cache);
//                 kFp32Rule  FP32 tables, FP32 products and sum, one BF16
//                            rounding (the action expert's view: FP32 tables
//                            under autocast, cast to BF16 at attention).
//
// Row addressing: cache row of work row r = (row_map ? row_map[r] : row0 + r)
// + (slot_ptr ? *slot_ptr : 0); an optional device bound (n_ptr) skips rows
// r >= *n_ptr (CUDA-graph friendly variable lengths). All strides are in
// elements; 8-byte aligned pointers and strides that are multiples of 4.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

// q/k/v      : projection outputs, row-major, leading dims q_ld / k_ld / v_ld (elements); bf16, or fp16 when
//              f16_in=1 (Marlin W4A8 outputs, rounded to bf16 on load exactly as .to(torch.bfloat16))
// q_norm_w/k_norm_w : [128] bf16
// cos/sin    : [rows, 128] position table rows of the work rows: bf16 (fp32_rule=0, the VLM) or fp32
//              (fp32_rule=1, the action expert: FP32 products and sum, one BF16 rounding)
// q_out      : [rows, Hq, 128] bf16, rotated
// k_raw      : FlashDrive cache keys, element (c, h, d) at c * kr_rs + h * kr_hs + d (normalized, un-rotated)
// k_rope     : attention copy, element (c, h, d) at c * k_rs + h * k_hs + d (rotated)
// v_out      : FlashDrive cache values, element (c, h, d) at c * v_rs + h * v_hs + d
int flashdrive_qk_prep_bf16(const void* q, const void* k, const void* v, int q_ld, int k_ld, int v_ld,
                            const void* q_norm_w, const void* k_norm_w, const void* cos, const void* sin, int fp32_rule,
                            int f16_in, void* q_out, void* k_raw, long long kr_rs, long long kr_hs,
                            void* k_rope, long long k_rs, long long k_hs,
                            void* v_out, long long v_rs, long long v_hs,
                            const long long* row_map, const long long* slot_ptr,
                            int rows, int Hq, int Hk, float eps, cudaStream_t stream);

// Consumers of FP16 Marlin outputs that apply FlashDrive's .to(torch.bfloat16) cast on load (identical results to
// the cast followed by the BF16 kernel):
//   rms_f16delta  residual [rows, dim] bf16 += bf16(delta fp16), out = Qwen3 RMSNorm(residual) * weight (the
//                 rounding contract of alpamayo15_rms_bf16)
//   silu_mul      out [n] bf16 = bf16(silu(bf16(gate))) * bf16(up), each product rounded to bf16 (silu_mul_qwen36_bf16)
int flashdrive_rms_f16delta_bf16(void* residual, const void* delta, const void* weight, void* out, int rows, int dim,
                                 float eps, cudaStream_t stream);
int flashdrive_silu_mul_f16in_bf16(const void* gate, const void* up, void* out, long long n, cudaStream_t stream);

// dst rows = rotation of src rows (same cache rows), Hk heads of 128.
// cos/sin: position table indexed by cache row ([S, 128], bf16 for fp32_rule=0, fp32 for fp32_rule=1).
// n_ptr  : optional int32 device bound on the work rows processed (r < *n_ptr).
int flashdrive_rope_rows(const void* src, long long s_rs, long long s_hs,
                         void* dst, long long d_rs, long long d_hs,
                         const void* cos, const void* sin, int fp32_rule,
                         const long long* row_map, long long row0, const long long* slot_ptr,
                         const int* n_ptr, int rows, int Hk, cudaStream_t stream);

// W4A8 activation preparation for FlashDrive's ParoQuant linears (bit-exact with the paroquant rotation op and
// vLLM's per-token INT8 quantization; see the .cu). For each of n_sets (1-3) rotation sets sharing input x
// ([rows, K] bf16, leading dim ld_x, K % 128 == 0): theta [8, K/2] fp16, pairs [8, K] int16, scales [K] fp16,
// igs = input_global_scale (fp32 device scalar); workspaces xr [rows, K] fp16 and part [rows, K/128] fp32;
// outputs xq [rows, K] int8 and a_scale [rows] fp32 (Marlin a_scales). Two launches (rotate, quantize).
int flashdrive_w4a8_act_prep(const void* x, int ld_x, int rows, int K, int n_sets,
                             const void* const* theta, const void* const* pairs, const void* const* scales,
                             const float* const* igs, void* const* xr, float* const* part, void* const* xq,
                             float* const* a_scale, cudaStream_t stream);

// The rotation pass of flashdrive_w4a8_act_prep alone (xr / part written; the quantization is left to the consumer, e.g.
// alpamayo15_w4a8_gemm_grouped_q). Same set arguments without igs / xq / a_scale.
int flashdrive_w4a8_rotate(const void* x, int ld_x, int rows, int K, int n_sets,
                           const void* const* theta, const void* const* pairs, const void* const* scales,
                           void* const* xr, float* const* part, cudaStream_t stream);

// FlashDrive's streaming frame-block shift of every layer's cache in one launch, plus the VLM view of the moved
// rows: for every view v and frame pair f (F-1 pairs), cache rows [ranges[v][f+1][0], ranges[v][f+1][1]) of keys and
// values move to rows [ranges[v][f][0], ranges[v][f][1]) (equal lengths; frames processed in increasing f so a
// block is read before it is overwritten), and kvlm[l][dst] = BF16-rule rotation of the moved key at dst.
//   k_ptrs / v_ptrs : device arrays of L pointers to the FlashDrive cache tensors (element (c, h, d) at
//                     c * kv_rs + h * kv_hs + d, in elements)
//   kvlm            : [L, S, Hk, 128] bf16 attention copy
//   ranges          : device int32 [n_views, F, 2] (start, end) rows per view and frame
//   cos / sin       : [S, 128] bf16 position tables
int flashdrive_shift_cache(const void* const* k_ptrs, const void* const* v_ptrs, long long kv_rs, long long kv_hs,
                           void* kvlm, const void* cos, const void* sin, const int* ranges, int n_views, int F,
                           int L, int S, int Hk, cudaStream_t stream);

// W4A8 GEMM for a few rows (verification block, trajectory-start row): Marlin's integer arithmetic
//   out[m, n] = half( float( sum_g s[g, n] * sum_{k in g} x[m, k] * (w[k, n] - z[g, n]) ) * a_scale[m] )
// on mma.sync m16n8k32 (int8 x int8 -> int32): weights are the A operand (16 output rows x 32 k), the INT8
// activations the B operand (up to 16 tokens as two 8-column tiles). Group partials and their integer scaling are
// exact int32 (as the tcgen05 prefill GEMM); the total is converted to float once and scaled per token.
//   xq      [rows_alloc >= ceil(M/8)*8, K] int8 row-major (rows beyond M are read, never written back)
//   a_scale [M] fp32
//   w_pack  [N/16][K/64][32 lanes][16 B]: nibbles of the two k32 A fragments of a 64-wide k chunk (low nibble =
//           first fragment), see pack_w4a8_small in the pipeline
//   zp      [K/128][N] uint8 zero points, sg [K/128][N] int32 group scales (0..4096)
//   out     [M, N] row-major, rows ld_out elements apart (0 = N): fp16 (Marlin's output type) or bf16 (out_bf16 = 1)
// M in [1, 16], N % 16 == 0, K % 1024 == 0 (eight 128-wide groups per K split).
int flashdrive_w4a8_gemm_small(const void* xq, const float* a_scale, const void* w_pack, const void* zp,
                               const int* sg, void* out, int M, int N, int K, cudaStream_t stream, int ld_out = 0,
                               int out_bf16 = 0);

// BF16 GEMM for a few rows (DFlash draft projections, LM head on the block rows):
//   y[m, n] = sum_k x[m, k] * w[n, k]   (mma.sync m16n8k16 bf16, FP32 accumulation; weights are the A operand read
//   straight from the row-major [N, K] tensor, so no repacked copy is kept)
//   x   [rows_alloc >= ceil(M/8)*8, K] bf16 row-major, leading dimension ld_x (rows beyond M are read, not written)
//   w   [N, K] bf16 row-major (any N; the last 16-row tile is guarded), K % 512 == 0
//   y   [M, N] row-major, leading dimension ld_y: bf16 (y_f32 = 0) or fp32 (y_f32 = 1)
// M in [1, 16]. Deterministic (fixed reduction order).
int flashdrive_bf16_gemm_small(const void* x, int ld_x, const void* w, void* y, int ld_y, int y_f32, int M, int N,
                               int K, cudaStream_t stream);

// DFlash draft attention operands: Qwen3 head RMSNorm + BF16-rule RoPE (as flashdrive_qk_prep_bf16, fp32_rule 0)
// of q rows (positions ctx .. ctx + nq - 1 of the table) and of nk = ctx + nq key rows (positions 0 .. nk - 1);
// values copied. q [nq, Hq*128], k/v [nk, Hk*128] bf16 with leading dims; cos/sin [nk, 128] bf16;
// q_out [nq, Hq, 128], k_out/v_out [nk, Hk, 128] bf16.
int flashdrive_draft_qkv_prep_bf16(const void* q, const void* k, const void* v, int q_ld, int k_ld, int v_ld,
                                   const void* q_norm_w, const void* k_norm_w, const void* cos, const void* sin,
                                   void* q_out, void* k_out, void* v_out, int nq, int ctx, int Hq, int Hk, float eps,
                                   cudaStream_t stream);

// Dense formulation of the ParoQuant rotation: y[m, g*128 + c] = sum_k R[g][c][k] * (half(x[m, g*128 + k]) * scale),
// one 128 x 128 FP16 matrix per 128-channel group (the product of the 8 Givens layers, precomputed), FP32
// accumulation on tensor cores, outputs rounded to FP16 once (not bit-identical to the layer-by-layer kernel: no
// per-layer FP16 rounding). Writes the rotated rows (xr) and the per-(row, group) absmax (part) consumed by the
// quantization pass of flashdrive_w4a8_act_prep. Per set: rot [K/128, 128, 128] fp16 (R[g][c][k]), scales [K] fp16,
// xr [rows, K] fp16, part [rows, K/128] fp32. x [rows, K] bf16 with leading dim ld_x; K % 128 == 0; 1-3 sets.
int flashdrive_w4a8_rotate_dense(const void* x, int ld_x, int rows, int K, int n_sets, const void* const* rot,
                                 const void* const* scales, void* const* xr, float* const* part, cudaStream_t stream);

// The quantization pass of flashdrive_w4a8_act_prep alone (xr/part -> xq/a_scale).
int flashdrive_w4a8_quant(int rows, int K, int n_sets, const void* const* xr, const float* const* part,
                          const float* const* igs, void* const* xq, float* const* a_scale, cudaStream_t stream);

// LayerNorm of BF16 rows with FP32 statistics and arithmetic, one BF16 rounding (F.layer_norm on BF16 input with
// BF16 weight/bias under autocast): out[r] = (x[r] - mean) * rsqrt(var + eps) * w + b. dim % 8 == 0 (8-byte vectors).
int alpamayo15_layernorm_bf16(const void* x, const void* w, const void* b, void* out, int rows, int dim, float eps,
                              cudaStream_t stream);

// The LayerNorm above with the per-row FP8 E4M3 quantization fused (the W8A8 vision tier): q [rows, dim] e4m3 bytes,
// scale [rows] fp32, same codes as the LayerNorm followed by alpamayo15_quantize_fp8_rows_bf16. dim % 8 == 0, <= 16384.
int alpamayo15_layernorm_fp8_rows_bf16(const void* x, const void* w, const void* b, void* q, float* scale, int rows, int dim,
                                       float eps, cudaStream_t stream);

// out[r, f] = bf16(silu(bf16 gate[r, f])) * bf16 up[r, f] (the rounding of silu_mul_qwen36_bf16); gate and up are
// bf16 rows of leading dimension ld: the halves of a fused gate | up GEMM output gu [rows, 2F] (gate = gu, up = gu + F,
// ld = 2F) or two separate [rows, F] outputs (ld = F). out [rows, F] bf16. F % 4 == 0, ld % 4 == 0.
int alpamayo15_silu_mul_split_bf16(const void* gate, const void* up, int ld, void* out, int rows, int F,
                                   cudaStream_t stream);

// Per-row FP8 E4M3 quantization of BF16 activations (the W8A8 expert tier): scale[r] = absmax(row) / 448 (1 for an
// all-zero row), q = round-to-nearest-even(x / scale) saturated to +-448. x [rows, K] bf16 with leading dim ld_x,
// q [rows, K] e4m3 bytes, scale [rows] fp32. K % 8 == 0.
int alpamayo15_quantize_fp8_rows_bf16(const void* x, int ld_x, void* q, float* scale, int rows, int K, cudaStream_t stream);

// Process-wide programmatic-dependent-launch switch for every FlashDrive-backend kernel (off by default).
void alpamayo15_set_pdl(bool on);
bool alpamayo15_pdl();

// Streaming variants of the few-row GEMMs: one warp per 16-row weight tile over the full K (no K split, no shared
// memory), four warps per CTA, 16-byte streaming loads -- the structure that reaches the DRAM rate on Thor for
// N >= 2048 (many independent streams); the CTA-split kernels above stay for small N.
//   W4A8: same operands as flashdrive_w4a8_gemm_small (M <= 16, N % 16 == 0, K % 128 == 0)
int flashdrive_w4a8_gemm_tile(const void* xq, const float* a_scale, const void* w_pack, const void* zp, const int* sg,
                              void* out, int M, int N, int K, cudaStream_t stream, int ld_out = 0, int out_bf16 = 0);
//   BF16 on fragment-ordered weights: w_frag [ceil(N/16)][K/16][32 lanes][16 B] (bf16_fragment_layout in the
//   pipeline: lane 4g+c holds a0 = w[16T+g, 16kc+2c..+1], a1 = row +8, a2 = cols +8, a3 = both), K % 16 == 0;
//   x [rows_alloc >= ceil(M/8)*8, K] bf16 with leading dim ld_x; y [M, N] bf16 (y_f32 = 0) or fp32, leading dim ld_y.
int flashdrive_bf16_gemm_frag(const void* x, int ld_x, const void* w_frag, void* y, int ld_y, int y_f32, int M, int N,
                              int K, cudaStream_t stream);

// Fused producer + W4A8 activation preparation for tiles of 8 or 16 rows (any row count: the streaming-window
// prefill as well as the verification block). The consumer kernel of a GEMM output -- mode 0: bf16 rows src; mode 1:
// residual (bf16, leading dim res_ld) += bf16(delta fp16 = src) (skipped when src is null) then Qwen3 RMSNorm * norm_w
// (the rms_f16delta / rms_bf16 contract, residual updated in place); mode 2: bf16(silu(bf16 src)) * bf16(src2) from
// fp16 halves (the silu_mul_f16in contract) -- followed, for each of n_sets rotation sets, by the ParoQuant rotation and
// per-token INT8 quantization, bit-exact with flashdrive_w4a8_act_prep on the producer's output. Same set arguments
// as flashdrive_w4a8_act_prep without the FP16 / partial workspaces; leading dims in elements; K % 128 == 0.
// tile_rows (0 = default, 8 or 16) rows per CTA; cluster (0 = automatic, 1..8) CTAs per tile, each owning a span of
// K; clusters of more than one CTA exchange the row absmax through distributed shared memory.
int flashdrive_w4a8_act_prep_tile(int mode, const void* src, const void* src2, int src_ld, void* residual, int res_ld,
                                  const void* norm_w, float eps, int rows, int K, int n_sets, const void* const* theta,
                                  const void* const* pairs, const void* const* scales, const float* const* igs,
                                  void* const* xq, float* const* a_scale, int tile_rows, int cluster,
                                  cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
