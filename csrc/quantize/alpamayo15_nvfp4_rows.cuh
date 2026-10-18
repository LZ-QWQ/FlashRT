// SPDX-License-Identifier: Apache-2.0
// NVFP4 row quantization for the Alpamayo 1.5 prefill tier (SM100/SM110): bf16 rows -> e2m1 nibbles + UE4M3 block-16
// scales in the CUTLASS Sm1xx tile-interleaved layout, with a scale gs = amax / (6 * 448) that places the block scales
// in the E4M3 normal range (block scale = e4m3(block_amax / 6 / gs), value = e2m1(x / (scale * gs))): per tensor for the
// static weights (absmax + quantize_rows), per row for the activations (the rowscale quantizer and the fused RMSNorm /
// SiLU-mul producers, whose per-row GEMM alphas gs_row * gs_weight feed the GEMM's per-row alpha epilogue).
#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace fp4 {

// Per-block absolute maxima of x [rows, K] bf16 (rows ld elements apart) into partials[0 .. n_partials); returns the
// number of partials written (<= max_partials) or a negative code.
int alpamayo15_nvfp4_absmax_bf16(const void* x, int ld, int rows, int K, float* partials, int max_partials, cudaStream_t stream);

// Quantize x [rows, K] bf16 into packed [rows, K/2] e2m1 and the SFA (is_sfb = 0) or SFB (is_sfb = 1) scale layout of a
// GEMM whose A / B operand it is. amax = max of partials[0 .. n_partials) (device). gs = amax / 2688 (1 for an all-zero
// tensor). gs_dev (optional) receives gs; for i < n_alpha, alpha_dev[i] = gs * gs_other[i] (the GEMM's alpha when the
// other operand was quantized with per-tensor scale gs_other[i]). K % 16 == 0; x and packed 16-byte aligned.
int alpamayo15_nvfp4_quantize_rows_bf16(const void* x, int ld, int rows, int K, const float* partials, int n_partials,
                                        void* packed, void* sf, int is_sfb, float* gs_dev, const float* gs_other, int n_alpha,
                                        float* alpha_dev, cudaStream_t stream);

// Bytes of the scale-factor buffer of a [rows, K] operand (rows padded to 128, K/16 blocks padded to 4).
size_t alpamayo15_nvfp4_sf_bytes(int rows, int K);


// Per-row scales (activations, the GEMM's A operand; SFA layout). Each entry: one CTA per row of 256 threads, K <= 16384,
// K % 16 == 0, 16-byte aligned rows; the row's alphas gs_row * gs_other[j] (j < n_alpha <= 4) are written to
// alpha_rows[j * alpha_ld + row] (alpha_ld >= rows); gs_rows (optional) receives gs_row.
// x [rows, K] bf16 (rows ld elements apart, ld % 8 == 0) -> packed [rows, K/2] + sf.
int alpamayo15_nvfp4_quantize_rows_rowscale_bf16(const void* x, int ld, int rows, int K, void* packed, void* sf,
                                                 const float* gs_other, int n_alpha, float* alpha_rows, int alpha_ld,
                                                 float* gs_rows, cudaStream_t stream);
// residual [rows, dim] bf16 += delta (optional, in place), then RMSNorm(residual) * weight (Qwen3 rounding, the
// arithmetic of alpamayo15_rms_bf16 with the FP8-tier producer's reduction order) -> packed + sf. dim % 8 == 0.
int alpamayo15_nvfp4_rms_rows_bf16(void* residual, const void* delta, const void* weight, int rows, int dim, float eps,
                                   void* packed, void* sf, const float* gs_other, int n_alpha, float* alpha_rows, int alpha_ld,
                                   cudaStream_t stream);
// bf16(silu(gate)) * up (rows ld elements apart, F % 8 == 0; the rounding of alpamayo15_silu_mul_split_bf16) -> packed + sf.
int alpamayo15_nvfp4_silu_mul_rows_bf16(const void* gate, const void* up, int ld, int rows, int F, void* packed, void* sf,
                                        const float* gs_other, int n_alpha, float* alpha_rows, int alpha_ld, cudaStream_t stream);

}  // namespace fp4
}  // namespace flash_rt
