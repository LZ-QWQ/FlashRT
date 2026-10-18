// SPDX-License-Identifier: Apache-2.0
// NVFP4 row quantization for the Alpamayo 1.5 prefill tier (SM100/SM110): bf16 rows -> e2m1 nibbles + UE4M3 block-16
// scales in the CUTLASS Sm1xx tile-interleaved layout, with a per-tensor scale gs = amax / (6 * 448) that places the
// block scales in the E4M3 normal range (block scale = e4m3(block_amax / 6 / gs), value = e2m1(x / (scale * gs))).
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

}  // namespace fp4
}  // namespace flash_rt
