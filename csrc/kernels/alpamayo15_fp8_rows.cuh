// SPDX-License-Identifier: Apache-2.0
//
// Per-row FP8 E4M3 quantization of a BF16 row held in shared memory, for the fused producers of the Alpamayo 1.5
// W8A8 tiers: scale = absmax / 448 (1 for an all-zero row), codes = round-to-nearest-even(x / scale) saturated to
// +-448 -- the same codes and scales as alpamayo15_quantize_fp8_rows_bf16 on the BF16 row. One CTA per row; dim % 4 == 0;
// smem holds >= 32 floats.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace flash_rt {
namespace kernels {
namespace fp8rows {

struct alignas(8) Bf4 { __nv_bfloat16 v[4]; };

__device__ __forceinline__ void quantize_row_fp8(const __nv_bfloat16* srow, uint8_t* __restrict__ q, float* __restrict__ scale,
                                                 int row, int dim, float* smem) {
  float mx = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) mx = fmaxf(mx, fabsf(__bfloat162float(srow[i])));
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (lane == 0) smem[warp] = mx;
  __syncthreads();
  if (warp == 0) {
    float t = threadIdx.x < ((blockDim.x + 31) >> 5) ? smem[threadIdx.x] : 0.0f;
    for (int o = 16; o > 0; o >>= 1) t = fmaxf(t, __shfl_xor_sync(0xffffffffu, t, o));
    if (lane == 0) smem[0] = t;
  }
  __syncthreads();
  const float amax = smem[0];
  const float s = amax > 0.0f ? amax / 448.0f : 1.0f;
  if (threadIdx.x == 0) scale[row] = s;
  uint32_t* qr = reinterpret_cast<uint32_t*>(q + (size_t)row * dim);
  const Bf4* s4 = reinterpret_cast<const Bf4*>(srow);
  for (int i = threadIdx.x; i < (dim >> 2); i += blockDim.x) {
    const Bf4 t = s4[i];
    float f[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) f[e] = fminf(fmaxf(__fdiv_rn(__bfloat162float(t.v[e]), s), -448.0f), 448.0f);
    const __nv_fp8x2_storage_t lo = __nv_cvt_float2_to_fp8x2(make_float2(f[0], f[1]), __NV_SATFINITE, __NV_E4M3);
    const __nv_fp8x2_storage_t hi = __nv_cvt_float2_to_fp8x2(make_float2(f[2], f[3]), __NV_SATFINITE, __NV_E4M3);
    qr[i] = (uint32_t)lo | ((uint32_t)hi << 16);
  }
  __syncthreads();
}

}  // namespace fp8rows
}  // namespace kernels
}  // namespace flash_rt
