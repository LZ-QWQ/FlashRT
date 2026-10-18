// SPDX-License-Identifier: Apache-2.0
// See alpamayo15_nvfp4_rows.cuh.

#include "alpamayo15_nvfp4_rows.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cute/tensor.hpp"

#include <cstdint>

namespace flash_rt {
namespace fp4 {

namespace {

using Cfg16 = cutlass::detail::Sm1xxBlockScaledConfig<16>;

__global__ void __launch_bounds__(256) nvfp4_absmax_kernel(const __nv_bfloat16* __restrict__ x, int ld, int rows, int K,
                                                           float* __restrict__ partials) {
  __shared__ float red[8];
  const long long n8 = (long long)rows * (K >> 3);
  float mx = 0.0f;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n8; i += (long long)gridDim.x * blockDim.x) {
    const int r = (int)(i / (K >> 3)), c = (int)(i - (long long)r * (K >> 3)) * 8;
    const uint4 raw = *reinterpret_cast<const uint4*>(x + (size_t)r * ld + c);
    const __nv_bfloat16* v = reinterpret_cast<const __nv_bfloat16*>(&raw);
#pragma unroll
    for (int e = 0; e < 8; ++e) mx = fmaxf(mx, fabsf(__bfloat162float(v[e])));
  }
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = mx;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = red[0];
    for (int w = 1; w < 8; ++w) t = fmaxf(t, red[w]);
    partials[blockIdx.x] = t;
  }
}

// Eight FP32 values -> eight e2m1 codes (element 0 in the low nibble of byte 0), round to nearest even, saturating
// (the form of cutlass::NumericArrayConverter<float_e2m1_t, float, 8>).
__device__ __forceinline__ unsigned cvt_e2m1x8(const float* v) {
  unsigned out;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(out) : "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]));
  return out;
}

template <class LayoutSF>
__global__ void __launch_bounds__(256) nvfp4_quantize_kernel(const __nv_bfloat16* __restrict__ x, int ld, int rows, int K,
                                                             const float* __restrict__ partials, int n_partials,
                                                             uint2* __restrict__ packed, uint8_t* __restrict__ sf, LayoutSF layout,
                                                             float* __restrict__ gs_dev, const float* __restrict__ gs_other, int n_alpha,
                                                             float* __restrict__ alpha_dev) {
  __shared__ float s_gs;
  if (threadIdx.x < 32) {
    float mx = 0.0f;
    for (int i = threadIdx.x; i < n_partials; i += 32) mx = fmaxf(mx, partials[i]);
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    if (threadIdx.x == 0) {
      const float gs = mx > 0.0f ? mx / 2688.0f : 1.0f;            // 6 (e2m1 max) x 448 (e4m3 max)
      s_gs = gs;
      if (blockIdx.x == 0 && blockIdx.y == 0) {
        if (gs_dev) *gs_dev = gs;
        for (int i = 0; i < n_alpha; ++i) alpha_dev[i] = gs * gs_other[i];
      }
    }
  }
  __syncthreads();
  const float gs = s_gs;
  const int nb = K >> 4;
  const int blk = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
  if (row >= rows || blk >= nb) return;
  const __nv_bfloat16* src = x + (size_t)row * ld + blk * 16;
  const uint4 r0 = *reinterpret_cast<const uint4*>(src), r1 = *reinterpret_cast<const uint4*>(src + 8);
  const __nv_bfloat16* h0 = reinterpret_cast<const __nv_bfloat16*>(&r0);
  const __nv_bfloat16* h1 = reinterpret_cast<const __nv_bfloat16*>(&r1);
  float v[16];
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) { v[i] = __bfloat162float(h0[i]); v[8 + i] = __bfloat162float(h1[i]); }
#pragma unroll
  for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(v[i]));
  const __nv_fp8_e4m3 bs = __nv_fp8_e4m3(fminf(amax / (6.0f * gs), 448.0f));
  const float bs_dq = static_cast<float>(bs);
  sf[layout(row, blk * 16, 0)] = *reinterpret_cast<const uint8_t*>(&bs);
  const float inv = bs_dq > 0.0f ? 1.0f / (bs_dq * gs) : 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) v[i] *= inv;
  uint2 out;
  out.x = cvt_e2m1x8(v);
  out.y = cvt_e2m1x8(v + 8);
  packed[(size_t)row * nb + blk] = out;
}

}  // namespace

size_t alpamayo15_nvfp4_sf_bytes(int rows, int K) {
  return (size_t)((rows + 127) / 128) * 128 * (size_t)((K / 16 + 3) / 4) * 4;
}

int alpamayo15_nvfp4_absmax_bf16(const void* x, int ld, int rows, int K, float* partials, int max_partials, cudaStream_t stream) {
  if (!x || !partials) return -1;
  if (rows <= 0 || K <= 0 || (K % 8) != 0 || ld < K || (ld % 8) != 0 || max_partials < 1) return -2;
  if (reinterpret_cast<uintptr_t>(x) & 15u) return -5;
  const long long n8 = (long long)rows * (K / 8);
  int blocks = (int)((n8 + 255) / 256);
  if (blocks > max_partials) blocks = max_partials;
  if (blocks > 1024) blocks = 1024;
  nvfp4_absmax_kernel<<<blocks, 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(x), ld, rows, K, partials);
  return cudaPeekAtLastError() == cudaSuccess ? blocks : -4;
}

int alpamayo15_nvfp4_quantize_rows_bf16(const void* x, int ld, int rows, int K, const float* partials, int n_partials,
                                        void* packed, void* sf, int is_sfb, float* gs_dev, const float* gs_other, int n_alpha,
                                        float* alpha_dev, cudaStream_t stream) {
  if (!x || !partials || !packed || !sf) return -1;
  if (rows <= 0 || K <= 0 || (K % 16) != 0 || ld < K || (ld % 8) != 0 || n_partials < 1 || n_alpha < 0 || n_alpha > 4) return -2;
  if (n_alpha > 0 && (!gs_other || !alpha_dev)) return -1;
  if ((reinterpret_cast<uintptr_t>(x) & 15u) || (reinterpret_cast<uintptr_t>(packed) & 7u)) return -5;
  const int nb = K / 16;
  const dim3 grid((nb + 255) / 256, rows);
  const auto shape = cute::make_shape(is_sfb ? 1 : rows, is_sfb ? rows : 1, K, 1);
  if (is_sfb) {
    auto layout = Cfg16::tile_atom_to_shape_SFB(shape);
    nvfp4_quantize_kernel<<<grid, 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(x), ld, rows, K, partials, n_partials,
                                                    static_cast<uint2*>(packed), static_cast<uint8_t*>(sf), layout, gs_dev, gs_other,
                                                    n_alpha, alpha_dev);
  } else {
    auto layout = Cfg16::tile_atom_to_shape_SFA(shape);
    nvfp4_quantize_kernel<<<grid, 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(x), ld, rows, K, partials, n_partials,
                                                    static_cast<uint2*>(packed), static_cast<uint8_t*>(sf), layout, gs_dev, gs_other,
                                                    n_alpha, alpha_dev);
  }
  return cudaPeekAtLastError() == cudaSuccess ? 0 : -4;
}

}  // namespace fp4
}  // namespace flash_rt
