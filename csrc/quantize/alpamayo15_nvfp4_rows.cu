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


// ---------------------------------------------------------------------------------------------------------------------
// Per-row scales: one CTA (256 threads) per row held in shared memory, gs = row_amax / 2688 (1 for an all-zero row),
// block-16 scales e4m3(block_amax / 6 / gs), e2m1 codes as above; the row's GEMM alphas gs * gs_other[j] go to
// alpha_rows[j * alpha_ld + row] (the per-row alpha of the GEMM whose B operand carries gs_other[j]).

constexpr int kRowMax = 16384;

struct alignas(8) Bf4 { __nv_bfloat16 v[4]; };

__device__ __forceinline__ float bf(const __nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ __nv_bfloat16 to_bf(float v) { return __float2bfloat16_rn(v); }

__device__ __forceinline__ float block_sum(float v, float* smem) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  const int nw = (blockDim.x + 31) >> 5;
  float t = (threadIdx.x < nw) ? smem[threadIdx.x] : 0.0f;
  if (warp == 0) {
    for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
    if (lane == 0) smem[0] = t;
  }
  __syncthreads();
  const float r = smem[0];
  __syncthreads();
  return r;
}

__device__ __forceinline__ float block_max(float v, float* smem) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  const int nw = (blockDim.x + 31) >> 5;
  float t = (threadIdx.x < nw) ? smem[threadIdx.x] : 0.0f;
  if (warp == 0) {
    for (int o = 16; o > 0; o >>= 1) t = fmaxf(t, __shfl_xor_sync(0xffffffffu, t, o));
    if (lane == 0) smem[0] = t;
  }
  __syncthreads();
  const float r = smem[0];
  __syncthreads();
  return r;
}

template <class LayoutSF>
__device__ __forceinline__ void quantize_row_nvfp4(const __nv_bfloat16* srow, int K, int row, uint2* __restrict__ packed,
                                                   uint8_t* __restrict__ sf, const LayoutSF& layout, const float* __restrict__ gs_other,
                                                   int n_alpha, float* __restrict__ alpha_rows, int alpha_ld,
                                                   float* __restrict__ gs_rows, float* smem) {
  const Bf4* s4 = reinterpret_cast<const Bf4*>(srow);
  float mx = 0.0f;
  for (int i = threadIdx.x; i < (K >> 2); i += blockDim.x) {
    const Bf4 t = s4[i];
#pragma unroll
    for (int e = 0; e < 4; ++e) mx = fmaxf(mx, fabsf(bf(t.v[e])));
  }
  const float amax = block_max(mx, smem);
  const float gs = amax > 0.0f ? amax / 2688.0f : 1.0f;
  if (threadIdx.x == 0) {
    for (int j = 0; j < n_alpha; ++j) alpha_rows[(size_t)j * alpha_ld + row] = gs * gs_other[j];
    if (gs_rows) gs_rows[row] = gs;
  }
  const int nb = K >> 4;
  for (int blk = threadIdx.x; blk < nb; blk += blockDim.x) {
    const uint4 r0 = *reinterpret_cast<const uint4*>(srow + blk * 16), r1 = *reinterpret_cast<const uint4*>(srow + blk * 16 + 8);
    const __nv_bfloat16* h0 = reinterpret_cast<const __nv_bfloat16*>(&r0);
    const __nv_bfloat16* h1 = reinterpret_cast<const __nv_bfloat16*>(&r1);
    float v[16];
    float bmax = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) { v[i] = bf(h0[i]); v[8 + i] = bf(h1[i]); }
#pragma unroll
    for (int i = 0; i < 16; ++i) bmax = fmaxf(bmax, fabsf(v[i]));
    const __nv_fp8_e4m3 bs = __nv_fp8_e4m3(fminf(bmax / (6.0f * gs), 448.0f));
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
}

// x [rows, K] bf16 rows (ld elements apart) -> per-row-scaled NVFP4 (the attention output of the W4A4 prefill tier).
template <class LayoutSF>
__global__ void __launch_bounds__(256) nvfp4_rows_rowscale_kernel(const __nv_bfloat16* __restrict__ x, int ld, int K,
                                                                  uint2* __restrict__ packed, uint8_t* __restrict__ sf, LayoutSF layout,
                                                                  const float* __restrict__ gs_other, int n_alpha,
                                                                  float* __restrict__ alpha_rows, int alpha_ld, float* __restrict__ gs_rows) {
  __shared__ float smem[32];
  __shared__ __align__(16) __nv_bfloat16 srow[kRowMax];
  const int row = blockIdx.x;
  const uint4* src = reinterpret_cast<const uint4*>(x + (size_t)row * ld);
  uint4* dst = reinterpret_cast<uint4*>(srow);
  for (int i = threadIdx.x; i < (K >> 3); i += blockDim.x) dst[i] = src[i];
  __syncthreads();
  quantize_row_nvfp4(srow, K, row, packed, sf, layout, gs_other, n_alpha, alpha_rows, alpha_ld, gs_rows, smem);
}

// Qwen3 RMSNorm on BF16 activations with an optional BF16 residual add (res += delta, in place), the normalized row
// quantized to per-row-scaled NVFP4 instead of a BF16 output. Per-element arithmetic of the engine's rms_bf16 kernel
// (BF16 residual update, BF16 rounding of the normalized value and of the weight product); the row is read as 8-byte
// vectors, thread t owning channels 4t..4t+3, 4(t+B).., and the FP32 sum of squares accumulated in that order (the
// FP8-tier producer's order; the separate rms_bf16 kernel sums channel t, t+B, .. and can differ in the scale's last
// bit). dim % 8 == 0, dim <= kRowMax.
template <class LayoutSF>
__global__ void __launch_bounds__(256) nvfp4_rms_rows_kernel(__nv_bfloat16* __restrict__ res, const __nv_bfloat16* __restrict__ delta,
                                                             const __nv_bfloat16* __restrict__ w, int dim, float eps,
                                                             uint2* __restrict__ packed, uint8_t* __restrict__ sf, LayoutSF layout,
                                                             const float* __restrict__ gs_other, int n_alpha,
                                                             float* __restrict__ alpha_rows, int alpha_ld) {
  __shared__ float smem[32];
  __shared__ __align__(16) __nv_bfloat16 srow[kRowMax];
  const int row = blockIdx.x, n4 = dim >> 2;
  Bf4* r4 = reinterpret_cast<Bf4*>(res + (size_t)row * dim);
  const Bf4* d4 = delta ? reinterpret_cast<const Bf4*>(delta + (size_t)row * dim) : nullptr;
  float ss = 0.0f;
  for (int i = threadIdx.x; i < n4; i += 256) {
    Bf4 v = r4[i];
    if (d4) {
      const Bf4 d = d4[i];
#pragma unroll
      for (int e = 0; e < 4; ++e) v.v[e] = to_bf(__fadd_rn(bf(v.v[e]), bf(d.v[e])));
      r4[i] = v;
    }
#pragma unroll
    for (int e = 0; e < 4; ++e) { const float f = bf(v.v[e]); ss += __fmul_rn(f, f); }
  }
  const float inv = rsqrtf(__fadd_rn(block_sum(ss, smem) / (float)dim, eps));
  const Bf4* w4 = reinterpret_cast<const Bf4*>(w);
  Bf4* s4 = reinterpret_cast<Bf4*>(srow);
  for (int i = threadIdx.x; i < n4; i += 256) {
    const Bf4 v = r4[i], ww = w4[i];
    Bf4 o;
#pragma unroll
    for (int e = 0; e < 4; ++e) o.v[e] = to_bf(__fmul_rn(bf(ww.v[e]), bf(to_bf(__fmul_rn(bf(v.v[e]), inv)))));
    s4[i] = o;
  }
  __syncthreads();
  quantize_row_nvfp4(srow, dim, row, packed, sf, layout, gs_other, n_alpha, alpha_rows, alpha_ld, nullptr, smem);
}

// bf16(silu(bf16 gate)) * bf16 up (the rounding of the engine's split SiLU-mul kernel) quantized to per-row-scaled
// NVFP4; gate / up rows ld elements apart. F % 8 == 0, F <= kRowMax.
template <class LayoutSF>
__global__ void __launch_bounds__(256) nvfp4_silu_mul_rows_kernel(const __nv_bfloat16* __restrict__ gate,
                                                                  const __nv_bfloat16* __restrict__ up, int ld, int F,
                                                                  uint2* __restrict__ packed, uint8_t* __restrict__ sf, LayoutSF layout,
                                                                  const float* __restrict__ gs_other, int n_alpha,
                                                                  float* __restrict__ alpha_rows, int alpha_ld) {
  __shared__ float smem[32];
  __shared__ __align__(16) __nv_bfloat16 srow[kRowMax];
  const int row = blockIdx.x, n4 = F >> 2, B = blockDim.x;
  const Bf4* g4 = reinterpret_cast<const Bf4*>(gate + (size_t)row * ld);
  const Bf4* u4 = reinterpret_cast<const Bf4*>(up + (size_t)row * ld);
  Bf4* s4 = reinterpret_cast<Bf4*>(srow);
  auto silu4 = [&](const Bf4 g, const Bf4 u) {
    Bf4 o;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float gf = bf(g.v[e]);
      const float sl = bf(to_bf(gf / (1.0f + expf(-gf))));
      o.v[e] = to_bf(__fmul_rn(sl, bf(u.v[e])));
    }
    return o;
  };
  int i = threadIdx.x;
  for (; i + 3 * B < n4; i += 4 * B) {                // four gate / up vectors in flight per thread
    const Bf4 g0 = g4[i], g1 = g4[i + B], g2 = g4[i + 2 * B], g3 = g4[i + 3 * B];
    const Bf4 u0 = u4[i], u1 = u4[i + B], u2 = u4[i + 2 * B], u3 = u4[i + 3 * B];
    s4[i] = silu4(g0, u0); s4[i + B] = silu4(g1, u1); s4[i + 2 * B] = silu4(g2, u2); s4[i + 3 * B] = silu4(g3, u3);
  }
  for (; i < n4; i += B) s4[i] = silu4(g4[i], u4[i]);
  __syncthreads();
  quantize_row_nvfp4(srow, F, row, packed, sf, layout, gs_other, n_alpha, alpha_rows, alpha_ld, nullptr, smem);
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


namespace {

// Argument checks shared by the per-row-scale entries: the row buffer bound, the vector widths, the alpha targets.
int check_rows(const void* x, int rows, int K, const void* packed, const void* sf, const float* gs_other, int n_alpha,
               const float* alpha_rows, int alpha_ld) {
  if (!x || !packed || !sf) return -1;
  if (rows <= 0 || K <= 0 || (K % 16) != 0 || K > kRowMax || n_alpha < 0 || n_alpha > 4) return -2;
  if (n_alpha > 0 && (!gs_other || !alpha_rows || alpha_ld < rows)) return -1;
  if ((reinterpret_cast<uintptr_t>(x) & 15u) || (reinterpret_cast<uintptr_t>(packed) & 7u)) return -5;
  return 0;
}

}  // namespace

int alpamayo15_nvfp4_quantize_rows_rowscale_bf16(const void* x, int ld, int rows, int K, void* packed, void* sf,
                                                 const float* gs_other, int n_alpha, float* alpha_rows, int alpha_ld,
                                                 float* gs_rows, cudaStream_t stream) {
  const int rc = check_rows(x, rows, K, packed, sf, gs_other, n_alpha, alpha_rows, alpha_ld);
  if (rc) return rc;
  if (ld < K || (ld % 8) != 0) return -2;
  auto layout = Cfg16::tile_atom_to_shape_SFA(cute::make_shape(rows, 1, K, 1));
  nvfp4_rows_rowscale_kernel<<<rows, 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(x), ld, K, static_cast<uint2*>(packed),
                                                       static_cast<uint8_t*>(sf), layout, gs_other, n_alpha, alpha_rows, alpha_ld, gs_rows);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : -4;
}

int alpamayo15_nvfp4_rms_rows_bf16(void* residual, const void* delta, const void* weight, int rows, int dim, float eps,
                                   void* packed, void* sf, const float* gs_other, int n_alpha, float* alpha_rows, int alpha_ld,
                                   cudaStream_t stream) {
  const int rc = check_rows(residual, rows, dim, packed, sf, gs_other, n_alpha, alpha_rows, alpha_ld);
  if (rc) return rc;
  if (!weight || (dim % 8) != 0) return -2;
  if ((reinterpret_cast<uintptr_t>(delta) & 7u) || (reinterpret_cast<uintptr_t>(weight) & 7u)) return -5;
  auto layout = Cfg16::tile_atom_to_shape_SFA(cute::make_shape(rows, 1, dim, 1));
  nvfp4_rms_rows_kernel<<<rows, 256, 0, stream>>>(static_cast<__nv_bfloat16*>(residual), static_cast<const __nv_bfloat16*>(delta),
                                                  static_cast<const __nv_bfloat16*>(weight), dim, eps, static_cast<uint2*>(packed),
                                                  static_cast<uint8_t*>(sf), layout, gs_other, n_alpha, alpha_rows, alpha_ld);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : -4;
}

int alpamayo15_nvfp4_silu_mul_rows_bf16(const void* gate, const void* up, int ld, int rows, int F, void* packed, void* sf,
                                        const float* gs_other, int n_alpha, float* alpha_rows, int alpha_ld, cudaStream_t stream) {
  const int rc = check_rows(gate, rows, F, packed, sf, gs_other, n_alpha, alpha_rows, alpha_ld);
  if (rc) return rc;
  if (!up || ld < F || (ld % 8) != 0 || (F % 8) != 0) return -2;
  if (reinterpret_cast<uintptr_t>(up) & 7u) return -5;
  auto layout = Cfg16::tile_atom_to_shape_SFA(cute::make_shape(rows, 1, F, 1));
  nvfp4_silu_mul_rows_kernel<<<rows, 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(gate), static_cast<const __nv_bfloat16*>(up), ld, F,
                                                       static_cast<uint2*>(packed), static_cast<uint8_t*>(sf), layout, gs_other, n_alpha,
                                                       alpha_rows, alpha_ld);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : -4;
}

}  // namespace fp4
}  // namespace flash_rt
