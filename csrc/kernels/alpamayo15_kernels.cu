// SPDX-License-Identifier: Apache-2.0
//
// Alpamayo 1.5 (Thor sm_110) model-local kernels. See alpamayo15_kernels.cuh
// for the rounding contract each kernel reproduces.

#include "alpamayo15_kernels.cuh"
#include "alpamayo15_flashdrive_kernels.cuh"   // alpamayo15_pdl(): the process-wide programmatic-dependent-launch switch
#include "alpamayo15_fp8_rows.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace flash_rt {
namespace kernels {

namespace {

constexpr int kHeadDim = 128;

inline bool aligned8(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 7u) == 0; }
inline bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

__device__ __forceinline__ float bf(const __nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ __nv_bfloat16 to_bf(float v) { return __float2bfloat16_rn(v); }

// Programmatic dependent launch (griddepcontrol): every kernel waits for its predecessor's completion before touching
// memory it may produce and lets its successor's prologue start early; launches carry the attribute only when the
// process-wide switch is on.
__device__ __forceinline__ void a15_pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
__device__ __forceinline__ void a15_pdl_trigger() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}
__device__ __forceinline__ void a15_pdl() { a15_pdl_wait(); a15_pdl_trigger(); }

template <typename... KArgs, typename... Args>
inline cudaError_t a15_launch(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t stream,
                              Args&&... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = alpamayo15_pdl() ? 1 : 0;
  return cudaLaunchKernelEx(&cfg, kernel, std::forward<Args>(args)...);
}

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

// Four consecutive BF16 values (one 8-byte access).
struct alignas(8) Bf4 { __nv_bfloat16 v[4]; };

// One warp per (row, head): rotate q or k in place. Lane l holds elements
// [4l, 4l+4); the rotate-half partner of lane l lives in lane l +/- D/8.
__global__ void vision_rope_qk_kernel(__nv_bfloat16* __restrict__ qkv,
                                      const float* __restrict__ cos,
                                      const float* __restrict__ sin,
                                      int S, int H, int D) {
  // One thread per pair of 4-element vectors (j, j + D/8) of a q or k head row: both halves of rotate-half are read
  // and written by the same thread (no idle lanes, no shuffles), D/8 threads per head row.
  a15_pdl();
  const int hl = D >> 3;
  const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const long long gw = t / hl;
  if (gw >= (long long)S * 2 * H) return;
  const int j = (int)(t - gw * hl);
  const int row = (int)(gw / (2 * H)), head = (int)(gw % (2 * H));  // heads 0..H-1 = q, H..2H-1 = k
  __nv_bfloat16* p = qkv + (size_t)row * 3 * H * D + (size_t)head * D;
  const float* c = cos + (size_t)row * D;
  const float* sn = sin + (size_t)row * D;
  const int lo = 4 * j, hi = lo + (D >> 1);
  const Bf4 xl = *reinterpret_cast<const Bf4*>(p + lo);
  const Bf4 xh = *reinterpret_cast<const Bf4*>(p + hi);
  const float4 cl = *reinterpret_cast<const float4*>(c + lo), ch = *reinterpret_cast<const float4*>(c + hi);
  const float4 sl = *reinterpret_cast<const float4*>(sn + lo), sh = *reinterpret_cast<const float4*>(sn + hi);
  const float cla[4] = {cl.x, cl.y, cl.z, cl.w}, cha[4] = {ch.x, ch.y, ch.z, ch.w};
  const float sla[4] = {sl.x, sl.y, sl.z, sl.w}, sha[4] = {sh.x, sh.y, sh.z, sh.w};
  Bf4 ol, oh;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float a = bf(xl.v[i]), b = bf(xh.v[i]);
    ol.v[i] = to_bf(__fadd_rn(__fmul_rn(a, cla[i]), __fmul_rn(-b, sla[i])));
    oh.v[i] = to_bf(__fadd_rn(__fmul_rn(b, cha[i]), __fmul_rn(a, sha[i])));
  }
  *reinterpret_cast<Bf4*>(p + lo) = ol;
  *reinterpret_cast<Bf4*>(p + hi) = oh;
}
// Row passes of the RMSNorm kernels. Thread t owns channels t, t + blockDim, ... in that order (its partial sum is
// sequential over them, as before); the loads of four channels are issued before their use so that more bytes are in
// flight per thread.
__device__ __forceinline__ void rms_bf16_accumulate(__nv_bfloat16* __restrict__ r, const __nv_bfloat16* __restrict__ dl,
                                                    int dim, float& ss) {
  const int B = blockDim.x;
  int i = threadIdx.x;
  for (; i + 3 * B < dim; i += 4 * B) {
    __nv_bfloat16 v0 = r[i], v1 = r[i + B], v2 = r[i + 2 * B], v3 = r[i + 3 * B];
    if (dl) {
      const __nv_bfloat16 d0 = dl[i], d1 = dl[i + B], d2 = dl[i + 2 * B], d3 = dl[i + 3 * B];
      v0 = to_bf(__fadd_rn(bf(v0), bf(d0))); v1 = to_bf(__fadd_rn(bf(v1), bf(d1)));
      v2 = to_bf(__fadd_rn(bf(v2), bf(d2))); v3 = to_bf(__fadd_rn(bf(v3), bf(d3)));
      r[i] = v0; r[i + B] = v1; r[i + 2 * B] = v2; r[i + 3 * B] = v3;
    }
    const float f0 = bf(v0), f1 = bf(v1), f2 = bf(v2), f3 = bf(v3);
    ss += __fmul_rn(f0, f0); ss += __fmul_rn(f1, f1); ss += __fmul_rn(f2, f2); ss += __fmul_rn(f3, f3);
  }
  for (; i < dim; i += B) {
    __nv_bfloat16 v = r[i];
    if (dl) {
      v = to_bf(__fadd_rn(bf(v), bf(dl[i])));
      r[i] = v;
    }
    const float f = bf(v);
    ss += __fmul_rn(f, f);
  }
}
template <class Store>
__device__ __forceinline__ void rms_bf16_normalize(const __nv_bfloat16* __restrict__ r, const __nv_bfloat16* __restrict__ w,
                                                   int dim, float inv, Store store) {
  const int B = blockDim.x;
  int i = threadIdx.x;
  for (; i + 3 * B < dim; i += 4 * B) {
    const __nv_bfloat16 r0 = r[i], r1 = r[i + B], r2 = r[i + 2 * B], r3 = r[i + 3 * B];
    const __nv_bfloat16 w0 = w[i], w1 = w[i + B], w2 = w[i + 2 * B], w3 = w[i + 3 * B];
    store(i, to_bf(__fmul_rn(bf(w0), bf(to_bf(__fmul_rn(bf(r0), inv))))));
    store(i + B, to_bf(__fmul_rn(bf(w1), bf(to_bf(__fmul_rn(bf(r1), inv))))));
    store(i + 2 * B, to_bf(__fmul_rn(bf(w2), bf(to_bf(__fmul_rn(bf(r2), inv))))));
    store(i + 3 * B, to_bf(__fmul_rn(bf(w3), bf(to_bf(__fmul_rn(bf(r3), inv))))));
  }
  for (; i < dim; i += B) store(i, to_bf(__fmul_rn(bf(w[i]), bf(to_bf(__fmul_rn(bf(r[i]), inv))))));
}
__device__ __forceinline__ void res_rms_f32_accumulate(float* __restrict__ r, const __nv_bfloat16* __restrict__ dl, int dim,
                                                       float& ss) {
  const int B = blockDim.x;
  int i = threadIdx.x;
  for (; i + 3 * B < dim; i += 4 * B) {
    float v0 = r[i], v1 = r[i + B], v2 = r[i + 2 * B], v3 = r[i + 3 * B];
    if (dl) {
      const __nv_bfloat16 d0 = dl[i], d1 = dl[i + B], d2 = dl[i + 2 * B], d3 = dl[i + 3 * B];
      v0 = __fadd_rn(v0, bf(d0)); v1 = __fadd_rn(v1, bf(d1)); v2 = __fadd_rn(v2, bf(d2)); v3 = __fadd_rn(v3, bf(d3));
      r[i] = v0; r[i + B] = v1; r[i + 2 * B] = v2; r[i + 3 * B] = v3;
    }
    ss += __fmul_rn(v0, v0); ss += __fmul_rn(v1, v1); ss += __fmul_rn(v2, v2); ss += __fmul_rn(v3, v3);
  }
  for (; i < dim; i += B) {
    float v = r[i];
    if (dl) {
      v = __fadd_rn(v, bf(dl[i]));
      r[i] = v;
    }
    ss += __fmul_rn(v, v);
  }
}
template <class Store>
__device__ __forceinline__ void res_rms_f32_normalize(const float* __restrict__ r, const __nv_bfloat16* __restrict__ w, int dim,
                                                      float inv, Store store) {
  const int B = blockDim.x;
  int i = threadIdx.x;
  for (; i + 3 * B < dim; i += 4 * B) {
    const float r0 = r[i], r1 = r[i + B], r2 = r[i + 2 * B], r3 = r[i + 3 * B];
    const __nv_bfloat16 w0 = w[i], w1 = w[i + B], w2 = w[i + 2 * B], w3 = w[i + 3 * B];
    store(i, to_bf(__fmul_rn(bf(w0), __fmul_rn(r0, inv))));
    store(i + B, to_bf(__fmul_rn(bf(w1), __fmul_rn(r1, inv))));
    store(i + 2 * B, to_bf(__fmul_rn(bf(w2), __fmul_rn(r2, inv))));
    store(i + 3 * B, to_bf(__fmul_rn(bf(w3), __fmul_rn(r3, inv))));
  }
  for (; i < dim; i += B) store(i, to_bf(__fmul_rn(bf(w[i]), __fmul_rn(r[i], inv))));
}


// One block per row: residual (+= delta) and RMSNorm -> BF16.
__global__ void res_rms_f32_bf16_kernel(float* __restrict__ res,
                                        const __nv_bfloat16* __restrict__ delta,
                                        const __nv_bfloat16* __restrict__ w,
                                        __nv_bfloat16* __restrict__ out,
                                        int dim, float eps) {
  a15_pdl();
  __shared__ float smem[32];
  const int row = blockIdx.x;
  float* r = res + (size_t)row * dim;
  const __nv_bfloat16* dl = delta ? delta + (size_t)row * dim : nullptr;
  float ss = 0.0f;
  res_rms_f32_accumulate(r, dl, dim, ss);
  const float inv = rsqrtf(__fadd_rn(block_sum(ss, smem) / (float)dim, eps));
  __nv_bfloat16* o = out + (size_t)row * dim;
  res_rms_f32_normalize(r, w, dim, inv, [&](int i, __nv_bfloat16 v) { o[i] = v; });
}

// One block per row: optional BF16 residual add, then Qwen3 RMSNorm on BF16 activations.
__global__ void rms_bf16_kernel(__nv_bfloat16* __restrict__ res,
                                const __nv_bfloat16* __restrict__ delta,
                                const __nv_bfloat16* __restrict__ w,
                                __nv_bfloat16* __restrict__ out,
                                int dim, float eps) {
  a15_pdl();
  __shared__ float smem[32];
  const int row = blockIdx.x;
  __nv_bfloat16* r = res + (size_t)row * dim;
  const __nv_bfloat16* dl = delta ? delta + (size_t)row * dim : nullptr;
  float ss = 0.0f;
  rms_bf16_accumulate(r, dl, dim, ss);
  const float inv = rsqrtf(__fadd_rn(block_sum(ss, smem) / (float)dim, eps));
  __nv_bfloat16* o = out + (size_t)row * dim;
  rms_bf16_normalize(r, w, dim, inv, [&](int i, __nv_bfloat16 v) { o[i] = v; });
}

constexpr int kFusedRowMax = 16384;
using fp8rows::quantize_row_fp8;

// rms_bf16_kernel whose normalized row goes to FP8 codes + a row scale instead of a BF16 output (the W8A8 prefill
// tier). Per-element arithmetic of rms_bf16 (BF16 residual update, BF16 rounding of the normalized value and of the
// weight product) with the row read and written as 8-byte vectors: thread t owns channels 4t..4t+3, 4(t+B).., so the
// FP32 sum of squares is accumulated in that order (the separate rms_bf16 kernel sums channel t, t+B, ..; the two can
// differ in the last bit of the scale). dim % 4 == 0, dim <= kFusedRowMax.
__global__ void __launch_bounds__(256) rms_fp8_rows_kernel(__nv_bfloat16* __restrict__ res,
                                                           const __nv_bfloat16* __restrict__ delta,
                                                           const __nv_bfloat16* __restrict__ w,
                                                           uint8_t* __restrict__ q, float* __restrict__ scale,
                                                           int dim, float eps) {
  a15_pdl_wait();
  __shared__ float smem[32];
  __shared__ __align__(16) __nv_bfloat16 srow[kFusedRowMax];
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
  quantize_row_fp8(srow, q, scale, row, dim, smem);
  a15_pdl_trigger();
}

// res_rms_f32_bf16_kernel whose normalized row goes to FP8 codes + a row scale (the W8A8 expert tier). dim <= kFusedRowMax.
__global__ void __launch_bounds__(256) res_rms_f32_fp8_rows_kernel(float* __restrict__ res,
                                                                   const __nv_bfloat16* __restrict__ delta,
                                                                   const __nv_bfloat16* __restrict__ w,
                                                                   uint8_t* __restrict__ q, float* __restrict__ scale,
                                                                   int dim, float eps) {
  a15_pdl_wait();
  __shared__ float smem[32];
  __shared__ __align__(16) __nv_bfloat16 srow[kFusedRowMax];
  const int row = blockIdx.x;
  float* r = res + (size_t)row * dim;
  const __nv_bfloat16* dl = delta ? delta + (size_t)row * dim : nullptr;
  float ss = 0.0f;
  res_rms_f32_accumulate(r, dl, dim, ss);
  const float inv = rsqrtf(__fadd_rn(block_sum(ss, smem) / (float)dim, eps));
  res_rms_f32_normalize(r, w, dim, inv, [&](int i, __nv_bfloat16 v) { srow[i] = v; });
  __syncthreads();
  quantize_row_fp8(srow, q, scale, row, dim, smem);
  a15_pdl_trigger();
}

// out = bf16(silu(bf16 gate)) * bf16 up (the rounding of the split SiLU-mul kernel) quantized per row to FP8 codes
// + a row scale; gate / up rows ld elements apart. F <= kFusedRowMax, F % 4 == 0.
__global__ void __launch_bounds__(256) silu_mul_fp8_rows_kernel(const __nv_bfloat16* __restrict__ gate,
                                                                const __nv_bfloat16* __restrict__ up, int ld,
                                                                uint8_t* __restrict__ q, float* __restrict__ scale,
                                                                int F) {
  a15_pdl_wait();
  __shared__ float smem[32];
  __shared__ __align__(16) __nv_bfloat16 srow[kFusedRowMax];
  const int row = blockIdx.x;
  const Bf4* g4 = reinterpret_cast<const Bf4*>(gate + (size_t)row * ld);
  const Bf4* u4 = reinterpret_cast<const Bf4*>(up + (size_t)row * ld);
  Bf4* s4 = reinterpret_cast<Bf4*>(srow);
  for (int i = threadIdx.x; i < (F >> 2); i += blockDim.x) {
    const Bf4 g = g4[i], u = u4[i];
    Bf4 o;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float gf = bf(g.v[e]);
      const float sl = bf(to_bf(gf / (1.0f + expf(-gf))));
      o.v[e] = to_bf(__fmul_rn(sl, bf(u.v[e])));
    }
    s4[i] = o;
  }
  __syncthreads();
  quantize_row_fp8(srow, q, scale, row, F, smem);
  a15_pdl_trigger();
}

// One warp per (row, head) over q | k | v heads of a fused projection row.
// Head RMSNorm (Qwen3: BF16 rounding of the normalized value and of the weight
// product), then rotate-half RoPE. kFp32Rope: FP32 tables and FP32 math with one
// BF16 rounding (expert under autocast); otherwise BF16 tables with each
// product and the sum rounded to BF16 (VLM). V heads are copied. K/V of row r
// go to cache row (r / rows_per_batch) * batch_stride + r % rows_per_batch + slot,
// slot = slot_ptr[slot_per_batch ? r / rows_per_batch : 0] or 0 when slot_ptr is null.
template <bool kFp32Rope>
__global__ void qk_prep_kernel(const __nv_bfloat16* __restrict__ qkv,
                               const __nv_bfloat16* __restrict__ qw,
                               const __nv_bfloat16* __restrict__ kw,
                               const void* __restrict__ cos_v,
                               const void* __restrict__ sin_v,
                               __nv_bfloat16* __restrict__ q_out,
                               __nv_bfloat16* __restrict__ k_cache,
                               __nv_bfloat16* __restrict__ v_cache,
                               const long long* __restrict__ slot_ptr,
                               int rows, int Hq, int Hk, int kv_row_stride,
                               int rows_per_batch, long long batch_stride, int slot_per_batch,
                               float eps) {
  a15_pdl();
  const int nh = Hq + 2 * Hk;
  const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (gw >= rows * nh) return;
  const int row = gw / nh, head = gw % nh;
  const __nv_bfloat16* src = qkv + (size_t)row * nh * kHeadDim + (size_t)head * kHeadDim + 4 * lane;
  const Bf4 x = *reinterpret_cast<const Bf4*>(src);
  const int batch = row / rows_per_batch;
  const long long dst_row = (long long)batch * batch_stride + row % rows_per_batch +
                            (slot_ptr ? slot_ptr[slot_per_batch ? batch : 0] : 0LL);
  if (head >= Hq + Hk) {
    *reinterpret_cast<Bf4*>(v_cache + dst_row * kv_row_stride + (size_t)(head - Hq - Hk) * kHeadDim + 4 * lane) = x;
    return;
  }
  const bool is_q = head < Hq;
  float ss = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) ss += __fmul_rn(bf(x.v[i]), bf(x.v[i]));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  const float inv = rsqrtf(__fadd_rn(ss / (float)kHeadDim, eps));
  const Bf4 w = *reinterpret_cast<const Bf4*>((is_q ? qw : kw) + 4 * lane);
  Bf4 out;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const __nv_bfloat16 n = to_bf(__fmul_rn(bf(w.v[i]), bf(to_bf(__fmul_rn(bf(x.v[i]), inv)))));
    const float other = __shfl_xor_sync(0xffffffffu, bf(n), 16);  // lane +/- 16 <-> d +/- 64
    const int d = 4 * lane + i;
    if (kFp32Rope) {
      const float c = static_cast<const float*>(cos_v)[(size_t)row * kHeadDim + d];
      const float s = static_cast<const float*>(sin_v)[(size_t)row * kHeadDim + d];
      const float rot = lane < 16 ? -other : other;
      out.v[i] = to_bf(__fadd_rn(__fmul_rn(bf(n), c), __fmul_rn(rot, s)));
    } else {
      const float c = bf(static_cast<const __nv_bfloat16*>(cos_v)[(size_t)row * kHeadDim + d]);
      const float s = bf(static_cast<const __nv_bfloat16*>(sin_v)[(size_t)row * kHeadDim + d]);
      const float rot = lane < 16 ? -other : other;  // negation of a BF16 value is exact
      const float a = bf(to_bf(__fmul_rn(bf(n), c)));
      const float b = bf(to_bf(__fmul_rn(rot, s)));
      out.v[i] = to_bf(__fadd_rn(a, b));
    }
  }
  __nv_bfloat16* dst = is_q ? q_out + (size_t)row * Hq * kHeadDim + (size_t)head * kHeadDim
                            : k_cache + dst_row * kv_row_stride + (size_t)(head - Hq) * kHeadDim;
  *reinterpret_cast<Bf4*>(dst + 4 * lane) = out;
}


// ---------------------------------------------------------------------------
// W8A16 skinny GEMM for the decode projections (opt-in FP8 weight tier).
// One warp owns RT 16-row output tiles over the full K; per 32-column chunk each
// lane loads its 16 fragment bytes (one coalesced 512-byte warp access per tile),
// widens e4m3 -> f16 -> f32 -> bf16 (exact for every e4m3 value) and issues
// mma.m16n8k16 with the activations as the 8-column B operand (NT column tiles
// cover up to 16 rows of x). The row scale multiplies the FP32 accumulators.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t e4m3x2_to_bf16x2(uint32_t v) {
  const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(v & 0xFFFFu), __NV_E4M3);
  const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h));
  const __nv_bfloat162 b = __float22bfloat162_rn(f);
  return *reinterpret_cast<const uint32_t*>(&b);
}

// Two INT8 codes (low byte first) -> bf16x2; |code| <= 127 is exact in bf16.
__device__ __forceinline__ uint32_t s8x2_to_bf16x2(uint32_t v) {
  const float lo = static_cast<float>(static_cast<signed char>(v & 0xFFu));
  const float hi = static_cast<float>(static_cast<signed char>((v >> 8) & 0xFFu));
  const __nv_bfloat162 b = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&b);
}

__device__ __forceinline__ void mma_bf16_16816(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                               uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Weights are read exactly once per call: bypass L1 allocation.
__device__ __forceinline__ uint4 ld_stream16(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

template <int NT, int RT, bool kInt8>
__global__ void __launch_bounds__(128) w8a16_gemm_kernel(const uint4* __restrict__ wfrag,
                                                         const float* __restrict__ scale,
                                                         const __nv_bfloat16* __restrict__ x,
                                                         __nv_bfloat16* __restrict__ y,
                                                         int M, int N, int K, int T) {
  a15_pdl_wait();
  const int lane = threadIdx.x & 31;
  const int tile0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * RT;
  if (tile0 >= T) return;
  const int KC = K >> 5;
  const int g = lane >> 2, c = lane & 3;
  float acc[RT][NT][4];
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[r][n][i] = 0.0f;
  const uint32_t* xrow[NT];
  bool xvalid[NT];
#pragma unroll
  for (int n = 0; n < NT; ++n) {
    const int m = n * 8 + g;
    xvalid[n] = m < M;
    xrow[n] = reinterpret_cast<const uint32_t*>(x + (size_t)(xvalid[n] ? m : 0) * K);
  }
  const uint4* wrow[RT];
#pragma unroll
  for (int r = 0; r < RT; ++r) wrow[r] = wfrag + (size_t)min(tile0 + r, T - 1) * KC * 32 + lane;
#pragma unroll 4
  for (int kc = 0; kc < KC; ++kc) {
    uint4 a[RT];
#pragma unroll
    for (int r = 0; r < RT; ++r) a[r] = ld_stream16(wrow[r] + (size_t)kc * 32);
    uint32_t b[NT][4];
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      if (xvalid[n]) {
        const uint32_t* xp = xrow[n] + kc * 16;
        b[n][0] = __ldg(xp + c); b[n][1] = __ldg(xp + 4 + c);
        b[n][2] = __ldg(xp + 8 + c); b[n][3] = __ldg(xp + 12 + c);
      } else {
        b[n][0] = b[n][1] = b[n][2] = b[n][3] = 0u;
      }
    }
#pragma unroll
    for (int r = 0; r < RT; ++r) {
      auto cvt = [](uint32_t v) { return kInt8 ? s8x2_to_bf16x2(v) : e4m3x2_to_bf16x2(v); };
      const uint32_t p0 = cvt(a[r].x), p1 = cvt(a[r].x >> 16);
      const uint32_t p2 = cvt(a[r].y), p3 = cvt(a[r].y >> 16);
      const uint32_t q0 = cvt(a[r].z), q1 = cvt(a[r].z >> 16);
      const uint32_t q2 = cvt(a[r].w), q3 = cvt(a[r].w >> 16);
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        mma_bf16_16816(acc[r][n], p0, p1, p2, p3, b[n][0], b[n][1]);
        mma_bf16_16816(acc[r][n], q0, q1, q2, q3, b[n][2], b[n][3]);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < RT; ++r) {
    const int tile = tile0 + r;
    if (tile >= T) break;
    const int lo = tile * 16 + g, hi = lo + 8;
    const float s_lo = lo < N ? scale[lo] : 0.0f, s_hi = hi < N ? scale[hi] : 0.0f;
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      const int m0 = n * 8 + 2 * c, m1 = m0 + 1;
      if (lo < N) {
        if (m0 < M) y[(size_t)m0 * N + lo] = to_bf(acc[r][n][0] * s_lo);
        if (m1 < M) y[(size_t)m1 * N + lo] = to_bf(acc[r][n][1] * s_lo);
      }
      if (hi < N) {
        if (m0 < M) y[(size_t)m0 * N + hi] = to_bf(acc[r][n][2] * s_hi);
        if (m1 < M) y[(size_t)m1 * N + hi] = to_bf(acc[r][n][3] * s_hi);
      }
    }
  }
  a15_pdl_trigger();
}

template <int NT, int RT>
void launch_w8a16(const void* wfrag, const float* scale, const void* x, void* y, int M, int N, int K, int fmt,
                  cudaStream_t stream) {
  constexpr int kWarps = 4;
  const int T = (N + 15) / 16;
  const int blocks = (T + kWarps * RT - 1) / (kWarps * RT);
  if (fmt == 1)
    a15_launch(w8a16_gemm_kernel<NT, RT, true>, dim3(blocks), dim3(kWarps * 32), (size_t)0, stream, 
        reinterpret_cast<const uint4*>(wfrag), scale, reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<__nv_bfloat16*>(y), M, N, K, T);
  else
    a15_launch(w8a16_gemm_kernel<NT, RT, false>, dim3(blocks), dim3(kWarps * 32), (size_t)0, stream, 
        reinterpret_cast<const uint4*>(wfrag), scale, reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<__nv_bfloat16*>(y), M, N, K, T);
}


// ---------------------------------------------------------------------------
// Grouped few-row W4A8 GEMM (decode tier): the ParoQuant linears that share one
// input (q|k|v, gate|up) in one launch. Marlin's integer arithmetic, as the
// FlashDrive few-row kernels:
//   out[m, n] = bf16( float( sum_g s[g, n] * sum_{k in g} x[m, k] * (w[k, n] - z[g, n]) ) * a_scale[m] )
// on mma.sync m16n8k32 (s8 x s8 -> s32); a 16-row weight tile is the A operand,
// the problem's INT8 activations (8-row tiles) the B operand. One warp per
// weight tile over the full K (KS warps may split a tile's groups; their int32
// partials are added in shared memory in a fixed order), W warps per CTA, PF
// weight groups in flight per warp through a register ring.
// Weight pack [tiles][K/128 groups][1088 B]: per tile and group the 16 zero
// points (u8), the 16 integer group scales (i16), 16 B of padding, then the
// 1 KB of nibbles of pack_w4a8_small (two 64-wide k chunks, 32 lanes x 16 B).
// The scales and zero points thus stream with the nibbles instead of being
// gathered from [K/128, N] tables (4 scattered sector reads per group).
// ---------------------------------------------------------------------------
constexpr int kW4Group = 128;
constexpr int kW4MaxProblems = 3;
constexpr int kW4Meta = 64;                 // bytes of zero points / scales / padding per tile and group
constexpr int kW4Block = kW4Meta + 1024;    // bytes per tile and group

struct W4GroupedArgs {
  const int8_t* xq[kW4MaxProblems];          // INT8 activations (separate quantization pass), or
  const float* a_scale[kW4MaxProblems];
  const __half* xr[kW4MaxProblems];          // rotated FP16 activations [rows, K] + per-group row absmax [rows, K/128] +
  const float* part[kW4MaxProblems];         //   input_global_scale, quantized in the kernel's prologue (kQuant)
  const float* igs[kW4MaxProblems];
  __nv_bfloat16* out[kW4MaxProblems];
  const float* col_scale[kW4MaxProblems];   // optional [N_i] fp32 per-output-channel factor (nullptr = none)
  int N[kW4MaxProblems];                    // packed output channels (multiple of 16)
  int Nv[kW4MaxProblems];                   // valid output channels written (<= N)
  int tile0[kW4MaxProblems + 1];    // prefix sums of the problems' 16-row tiles
  const unsigned char* w_pack;
  const unsigned char* pf_ptr;      // optional region to prefetch into the L2 during the weight stream (the next
  unsigned pf_bytes;                //   group's rotation tables: a cold read before a streaming kernel slows it by ~30%)
  int ld_out, M, K, n_problems;
};

// vLLM's per-token INT8 quantization arithmetic (Triton PTX), as the FlashDrive activation preparation
__device__ __forceinline__ float w4_div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float w4_round_half_away(float y) {
  float h = copysignf(0.5f, y), t;
  asm("add.rz.ftz.f32 %0, %1, %2;" : "=f"(t) : "f"(y), "f"(h));
  return truncf(t);
}

__device__ __forceinline__ void mma_s8_16832(int* c, const unsigned* a, const unsigned* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// kQuant: the CTA quantizes the M (<= 4) rows of its problem's rotated FP16 activations into shared memory first (the
// weight prefetch is issued before, so the pass hides behind the first DRAM latency), bit-exact with the separate
// quantization kernel; the B fragments then come from shared memory. Dynamic shared memory: M * K int8 + 8 floats.
template <int MT, int KS, int W, int PF, bool kQuant>
__global__ void __launch_bounds__(W * 32) w4a8_grouped_kernel(W4GroupedArgs a) {
  a15_pdl_wait();
  __shared__ int red[KS > 1 ? W : 1][MT * 4][32];
  extern __shared__ __align__(16) unsigned char w4_dyn[];
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3, warp = threadIdx.x >> 5;
  const int NT = a.tile0[a.n_problems];
  const int T = blockIdx.x * (W / KS) + warp / KS, ks_id = warp % KS;
  const bool valid = T < NT;
  int p = 0;
#pragma unroll
  for (int i = 1; i < kW4MaxProblems; ++i)
    if (i < a.n_problems && T >= a.tile0[i]) p = i;
  const int Tl = T - a.tile0[p], K = a.K, M = a.M;
  int8_t* xq_s = reinterpret_cast<int8_t*>(w4_dyn);                       // [M][K] (kQuant)
  float* a_scale_s = reinterpret_cast<float*>(w4_dyn + (size_t)M * K);    // [8]
  const int8_t* __restrict__ xq = kQuant ? xq_s : a.xq[p];
  const int G = K / kW4Group, GS = G / KS;
  const unsigned char* tb = a.w_pack + (size_t)(valid ? T : 0) * G * kW4Block;
  auto meta = [&](int gi, int part) { return reinterpret_cast<const uint4*>(tb + (size_t)gi * kW4Block) + part; };
  auto nib = [&](int gi, int c) { return reinterpret_cast<const uint4*>(tb + (size_t)gi * kW4Block + kW4Meta + 512 * c) + lane; };
  const int n0 = 16 * Tl + g, n1 = n0 + 8;
  int total[MT][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int i = 0; i < 4; ++i) total[mt][i] = 0;
  if (a.pf_bytes) {                        // spread the L2 prefetch of the region over the CTAs, 128 B per thread
    const unsigned lines = a.pf_bytes >> 7, per = (lines + gridDim.x - 1) / gridDim.x, l0 = blockIdx.x * per;
    const unsigned l1 = min(lines, l0 + per);
    for (unsigned l = l0 + threadIdx.x; l < l1; l += W * 32)
      asm volatile("prefetch.global.L2 [%0];" ::"l"(a.pf_ptr + (size_t)l * 128));
  }
  const int g0 = ks_id * GS;
  uint4 ring[PF][5];                       // per group in flight: two nibble chunks, zero points, two scale halves
  auto issue = [&](int gi, int i) {
    ring[i][0] = ld_stream16(nib(gi, 0));
    ring[i][1] = ld_stream16(nib(gi, 1));
    ring[i][2] = ld_stream16(meta(gi, 0));
    ring[i][3] = ld_stream16(meta(gi, 1));
    ring[i][4] = ld_stream16(meta(gi, 2));
  };
  if (valid) {
#pragma unroll
    for (int i = 0; i < PF; ++i) issue(g0 + i, i);
  }
  if (kQuant) {
    // The CTA's warps all belong to one problem (a tile never spans problems and a CTA's tiles are consecutive;
    // the host rejects groups whose tile counts would mix problems inside a CTA -- see the entry point).
    const int pc = p;
    const __half* __restrict__ xr = a.xr[pc];
    const float* __restrict__ part = a.part[pc];
    __shared__ float inv_s[8];
    if (threadIdx.x < 32 * M) {              // warp m: absmax of row m over its K/128 group partials
      const int m = threadIdx.x >> 5;
      float mx = 0.0f;
      for (int i = lane; i < G; i += 32) mx = fmaxf(mx, part[(size_t)m * G + i]);
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
      if (lane == 0) {
        const float absmax = fmaxf(mx, 1e-10f);
        inv_s[m] = w4_div_full(127.0f, absmax);
        a_scale_s[m] = __fmul_rn(w4_div_full(absmax, 127.0f), *a.igs[pc]);
      }
    }
    __syncthreads();
    const __half2* x2 = reinterpret_cast<const __half2*>(xr);
    char2* q2 = reinterpret_cast<char2*>(xq_s);
    const int half_k = K >> 1;
    for (int idx = threadIdx.x; idx < M * half_k; idx += W * 32) {
      const int m = idx / half_k, c = idx - m * half_k;
      const float2 f = __half22float2(x2[(size_t)m * half_k + c]);
      const float inv = inv_s[m];
      char2 o;
      o.x = static_cast<signed char>(static_cast<int>(w4_round_half_away(__fmul_rn(inv, f.x))));
      o.y = static_cast<signed char>(static_cast<int>(w4_round_half_away(__fmul_rn(inv, f.y))));
      q2[idx] = o;
    }
    __syncthreads();
  }
  if (valid) {
    const int gq = g >> 2, gr = 8 * (g & 3), gh = g >> 1, hs = 16 * (g & 1);
    for (int base = 0; base < GS; base += PF) {
#pragma unroll
      for (int i = 0; i < PF; ++i) {
        const int gl = base + i, gi = g0 + gl;
        const uint4 w0 = ring[i][0], w1 = ring[i][1], m0 = ring[i][2], m1 = ring[i][3], m2 = ring[i][4];
        if (gl + PF < GS) issue(gi + PF, i);
        // zero points: bytes g and g + 8 of m0; scales: int16 g of m1 (n0) and of m2 (n1)
        const unsigned z0 = 0x01010101u * (((gq ? m0.y : m0.x) >> gr) & 0xFFu);
        const unsigned z1 = 0x01010101u * (((gq ? m0.w : m0.z) >> gr) & 0xFFu);
        const unsigned sw0 = gh == 0 ? m1.x : gh == 1 ? m1.y : gh == 2 ? m1.z : m1.w;
        const unsigned sw1 = gh == 0 ? m2.x : gh == 1 ? m2.y : gh == 2 ? m2.z : m2.w;
        const int s0 = (int)((sw0 >> hs) & 0xFFFFu), s1 = (int)((sw1 >> hs) & 0xFFFFu);
        int acc[MT][4];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int j = 0; j < 4; ++j) acc[mt][j] = 0;
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          const uint4 w4 = c ? w1 : w0;
          const unsigned q[4] = {w4.x, w4.y, w4.z, w4.w};
#pragma unroll
          for (int kk = 0; kk < 2; ++kk) {
            unsigned af[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              const unsigned nib = kk ? ((q[j] >> 4) & 0x0F0F0F0Fu) : (q[j] & 0x0F0F0F0Fu);
              af[j] = __vsub4(nib, (j & 1) ? z1 : z0);
            }
            const int k0 = gi * kW4Group + 64 * c + 32 * kk;
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
              const int row = kQuant ? min(8 * mt + g, M - 1) : 8 * mt + g;
              const int8_t* xrow = xq + (size_t)row * K + k0 + 4 * t;
              unsigned bb[2];
              if (kQuant) {
                bb[0] = *reinterpret_cast<const unsigned*>(xrow);
                bb[1] = *reinterpret_cast<const unsigned*>(xrow + 16);
              } else {
                bb[0] = __ldg(reinterpret_cast<const unsigned*>(xrow));
                bb[1] = __ldg(reinterpret_cast<const unsigned*>(xrow + 16));
              }
              mma_s8_16832(acc[mt], af, bb);
            }
          }
        }
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          total[mt][0] += acc[mt][0] * s0;
          total[mt][1] += acc[mt][1] * s0;
          total[mt][2] += acc[mt][2] * s1;
          total[mt][3] += acc[mt][3] * s1;
        }
      }
    }
  }
  if (KS > 1) {
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int i = 0; i < 4; ++i) red[warp][mt * 4 + i][lane] = total[mt][i];
    __syncthreads();
    if (ks_id != 0) return;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        int v = 0;
#pragma unroll
        for (int w = 0; w < KS; ++w) v += red[warp + w][mt * 4 + i][lane];
        total[mt][i] = v;
      }
  }
  if (!valid) return;
  __nv_bfloat16* __restrict__ out = a.out[p];
  const float* __restrict__ a_scale = kQuant ? a_scale_s : a.a_scale[p];
  const float* __restrict__ cs = a.col_scale[p];
  const int ld = a.ld_out, Nv = a.Nv[p];
  // Marlin's epilogue, half(float(total) * a_scale[m]) as BF16, or with a per-channel factor (own quantization of
  // a linear whose integer group scales are normalized per output channel); padding channels are not written.
  auto store = [&](int m, int n, int v) {
    if (n >= Nv) return;
    const float r = __fmul_rn(__int2float_rn(v), a_scale[m]);
    out[(size_t)m * ld + n] = to_bf(cs ? __fmul_rn(r, cs[n]) : r);
  };
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    const int m0 = 8 * mt + 2 * t, m1 = m0 + 1;
    if (m0 < M) { store(m0, n0, total[mt][0]); store(m0, n1, total[mt][2]); }
    if (m1 < M) { store(m1, n0, total[mt][1]); store(m1, n1, total[mt][3]); }
  }
  a15_pdl_trigger();
}

template <int MT, int KS, int W, int PF, bool kQuant>
cudaError_t launch_w4a8_grouped(const W4GroupedArgs& a, cudaStream_t stream) {
  const int NT = a.tile0[a.n_problems];
  const dim3 grid((NT * KS + W - 1) / W);
  size_t smem = 0;
  if (kQuant) {
    smem = (size_t)a.M * a.K + 32;
    static bool configured = false;        // per instantiation: allow more than the default dynamic shared memory
    if (!configured && smem > 48 * 1024) {
      if (cudaFuncSetAttribute(w4a8_grouped_kernel<MT, KS, W, PF, kQuant>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                               (int)(4 * 12288 + 32)) != cudaSuccess)
        return cudaGetLastError();
      configured = true;
    }
  }
  return a15_launch(w4a8_grouped_kernel<MT, KS, W, PF, kQuant>, grid, dim3(W * 32), smem, stream, a);
}

template <int MT, int KS, bool kQuant>
cudaError_t launch_w4a8_grouped_wp(const W4GroupedArgs& a, int warps, int prefetch, cudaStream_t stream) {
  if (warps == 8) {
    if (prefetch == 4) return launch_w4a8_grouped<MT, KS, 8, 4, kQuant>(a, stream);
    return launch_w4a8_grouped<MT, KS, 8, 2, kQuant>(a, stream);
  }
  if (prefetch == 4) return launch_w4a8_grouped<MT, KS, 4, 4, kQuant>(a, stream);
  return launch_w4a8_grouped<MT, KS, 4, 2, kQuant>(a, stream);
}

template <bool kQuant>
cudaError_t launch_w4a8_grouped_dispatch(const W4GroupedArgs& a, int ks, int warps, int prefetch, cudaStream_t stream) {
  if (a.M <= 8) return ks == 4 ? launch_w4a8_grouped_wp<1, 4, kQuant>(a, warps, prefetch, stream)
                    : ks == 2 ? launch_w4a8_grouped_wp<1, 2, kQuant>(a, warps, prefetch, stream)
                              : launch_w4a8_grouped_wp<1, 1, kQuant>(a, warps, prefetch, stream);
  if (kQuant) return cudaErrorInvalidValue;          // the fused quantization serves M <= 4
  return ks == 4 ? launch_w4a8_grouped_wp<2, 4, false>(a, warps, prefetch, stream)
       : ks == 2 ? launch_w4a8_grouped_wp<2, 2, false>(a, warps, prefetch, stream)
                 : launch_w4a8_grouped_wp<2, 1, false>(a, warps, prefetch, stream);
}


// ---------------------------------------------------------------------------
// Per-token INT8 quantization of BF16 rows (the activation of the INT4 LM head):
// scale[m] = absmax / 127 (1 for an all-zero row), q = round-to-nearest-even(x / scale) in [-127, 127].
// One CTA per row; K % 4 == 0.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256) quantize_int8_rows_kernel(const __nv_bfloat16* __restrict__ x, int ld_x,
                                                                 int8_t* __restrict__ q, float* __restrict__ scale, int K) {
  a15_pdl();
  __shared__ float smem[32];
  const int row = blockIdx.x;
  const __nv_bfloat16* xr = x + (size_t)row * ld_x;
  float mx = 0.0f;
  for (int i = threadIdx.x; i < K; i += 256) mx = fmaxf(mx, fabsf(bf(xr[i])));
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (lane == 0) smem[warp] = mx;
  __syncthreads();
  if (warp == 0) {
    float t = lane < 8 ? smem[lane] : 0.0f;
    for (int o = 16; o > 0; o >>= 1) t = fmaxf(t, __shfl_xor_sync(0xffffffffu, t, o));
    if (lane == 0) smem[0] = t;
  }
  __syncthreads();
  const float amax = smem[0];
  const float s = amax > 0.0f ? __fdiv_rn(amax, 127.0f) : 1.0f;
  if (threadIdx.x == 0) scale[row] = s;
  const fp8rows::Bf4* x4 = reinterpret_cast<const fp8rows::Bf4*>(xr);
  char4* q4 = reinterpret_cast<char4*>(q + (size_t)row * K);
  for (int i = threadIdx.x; i < (K >> 2); i += 256) {
    const fp8rows::Bf4 t = x4[i];
    int v[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) v[e] = max(-127, min(127, __float2int_rn(__fdiv_rn(bf(t.v[e]), s))));
    q4[i] = make_char4((signed char)v[0], (signed char)v[1], (signed char)v[2], (signed char)v[3]);
  }
}

}  // namespace

int alpamayo15_rms_bf16(void* residual, const void* delta, const void* weight, void* out,
                        int rows, int dim, float eps, cudaStream_t stream) {
  if (!residual || !weight || !out) return 1;
  if (rows <= 0 || dim <= 0) return 2;
  a15_launch(rms_bf16_kernel, dim3(rows), dim3(256), (size_t)0, stream, 
      reinterpret_cast<__nv_bfloat16*>(residual), reinterpret_cast<const __nv_bfloat16*>(delta),
      reinterpret_cast<const __nv_bfloat16*>(weight), reinterpret_cast<__nv_bfloat16*>(out), dim, eps);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int alpamayo15_rms_fp8_rows_bf16(void* residual, const void* delta, const void* weight, void* q, float* scale,
                                 int rows, int dim, float eps, cudaStream_t stream) {
  if (!residual || !weight || !q || !scale) return 1;
  if (rows <= 0 || dim <= 0 || dim > kFusedRowMax || (dim % 4) != 0) return 2;
  if (!aligned8(residual) || !aligned8(weight) || (delta && !aligned8(delta)) || (reinterpret_cast<uintptr_t>(q) & 3u) != 0) return 5;
  a15_launch(rms_fp8_rows_kernel, dim3(rows), dim3(256), (size_t)0, stream, 
      reinterpret_cast<__nv_bfloat16*>(residual), reinterpret_cast<const __nv_bfloat16*>(delta),
      reinterpret_cast<const __nv_bfloat16*>(weight), static_cast<uint8_t*>(q), scale, dim, eps);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int alpamayo15_res_rms_f32_fp8_rows_bf16(float* residual, const void* delta, const void* weight, void* q, float* scale,
                                         int rows, int dim, float eps, cudaStream_t stream) {
  if (!residual || !weight || !q || !scale) return 1;
  if (rows <= 0 || dim <= 0 || dim > kFusedRowMax || (dim % 4) != 0) return 2;
  if (!aligned8(weight) || (reinterpret_cast<uintptr_t>(q) & 3u) != 0) return 5;
  a15_launch(res_rms_f32_fp8_rows_kernel, dim3(rows), dim3(256), (size_t)0, stream, residual,
             reinterpret_cast<const __nv_bfloat16*>(delta), reinterpret_cast<const __nv_bfloat16*>(weight),
             static_cast<uint8_t*>(q), scale, dim, eps);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int alpamayo15_silu_mul_fp8_rows_bf16(const void* gate, const void* up, int ld, void* q, float* scale, int rows, int F,
                                      cudaStream_t stream) {
  if (!gate || !up || !q || !scale) return 1;
  if (rows <= 0 || F <= 0 || F > kFusedRowMax || (F % 4) != 0 || ld < F || (ld % 4) != 0) return 2;
  if (!aligned8(gate) || !aligned8(up) || (reinterpret_cast<uintptr_t>(q) & 3u) != 0) return 5;
  a15_launch(silu_mul_fp8_rows_kernel, dim3(rows), dim3(256), (size_t)0, stream, 
      reinterpret_cast<const __nv_bfloat16*>(gate), reinterpret_cast<const __nv_bfloat16*>(up), ld,
      static_cast<uint8_t*>(q), scale, F);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int alpamayo15_vlm_qk_prep_bf16(const void* qkv, const void* q_norm_w, const void* k_norm_w,
                                const void* cos, const void* sin, void* q_out,
                                void* k_cache, void* v_cache, const long long* slot_ptr,
                                int rows, int Hq, int Hk, int kv_row_stride,
                                int rows_per_batch, long long batch_stride, float eps,
                                cudaStream_t stream) {
  // VLM rows share one device slot (decode position) or none (prefill).
  if (!qkv || !q_norm_w || !k_norm_w || !cos || !sin || !q_out || !k_cache || !v_cache) return 1;
  if (rows <= 0 || Hq <= 0 || Hk <= 0 || kv_row_stride < Hk * kHeadDim || rows_per_batch <= 0 ||
      batch_stride < 0) return 2;
  if (!aligned8(qkv) || !aligned8(q_norm_w) || !aligned8(k_norm_w) || !aligned8(q_out) ||
      !aligned8(k_cache) || !aligned8(v_cache) || (kv_row_stride % 4) != 0) return 5;
  const long long warps = (long long)rows * (Hq + 2 * Hk);
  a15_launch(qk_prep_kernel<false>, dim3((unsigned)((warps + 7) / 8)), dim3(256), (size_t)0, stream, 
      reinterpret_cast<const __nv_bfloat16*>(qkv), reinterpret_cast<const __nv_bfloat16*>(q_norm_w),
      reinterpret_cast<const __nv_bfloat16*>(k_norm_w), cos, sin, reinterpret_cast<__nv_bfloat16*>(q_out),
      reinterpret_cast<__nv_bfloat16*>(k_cache), reinterpret_cast<__nv_bfloat16*>(v_cache),
      slot_ptr, rows, Hq, Hk, kv_row_stride, rows_per_batch, batch_stride, 0, eps);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int alpamayo15_vision_rope_qk_bf16(void* qkv, const float* cos, const float* sin,
                                   int S, int H, int D, cudaStream_t stream) {
  if (!qkv || !cos || !sin) return 1;
  if (S <= 0 || H <= 0 || D <= 0 || (D % 8) != 0 || D > 128) return 2;
  if (!aligned8(qkv) || !aligned16(cos) || !aligned16(sin)) return 5;
  const long long threads = (long long)S * 2 * H * (D / 8);
  a15_launch(vision_rope_qk_kernel, dim3((unsigned)((threads + 255) / 256)), dim3(256), (size_t)0, stream,
      reinterpret_cast<__nv_bfloat16*>(qkv), cos, sin, S, H, D);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int alpamayo15_res_rms_f32_bf16(float* residual, const void* delta, const void* weight,
                                void* out, int rows, int dim, float eps, cudaStream_t stream) {
  if (!residual || !weight || !out) return 1;
  if (rows <= 0 || dim <= 0) return 2;
  a15_launch(res_rms_f32_bf16_kernel, dim3(rows), dim3(256), (size_t)0, stream, 
      residual, reinterpret_cast<const __nv_bfloat16*>(delta),
      reinterpret_cast<const __nv_bfloat16*>(weight),
      reinterpret_cast<__nv_bfloat16*>(out), dim, eps);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int alpamayo15_expert_qk_prep_bf16(const void* qkv, const void* q_norm_w, const void* k_norm_w,
                                   const float* cos, const float* sin,
                                   void* q_out, void* k_out, void* v_out, const long long* slot_ptr,
                                   int rows, int Hq, int Hk, int k_row_stride,
                                   int rows_per_batch, long long batch_stride, float eps,
                                   cudaStream_t stream) {
  if (!qkv || !q_norm_w || !k_norm_w || !cos || !sin || !q_out || !k_out || !v_out) return 1;
  if (rows <= 0 || Hq <= 0 || Hk <= 0 || k_row_stride < Hk * kHeadDim || rows_per_batch <= 0 ||
      batch_stride < 0) return 2;
  if (!aligned8(qkv) || !aligned8(q_norm_w) || !aligned8(k_norm_w) || !aligned8(q_out) ||
      !aligned8(k_out) || !aligned8(v_out) || (k_row_stride % 4) != 0) return 5;
  const long long warps = (long long)rows * (Hq + 2 * Hk);
  a15_launch(qk_prep_kernel<true>, dim3((unsigned)((warps + 7) / 8)), dim3(256), (size_t)0, stream, 
      reinterpret_cast<const __nv_bfloat16*>(qkv),
      reinterpret_cast<const __nv_bfloat16*>(q_norm_w),
      reinterpret_cast<const __nv_bfloat16*>(k_norm_w), cos, sin,
      reinterpret_cast<__nv_bfloat16*>(q_out), reinterpret_cast<__nv_bfloat16*>(k_out),
      reinterpret_cast<__nv_bfloat16*>(v_out), slot_ptr, rows, Hq, Hk, k_row_stride, rows_per_batch,
      batch_stride, 1, eps);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}


int alpamayo15_w8a16_gemm_bf16(const void* w_frag, const float* scale, const void* x, void* y,
                               int M, int N, int K, int fmt, cudaStream_t stream) {
  if (!w_frag || !scale || !x || !y) return 1;
  if (M <= 0 || M > 16 || N <= 0 || K <= 0 || (K % 32) != 0 || fmt < 0 || fmt > 1) return 2;
  if (!aligned16(w_frag) || (reinterpret_cast<uintptr_t>(x) & 3u) != 0) return 5;
  // Large-N projections (lm_head) amortize the activation loads over two tiles per warp.
  const bool wide = N >= 65536;
  if (M <= 8) {
    if (wide) launch_w8a16<1, 2>(w_frag, scale, x, y, M, N, K, fmt, stream);
    else launch_w8a16<1, 1>(w_frag, scale, x, y, M, N, K, fmt, stream);
  } else {
    if (wide) launch_w8a16<2, 2>(w_frag, scale, x, y, M, N, K, fmt, stream);
    else launch_w8a16<2, 1>(w_frag, scale, x, y, M, N, K, fmt, stream);
  }
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int alpamayo15_quantize_int8_rows_bf16(const void* x, int ld_x, void* q, float* scale, int rows, int K, cudaStream_t stream) {
  if (!x || !q || !scale) return 1;
  if (rows <= 0 || K <= 0 || (K % 4) != 0 || ld_x < K || (ld_x % 4) != 0) return 2;
  if (!aligned8(x) || (reinterpret_cast<uintptr_t>(q) & 3u) != 0) return 5;
  a15_launch(quantize_int8_rows_kernel, dim3(rows), dim3(256), (size_t)0, stream, static_cast<const __nv_bfloat16*>(x), ld_x,
             static_cast<int8_t*>(q), scale, K);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

namespace {
// Shared validation / argument block of the grouped few-row W4A8 GEMM entry points; returns the rc (0 = ok) and the
// K split to launch with.
int w4_grouped_args(W4GroupedArgs& a, int& ks, int n_problems, const void* w_pack, void* const* out, int M, const int* N, int K,
                    int ld_out, int prefetch, int warps, int ksplit, const float* const* col_scale, const int* n_valid,
                    const void* pf_ptr, size_t pf_bytes) {
  if (pf_bytes && (!pf_ptr || (reinterpret_cast<uintptr_t>(pf_ptr) & 127u) != 0 || pf_bytes > (1u << 30))) return 5;
  a.pf_ptr = static_cast<const unsigned char*>(pf_ptr);
  a.pf_bytes = static_cast<unsigned>(pf_bytes);
  if (n_problems < 1 || n_problems > kW4MaxProblems || !w_pack) return 1;
  if (M <= 0 || M > 16 || K <= 0 || (K % kW4Group) != 0 || ld_out <= 0) return 2;     // any ld_out: BF16 elements are stored singly
  if ((prefetch != 2 && prefetch != 4) || (warps != 4 && warps != 8) || (ksplit != 0 && ksplit != 1 && ksplit != 2 && ksplit != 4))
    return 2;
  if (!aligned16(w_pack)) return 5;
  a.tile0[0] = 0;
  for (int i = 0; i < n_problems; ++i) {
    if (!out[i]) return 1;
    const int nv = n_valid ? n_valid[i] : N[i];
    if (N[i] <= 0 || (N[i] % 16) != 0 || nv <= 0 || nv > N[i] || ld_out < nv) return 2;
    if ((reinterpret_cast<uintptr_t>(out[i]) & 1u) != 0) return 5;
    a.out[i] = static_cast<__nv_bfloat16*>(out[i]);
    a.col_scale[i] = col_scale ? col_scale[i] : nullptr;
    if (a.col_scale[i] && (reinterpret_cast<uintptr_t>(a.col_scale[i]) & 3u) != 0) return 5;
    a.N[i] = N[i];
    a.Nv[i] = nv;
    a.tile0[i + 1] = a.tile0[i] + N[i] / 16;
  }
  a.w_pack = static_cast<const unsigned char*>(w_pack);
  a.ld_out = ld_out; a.M = M; a.K = K; a.n_problems = n_problems;
  const int G = K / kW4Group, NT = a.tile0[n_problems];
  ks = ksplit;
  if (ks == 0) ks = (NT >= 1024 && (G / 2) % prefetch == 0) ? 2 : 1;   // wide problems (gate|up): two warps per tile
  if ((G % ks) != 0 || (G / ks) % prefetch != 0 || ks > warps) return 2;
  return 0;
}
}  // namespace

int alpamayo15_w4a8_gemm_grouped(int n_problems, const void* const* xq, const float* const* a_scale, const void* w_pack,
                                 void* const* out, int M, const int* N, int K, int ld_out, int prefetch, int warps,
                                 int ksplit, cudaStream_t stream, const float* const* col_scale, const int* n_valid,
                                 const void* pf_ptr, size_t pf_bytes) {
  W4GroupedArgs a = {};
  int ks = 1;
  const int rc = w4_grouped_args(a, ks, n_problems, w_pack, out, M, N, K, ld_out, prefetch, warps, ksplit, col_scale, n_valid,
                                 pf_ptr, pf_bytes);
  if (rc) return rc;
  for (int i = 0; i < n_problems; ++i) {
    if (!xq[i] || !a_scale[i]) return 1;
    if ((reinterpret_cast<uintptr_t>(xq[i]) & 3u) != 0) return 5;
    a.xq[i] = static_cast<const int8_t*>(xq[i]);
    a.a_scale[i] = a_scale[i];
  }
  const cudaError_t err = launch_w4a8_grouped_dispatch<false>(a, ks, warps, prefetch, stream);
  return err == cudaSuccess && cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int alpamayo15_w4a8_gemm_grouped_q(int n_problems, const void* const* xr, const float* const* part, const float* const* igs,
                                   const void* w_pack, void* const* out, int M, const int* N, int K, int ld_out, int prefetch,
                                   int warps, int ksplit, cudaStream_t stream, const float* const* col_scale, const int* n_valid,
                                   const void* pf_ptr, size_t pf_bytes) {
  W4GroupedArgs a = {};
  int ks = 1;
  const int rc = w4_grouped_args(a, ks, n_problems, w_pack, out, M, N, K, ld_out, prefetch, warps, ksplit, col_scale, n_valid,
                                 pf_ptr, pf_bytes);
  if (rc) return rc;
  if (M > 4 || M > warps) return 2;                     // one warp per row for the absmax pass, M * K bytes of shared memory
  for (int i = 0; i < n_problems; ++i) {
    if (!xr[i] || !part[i] || !igs[i]) return 1;
    if ((reinterpret_cast<uintptr_t>(xr[i]) & 3u) != 0 || (reinterpret_cast<uintptr_t>(part[i]) & 3u) != 0) return 5;
    a.xr[i] = static_cast<const __half*>(xr[i]);
    a.part[i] = part[i];
    a.igs[i] = igs[i];
    // a CTA quantizes one problem: its tiles must not straddle a problem boundary
    if (i > 0 && (a.tile0[i] % (warps / ks)) != 0) return 2;
  }
  const cudaError_t err = launch_w4a8_grouped_dispatch<true>(a, ks, warps, prefetch, stream);
  return err == cudaSuccess && cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

}  // namespace kernels
}  // namespace flash_rt
