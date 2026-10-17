// SPDX-License-Identifier: Apache-2.0
//
// Alpamayo 1.5 (Thor sm_110) model-local kernels. See alpamayo15_kernels.cuh
// for the rounding contract each kernel reproduces.

#include "alpamayo15_kernels.cuh"

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
  const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (gw >= S * 2 * H) return;
  const int row = gw / (2 * H), head = gw % (2 * H);  // heads 0..H-1 = q, H..2H-1 = k
  const int nl = D >> 2, hl = D >> 3;                   // active lanes, partner lane offset
  const bool active = lane < nl;
  __nv_bfloat16* p = qkv + (size_t)row * 3 * H * D + (size_t)head * D + 4 * lane;
  Bf4 x = {};
  if (active) x = *reinterpret_cast<const Bf4*>(p);
  const int src = lane < hl ? lane + hl : lane - hl;
  float y[4];
  const float* c = cos + (size_t)row * D + 4 * lane;
  const float* sn = sin + (size_t)row * D + 4 * lane;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float other = __shfl_sync(0xffffffffu, bf(x.v[i]), src);
    const float rot = lane < hl ? -other : other;
    if (active) y[i] = __fadd_rn(__fmul_rn(bf(x.v[i]), c[i]), __fmul_rn(rot, sn[i]));
  }
  if (active) {
    Bf4 o;
#pragma unroll
    for (int i = 0; i < 4; ++i) o.v[i] = to_bf(y[i]);
    *reinterpret_cast<Bf4*>(p) = o;
  }
}

// One block per row: residual (+= delta) and RMSNorm -> BF16.
__global__ void res_rms_f32_bf16_kernel(float* __restrict__ res,
                                        const __nv_bfloat16* __restrict__ delta,
                                        const __nv_bfloat16* __restrict__ w,
                                        __nv_bfloat16* __restrict__ out,
                                        int dim, float eps) {
  __shared__ float smem[32];
  const int row = blockIdx.x;
  float* r = res + (size_t)row * dim;
  const __nv_bfloat16* dl = delta ? delta + (size_t)row * dim : nullptr;
  float ss = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    float v = r[i];
    if (dl) {
      v = __fadd_rn(v, bf(dl[i]));
      r[i] = v;
    }
    ss += __fmul_rn(v, v);
  }
  const float inv = rsqrtf(__fadd_rn(block_sum(ss, smem) / (float)dim, eps));
  __nv_bfloat16* o = out + (size_t)row * dim;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    o[i] = to_bf(__fmul_rn(bf(w[i]), __fmul_rn(r[i], inv)));
  }
}

// One block per row: optional BF16 residual add, then Qwen3 RMSNorm on BF16 activations.
__global__ void rms_bf16_kernel(__nv_bfloat16* __restrict__ res,
                                const __nv_bfloat16* __restrict__ delta,
                                const __nv_bfloat16* __restrict__ w,
                                __nv_bfloat16* __restrict__ out,
                                int dim, float eps) {
  __shared__ float smem[32];
  const int row = blockIdx.x;
  __nv_bfloat16* r = res + (size_t)row * dim;
  const __nv_bfloat16* dl = delta ? delta + (size_t)row * dim : nullptr;
  float ss = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    __nv_bfloat16 v = r[i];
    if (dl) {
      v = to_bf(__fadd_rn(bf(v), bf(dl[i])));
      r[i] = v;
    }
    const float f = bf(v);
    ss += __fmul_rn(f, f);
  }
  const float inv = rsqrtf(__fadd_rn(block_sum(ss, smem) / (float)dim, eps));
  __nv_bfloat16* o = out + (size_t)row * dim;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const __nv_bfloat16 n1 = to_bf(__fmul_rn(bf(r[i]), inv));
    o[i] = to_bf(__fmul_rn(bf(w[i]), bf(n1)));
  }
}

// Per-row FP8 E4M3 quantization of a BF16 row held in shared memory (the W8A8 prefill tier): scale = absmax / 448
// (1 for an all-zero row), codes = round-to-nearest-even(x / scale) saturated to +-448, as alpamayo15_quantize_fp8_rows_bf16.
__device__ __forceinline__ void quantize_row_fp8(const __nv_bfloat16* srow, uint8_t* __restrict__ q, float* __restrict__ scale,
                                                 int row, int dim, float* smem) {
  float mx = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) mx = fmaxf(mx, fabsf(bf(srow[i])));
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
    for (int e = 0; e < 4; ++e) f[e] = fminf(fmaxf(__fdiv_rn(bf(t.v[e]), s), -448.0f), 448.0f);
    const __nv_fp8x2_storage_t lo = __nv_cvt_float2_to_fp8x2(make_float2(f[0], f[1]), __NV_SATFINITE, __NV_E4M3);
    const __nv_fp8x2_storage_t hi = __nv_cvt_float2_to_fp8x2(make_float2(f[2], f[3]), __NV_SATFINITE, __NV_E4M3);
    qr[i] = (uint32_t)lo | ((uint32_t)hi << 16);
  }
}

// rms_bf16_kernel whose normalized row goes to FP8 codes + a row scale instead of a BF16 output (identical codes
// to rms_bf16 followed by alpamayo15_quantize_fp8_rows_bf16). dim <= kFusedRowMax.
constexpr int kFusedRowMax = 16384;
__global__ void __launch_bounds__(256) rms_fp8_rows_kernel(__nv_bfloat16* __restrict__ res,
                                                           const __nv_bfloat16* __restrict__ delta,
                                                           const __nv_bfloat16* __restrict__ w,
                                                           uint8_t* __restrict__ q, float* __restrict__ scale,
                                                           int dim, float eps) {
  __shared__ float smem[32];
  __shared__ __align__(16) __nv_bfloat16 srow[kFusedRowMax];
  const int row = blockIdx.x;
  __nv_bfloat16* r = res + (size_t)row * dim;
  const __nv_bfloat16* dl = delta ? delta + (size_t)row * dim : nullptr;
  float ss = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    __nv_bfloat16 v = r[i];
    if (dl) {
      v = to_bf(__fadd_rn(bf(v), bf(dl[i])));
      r[i] = v;
    }
    const float f = bf(v);
    ss += __fmul_rn(f, f);
  }
  const float inv = rsqrtf(__fadd_rn(block_sum(ss, smem) / (float)dim, eps));
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const __nv_bfloat16 n1 = to_bf(__fmul_rn(bf(r[i]), inv));
    srow[i] = to_bf(__fmul_rn(bf(w[i]), bf(n1)));
  }
  __syncthreads();
  quantize_row_fp8(srow, q, scale, row, dim, smem);
}

// out = bf16(silu(bf16 gate)) * bf16 up (the rounding of the split SiLU-mul kernel) quantized per row to FP8 codes
// + a row scale; gate / up rows ld elements apart. F <= kFusedRowMax, F % 4 == 0.
__global__ void __launch_bounds__(256) silu_mul_fp8_rows_kernel(const __nv_bfloat16* __restrict__ gate,
                                                                const __nv_bfloat16* __restrict__ up, int ld,
                                                                uint8_t* __restrict__ q, float* __restrict__ scale,
                                                                int F) {
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
}

template <int NT, int RT>
void launch_w8a16(const void* wfrag, const float* scale, const void* x, void* y, int M, int N, int K, int fmt,
                  cudaStream_t stream) {
  constexpr int kWarps = 4;
  const int T = (N + 15) / 16;
  const int blocks = (T + kWarps * RT - 1) / (kWarps * RT);
  if (fmt == 1)
    w8a16_gemm_kernel<NT, RT, true><<<blocks, kWarps * 32, 0, stream>>>(
        reinterpret_cast<const uint4*>(wfrag), scale, reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<__nv_bfloat16*>(y), M, N, K, T);
  else
    w8a16_gemm_kernel<NT, RT, false><<<blocks, kWarps * 32, 0, stream>>>(
        reinterpret_cast<const uint4*>(wfrag), scale, reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<__nv_bfloat16*>(y), M, N, K, T);
}

}  // namespace

int alpamayo15_rms_bf16(void* residual, const void* delta, const void* weight, void* out,
                        int rows, int dim, float eps, cudaStream_t stream) {
  if (!residual || !weight || !out) return 1;
  if (rows <= 0 || dim <= 0) return 2;
  rms_bf16_kernel<<<rows, 256, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(residual), reinterpret_cast<const __nv_bfloat16*>(delta),
      reinterpret_cast<const __nv_bfloat16*>(weight), reinterpret_cast<__nv_bfloat16*>(out), dim, eps);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int alpamayo15_rms_fp8_rows_bf16(void* residual, const void* delta, const void* weight, void* q, float* scale,
                                 int rows, int dim, float eps, cudaStream_t stream) {
  if (!residual || !weight || !q || !scale) return 1;
  if (rows <= 0 || dim <= 0 || dim > kFusedRowMax || (dim % 4) != 0) return 2;
  if (!aligned8(residual) || !aligned8(weight) || (reinterpret_cast<uintptr_t>(q) & 3u) != 0) return 5;
  rms_fp8_rows_kernel<<<rows, 256, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(residual), reinterpret_cast<const __nv_bfloat16*>(delta),
      reinterpret_cast<const __nv_bfloat16*>(weight), static_cast<uint8_t*>(q), scale, dim, eps);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int alpamayo15_silu_mul_fp8_rows_bf16(const void* gate, const void* up, int ld, void* q, float* scale, int rows, int F,
                                      cudaStream_t stream) {
  if (!gate || !up || !q || !scale) return 1;
  if (rows <= 0 || F <= 0 || F > kFusedRowMax || (F % 4) != 0 || ld < F || (ld % 4) != 0) return 2;
  if (!aligned8(gate) || !aligned8(up) || (reinterpret_cast<uintptr_t>(q) & 3u) != 0) return 5;
  silu_mul_fp8_rows_kernel<<<rows, 256, 0, stream>>>(
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
  qk_prep_kernel<false><<<(unsigned)((warps + 7) / 8), 256, 0, stream>>>(
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
  if (!aligned8(qkv)) return 5;
  const long long warps = (long long)S * 2 * H;
  vision_rope_qk_kernel<<<(unsigned)((warps + 7) / 8), 256, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(qkv), cos, sin, S, H, D);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int alpamayo15_res_rms_f32_bf16(float* residual, const void* delta, const void* weight,
                                void* out, int rows, int dim, float eps, cudaStream_t stream) {
  if (!residual || !weight || !out) return 1;
  if (rows <= 0 || dim <= 0) return 2;
  res_rms_f32_bf16_kernel<<<rows, 256, 0, stream>>>(
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
  qk_prep_kernel<true><<<(unsigned)((warps + 7) / 8), 256, 0, stream>>>(
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

}  // namespace kernels
}  // namespace flash_rt
