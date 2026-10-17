// SPDX-License-Identifier: Apache-2.0
//
// FlashDrive (Alpamayo 1.5) backend kernels for Thor (sm_110). See
// alpamayo15_flashdrive_kernels.cuh for the cache contract and rounding rules.

#include "alpamayo15_flashdrive_kernels.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <utility>

namespace flash_rt {
namespace kernels {
namespace {
// Programmatic dependent launch: when the process-wide flag is on, kernels are launched with the programmatic
// stream-serialization attribute (recorded as programmatic edges inside CUDA graphs) and execute
// griddepcontrol.wait before touching their inputs / launch_dependents right after, so the next kernel's prologue
// overlaps this kernel's tail. No-ops when the grid was launched without the attribute.
bool g_fd_pdl = false;

__device__ __forceinline__ void fd_pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
__device__ __forceinline__ void fd_pdl_trigger() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}
// Short kernels: wait, then let the successor's prologue start right away.
__device__ __forceinline__ void fd_pdl() { fd_pdl_wait(); fd_pdl_trigger(); }

template <typename... KArgs, typename... Args>
inline cudaError_t fd_launch(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t stream,
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
  cfg.numAttrs = g_fd_pdl ? 1 : 0;
  return cudaLaunchKernelEx(&cfg, kernel, std::forward<Args>(args)...);
}
}  // namespace
}  // namespace kernels
}  // namespace flash_rt

namespace flash_rt {
namespace kernels {

namespace {

constexpr int kHeadDim = 128;

inline bool aligned8(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 7u) == 0; }

__device__ __forceinline__ float bf(const __nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ __nv_bfloat16 to_bf(float v) { return __float2bfloat16_rn(v); }

struct alignas(8) Bf4 { __nv_bfloat16 v[4]; };

__device__ __forceinline__ long long cache_row(int r, const long long* row_map, long long row0,
                                               const long long* slot_ptr) {
  return (row_map ? row_map[r] : row0 + r) + (slot_ptr ? *slot_ptr : 0LL);
}

// BF16-table RoPE of one lane's 4 values (lane l holds d = 4l..4l+3; partner d +/- 64 = lane l ^ 16).
__device__ __forceinline__ Bf4 rope_bf16_rule(const Bf4& x, const __nv_bfloat16* cos, const __nv_bfloat16* sin,
                                              int lane) {
  Bf4 out;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float other = __shfl_xor_sync(0xffffffffu, bf(x.v[i]), 16);
    const int d = 4 * lane + i;
    const float c = bf(cos[d]), s = bf(sin[d]);
    const float rot = lane < 16 ? -other : other;  // negation of a BF16 value is exact
    const float a = bf(to_bf(__fmul_rn(bf(x.v[i]), c)));
    const float b = bf(to_bf(__fmul_rn(rot, s)));
    out.v[i] = to_bf(__fadd_rn(a, b));
  }
  return out;
}

__device__ __forceinline__ Bf4 rope_fp32_rule(const Bf4& x, const float* cos, const float* sin, int lane) {
  Bf4 out;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float other = __shfl_xor_sync(0xffffffffu, bf(x.v[i]), 16);
    const int d = 4 * lane + i;
    const float rot = lane < 16 ? -other : other;
    out.v[i] = to_bf(__fadd_rn(__fmul_rn(bf(x.v[i]), cos[d]), __fmul_rn(rot, sin[d])));
  }
  return out;
}

struct alignas(8) H4 { __half v[4]; };

// Four input values of a projection row: BF16, or FP16 (Marlin output) rounded to BF16 as .to(bfloat16) does.
template <bool kF16In>
__device__ __forceinline__ Bf4 load4(const void* base, size_t off) {
  if (kF16In) {
    const H4 h = *reinterpret_cast<const H4*>(static_cast<const __half*>(base) + off);
    Bf4 r;
#pragma unroll
    for (int i = 0; i < 4; ++i) r.v[i] = to_bf(__half2float(h.v[i]));
    return r;
  }
  return *reinterpret_cast<const Bf4*>(static_cast<const __nv_bfloat16*>(base) + off);
}

// One warp per (row, head) over q | k | v heads.
template <bool kFp32Rule, bool kF16In>
__global__ void fd_qk_prep_kernel(const void* __restrict__ q, const void* __restrict__ k,
                                  const void* __restrict__ v, int q_ld, int k_ld, int v_ld,
                                  const __nv_bfloat16* __restrict__ qw, const __nv_bfloat16* __restrict__ kw,
                                  const void* __restrict__ cos, const void* __restrict__ sin,
                                  __nv_bfloat16* __restrict__ q_out,
                                  __nv_bfloat16* __restrict__ k_raw, long long kr_rs, long long kr_hs,
                                  __nv_bfloat16* __restrict__ k_rope, long long k_rs, long long k_hs,
                                  __nv_bfloat16* __restrict__ v_out, long long v_rs, long long v_hs,
                                  const long long* __restrict__ row_map, const long long* __restrict__ slot_ptr,
                                  int rows, int Hq, int Hk, float eps) {
  fd_pdl();
  const int nh = Hq + 2 * Hk;
  const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (gw >= rows * nh) return;
  const int row = gw / nh, head = gw % nh;
  const long long c = cache_row(row, row_map, 0, slot_ptr);
  if (head >= Hq + Hk) {
    const int h = head - Hq - Hk;
    const Bf4 x = load4<kF16In>(v, (size_t)row * v_ld + (size_t)h * kHeadDim + 4 * lane);
    *reinterpret_cast<Bf4*>(v_out + c * v_rs + h * v_hs + 4 * lane) = x;
    return;
  }
  const bool is_q = head < Hq;
  const int h = is_q ? head : head - Hq;
  const Bf4 x = is_q ? load4<kF16In>(q, (size_t)row * q_ld + (size_t)h * kHeadDim + 4 * lane)
                     : load4<kF16In>(k, (size_t)row * k_ld + (size_t)h * kHeadDim + 4 * lane);
  float ss = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) ss += __fmul_rn(bf(x.v[i]), bf(x.v[i]));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  const float inv = rsqrtf(__fadd_rn(ss / (float)kHeadDim, eps));
  const Bf4 w = *reinterpret_cast<const Bf4*>((is_q ? qw : kw) + 4 * lane);
  Bf4 n;
#pragma unroll
  for (int i = 0; i < 4; ++i) n.v[i] = to_bf(__fmul_rn(bf(w.v[i]), bf(to_bf(__fmul_rn(bf(x.v[i]), inv)))));
  const Bf4 r = kFp32Rule
      ? rope_fp32_rule(n, static_cast<const float*>(cos) + (size_t)row * kHeadDim,
                       static_cast<const float*>(sin) + (size_t)row * kHeadDim, lane)
      : rope_bf16_rule(n, static_cast<const __nv_bfloat16*>(cos) + (size_t)row * kHeadDim,
                       static_cast<const __nv_bfloat16*>(sin) + (size_t)row * kHeadDim, lane);
  if (is_q) {
    *reinterpret_cast<Bf4*>(q_out + ((size_t)row * Hq + h) * kHeadDim + 4 * lane) = r;
  } else {
    *reinterpret_cast<Bf4*>(k_raw + c * kr_rs + h * kr_hs + 4 * lane) = n;
    *reinterpret_cast<Bf4*>(k_rope + c * k_rs + h * k_hs + 4 * lane) = r;
  }
}

template <bool kFp32Rule>
__global__ void fd_rope_rows_kernel(const __nv_bfloat16* __restrict__ src, long long s_rs, long long s_hs,
                                    __nv_bfloat16* __restrict__ dst, long long d_rs, long long d_hs,
                                    const void* __restrict__ cos_v, const void* __restrict__ sin_v,
                                    const long long* __restrict__ row_map, long long row0,
                                    const long long* __restrict__ slot_ptr, const int* __restrict__ n_ptr,
                                    int rows, int Hk) {
  fd_pdl();
  const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (gw >= rows * Hk) return;
  const int r = gw / Hk, h = gw % Hk;
  if (n_ptr && r >= *n_ptr) return;
  const long long c = cache_row(r, row_map, row0, slot_ptr);
  const Bf4 x = *reinterpret_cast<const Bf4*>(src + c * s_rs + h * s_hs + 4 * lane);
  Bf4 out;
  if (kFp32Rule) {
    out = rope_fp32_rule(x, static_cast<const float*>(cos_v) + c * kHeadDim,
                         static_cast<const float*>(sin_v) + c * kHeadDim, lane);
  } else {
    out = rope_bf16_rule(x, static_cast<const __nv_bfloat16*>(cos_v) + c * kHeadDim,
                         static_cast<const __nv_bfloat16*>(sin_v) + c * kHeadDim, lane);
  }
  *reinterpret_cast<Bf4*>(dst + c * d_rs + h * d_hs + 4 * lane) = out;
}

inline bool stride_ok(long long s) { return s >= 0 && (s % 4) == 0; }

__device__ __forceinline__ float block_sum256(float v, float* smem) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  float t = (threadIdx.x < (blockDim.x >> 5)) ? smem[threadIdx.x] : 0.0f;
  if (warp == 0) {
    for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
    if (lane == 0) smem[0] = t;
  }
  __syncthreads();
  const float r = smem[0];
  __syncthreads();
  return r;
}

// residual (BF16) += bf16(delta FP16) (BF16 add rounding), out = Qwen3 RMSNorm(residual) * w: the Alpamayo 1.5
// engine's rms_bf16 contract with the delta read straight from a Marlin FP16 output.
__global__ void fd_rms_f16delta_kernel(__nv_bfloat16* __restrict__ res, const __half* __restrict__ delta,
                                       const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ out,
                                       int dim, float eps) {
  fd_pdl();
  __shared__ float smem[32];
  const int row = blockIdx.x;
  __nv_bfloat16* r = res + (size_t)row * dim;
  const __half* dl = delta + (size_t)row * dim;
  float ss = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const __nv_bfloat16 v = to_bf(__fadd_rn(bf(r[i]), bf(to_bf(__half2float(dl[i])))));
    r[i] = v;
    const float f = bf(v);
    ss += __fmul_rn(f, f);
  }
  const float inv = rsqrtf(__fadd_rn(block_sum256(ss, smem) / (float)dim, eps));
  __nv_bfloat16* o = out + (size_t)row * dim;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const __nv_bfloat16 n1 = to_bf(__fmul_rn(bf(r[i]), inv));
    o[i] = to_bf(__fmul_rn(bf(w[i]), bf(n1)));
  }
}

// out = bf16(silu(bf16(g))) * bf16(u) with BF16 roundings of the SiLU and of the product (act_fn(gate) * up on
// the BF16 outputs of FlashDrive's W4A8 wrappers), g/u read as Marlin FP16 outputs.
__global__ void fd_silu_mul_f16in_kernel(const __half* __restrict__ g, const __half* __restrict__ u,
                                         __nv_bfloat16* __restrict__ out, long long n) {
  fd_pdl();
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
    const float gb = bf(to_bf(__half2float(g[i])));
    const float ub = bf(to_bf(__half2float(u[i])));
    const float sl = bf(to_bf(gb / (1.0f + expf(-gb))));
    out[i] = to_bf(__fmul_rn(sl, ub));
  }
}

// ---------------------------------------------------------------------------
// W4A8 activation preparation (FlashDrive's ParoQuant linears), bit-exact with
//   x.to(float16) -> torch.ops.rotation.rotate(x, pairs, theta, channel_scales, 128)   (paroquant 0.1.15)
//   -> vLLM per_token_quant_int8 (Triton) -> a_scales * input_global_scale,
// for up to 3 rotation sets sharing one input row (q|k|v, gate|up).
// Rotation (paroquant RotateAccessHalf, KROT=8, group 128, CTA_M=4): x_half * scale_half (__hmul), then per
// layer r each of the 64 pairs (i, j) of a group: yi = fma(c, xi, s*xj), yj = fma(c, xj, -s*xi) in FP32 with
// (s, c) = __sincosf(theta), rounded to half per layer. Quantization (Triton PTX): absmax = max(max|x|, 1e-10),
// scale = div.full(absmax, 127), inv = div.full(127, absmax), q = cvt.rzi(add.rz.ftz(x*inv, copysign(0.5, x*inv))).
// ---------------------------------------------------------------------------
constexpr int kRotGroup = 128;
constexpr int kRotLayers = 8;
constexpr int kRotRows = 16;  // rows per rotation CTA (sin/cos and pair indices loaded once per 16 rows)

struct ActPrepSet {
  const __half* theta;      // [8, K/2] fp16
  const int16_t* pairs;     // [8, K] int16 (in-group pair indices, i low / j high per int32)
  const __half* scales;     // [K] fp16 channel scales
  const float* igs;         // input_global_scale (fp32 scalar)
  __half* xr;               // [rows, K] fp16 rotated activations (workspace)
  float* part;              // [rows, K/128] per-group row absmax (workspace)
  int8_t* xq;               // [rows, K] int8
  float* a_scale;           // [rows] fp32 = per-token scale * input_global_scale
};
struct ActPrepSets { ActPrepSet s[3]; };

__device__ __forceinline__ float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float round_half_away_triton(float y) {
  float h = copysignf(0.5f, y), t;
  asm("add.rz.ftz.f32 %0, %1, %2;" : "=f"(t) : "f"(y), "f"(h));
  return truncf(t);
}

// Pass 1: grid (row blocks, groups, sets), 64 threads = the 64 rotation pairs of a 128-channel group.
// Per element the sequence is paroquant's: half(x) * half(scale) (__hmul), then 8 layers of
// fma(c, xi, s*xj) / fma(c, xj, -s*xi) in FP32 with __sincosf, rounded to half after each layer.
// Shared tile is row-major [ROWS][128] so the random pair indices of a layer spread over the banks.
template <int ROWS>
__global__ void __launch_bounds__(64) fd_w4a8_rotate_kernel(const __nv_bfloat16* __restrict__ x, int ld_x,
                                                             ActPrepSets sets, int rows, int K) {
  fd_pdl_wait();
  __shared__ __half xs[ROWS][kRotGroup];
  __shared__ float wmax[2][ROWS];
  const ActPrepSet& st = sets.s[blockIdx.z];
  const int row0 = blockIdx.x * ROWS, g = blockIdx.y, t = threadIdx.x;
  const int nrow = min(ROWS, rows - row0);
  const int off = kRotGroup * g + 2 * t;
  const __half sc_i = st.scales[off], sc_j = st.scales[off + 1];
#pragma unroll
  for (int i = 0; i < ROWS; ++i) {
    if (i < nrow) {
      const __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(x + (size_t)(row0 + i) * ld_x + off);
      xs[i][2 * t] = __hmul(__float2half_rn(__bfloat162float(v.x)), sc_i);
      xs[i][2 * t + 1] = __hmul(__float2half_rn(__bfloat162float(v.y)), sc_j);
    }
  }
  float th[kRotLayers];
  int ij[kRotLayers];
#pragma unroll
  for (int r = 0; r < kRotLayers; ++r) {
    th[r] = __half2float(st.theta[(size_t)r * K / 2 + g * kRotGroup / 2 + t]);
    ij[r] = *reinterpret_cast<const int*>(st.pairs + (size_t)r * K + g * kRotGroup + 2 * t);
  }
  __syncthreads();
#pragma unroll
  for (int r = 0; r < kRotLayers; ++r) {
    const int16_t i = ij[r] & 0xFFFF, j = ij[r] >> 16;
    float s_, c_;
    __sincosf(th[r], &s_, &c_);
#pragma unroll
    for (int m = 0; m < ROWS; ++m) {
      if (m < nrow) {
        const float xi = __half2float(xs[m][i]);
        const float xj = __half2float(xs[m][j]);
        xs[m][i] = __float2half_rn(fmaf(c_, xi, s_ * xj));
        xs[m][j] = __float2half_rn(fmaf(c_, xj, -s_ * xi));
      }
    }
    __syncthreads();
  }
  const int warp = t >> 5, lane = t & 31;
#pragma unroll
  for (int i = 0; i < ROWS; ++i) {
    if (i < nrow) {                                    // nrow is uniform across the block
      const __half a = xs[i][2 * t], b = xs[i][2 * t + 1];
      __half2 v; v.x = a; v.y = b;
      *reinterpret_cast<__half2*>(st.xr + (size_t)(row0 + i) * K + off) = v;
      float m = fmaxf(fabsf(__half2float(a)), fabsf(__half2float(b)));
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
      if (lane == 0) wmax[warp][i] = m;
    }
  }
  __syncthreads();
  if (t < nrow) st.part[(size_t)(row0 + t) * (K / kRotGroup) + g] = fmaxf(wmax[0][t], wmax[1][t]);
  fd_pdl_trigger();
}

// Pass 2: grid (rows, sets): row absmax from the group partials, then vLLM's per-token INT8 quantization.
__global__ void __launch_bounds__(256) fd_w4a8_quant_kernel(ActPrepSets sets, int K) {
  fd_pdl();
  __shared__ float red[256];
  const ActPrepSet& st = sets.s[blockIdx.y];
  const int row = blockIdx.x, tid = threadIdx.x, ng = K / kRotGroup;
  float m = 0.0f;
  for (int gi = tid; gi < ng; gi += 256) m = fmaxf(m, st.part[(size_t)row * ng + gi]);
  red[tid] = m;
  __syncthreads();
  for (int o = 128; o > 0; o >>= 1) {
    if (tid < o) red[tid] = fmaxf(red[tid], red[tid + o]);
    __syncthreads();
  }
  const float absmax = fmaxf(red[0], 1e-10f);
  const float inv = div_full(127.0f, absmax);
  const __half2* xr = reinterpret_cast<const __half2*>(st.xr + (size_t)row * K);
  char2* q = reinterpret_cast<char2*>(st.xq + (size_t)row * K);
  for (int c = tid; c < K / 2; c += 256) {
    const float2 f = __half22float2(xr[c]);
    char2 o;
    o.x = static_cast<signed char>(static_cast<int>(round_half_away_triton(__fmul_rn(inv, f.x))));
    o.y = static_cast<signed char>(static_cast<int>(round_half_away_triton(__fmul_rn(inv, f.y))));
    q[c] = o;
  }
  if (tid == 0) st.a_scale[row] = __fmul_rn(div_full(absmax, 127.0f), *st.igs);
}


// ---------------------------------------------------------------------------
// Frame-block shift of the whole cache: one CTA per (layer, view, keys|values),
// frame pairs in increasing order (block f+1 is read completely before it is
// written by the next pair), rows x heads over the warps, 8 bytes per lane.
// ---------------------------------------------------------------------------
struct ShiftArgs {
  const void* const* k_ptrs;
  const void* const* v_ptrs;
  long long kv_rs, kv_hs;
  __nv_bfloat16* kvlm;
  const __nv_bfloat16* cos;
  const __nv_bfloat16* sin;
  const int* ranges;      // [n_views][F][2]
  int F, S, Hk;
};

__global__ void __launch_bounds__(256) fd_shift_cache_kernel(ShiftArgs a) {
  fd_pdl_wait();
  const int l = blockIdx.x, v = blockIdx.y, is_v = blockIdx.z;
  __nv_bfloat16* base = static_cast<__nv_bfloat16*>(const_cast<void*>(is_v ? a.v_ptrs[l] : a.k_ptrs[l]));
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int* rv = a.ranges + (size_t)v * a.F * 2;
  for (int f = 0; f + 1 < a.F; ++f) {
    const int d0 = rv[2 * f], d1 = rv[2 * f + 1], s0 = rv[2 * (f + 1)];
    const int n = (d1 - d0) * a.Hk;
    for (int idx = warp; idx < n; idx += 8) {
      const int r = idx / a.Hk, h = idx - r * a.Hk;
      const long long cs = s0 + r, cd = d0 + r;
      const Bf4 x = *reinterpret_cast<const Bf4*>(base + cs * a.kv_rs + h * a.kv_hs + 4 * lane);
      *reinterpret_cast<Bf4*>(base + cd * a.kv_rs + h * a.kv_hs + 4 * lane) = x;
      if (!is_v) {
        const Bf4 rr = rope_bf16_rule(x, a.cos + cd * kHeadDim, a.sin + cd * kHeadDim, lane);
        *reinterpret_cast<Bf4*>(a.kvlm + (((size_t)l * a.S + cd) * a.Hk + h) * kHeadDim + 4 * lane) = rr;
      }
    }
    __syncthreads();
  }
  fd_pdl_trigger();
}


// ---------------------------------------------------------------------------
// W4A8 few-row GEMM on mma.sync m16n8k32 (s8 x s8 -> s32). Persistent CTAs
// walk the 16-row output tiles; inside a tile the 8 warps split the 128-wide
// K groups (group g -> warp g % 8). Each warp streams its groups (1 KB of
// nibbles each) through a kSmStages-deep cp.async ring in shared memory, so the
// weight stream stays kSmStages - 1 groups ahead without holding registers, and
// prefetches the zero points / integer scales one group ahead; the int32 totals
// of the warps are added in shared memory (exact) before the epilogue.
// ---------------------------------------------------------------------------
constexpr int kSmWarps = 8;
constexpr int kSmStages = 8;     // groups in flight per warp (8 KB of shared memory per warp)

__device__ __forceinline__ void mma_s8_16x8x32(int* c, const unsigned* a, const unsigned* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

struct SmMeta { unsigned z0, z1; int s0, s1; };

template <int MT>
__global__ void __launch_bounds__(kSmWarps * 32) fd_w4a8_small_kernel(
    const int8_t* __restrict__ xq, const float* __restrict__ a_scale, const uint4* __restrict__ w_pack,
    const unsigned char* __restrict__ zp, const int* __restrict__ sg, __half* __restrict__ out, int M, int N, int K) {
  fd_pdl_wait();
  extern __shared__ __align__(16) unsigned char fd_sm_dyn[];
  uint4 (*wring)[kSmStages][2][32] = reinterpret_cast<uint4 (*)[kSmStages][2][32]>(fd_sm_dyn);   // [warp][stage][chunk][lane]
  __shared__ int red[2][kSmWarps][16][MT * 8];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int G = K / kRotGroup, GW = G / kSmWarps;                            // G % 8 == 0 (host check): uniform
  const int NT = N / 16;
  const int my_tiles = (NT - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x;
  const int n_items = my_tiles * GW;                                       // (tile, group) items of this warp
  auto item_tg = [&](int item, int& T, int& gi) {
    const int ti = item / GW;
    gi = warp + (item - ti * GW) * kSmWarps;
    T = blockIdx.x + ti * gridDim.x;
  };
  auto issue = [&](int item, int stage) {                                   // the item's two 512-byte chunks
    int T, gi; item_tg(item, T, gi);
    const uint4* wt = w_pack + ((size_t)T * (K / 64) + 2 * gi) * 32 + lane;
    cp_async16(&wring[warp][stage][0][lane], wt);
    cp_async16(&wring[warp][stage][1][lane], wt + 32);
  };
  auto meta = [&](int item) {
    int T, gi; item_tg(item, T, gi);
    const int n0 = 16 * T + g;
    SmMeta m;
    m.z0 = 0x01010101u * __ldg(zp + (size_t)gi * N + n0);
    m.z1 = 0x01010101u * __ldg(zp + (size_t)gi * N + n0 + 8);
    m.s0 = __ldg(sg + (size_t)gi * N + n0);
    m.s1 = __ldg(sg + (size_t)gi * N + n0 + 8);
    return m;
  };
  int total[MT][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int i = 0; i < 4; ++i) total[mt][i] = 0;
  auto tile_epilogue = [&](int ti) {       // exact int32 sum over the warps, then half(float(total) * a_scale[m])
    const int T = blockIdx.x + ti * gridDim.x;
    int (*rt)[16][MT * 8] = red[ti & 1];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      rt[warp][g][8 * mt + 2 * t] = total[mt][0];
      rt[warp][g][8 * mt + 2 * t + 1] = total[mt][1];
      rt[warp][g + 8][8 * mt + 2 * t] = total[mt][2];
      rt[warp][g + 8][8 * mt + 2 * t + 1] = total[mt][3];
      total[mt][0] = total[mt][1] = total[mt][2] = total[mt][3] = 0;
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < 16 * MT * 8; idx += kSmWarps * 32) {
      const int r = idx / (MT * 8), m = idx - r * (MT * 8);
      if (m >= M) continue;
      int v = 0;
#pragma unroll
      for (int w = 0; w < kSmWarps; ++w) v += rt[w][r][m];
      out[(size_t)m * N + 16 * T + r] = __float2half_rn(__fmul_rn(__int2float_rn(v), a_scale[m]));
    }
  };
  // prologue: fill the ring
#pragma unroll
  for (int st = 0; st < kSmStages - 1; ++st) {
    if (st < n_items) issue(st, st);
    cp_async_commit();
  }
  SmMeta mcur = meta(0);
  for (int base = 0; base < n_items; base += kSmStages) {
#pragma unroll
    for (int b = 0; b < kSmStages; ++b) {
      const int it = base + b;
      if (it < n_items) {
        // stage b holds item `it`; keep kSmStages - 1 groups in flight
        const int nxt = it + kSmStages - 1;
        if (nxt < n_items) issue(nxt, (b + kSmStages - 1) % kSmStages);
        cp_async_commit();
        cp_async_wait<kSmStages - 1>();
        __syncwarp();
        const SmMeta mn = (it + 1 < n_items) ? meta(it + 1) : mcur;
        int T, gi; item_tg(it, T, gi);
        int acc[MT][4];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int i = 0; i < 4; ++i) acc[mt][i] = 0;
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          const uint4 w4 = wring[warp][b][c][lane];
          const unsigned q[4] = {w4.x, w4.y, w4.z, w4.w};
#pragma unroll
          for (int ks = 0; ks < 2; ++ks) {
            unsigned a[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              const unsigned nib = ks ? ((q[i] >> 4) & 0x0F0F0F0Fu) : (q[i] & 0x0F0F0F0Fu);
              a[i] = __vsub4(nib, (i & 1) ? mcur.z1 : mcur.z0);
            }
            const int k0 = gi * kRotGroup + 64 * c + 32 * ks;
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
              const int8_t* xr = xq + (size_t)(8 * mt + g) * K + k0 + 4 * t;
              unsigned bb[2];
              bb[0] = *reinterpret_cast<const unsigned*>(xr);
              bb[1] = *reinterpret_cast<const unsigned*>(xr + 16);
              mma_s8_16x8x32(acc[mt], a, bb);
            }
          }
        }
        __syncwarp();                       // the stage is free for the next issue
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          total[mt][0] += acc[mt][0] * mcur.s0;
          total[mt][1] += acc[mt][1] * mcur.s0;
          total[mt][2] += acc[mt][2] * mcur.s1;
          total[mt][3] += acc[mt][3] * mcur.s1;
        }
        mcur = mn;
        if ((it + 1) % GW == 0) tile_epilogue(it / GW);
      }
    }
  }
  fd_pdl_trigger();
}

// ---------------------------------------------------------------------------
// BF16 few-row GEMM on mma.sync m16n8k16 (bf16 -> f32). Persistent CTAs walk
// the 16-row weight tiles; 8 warps split K in 64-wide chunks (chunk c -> warp
// c % 8), each lane loading its A fragments (rows g, g+8; k pairs 2t, 8+2t of
// every 16-wide step: the four lanes of a row read 32 contiguous bytes, so
// every sector of the weight tile is consumed) with a ring of chunks in flight
// that runs on across tiles; the FP32 partials of the warps are added in shared
// memory in a fixed order.
// ---------------------------------------------------------------------------
constexpr int kBfWarps = 8;
constexpr int kBfNB = 4;

__device__ __forceinline__ void mma_bf16_16x8x16(float* c, const unsigned* a, const unsigned* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

struct BfChunk { unsigned a[4][4]; };   // 4 k16 steps x (a0..a3)

template <int MT, bool kF32Out>
__global__ void __launch_bounds__(kBfWarps * 32) fd_bf16_small_kernel(
    const __nv_bfloat16* __restrict__ x, int ld_x, const __nv_bfloat16* __restrict__ w, void* __restrict__ y, int ld_y,
    int M, int N, int K) {
  fd_pdl_wait();
  __shared__ float red[2][kBfWarps][16][MT * 8];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int NC = K / 64, CW = NC / kBfWarps;                               // K % 512 == 0 (host check): uniform
  const int NT = (N + 15) / 16;
  const int my_tiles = (NT - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x;
  const int n_items = my_tiles * CW;
  auto load_item = [&](int item, BfChunk& ch) {
    const int ti = item / CW, c = warp + (item - ti * CW) * kBfWarps;
    const int T = blockIdx.x + ti * gridDim.x;
    const int n0 = 16 * T + g, n1 = n0 + 8;
    const __nv_bfloat16* w0 = w + (size_t)(n0 < N ? n0 : 0) * K;     // guarded rows read row 0 and are dropped
    const __nv_bfloat16* w1 = w + (size_t)(n1 < N ? n1 : 0) * K;
    const int k0 = 64 * c;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const int kk = k0 + 16 * s + 2 * t;
      ch.a[s][0] = __ldg(reinterpret_cast<const unsigned*>(w0 + kk));
      ch.a[s][1] = __ldg(reinterpret_cast<const unsigned*>(w1 + kk));
      ch.a[s][2] = __ldg(reinterpret_cast<const unsigned*>(w0 + kk + 8));
      ch.a[s][3] = __ldg(reinterpret_cast<const unsigned*>(w1 + kk + 8));
    }
  };
  float acc[MT][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[mt][i] = 0.0f;
  auto compute = [&](const BfChunk& ch, int c) {
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const int kk = 64 * c + 16 * s + 2 * t;
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const __nv_bfloat16* xr = x + (size_t)(8 * mt + g) * ld_x + kk;
        unsigned bb[2];
        bb[0] = *reinterpret_cast<const unsigned*>(xr);
        bb[1] = *reinterpret_cast<const unsigned*>(xr + 8);
        mma_bf16_16x8x16(acc[mt], ch.a[s], bb);
      }
    }
  };
  auto tile_epilogue = [&](int ti) {
    const int T = blockIdx.x + ti * gridDim.x;
    float (*rt)[16][MT * 8] = red[ti & 1];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      rt[warp][g][8 * mt + 2 * t] = acc[mt][0];
      rt[warp][g][8 * mt + 2 * t + 1] = acc[mt][1];
      rt[warp][g + 8][8 * mt + 2 * t] = acc[mt][2];
      rt[warp][g + 8][8 * mt + 2 * t + 1] = acc[mt][3];
      acc[mt][0] = acc[mt][1] = acc[mt][2] = acc[mt][3] = 0.0f;
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < 16 * MT * 8; idx += kBfWarps * 32) {
      const int r = idx / (MT * 8), m = idx - r * (MT * 8);
      const int n = 16 * T + r;
      if (m >= M || n >= N) continue;
      float v = 0.0f;
#pragma unroll
      for (int wi = 0; wi < kBfWarps; ++wi) v += rt[wi][r][m];
      if (kF32Out) static_cast<float*>(y)[(size_t)m * ld_y + n] = v;
      else static_cast<__nv_bfloat16*>(y)[(size_t)m * ld_y + n] = __float2bfloat16_rn(v);
    }
  };
  BfChunk ring[kBfNB];
#pragma unroll
  for (int b = 0; b < kBfNB; ++b)
    if (b < n_items) load_item(b, ring[b]);
  for (int base = 0; base < n_items; base += kBfNB) {
#pragma unroll
    for (int b = 0; b < kBfNB; ++b) {
      const int it = base + b;
      if (it < n_items) {
        const BfChunk ch = ring[b];
        if (it + kBfNB < n_items) load_item(it + kBfNB, ring[b]);
        const int ti = it / CW, cl = it - ti * CW;
        compute(ch, warp + cl * kBfWarps);
        if (cl + 1 == CW) tile_epilogue(ti);
      }
    }
  }
  fd_pdl_trigger();
}

// Draft attention operands: one warp per (row, head) over q rows then k|v rows.
__global__ void fd_draft_qkv_prep_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
                                         const __nv_bfloat16* __restrict__ v, int q_ld, int k_ld, int v_ld,
                                         const __nv_bfloat16* __restrict__ qw, const __nv_bfloat16* __restrict__ kw,
                                         const __nv_bfloat16* __restrict__ cos, const __nv_bfloat16* __restrict__ sin,
                                         __nv_bfloat16* __restrict__ q_out, __nv_bfloat16* __restrict__ k_out,
                                         __nv_bfloat16* __restrict__ v_out, int nq, int ctx, int Hq, int Hk, float eps) {
  fd_pdl();
  const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  const int nk = ctx + nq;
  const int nwq = nq * Hq, nwk = nk * Hk;
  if (gw >= nwq + 2 * nwk) return;
  if (gw >= nwq + nwk) {                                   // values
    const int i = gw - nwq - nwk, row = i / Hk, h = i - row * Hk;
    *reinterpret_cast<Bf4*>(v_out + ((size_t)row * Hk + h) * kHeadDim + 4 * lane) =
        *reinterpret_cast<const Bf4*>(v + (size_t)row * v_ld + h * kHeadDim + 4 * lane);
    return;
  }
  const bool is_q = gw < nwq;
  const int i = is_q ? gw : gw - nwq;
  const int H = is_q ? Hq : Hk;
  const int row = i / H, h = i - row * H;
  const int pos = is_q ? ctx + row : row;
  const Bf4 x = *reinterpret_cast<const Bf4*>((is_q ? q + (size_t)row * q_ld : k + (size_t)row * k_ld) + h * kHeadDim + 4 * lane);
  float ss = 0.0f;
#pragma unroll
  for (int e = 0; e < 4; ++e) ss += __fmul_rn(bf(x.v[e]), bf(x.v[e]));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  const float inv = rsqrtf(__fadd_rn(ss / (float)kHeadDim, eps));
  const Bf4 wv = *reinterpret_cast<const Bf4*>((is_q ? qw : kw) + 4 * lane);
  Bf4 n;
#pragma unroll
  for (int e = 0; e < 4; ++e) n.v[e] = to_bf(__fmul_rn(bf(wv.v[e]), bf(to_bf(__fmul_rn(bf(x.v[e]), inv)))));
  const Bf4 r = rope_bf16_rule(n, cos + (size_t)pos * kHeadDim, sin + (size_t)pos * kHeadDim, lane);
  *reinterpret_cast<Bf4*>((is_q ? q_out : k_out) + ((size_t)row * H + h) * kHeadDim + 4 * lane) = r;
}


// ---------------------------------------------------------------------------
// Dense rotation: one CTA per (128-row tile, group, set); the group's 128 x 128
// FP16 matrix sits in shared memory (row stride 136 halves: conflict-free
// fragment reads), each warp computes a 16-row x 128-column tile on
// mma.m16n8k16 (f16 -> f32) reading its A fragments straight from x (scaled
// and converted on load). Small row counts (<= 16) use one CTA per group whose
// warps split the 16 column tiles instead.
// ---------------------------------------------------------------------------
struct RotDenseSet {
  const __half* rot;      // [G][128][128]
  const __half* scales;   // [K]
  __half* xr;             // [rows, K]
  float* part;            // [rows, G]
};
struct RotDenseSets { RotDenseSet s[3]; };
constexpr int kRdPad = 136;

__device__ __forceinline__ void mma_f16_16x8x16(float* c, const unsigned* a, const unsigned* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// two scaled FP16 values of row `row` at channels k, k+1 (0 when the row is out of range)
__device__ __forceinline__ unsigned rd_load_a(const __nv_bfloat16* x, int ld_x, int row, int k, const __half* sc, bool ok) {
  if (!ok) return 0u;
  const __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(x + (size_t)row * ld_x + k);
  const __half2 s2 = *reinterpret_cast<const __half2*>(sc + k);
  __half2 h;
  h.x = __hmul(__float2half_rn(__bfloat162float(v.x)), s2.x);
  h.y = __hmul(__float2half_rn(__bfloat162float(v.y)), s2.y);
  return *reinterpret_cast<const unsigned*>(&h);
}

template <bool kSmallM>
__global__ void __launch_bounds__(256) fd_w4a8_rotate_dense_kernel(const __nv_bfloat16* __restrict__ x, int ld_x,
                                                                   RotDenseSets sets, int rows, int K) {
  fd_pdl_wait();
  __shared__ __align__(16) __half rs[kRotGroup * kRdPad];
  const RotDenseSet& st = sets.s[blockIdx.z];
  const int g = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int lg = lane >> 2, lt = lane & 3;
  const int G = K / kRotGroup;
  // the group's matrix -> shared memory (16 B per thread per step)
  {
    const uint4* src = reinterpret_cast<const uint4*>(st.rot + (size_t)g * kRotGroup * kRotGroup);
    for (int i = threadIdx.x; i < kRotGroup * kRotGroup / 8; i += 256) {
      const int r = i >> 4, c8 = i & 15;
      *reinterpret_cast<uint4*>(rs + r * kRdPad + c8 * 8) = src[i];
    }
  }
  __syncthreads();
  const int row_base = kSmallM ? 0 : blockIdx.x * 128 + warp * 16;
  const int nt0 = kSmallM ? warp * 2 : 0, ntn = kSmallM ? 2 : 16;
  const int r0 = row_base + lg, r1 = row_base + lg + 8;
  const bool ok0 = r0 < rows, ok1 = r1 < rows;
  const int kbase = g * kRotGroup;
  float acc[16][4];
#pragma unroll
  for (int i = 0; i < 16; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;
#pragma unroll
  for (int ks = 0; ks < 8; ++ks) {
    unsigned a[4];
    const int k = 16 * ks + 2 * lt;
    a[0] = rd_load_a(x, ld_x, r0, kbase + k, st.scales, ok0);
    a[1] = rd_load_a(x, ld_x, r1, kbase + k, st.scales, ok1);
    a[2] = rd_load_a(x, ld_x, r0, kbase + k + 8, st.scales, ok0);
    a[3] = rd_load_a(x, ld_x, r1, kbase + k + 8, st.scales, ok1);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      if (i < ntn) {
        const int nt = nt0 + i;
        const __half* rrow = rs + (nt * 8 + lg) * kRdPad + 16 * ks + 2 * lt;
        unsigned b[2];
        b[0] = *reinterpret_cast<const unsigned*>(rrow);
        b[1] = *reinterpret_cast<const unsigned*>(rrow + 8);
        mma_f16_16x8x16(acc[i], a, b);
      }
    }
  }
  // epilogue: FP16 rows + per-row absmax of this group
  float m0 = 0.0f, m1 = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    if (i < ntn) {
      const int col = kbase + (nt0 + i) * 8 + 2 * lt;
      const __half2 h0 = __floats2half2_rn(acc[i][0], acc[i][1]);
      const __half2 h1 = __floats2half2_rn(acc[i][2], acc[i][3]);
      if (ok0) *reinterpret_cast<__half2*>(st.xr + (size_t)r0 * K + col) = h0;
      if (ok1) *reinterpret_cast<__half2*>(st.xr + (size_t)r1 * K + col) = h1;
      const float2 f0 = __half22float2(h0), f1 = __half22float2(h1);
      m0 = fmaxf(m0, fmaxf(fabsf(f0.x), fabsf(f0.y)));
      m1 = fmaxf(m1, fmaxf(fabsf(f1.x), fabsf(f1.y)));
    }
  }
  if (kSmallM) {         // the warps hold different column tiles of the same rows: reduce across the block
    __shared__ float red[8][16];
    m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 1));
    m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
    m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 1));
    m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
    if (lt == 0) { red[warp][lg] = m0; red[warp][lg + 8] = m1; }
    __syncthreads();
    if (threadIdx.x < 16 && threadIdx.x < rows) {
      float t = 0.0f;
#pragma unroll
      for (int w = 0; w < 8; ++w) t = fmaxf(t, red[w][threadIdx.x]);
      st.part[(size_t)threadIdx.x * G + g] = t;
    }
  } else {
    m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 1));
    m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
    m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 1));
    m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
    if (lt == 0) {
      if (ok0) st.part[(size_t)r0 * G + g] = m0;
      if (ok1) st.part[(size_t)r1 * G + g] = m1;
    }
  }
  fd_pdl_trigger();
}


// ---------------------------------------------------------------------------
// LayerNorm (vision tower): one CTA per row, FP32 two-pass statistics (mean,
// then centred variance) over BF16 inputs, FP32 affine, one BF16 rounding.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256) fd_layernorm_kernel(const __nv_bfloat16* __restrict__ x,
                                                           const __nv_bfloat16* __restrict__ w,
                                                           const __nv_bfloat16* __restrict__ b,
                                                           __nv_bfloat16* __restrict__ out, int dim, float eps) {
  fd_pdl();
  __shared__ float smem[32];
  const int row = blockIdx.x;
  const Bf4* xr = reinterpret_cast<const Bf4*>(x + (size_t)row * dim);
  const int n4 = dim >> 2;
  float v[16][4];                       // up to 16 vectors of 4 per thread (dim <= 16384)
  float sum = 0.0f;
  int cnt = 0;
  for (int i = threadIdx.x; i < n4; i += 256, ++cnt) {
    const Bf4 t = xr[i];
#pragma unroll
    for (int e = 0; e < 4; ++e) { v[cnt][e] = bf(t.v[e]); sum += v[cnt][e]; }
  }
  const float mean = block_sum256(sum, smem) / (float)dim;
  float sq = 0.0f;
  for (int c = 0; c < cnt; ++c)
#pragma unroll
    for (int e = 0; e < 4; ++e) { const float d = v[c][e] - mean; sq += d * d; }
  const float rstd = rsqrtf(block_sum256(sq, smem) / (float)dim + eps);
  const Bf4* wr = reinterpret_cast<const Bf4*>(w);
  const Bf4* br = reinterpret_cast<const Bf4*>(b);
  Bf4* o = reinterpret_cast<Bf4*>(out + (size_t)row * dim);
  cnt = 0;
  for (int i = threadIdx.x; i < n4; i += 256, ++cnt) {
    const Bf4 ww = wr[i], bb = br[i];
    Bf4 r;
#pragma unroll
    for (int e = 0; e < 4; ++e) r.v[e] = to_bf((v[cnt][e] - mean) * rstd * bf(ww.v[e]) + bf(bb.v[e]));
    o[i] = r;
  }
}


// gate | up halves of one GEMM output row -> bf16(silu(g)) * u, rounded as silu_mul_qwen36_bf16.
__global__ void fd_silu_mul_split_kernel(const __nv_bfloat16* __restrict__ gu, __nv_bfloat16* __restrict__ out,
                                         int rows, int F) {
  fd_pdl();
  const long long n4 = (long long)rows * (F >> 2);
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (long long)gridDim.x * blockDim.x) {
    const int r = (int)(i / (F >> 2)), c = (int)(i - (long long)r * (F >> 2)) * 4;
    const Bf4 g = *reinterpret_cast<const Bf4*>(gu + (size_t)r * 2 * F + c);
    const Bf4 u = *reinterpret_cast<const Bf4*>(gu + (size_t)r * 2 * F + F + c);
    Bf4 o;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float gf = bf(g.v[e]);
      const float sl = bf(to_bf(gf / (1.0f + expf(-gf))));
      o.v[e] = to_bf(__fmul_rn(sl, bf(u.v[e])));
    }
    *reinterpret_cast<Bf4*>(out + (size_t)r * F + c) = o;
  }
}


// Per-row FP8 E4M3 quantization: one CTA per row, FP32 absmax, RNE conversion (saturating) as torch's .to(e4m3fn)
// after clamping to +-448.
__global__ void __launch_bounds__(256) fd_quantize_fp8_rows_kernel(const __nv_bfloat16* __restrict__ x, int ld_x,
                                                                   uint8_t* __restrict__ q, float* __restrict__ scale,
                                                                   int K) {
  fd_pdl();
  __shared__ float smem[32];
  const int row = blockIdx.x;
  const Bf4* xr = reinterpret_cast<const Bf4*>(x + (size_t)row * ld_x);
  const int n4 = K >> 2;
  float mx = 0.0f;
  for (int i = threadIdx.x; i < n4; i += 256) {
    const Bf4 t = xr[i];
#pragma unroll
    for (int e = 0; e < 4; ++e) mx = fmaxf(mx, fabsf(bf(t.v[e])));
  }
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = mx;
  __syncthreads();
  if (threadIdx.x < 32) {
    float t = threadIdx.x < 8 ? smem[threadIdx.x] : 0.0f;
    for (int o = 16; o > 0; o >>= 1) t = fmaxf(t, __shfl_xor_sync(0xffffffffu, t, o));
    if (threadIdx.x == 0) smem[0] = t;
  }
  __syncthreads();
  const float amax = smem[0];
  const float s = amax > 0.0f ? amax / 448.0f : 1.0f;
  const float inv = 1.0f / s;
  if (threadIdx.x == 0) scale[row] = s;
  uint32_t* qr = reinterpret_cast<uint32_t*>(q + (size_t)row * K);
  for (int i = threadIdx.x; i < n4; i += 256) {
    const Bf4 t = xr[i];
    float f[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) f[e] = fminf(fmaxf(bf(t.v[e]) * inv, -448.0f), 448.0f);
    const __nv_fp8x2_storage_t lo = __nv_cvt_float2_to_fp8x2(make_float2(f[0], f[1]), __NV_SATFINITE, __NV_E4M3);
    const __nv_fp8x2_storage_t hi = __nv_cvt_float2_to_fp8x2(make_float2(f[2], f[3]), __NV_SATFINITE, __NV_E4M3);
    qr[i] = (uint32_t)lo | ((uint32_t)hi << 16);
  }
}


// ---------------------------------------------------------------------------
// Streaming few-row GEMMs: warp per 16-row tile, full K, 16-byte streaming
// loads, four warps per CTA (the structure of the W8A16 decode kernel).
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint4 fd_ld_stream16(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

// KS warps share one 16-row tile (each streams K / KS); the partials are summed through shared memory in a fixed
// order. The CTA always holds 4 warps, i.e. 4 / KS tiles; grid = ceil(tiles * KS / 4).
template <int MT, int KS>
__global__ void __launch_bounds__(128) fd_w4a8_tile_kernel(const int8_t* __restrict__ xq, const float* __restrict__ a_scale,
                                                           const uint4* __restrict__ w_pack, const unsigned char* __restrict__ zp,
                                                           const int* __restrict__ sg, __half* __restrict__ out, int M, int N, int K) {
  fd_pdl_wait();
  __shared__ int red[4][MT * 4][32];
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3, warp = threadIdx.x >> 5;
  const int T = blockIdx.x * (4 / KS) + warp / KS, ks_id = warp % KS;
  const int NT = N / 16;
  const bool valid = T < NT;
  const int G = K / kRotGroup, GS = G / KS;
  const uint4* wt = w_pack + (size_t)(valid ? T : 0) * (K / 64) * 32 + lane;
  const int n0 = 16 * (valid ? T : 0) + g, n1 = n0 + 8;
  int total[MT][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int i = 0; i < 4; ++i) total[mt][i] = 0;
  if (valid) {
#pragma unroll 2
    for (int gl = 0; gl < GS; ++gl) {
      const int gi = ks_id * GS + gl;
      const uint4 w0 = fd_ld_stream16(wt + (size_t)(2 * gi) * 32);
      const uint4 w1 = fd_ld_stream16(wt + (size_t)(2 * gi + 1) * 32);
      const unsigned z0 = 0x01010101u * __ldg(zp + (size_t)gi * N + n0);
      const unsigned z1 = 0x01010101u * __ldg(zp + (size_t)gi * N + n1);
      const int s0 = __ldg(sg + (size_t)gi * N + n0), s1 = __ldg(sg + (size_t)gi * N + n1);
      int acc[MT][4];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[mt][i] = 0;
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const uint4 w4 = c ? w1 : w0;
        const unsigned q[4] = {w4.x, w4.y, w4.z, w4.w};
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
          unsigned a[4];
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const unsigned nib = kk ? ((q[i] >> 4) & 0x0F0F0F0Fu) : (q[i] & 0x0F0F0F0Fu);
            a[i] = __vsub4(nib, (i & 1) ? z1 : z0);
          }
          const int k0 = gi * kRotGroup + 64 * c + 32 * kk;
#pragma unroll
          for (int mt = 0; mt < MT; ++mt) {
            const int8_t* xr = xq + (size_t)(8 * mt + g) * K + k0 + 4 * t;
            unsigned bb[2];
            bb[0] = __ldg(reinterpret_cast<const unsigned*>(xr));
            bb[1] = __ldg(reinterpret_cast<const unsigned*>(xr + 16));
            mma_s8_16x8x32(acc[mt], a, bb);
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
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    const int m0 = 8 * mt + 2 * t, m1 = m0 + 1;
    if (m0 < M) {
      out[(size_t)m0 * N + n0] = __float2half_rn(__fmul_rn(__int2float_rn(total[mt][0]), a_scale[m0]));
      out[(size_t)m0 * N + n1] = __float2half_rn(__fmul_rn(__int2float_rn(total[mt][2]), a_scale[m0]));
    }
    if (m1 < M) {
      out[(size_t)m1 * N + n0] = __float2half_rn(__fmul_rn(__int2float_rn(total[mt][1]), a_scale[m1]));
      out[(size_t)m1 * N + n1] = __float2half_rn(__fmul_rn(__int2float_rn(total[mt][3]), a_scale[m1]));
    }
  }
  fd_pdl_trigger();
}

template <int MT, int KS, bool kF32Out>
__global__ void __launch_bounds__(128) fd_bf16_frag_kernel(const __nv_bfloat16* __restrict__ x, int ld_x,
                                                           const uint4* __restrict__ w_frag, void* __restrict__ y, int ld_y,
                                                           int M, int N, int K) {
  fd_pdl_wait();
  __shared__ float red[4][MT * 4][32];
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3, warp = threadIdx.x >> 5;
  const int T = blockIdx.x * (4 / KS) + warp / KS, ks_id = warp % KS;
  const int NT = (N + 15) / 16;
  const bool valid = T < NT;
  const int KC = K / 16, KCS = KC / KS;
  const uint4* wt = w_frag + (size_t)(valid ? T : 0) * KC * 32 + lane;
  float acc[MT][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[mt][i] = 0.0f;
  if (valid) {
#pragma unroll 4
    for (int kl = 0; kl < KCS; ++kl) {
      const int kc = ks_id * KCS + kl;
      const uint4 w4 = fd_ld_stream16(wt + (size_t)kc * 32);
      const unsigned a[4] = {w4.x, w4.y, w4.z, w4.w};
      const int kk = 16 * kc + 2 * t;
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const __nv_bfloat16* xr = x + (size_t)(8 * mt + g) * ld_x + kk;
        unsigned bb[2];
        bb[0] = __ldg(reinterpret_cast<const unsigned*>(xr));
        bb[1] = __ldg(reinterpret_cast<const unsigned*>(xr + 8));
        mma_bf16_16x8x16(acc[mt], a, bb);
      }
    }
  }
  if (KS > 1) {
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int i = 0; i < 4; ++i) red[warp][mt * 4 + i][lane] = acc[mt][i];
    __syncthreads();
    if (ks_id != 0) return;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < KS; ++w) v += red[warp + w][mt * 4 + i][lane];
        acc[mt][i] = v;
      }
  }
  if (!valid) return;
  const int n0 = 16 * T + g, n1 = n0 + 8;
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    const int m0 = 8 * mt + 2 * t, m1 = m0 + 1;
    float v[4] = {acc[mt][0], acc[mt][1], acc[mt][2], acc[mt][3]};
    const int ms[4] = {m0, m1, m0, m1}, ns[4] = {n0, n0, n1, n1};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      if (ms[i] < M && ns[i] < N) {
        if (kF32Out) static_cast<float*>(y)[(size_t)ms[i] * ld_y + ns[i]] = v[i];
        else static_cast<__nv_bfloat16*>(y)[(size_t)ms[i] * ld_y + ns[i]] = __float2bfloat16_rn(v[i]);
      }
    }
  }
  fd_pdl_trigger();
}

}  // namespace

int flashdrive_qk_prep_bf16(const void* q, const void* k, const void* v, int q_ld, int k_ld, int v_ld,
                            const void* q_norm_w, const void* k_norm_w, const void* cos, const void* sin, int fp32_rule,
                            int f16_in, void* q_out, void* k_raw, long long kr_rs, long long kr_hs,
                            void* k_rope, long long k_rs, long long k_hs,
                            void* v_out, long long v_rs, long long v_hs,
                            const long long* row_map, const long long* slot_ptr,
                            int rows, int Hq, int Hk, float eps, cudaStream_t stream) {
  if (!q || !k || !v || !q_norm_w || !k_norm_w || !cos || !sin || !q_out || !k_raw || !k_rope || !v_out) return 1;
  if (rows <= 0 || Hq <= 0 || Hk <= 0 || q_ld < Hq * kHeadDim || k_ld < Hk * kHeadDim || v_ld < Hk * kHeadDim)
    return 2;
  if (!aligned8(q) || !aligned8(k) || !aligned8(v) || !aligned8(q_norm_w) || !aligned8(k_norm_w) || !aligned8(cos) ||
      !aligned8(sin) || !aligned8(q_out) || !aligned8(k_raw) || !aligned8(k_rope) || !aligned8(v_out) ||
      (q_ld % 4) || (k_ld % 4) || (v_ld % 4) || !stride_ok(kr_rs) || !stride_ok(kr_hs) || !stride_ok(k_rs) ||
      !stride_ok(k_hs) || !stride_ok(v_rs) || !stride_ok(v_hs))
    return 5;
  const long long warps = (long long)rows * (Hq + 2 * Hk);
  const unsigned blocks = (unsigned)((warps + 7) / 8);
#define FD_QK_PREP_ARGS                                                                                        \
  q, k, v, q_ld, k_ld, v_ld, static_cast<const __nv_bfloat16*>(q_norm_w), static_cast<const __nv_bfloat16*>(k_norm_w),  \
      cos, sin, static_cast<__nv_bfloat16*>(q_out), static_cast<__nv_bfloat16*>(k_raw), kr_rs, kr_hs,               \
      static_cast<__nv_bfloat16*>(k_rope), k_rs, k_hs, static_cast<__nv_bfloat16*>(v_out), v_rs, v_hs, row_map,       \
      slot_ptr, rows, Hq, Hk, eps
  if (fp32_rule && f16_in) {
    fd_launch(fd_qk_prep_kernel<true, true>, dim3(blocks), dim3(256), (size_t)(0), stream, FD_QK_PREP_ARGS);
  } else if (fp32_rule) {
    fd_launch(fd_qk_prep_kernel<true, false>, dim3(blocks), dim3(256), (size_t)(0), stream, FD_QK_PREP_ARGS);
  } else if (f16_in) {
    fd_launch(fd_qk_prep_kernel<false, true>, dim3(blocks), dim3(256), (size_t)(0), stream, FD_QK_PREP_ARGS);
  } else {
    fd_launch(fd_qk_prep_kernel<false, false>, dim3(blocks), dim3(256), (size_t)(0), stream, FD_QK_PREP_ARGS);
  }
#undef FD_QK_PREP_ARGS
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int flashdrive_rms_f16delta_bf16(void* residual, const void* delta, const void* weight, void* out, int rows, int dim,
                                 float eps, cudaStream_t stream) {
  if (!residual || !delta || !weight || !out) return 1;
  if (rows <= 0 || dim <= 0) return 2;
  fd_launch(fd_rms_f16delta_kernel, dim3(rows), dim3(256), (size_t)(0), stream, static_cast<__nv_bfloat16*>(residual), static_cast<const __half*>(delta),
                                                   static_cast<const __nv_bfloat16*>(weight),
                                                   static_cast<__nv_bfloat16*>(out), dim, eps);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int flashdrive_silu_mul_f16in_bf16(const void* gate, const void* up, void* out, long long n, cudaStream_t stream) {
  if (!gate || !up || !out) return 1;
  if (n <= 0) return 2;
  const long long blocks = (n + 255) / 256;
  fd_launch(fd_silu_mul_f16in_kernel, dim3((unsigned)(blocks > 4096 ? 4096 : blocks)), dim3(256), (size_t)(0), stream, 
      static_cast<const __half*>(gate), static_cast<const __half*>(up), static_cast<__nv_bfloat16*>(out), n);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int flashdrive_rope_rows(const void* src, long long s_rs, long long s_hs,
                         void* dst, long long d_rs, long long d_hs,
                         const void* cos, const void* sin, int fp32_rule,
                         const long long* row_map, long long row0, const long long* slot_ptr,
                         const int* n_ptr, int rows, int Hk, cudaStream_t stream) {
  if (!src || !dst || !cos || !sin) return 1;
  if (rows <= 0 || Hk <= 0 || row0 < 0) return 2;
  if (!aligned8(src) || !aligned8(dst) || !aligned8(cos) || !aligned8(sin) || !stride_ok(s_rs) ||
      !stride_ok(s_hs) || !stride_ok(d_rs) || !stride_ok(d_hs))
    return 5;
  const long long warps = (long long)rows * Hk;
  const unsigned blocks = (unsigned)((warps + 7) / 8);
  if (fp32_rule) {
    fd_launch(fd_rope_rows_kernel<true>, dim3(blocks), dim3(256), (size_t)(0), stream, 
        static_cast<const __nv_bfloat16*>(src), s_rs, s_hs, static_cast<__nv_bfloat16*>(dst), d_rs, d_hs, cos, sin,
        row_map, row0, slot_ptr, n_ptr, rows, Hk);
  } else {
    fd_launch(fd_rope_rows_kernel<false>, dim3(blocks), dim3(256), (size_t)(0), stream, 
        static_cast<const __nv_bfloat16*>(src), s_rs, s_hs, static_cast<__nv_bfloat16*>(dst), d_rs, d_hs, cos, sin,
        row_map, row0, slot_ptr, n_ptr, rows, Hk);
  }
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int flashdrive_w4a8_act_prep(const void* x, int ld_x, int rows, int K, int n_sets,
                             const void* const* theta, const void* const* pairs, const void* const* scales,
                             const float* const* igs, void* const* xr, float* const* part, void* const* xq,
                             float* const* a_scale, cudaStream_t stream) {
  if (!x || rows <= 0 || K <= 0 || n_sets < 1 || n_sets > 3) return 2;
  if ((K % kRotGroup) != 0 || ld_x < K || (ld_x % 2) != 0) return 2;
  if ((reinterpret_cast<uintptr_t>(x) & 3u) != 0 || (ld_x % 2) != 0) return 5;
  ActPrepSets sets{};
  for (int i = 0; i < n_sets; ++i) {
    if (!theta[i] || !pairs[i] || !scales[i] || !igs[i] || !xr[i] || !part[i] || !xq[i] || !a_scale[i]) return 1;
    if ((reinterpret_cast<uintptr_t>(pairs[i]) & 3u) != 0 || (reinterpret_cast<uintptr_t>(scales[i]) & 3u) != 0 ||
        (reinterpret_cast<uintptr_t>(xr[i]) & 3u) != 0 || (reinterpret_cast<uintptr_t>(xq[i]) & 1u) != 0)
      return 5;
    sets.s[i] = ActPrepSet{static_cast<const __half*>(theta[i]), static_cast<const int16_t*>(pairs[i]),
                           static_cast<const __half*>(scales[i]), igs[i], static_cast<__half*>(xr[i]), part[i],
                           static_cast<int8_t*>(xq[i]), a_scale[i]};
  }
  const auto* xb = static_cast<const __nv_bfloat16*>(x);
  if (rows <= 2) {
    fd_launch(fd_w4a8_rotate_kernel<2>, dim3(dim3((rows + 1) / 2, K / kRotGroup, n_sets)), dim3(64), (size_t)(0), stream, xb, ld_x, sets, rows, K);
  } else if (rows <= 8) {
    fd_launch(fd_w4a8_rotate_kernel<4>, dim3(dim3((rows + 3) / 4, K / kRotGroup, n_sets)), dim3(64), (size_t)(0), stream, xb, ld_x, sets, rows, K);
  } else {
    fd_launch(fd_w4a8_rotate_kernel<kRotRows>, dim3(dim3((rows + kRotRows - 1) / kRotRows, K / kRotGroup, n_sets)), dim3(64), (size_t)(0), stream, 
        xb, ld_x, sets, rows, K);
  }
  fd_launch(fd_w4a8_quant_kernel, dim3(dim3(rows, n_sets)), dim3(256), (size_t)(0), stream, sets, K);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int flashdrive_shift_cache(const void* const* k_ptrs, const void* const* v_ptrs, long long kv_rs, long long kv_hs,
                           void* kvlm, const void* cos, const void* sin, const int* ranges, int n_views, int F,
                           int L, int S, int Hk, cudaStream_t stream) {
  if (!k_ptrs || !v_ptrs || !kvlm || !cos || !sin || !ranges) return 1;
  if (n_views <= 0 || F < 2 || L <= 0 || S <= 0 || Hk <= 0) return 2;
  if (!aligned8(kvlm) || !aligned8(cos) || !aligned8(sin) || !stride_ok(kv_rs) || !stride_ok(kv_hs)) return 5;
  ShiftArgs a{k_ptrs, v_ptrs, kv_rs, kv_hs, static_cast<__nv_bfloat16*>(kvlm), static_cast<const __nv_bfloat16*>(cos),
              static_cast<const __nv_bfloat16*>(sin), ranges, F, S, Hk};
  fd_launch(fd_shift_cache_kernel, dim3(dim3(L, n_views, 2)), dim3(256), (size_t)(0), stream, a);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int flashdrive_w4a8_gemm_small(const void* xq, const float* a_scale, const void* w_pack, const void* zp,
                               const int* sg, void* out, int M, int N, int K, cudaStream_t stream) {
  if (!xq || !a_scale || !w_pack || !zp || !sg || !out) return 1;
  if (M <= 0 || M > 16 || N <= 0 || (N % 16) != 0 || K <= 0 || (K % (kRotGroup * kSmWarps)) != 0) return 2;
  if ((reinterpret_cast<uintptr_t>(w_pack) & 15u) != 0 || (reinterpret_cast<uintptr_t>(xq) & 3u) != 0 ||
      (reinterpret_cast<uintptr_t>(sg) & 3u) != 0 || (reinterpret_cast<uintptr_t>(out) & 1u) != 0)
    return 5;
  static int n_sm = 0;
  if (!n_sm) cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, 0);
  const int NT = N / 16;
  const dim3 grid(NT < 2 * n_sm ? NT : 2 * n_sm);          // persistent: up to 2 CTAs per SM (shared-memory ring)
  constexpr int kRingBytes = kSmWarps * kSmStages * 2 * 32 * 16;
  static bool configured = false;
  if (!configured) {
    if (cudaFuncSetAttribute(fd_w4a8_small_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, kRingBytes) != cudaSuccess ||
        cudaFuncSetAttribute(fd_w4a8_small_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, kRingBytes) != cudaSuccess)
      return 6;
    configured = true;
  }
  if (M <= 8) {
    fd_launch(fd_w4a8_small_kernel<1>, dim3(grid), dim3(kSmWarps * 32), (size_t)(kRingBytes), stream, 
        static_cast<const int8_t*>(xq), a_scale, static_cast<const uint4*>(w_pack),
        static_cast<const unsigned char*>(zp), sg, static_cast<__half*>(out), M, N, K);
  } else {
    fd_launch(fd_w4a8_small_kernel<2>, dim3(grid), dim3(kSmWarps * 32), (size_t)(kRingBytes), stream, 
        static_cast<const int8_t*>(xq), a_scale, static_cast<const uint4*>(w_pack),
        static_cast<const unsigned char*>(zp), sg, static_cast<__half*>(out), M, N, K);
  }
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int flashdrive_bf16_gemm_small(const void* x, int ld_x, const void* w, void* y, int ld_y, int y_f32, int M, int N,
                               int K, cudaStream_t stream) {
  if (!x || !w || !y) return 1;
  if (M <= 0 || M > 16 || N <= 0 || K <= 0 || (K % (64 * kBfWarps)) != 0 || ld_x < K || (ld_x % 2) != 0 || ld_y < N) return 2;
  if ((reinterpret_cast<uintptr_t>(x) & 3u) != 0 || (reinterpret_cast<uintptr_t>(w) & 3u) != 0 ||
      (y_f32 && (reinterpret_cast<uintptr_t>(y) & 3u) != 0) || (!y_f32 && (reinterpret_cast<uintptr_t>(y) & 1u) != 0))
    return 5;
  static int n_sm = 0;
  if (!n_sm) cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, 0);
  const int NT = (N + 15) / 16;
  const dim3 grid(NT < 3 * n_sm ? NT : 3 * n_sm);          // persistent: up to 3 CTAs per SM
  const auto* xb = static_cast<const __nv_bfloat16*>(x);
  const auto* wb = static_cast<const __nv_bfloat16*>(w);
  if (M <= 8) {
    if (y_f32) fd_launch(fd_bf16_small_kernel<1, true>, dim3(grid), dim3(kBfWarps * 32), (size_t)(0), stream, xb, ld_x, wb, y, ld_y, M, N, K);
    else fd_launch(fd_bf16_small_kernel<1, false>, dim3(grid), dim3(kBfWarps * 32), (size_t)(0), stream, xb, ld_x, wb, y, ld_y, M, N, K);
  } else {
    if (y_f32) fd_launch(fd_bf16_small_kernel<2, true>, dim3(grid), dim3(kBfWarps * 32), (size_t)(0), stream, xb, ld_x, wb, y, ld_y, M, N, K);
    else fd_launch(fd_bf16_small_kernel<2, false>, dim3(grid), dim3(kBfWarps * 32), (size_t)(0), stream, xb, ld_x, wb, y, ld_y, M, N, K);
  }
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int flashdrive_draft_qkv_prep_bf16(const void* q, const void* k, const void* v, int q_ld, int k_ld, int v_ld,
                                   const void* q_norm_w, const void* k_norm_w, const void* cos, const void* sin,
                                   void* q_out, void* k_out, void* v_out, int nq, int ctx, int Hq, int Hk, float eps,
                                   cudaStream_t stream) {
  if (!q || !k || !v || !q_norm_w || !k_norm_w || !cos || !sin || !q_out || !k_out || !v_out) return 1;
  if (nq <= 0 || ctx < 0 || Hq <= 0 || Hk <= 0 || q_ld < Hq * kHeadDim || k_ld < Hk * kHeadDim || v_ld < Hk * kHeadDim)
    return 2;
  if (!aligned8(q) || !aligned8(k) || !aligned8(v) || !aligned8(cos) || !aligned8(sin) || !aligned8(q_out) ||
      !aligned8(k_out) || !aligned8(v_out) || (q_ld % 4) || (k_ld % 4) || (v_ld % 4))
    return 5;
  const long long warps = (long long)nq * Hq + 2LL * (ctx + nq) * Hk;
  fd_launch(fd_draft_qkv_prep_kernel, dim3((unsigned)((warps + 7) / 8)), dim3(256), (size_t)(0), stream, 
      static_cast<const __nv_bfloat16*>(q), static_cast<const __nv_bfloat16*>(k), static_cast<const __nv_bfloat16*>(v),
      q_ld, k_ld, v_ld, static_cast<const __nv_bfloat16*>(q_norm_w), static_cast<const __nv_bfloat16*>(k_norm_w),
      static_cast<const __nv_bfloat16*>(cos), static_cast<const __nv_bfloat16*>(sin), static_cast<__nv_bfloat16*>(q_out),
      static_cast<__nv_bfloat16*>(k_out), static_cast<__nv_bfloat16*>(v_out), nq, ctx, Hq, Hk, eps);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int flashdrive_w4a8_rotate_dense(const void* x, int ld_x, int rows, int K, int n_sets, const void* const* rot,
                                 const void* const* scales, void* const* xr, float* const* part, cudaStream_t stream) {
  if (!x || rows <= 0 || K <= 0 || n_sets < 1 || n_sets > 3) return 2;
  if ((K % kRotGroup) != 0 || ld_x < K || (ld_x % 2) != 0) return 2;
  if ((reinterpret_cast<uintptr_t>(x) & 3u) != 0) return 5;
  RotDenseSets sets{};
  for (int i = 0; i < n_sets; ++i) {
    if (!rot[i] || !scales[i] || !xr[i] || !part[i]) return 1;
    if ((reinterpret_cast<uintptr_t>(rot[i]) & 15u) != 0 || (reinterpret_cast<uintptr_t>(scales[i]) & 3u) != 0 ||
        (reinterpret_cast<uintptr_t>(xr[i]) & 3u) != 0)
      return 5;
    sets.s[i] = RotDenseSet{static_cast<const __half*>(rot[i]), static_cast<const __half*>(scales[i]),
                            static_cast<__half*>(xr[i]), part[i]};
  }
  const auto* xb = static_cast<const __nv_bfloat16*>(x);
  if (rows <= 16) {
    fd_launch(fd_w4a8_rotate_dense_kernel<true>, dim3(dim3(1, K / kRotGroup, n_sets)), dim3(256), (size_t)(0), stream, xb, ld_x, sets, rows, K);
  } else {
    fd_launch(fd_w4a8_rotate_dense_kernel<false>, dim3(dim3((rows + 127) / 128, K / kRotGroup, n_sets)), dim3(256), (size_t)(0), stream, 
        xb, ld_x, sets, rows, K);
  }
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int flashdrive_w4a8_quant(int rows, int K, int n_sets, const void* const* xr, const float* const* part,
                          const float* const* igs, void* const* xq, float* const* a_scale, cudaStream_t stream) {
  if (rows <= 0 || K <= 0 || (K % kRotGroup) != 0 || n_sets < 1 || n_sets > 3) return 2;
  ActPrepSets sets{};
  for (int i = 0; i < n_sets; ++i) {
    if (!xr[i] || !part[i] || !igs[i] || !xq[i] || !a_scale[i]) return 1;
    if ((reinterpret_cast<uintptr_t>(xr[i]) & 3u) != 0 || (reinterpret_cast<uintptr_t>(xq[i]) & 1u) != 0) return 5;
    sets.s[i] = ActPrepSet{nullptr, nullptr, nullptr, igs[i], static_cast<__half*>(const_cast<void*>(xr[i])),
                           const_cast<float*>(part[i]), static_cast<int8_t*>(xq[i]), a_scale[i]};
  }
  fd_launch(fd_w4a8_quant_kernel, dim3(dim3(rows, n_sets)), dim3(256), (size_t)(0), stream, sets, K);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int flashdrive_layernorm_bf16(const void* x, const void* w, const void* b, void* out, int rows, int dim, float eps,
                              cudaStream_t stream) {
  if (!x || !w || !b || !out) return 1;
  if (rows <= 0 || dim <= 0 || (dim % 8) != 0 || dim > 16384) return 2;
  if (!aligned8(x) || !aligned8(w) || !aligned8(b) || !aligned8(out)) return 5;
  fd_launch(fd_layernorm_kernel, dim3(rows), dim3(256), (size_t)(0), stream, static_cast<const __nv_bfloat16*>(x), static_cast<const __nv_bfloat16*>(w),
                                                static_cast<const __nv_bfloat16*>(b), static_cast<__nv_bfloat16*>(out),
                                                dim, eps);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int flashdrive_silu_mul_split_bf16(const void* gu, void* out, int rows, int F, cudaStream_t stream) {
  if (!gu || !out) return 1;
  if (rows <= 0 || F <= 0 || (F % 4) != 0) return 2;
  if (!aligned8(gu) || !aligned8(out)) return 5;
  const long long n4 = (long long)rows * (F / 4);
  const long long blocks = (n4 + 255) / 256;
  fd_launch(fd_silu_mul_split_kernel, dim3((unsigned)(blocks > 4096 ? 4096 : blocks)), dim3(256), (size_t)(0), stream, 
      static_cast<const __nv_bfloat16*>(gu), static_cast<__nv_bfloat16*>(out), rows, F);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int flashdrive_quantize_fp8_rows_bf16(const void* x, int ld_x, void* q, float* scale, int rows, int K, cudaStream_t stream) {
  if (!x || !q || !scale) return 1;
  if (rows <= 0 || K <= 0 || (K % 8) != 0 || ld_x < K || (ld_x % 4) != 0) return 2;
  if (!aligned8(x) || (reinterpret_cast<uintptr_t>(q) & 3u) != 0) return 5;
  fd_launch(fd_quantize_fp8_rows_kernel, dim3(rows), dim3(256), (size_t)(0), stream, static_cast<const __nv_bfloat16*>(x), ld_x, static_cast<uint8_t*>(q),
                                                        scale, K);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

void flashdrive_set_pdl(bool on) { g_fd_pdl = on; }
bool flashdrive_pdl() { return g_fd_pdl; }

namespace {
// K splits per tile so that at least ~128 CTAs stream concurrently (and K stays divisible).
int tile_ks(int NT, int K, int kunit) {
  int ks = 1;
  while (ks < 4 && NT * ks / 4 < 128 && (K / kunit) % (ks * 2) == 0) ks *= 2;
  return ks;
}
}  // namespace

int flashdrive_w4a8_gemm_tile(const void* xq, const float* a_scale, const void* w_pack, const void* zp, const int* sg,
                              void* out, int M, int N, int K, cudaStream_t stream) {
  if (!xq || !a_scale || !w_pack || !zp || !sg || !out) return 1;
  if (M <= 0 || M > 16 || N <= 0 || (N % 16) != 0 || K <= 0 || (K % kRotGroup) != 0) return 2;
  if ((reinterpret_cast<uintptr_t>(w_pack) & 15u) != 0 || (reinterpret_cast<uintptr_t>(xq) & 3u) != 0 ||
      (reinterpret_cast<uintptr_t>(sg) & 3u) != 0 || (reinterpret_cast<uintptr_t>(out) & 1u) != 0)
    return 5;
  const int NT = N / 16, ks = tile_ks(NT, K, kRotGroup);
  const dim3 grid((NT * ks + 3) / 4);
  const auto* x8 = static_cast<const int8_t*>(xq);
  const auto* wp = static_cast<const uint4*>(w_pack);
  const auto* z8 = static_cast<const unsigned char*>(zp);
  auto* o = static_cast<__half*>(out);
#define FD_TILE(MT, KS) fd_launch(fd_w4a8_tile_kernel<MT, KS>, grid, dim3(128), (size_t)0, stream, x8, a_scale, wp, z8, sg, o, M, N, K)
  if (M <= 8) { if (ks == 1) FD_TILE(1, 1); else if (ks == 2) FD_TILE(1, 2); else FD_TILE(1, 4); }
  else { if (ks == 1) FD_TILE(2, 1); else if (ks == 2) FD_TILE(2, 2); else FD_TILE(2, 4); }
#undef FD_TILE
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int flashdrive_bf16_gemm_frag(const void* x, int ld_x, const void* w_frag, void* y, int ld_y, int y_f32, int M, int N,
                              int K, cudaStream_t stream) {
  if (!x || !w_frag || !y) return 1;
  if (M <= 0 || M > 16 || N <= 0 || K <= 0 || (K % 16) != 0 || ld_x < K || (ld_x % 2) != 0 || ld_y < N) return 2;
  if ((reinterpret_cast<uintptr_t>(x) & 3u) != 0 || (reinterpret_cast<uintptr_t>(w_frag) & 15u) != 0 ||
      (y_f32 && (reinterpret_cast<uintptr_t>(y) & 3u) != 0) || (!y_f32 && (reinterpret_cast<uintptr_t>(y) & 1u) != 0))
    return 5;
  const int NT = (N + 15) / 16, ks = tile_ks(NT, K, 16);
  const dim3 grid((NT * ks + 3) / 4);
  const auto* xb = static_cast<const __nv_bfloat16*>(x);
  const auto* wf = static_cast<const uint4*>(w_frag);
#define FD_FRAG(MT, KS, F) fd_launch(fd_bf16_frag_kernel<MT, KS, F>, grid, dim3(128), (size_t)0, stream, xb, ld_x, wf, y, ld_y, M, N, K)
  if (M <= 8) {
    if (y_f32) { if (ks == 1) FD_FRAG(1, 1, true); else if (ks == 2) FD_FRAG(1, 2, true); else FD_FRAG(1, 4, true); }
    else { if (ks == 1) FD_FRAG(1, 1, false); else if (ks == 2) FD_FRAG(1, 2, false); else FD_FRAG(1, 4, false); }
  } else {
    if (y_f32) { if (ks == 1) FD_FRAG(2, 1, true); else if (ks == 2) FD_FRAG(2, 2, true); else FD_FRAG(2, 4, true); }
    else { if (ks == 1) FD_FRAG(2, 1, false); else if (ks == 2) FD_FRAG(2, 2, false); else FD_FRAG(2, 4, false); }
  }
#undef FD_FRAG
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

}  // namespace kernels
}  // namespace flash_rt
