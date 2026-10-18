// SPDX-License-Identifier: Apache-2.0
//
// FlashDrive (Alpamayo 1.5) backend kernels for Thor (sm_110). See
// alpamayo15_flashdrive_kernels.cuh for the cache contract and rounding rules.

#include "alpamayo15_flashdrive_kernels.cuh"
#include "alpamayo15_fp8_rows.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cooperative_groups.h>

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

// Same, with a thread-block cluster of cluster_y CTAs along grid.y when cluster_y > 1.
template <typename... KArgs, typename... Args>
inline cudaError_t fd_launch_cluster(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t stream,
                                     unsigned cluster_y, Args&&... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attr[2];
  int n = 0;
  if (g_fd_pdl) {
    attr[n].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[n].val.programmaticStreamSerializationAllowed = 1;
    ++n;
  }
  if (cluster_y > 1) {
    attr[n].id = cudaLaunchAttributeClusterDimension;
    attr[n].val.clusterDim.x = 1;
    attr[n].val.clusterDim.y = cluster_y;
    attr[n].val.clusterDim.z = 1;
    ++n;
  }
  cfg.attrs = attr;
  cfg.numAttrs = n;
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

// GEMM output element: fp16 (Marlin's output type) or bf16, rows ld_out elements apart.
__device__ __forceinline__ void fd_store_out(void* out, int out_bf16, size_t idx, float v) {
  if (out_bf16) static_cast<__nv_bfloat16*>(out)[idx] = __float2bfloat16_rn(v);
  else static_cast<__half*>(out)[idx] = __float2half_rn(v);
}

template <int MT>
__global__ void __launch_bounds__(kSmWarps * 32) fd_w4a8_small_kernel(
    const int8_t* __restrict__ xq, const float* __restrict__ a_scale, const uint4* __restrict__ w_pack,
    const unsigned char* __restrict__ zp, const int* __restrict__ sg, void* __restrict__ out, int ld_out, int out_bf16,
    int M, int N, int K) {
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
      fd_store_out(out, out_bf16, (size_t)m * ld_out + 16 * T + r, __fmul_rn(__int2float_rn(v), a_scale[m]));
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


// LayerNorm, one warp per row for the vision tower's width (dim = 128 * NV; NV Bf4 vectors per lane): the same
// per-element arithmetic as fd_layernorm_kernel, the FP32 statistics reduced over 32 lanes instead of a 256-thread
// block (eight independent rows per CTA, no block barriers: 2x the throughput of the one-CTA-per-row form on Thor).
template <int NV>
__global__ void __launch_bounds__(256) fd_layernorm_warp_kernel(const __nv_bfloat16* __restrict__ x,
                                                                const __nv_bfloat16* __restrict__ w,
                                                                const __nv_bfloat16* __restrict__ b,
                                                                __nv_bfloat16* __restrict__ out, int rows, float eps) {
  fd_pdl();
  constexpr int dim = 128 * NV;
  const int lane = threadIdx.x & 31, row = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (row >= rows) return;
  const Bf4* xr = reinterpret_cast<const Bf4*>(x + (size_t)row * dim);
  float v[NV][4];
  float sum = 0.0f;
#pragma unroll
  for (int k = 0; k < NV; ++k) {
    const Bf4 t = xr[lane + 32 * k];
#pragma unroll
    for (int e = 0; e < 4; ++e) { v[k][e] = bf(t.v[e]); sum += v[k][e]; }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
  const float mean = sum / (float)dim;
  float sq = 0.0f;
#pragma unroll
  for (int k = 0; k < NV; ++k)
#pragma unroll
    for (int e = 0; e < 4; ++e) { const float d = v[k][e] - mean; sq += d * d; }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
  const float rstd = rsqrtf(sq / (float)dim + eps);
  const Bf4* wr = reinterpret_cast<const Bf4*>(w);
  const Bf4* br = reinterpret_cast<const Bf4*>(b);
  Bf4* o = reinterpret_cast<Bf4*>(out + (size_t)row * dim);
#pragma unroll
  for (int k = 0; k < NV; ++k) {
    const Bf4 ww = wr[lane + 32 * k], bb = br[lane + 32 * k];
    Bf4 r;
#pragma unroll
    for (int e = 0; e < 4; ++e) r.v[e] = to_bf((v[k][e] - mean) * rstd * bf(ww.v[e]) + bf(bb.v[e]));
    o[lane + 32 * k] = r;
  }
}


// fd_layernorm_kernel whose BF16 row goes to FP8 codes + a row scale (the W8A8 vision tier): identical codes to the
// LayerNorm followed by alpamayo15_quantize_fp8_rows_bf16. dim <= 16384.
__global__ void __launch_bounds__(256) fd_layernorm_fp8_rows_kernel(const __nv_bfloat16* __restrict__ x,
                                                                    const __nv_bfloat16* __restrict__ w,
                                                                    const __nv_bfloat16* __restrict__ b,
                                                                    uint8_t* __restrict__ q, float* __restrict__ scale,
                                                                    int dim, float eps) {
  fd_pdl_wait();
  __shared__ float smem[32];
  __shared__ __align__(16) __nv_bfloat16 srow[16384];
  const int row = blockIdx.x;
  const Bf4* xr = reinterpret_cast<const Bf4*>(x + (size_t)row * dim);
  const int n4 = dim >> 2;
  float v[16][4];
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
  Bf4* o = reinterpret_cast<Bf4*>(srow);
  cnt = 0;
  for (int i = threadIdx.x; i < n4; i += 256, ++cnt) {
    const Bf4 ww = wr[i], bb = br[i];
    Bf4 r;
#pragma unroll
    for (int e = 0; e < 4; ++e) r.v[e] = to_bf((v[cnt][e] - mean) * rstd * bf(ww.v[e]) + bf(bb.v[e]));
    o[i] = r;
  }
  __syncthreads();
  fp8rows::quantize_row_fp8(srow, q, scale, row, dim, smem);
  fd_pdl_trigger();
}

// fd_layernorm_warp_kernel whose BF16 row goes to FP8 codes + a row scale (same statistics as the warp LayerNorm,
// same codes as that output followed by alpamayo15_quantize_fp8_rows_bf16). dim = 128 * NV.
template <int NV>
__global__ void __launch_bounds__(256) fd_layernorm_fp8_rows_warp_kernel(const __nv_bfloat16* __restrict__ x,
                                                                         const __nv_bfloat16* __restrict__ w,
                                                                         const __nv_bfloat16* __restrict__ b,
                                                                         uint8_t* __restrict__ q, float* __restrict__ scale,
                                                                         int rows, float eps) {
  fd_pdl();
  constexpr int dim = 128 * NV;
  const int lane = threadIdx.x & 31, row = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (row >= rows) return;
  const Bf4* xr = reinterpret_cast<const Bf4*>(x + (size_t)row * dim);
  float v[NV][4];
  float sum = 0.0f;
#pragma unroll
  for (int k = 0; k < NV; ++k) {
    const Bf4 t = xr[lane + 32 * k];
#pragma unroll
    for (int e = 0; e < 4; ++e) { v[k][e] = bf(t.v[e]); sum += v[k][e]; }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
  const float mean = sum / (float)dim;
  float sq = 0.0f;
#pragma unroll
  for (int k = 0; k < NV; ++k)
#pragma unroll
    for (int e = 0; e < 4; ++e) { const float d = v[k][e] - mean; sq += d * d; }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
  const float rstd = rsqrtf(sq / (float)dim + eps);
  const Bf4* wr = reinterpret_cast<const Bf4*>(w);
  const Bf4* br = reinterpret_cast<const Bf4*>(b);
  float mx = 0.0f;
#pragma unroll
  for (int k = 0; k < NV; ++k) {
    const Bf4 ww = wr[lane + 32 * k], bb = br[lane + 32 * k];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      v[k][e] = bf(to_bf((v[k][e] - mean) * rstd * bf(ww.v[e]) + bf(bb.v[e])));   // the BF16 LayerNorm output
      mx = fmaxf(mx, fabsf(v[k][e]));
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  const float sc = mx > 0.0f ? mx / 448.0f : 1.0f;
  if (lane == 0) scale[row] = sc;
  uint32_t* qr = reinterpret_cast<uint32_t*>(q + (size_t)row * dim);
#pragma unroll
  for (int k = 0; k < NV; ++k) {
    float f[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) f[e] = fminf(fmaxf(__fdiv_rn(v[k][e], sc), -448.0f), 448.0f);
    const __nv_fp8x2_storage_t lo = __nv_cvt_float2_to_fp8x2(make_float2(f[0], f[1]), __NV_SATFINITE, __NV_E4M3);
    const __nv_fp8x2_storage_t hi = __nv_cvt_float2_to_fp8x2(make_float2(f[2], f[3]), __NV_SATFINITE, __NV_E4M3);
    qr[lane + 32 * k] = (uint32_t)lo | ((uint32_t)hi << 16);
  }
}

// out[r, f] = bf16(silu(gate[r, f])) * up[r, f] over rows of leading dimension ld (the gate | up halves of one GEMM
// output row with up = gate + F, ld = 2F, or two separate outputs with ld = F), rounded as silu_mul_qwen36_bf16.
__global__ void fd_silu_mul_split_kernel(const __nv_bfloat16* __restrict__ gate, const __nv_bfloat16* __restrict__ up,
                                         int ld, __nv_bfloat16* __restrict__ out, int rows, int F) {
  fd_pdl();
  const long long n4 = (long long)rows * (F >> 2);
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (long long)gridDim.x * blockDim.x) {
    const int r = (int)(i / (F >> 2)), c = (int)(i - (long long)r * (F >> 2)) * 4;
    const Bf4 g = *reinterpret_cast<const Bf4*>(gate + (size_t)r * ld + c);
    const Bf4 u = *reinterpret_cast<const Bf4*>(up + (size_t)r * ld + c);
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
  if (threadIdx.x == 0) scale[row] = s;
  uint32_t* qr = reinterpret_cast<uint32_t*>(q + (size_t)row * K);
  for (int i = threadIdx.x; i < n4; i += 256) {
    const Bf4 t = xr[i];
    float f[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) f[e] = fminf(fmaxf(__fdiv_rn(bf(t.v[e]), s), -448.0f), 448.0f);   // x / s as torch
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
                                                           const int* __restrict__ sg, void* __restrict__ out, int ld_out,
                                                           int out_bf16, int M, int N, int K) {
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
      fd_store_out(out, out_bf16, (size_t)m0 * ld_out + n0, __fmul_rn(__int2float_rn(total[mt][0]), a_scale[m0]));
      fd_store_out(out, out_bf16, (size_t)m0 * ld_out + n1, __fmul_rn(__int2float_rn(total[mt][2]), a_scale[m0]));
    }
    if (m1 < M) {
      fd_store_out(out, out_bf16, (size_t)m1 * ld_out + n0, __fmul_rn(__int2float_rn(total[mt][1]), a_scale[m1]));
      fd_store_out(out, out_bf16, (size_t)m1 * ld_out + n1, __fmul_rn(__int2float_rn(total[mt][3]), a_scale[m1]));
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


// ---------------------------------------------------------------------------
// Fused producer + W4A8 activation preparation for a tile of R rows.
// Grid (row tiles, C): CTA (t, c) owns rows [R t, R t + R) and column span c of
// K; C > 1 CTAs of one tile form a thread-block cluster and exchange the row
// absmax (and nothing else) through distributed shared memory. Shared memory
// holds the tile column-major in 16-byte chunks of 8 rows, xs[channel][R], one
// pad chunk per 8 channels, so one access of a rotation lane moves 8 rows of a
// channel and the producers' transposing stores stay conflict-free.
//   producer (per set; recomputed from the L2-resident inputs instead of
//   keeping a BF16 copy; 16-byte loads of 8 channels x 8 rows):  mode 0 bf16 x;
//   mode 1 residual += bf16(delta fp16) (first set only; skipped when src is
//   null) then Qwen3 RMSNorm * w with the rms_f16delta summation order (the
//   sequential partial of "thread t over channels t, t + 256, ..." then the
//   256-thread tree; every CTA of the cluster sums the whole row, each writes
//   its own span of the residual after the cluster has read it);  mode 2
//   bf16(silu(bf16 g)) * bf16(u) from fp16 gate | up (silu_mul_f16in).
//   then per set: half(bf16) * scale, the 8 ParoQuant layers (warp per
//   128-channel group, pair tables loaded once per tile), row absmax, INT8
//   quantization straight from shared memory. Arithmetic per element is that of
//   fd_rms_f16delta_kernel / fd_silu_mul_f16in_kernel / fd_w4a8_rotate_kernel /
//   fd_w4a8_quant_kernel (bit-exact with the separate kernels).
// ---------------------------------------------------------------------------
struct ActPrepRowSet {
  const __half* theta;
  const int16_t* pairs;
  const __half* scales;
  const float* igs;
  int8_t* xq;
  float* a_scale;
};
struct ActPrepRowSets { ActPrepRowSet s[3]; };

struct ActPrepTileArgs {
  int mode;
  const void* src;
  const void* src2;
  int src_ld;
  __nv_bfloat16* residual;
  int res_ld;
  const __nv_bfloat16* norm_w;
  float eps;
  ActPrepRowSets sets;
  int n_sets;
  int rows;
  int K;
  int span;       // channels per CTA (multiple of 128)
};

constexpr int kPrepThreads = 256;
constexpr size_t kPrepSmemMax = 112 * 1024;   // dynamic shared memory per CTA the host plans with (two CTAs per SM)

__host__ __device__ constexpr size_t fd_prep_smem_bytes(int span, int R) {
  // xs chunks (+ one pad chunk per 8 channels), or the 256 partial sums of each row of the mode-1 reduction
  const size_t xs = (size_t)(span * (R / 8) + span / 8) * 16;
  const size_t part = (size_t)R * kPrepThreads * sizeof(float);
  return xs > part ? xs : part;
}

__device__ __forceinline__ unsigned fd_pack2(__half lo, __half hi) {
  return (unsigned)__half_as_ushort(lo) | ((unsigned)__half_as_ushort(hi) << 16);
}
__device__ __forceinline__ __half fd_lo(unsigned u) { return __ushort_as_half((unsigned short)(u & 0xFFFFu)); }
__device__ __forceinline__ __half fd_hi(unsigned u) { return __ushort_as_half((unsigned short)(u >> 16)); }
__device__ __forceinline__ __nv_bfloat16 fd_blo(unsigned u) { return __ushort_as_bfloat16((unsigned short)(u & 0xFFFFu)); }
__device__ __forceinline__ __nv_bfloat16 fd_bhi(unsigned u) { return __ushort_as_bfloat16((unsigned short)(u >> 16)); }
__device__ __forceinline__ unsigned fd_u4(const uint4& v, int k) {   // k is a compile-time constant after unrolling
  return k == 0 ? v.x : k == 1 ? v.y : k == 2 ? v.z : v.w;
}
__device__ __forceinline__ __half fd_h8(const uint4& v, int q) { return q & 1 ? fd_hi(fd_u4(v, q >> 1)) : fd_lo(fd_u4(v, q >> 1)); }
__device__ __forceinline__ __nv_bfloat16 fd_b8(const uint4& v, int q) {
  return q & 1 ? fd_bhi(fd_u4(v, q >> 1)) : fd_blo(fd_u4(v, q >> 1));
}
__device__ __forceinline__ __half2 fd_u2h(unsigned u) { union { unsigned u; __half2 h; } c; c.u = u; return c.h; }
__device__ __forceinline__ unsigned fd_h2u(__half2 h) { union { unsigned u; __half2 h; } c; c.h = h; return c.u; }
__device__ __forceinline__ __half fd_prep_scale(__nv_bfloat16 v, __half sc) {
  return __hmul(__float2half_rn(bf(v)), sc);
}
__device__ __forceinline__ __nv_bfloat16 fd_res_delta(__nv_bfloat16 r, __half d) {
  return to_bf(__fadd_rn(bf(r), bf(to_bf(__half2float(d)))));
}

template <int R, bool kCluster>
__global__ void __launch_bounds__(kPrepThreads) fd_w4a8_act_prep_tile_kernel(ActPrepTileArgs a) {
  static_assert(R == 8 || R == 16, "tile rows: 8 or 16 (one or two 16-byte chunks per channel)");
  constexpr int kChunks = R / 8;
  fd_pdl_wait();
  extern __shared__ __align__(16) unsigned char fd_tile_smem[];
  uint4* xs4 = reinterpret_cast<uint4*>(fd_tile_smem);            // chunk (channel, ch) at slot channel*kChunks + ch + channel/8
  float* part = reinterpret_cast<float*>(fd_tile_smem);           // [R][256] partial sums (mode 1, before xs is written)
  __shared__ float red[32];
  __shared__ float wmax[8][R];
  __shared__ float rmax[R];
  __shared__ float rall[R];
  __shared__ float inv_s[R];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int row0 = blockIdx.x * R;
  const int nrow = min(R, a.rows - row0);
  const int K = a.K, span = a.span;
  const int c0 = blockIdx.y * span;
  const int span_here = max(0, min(span, K - c0));
  const int ng = span_here / kRotGroup;
  const int nblk = span_here / 8;                                  // 8-channel blocks of the span
  auto slot = [](int col, int ch) { return col * kChunks + ch + (col >> 3); };
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
#endif
  auto tile_sync = [&]() {
    if constexpr (kCluster) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
      cluster.sync();
#endif
    } else {
      __syncthreads();
    }
  };

  for (int si = 0; si < a.n_sets; ++si) {
    const ActPrepRowSet& st = a.sets.s[si];
    // ---------------- producer -> xs (FP16, channel-scaled) ----------------
    if (a.mode == 1) {
      const bool with_delta = (si == 0) && a.src != nullptr;
      const __half* dl = static_cast<const __half*>(a.src);
      if (si == 0) {
        // pass 1: sum of squares of every row of the tile over the whole row. Thread (chain c = tid % 32, row group
        // tid / 32) keeps the 8 sequential partials of rms_f16delta's threads 8c..8c+7 (channels 8c + q + 256 k, k
        // ascending), then the 256 partials of a row go through the same 256-thread tree.
        const int chain = tid & 31, rg = tid >> 5;
#pragma unroll
        for (int h = 0; h < kChunks; ++h) {
          const int m = rg + 8 * h;
          float p[8];
#pragma unroll
          for (int q = 0; q < 8; ++q) p[q] = 0.0f;
          if (m < nrow) {
            const __nv_bfloat16* rrow = a.residual + (size_t)(row0 + m) * a.res_ld;
            const __half* drow = with_delta ? dl + (size_t)(row0 + m) * a.src_ld : nullptr;
            for (int k = 0; k < K; k += kPrepThreads) {
              const int col = k + 8 * chain;
              const uint4 rv = *reinterpret_cast<const uint4*>(rrow + col);
              uint4 dv = make_uint4(0u, 0u, 0u, 0u);
              if (with_delta) dv = *reinterpret_cast<const uint4*>(drow + col);
#pragma unroll
              for (int q = 0; q < 8; ++q) {
                __nv_bfloat16 v = fd_b8(rv, q);
                if (with_delta) v = fd_res_delta(v, fd_h8(dv, q));
                const float f = bf(v);
                p[q] += __fmul_rn(f, f);
              }
            }
          }
#pragma unroll
          for (int q = 0; q < 8; ++q) part[m * kPrepThreads + 8 * chain + q] = p[q];
        }
        __syncthreads();
        for (int m = 0; m < nrow; ++m) {
          const float tot = block_sum256(part[m * kPrepThreads + tid], red);
          if (tid == 0) inv_s[m] = rsqrtf(__fadd_rn(tot / (float)K, a.eps));
        }
        tile_sync();   // every CTA of the tile has read the residual (and part is dead) before any span is rewritten
      }
      // pass 2: own span -- residual update (first set), normalization, channel scale
      for (int cb = tid; cb < nblk; cb += kPrepThreads) {
        const int col = c0 + 8 * cb;
#pragma unroll
        for (int ch = 0; ch < kChunks; ++ch) {
        const uint4 sc = *reinterpret_cast<const uint4*>(st.scales + col);
        const uint4 wv = *reinterpret_cast<const uint4*>(a.norm_w + col);
        uint4 rv[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
          const int m = 8 * ch + r;
          rv[r] = make_uint4(0u, 0u, 0u, 0u);
          if (m < nrow) {
            __nv_bfloat16* rp = a.residual + (size_t)(row0 + m) * a.res_ld + col;
            rv[r] = *reinterpret_cast<const uint4*>(rp);
            if (with_delta) {
              const uint4 dv = *reinterpret_cast<const uint4*>(dl + (size_t)(row0 + m) * a.src_ld + col);
              auto upd = [&](int k) {
                const __nv_bfloat16 lo = fd_res_delta(fd_blo(fd_u4(rv[r], k)), fd_lo(fd_u4(dv, k)));
                const __nv_bfloat16 hi = fd_res_delta(fd_bhi(fd_u4(rv[r], k)), fd_hi(fd_u4(dv, k)));
                return (unsigned)__bfloat16_as_ushort(lo) | ((unsigned)__bfloat16_as_ushort(hi) << 16);
              };
              const uint4 nv = make_uint4(upd(0), upd(1), upd(2), upd(3));
              rv[r] = nv;
              *reinterpret_cast<uint4*>(rp) = nv;
            }
          }
        }
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const __half sq = fd_h8(sc, q);
          const float w = bf(fd_b8(wv, q));
          __half h[8];
#pragma unroll
          for (int r = 0; r < 8; ++r) {
            const int m = 8 * ch + r;
            const __nv_bfloat16 n1 = to_bf(__fmul_rn(bf(fd_b8(rv[r], q)), m < nrow ? inv_s[m] : 0.0f));
            h[r] = fd_prep_scale(to_bf(__fmul_rn(w, bf(n1))), sq);
          }
          xs4[slot(8 * cb + q, ch)] = make_uint4(fd_pack2(h[0], h[1]), fd_pack2(h[2], h[3]), fd_pack2(h[4], h[5]),
                                                fd_pack2(h[6], h[7]));
        }
        }
      }
    } else if (a.mode == 2) {
      const __half* g = static_cast<const __half*>(a.src);
      const __half* u = static_cast<const __half*>(a.src2);
      for (int cb = tid; cb < nblk; cb += kPrepThreads) {
        const int col = c0 + 8 * cb;
#pragma unroll
        for (int ch = 0; ch < kChunks; ++ch) {
        const uint4 sc = *reinterpret_cast<const uint4*>(st.scales + col);
        uint4 gv[8], uv[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
          const int m = 8 * ch + r;
          gv[r] = make_uint4(0u, 0u, 0u, 0u);
          uv[r] = gv[r];
          if (m < nrow) {
            const size_t off = (size_t)(row0 + m) * a.src_ld + col;
            gv[r] = *reinterpret_cast<const uint4*>(g + off);
            uv[r] = *reinterpret_cast<const uint4*>(u + off);
          }
        }
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const __half sq = fd_h8(sc, q);
          __half h[8];
#pragma unroll
          for (int r = 0; r < 8; ++r) {
            const float gb = bf(to_bf(__half2float(fd_h8(gv[r], q))));
            const float ub = bf(to_bf(__half2float(fd_h8(uv[r], q))));
            const float sl = bf(to_bf(gb / (1.0f + expf(-gb))));
            h[r] = fd_prep_scale(to_bf(__fmul_rn(sl, ub)), sq);
          }
          xs4[slot(8 * cb + q, ch)] = make_uint4(fd_pack2(h[0], h[1]), fd_pack2(h[2], h[3]), fd_pack2(h[4], h[5]),
                                                fd_pack2(h[6], h[7]));
        }
        }
      }
    } else {
      const __nv_bfloat16* x = static_cast<const __nv_bfloat16*>(a.src);
      for (int cb = tid; cb < nblk; cb += kPrepThreads) {
        const int col = c0 + 8 * cb;
#pragma unroll
        for (int ch = 0; ch < kChunks; ++ch) {
        const uint4 sc = *reinterpret_cast<const uint4*>(st.scales + col);
        uint4 xv[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
          const int m = 8 * ch + r;
          xv[r] = make_uint4(0u, 0u, 0u, 0u);
          if (m < nrow) xv[r] = *reinterpret_cast<const uint4*>(x + (size_t)(row0 + m) * a.src_ld + col);
        }
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const __half sq = fd_h8(sc, q);
          __half h[8];
#pragma unroll
          for (int r = 0; r < 8; ++r) h[r] = fd_prep_scale(fd_b8(xv[r], q), sq);
          xs4[slot(8 * cb + q, ch)] = make_uint4(fd_pack2(h[0], h[1]), fd_pack2(h[2], h[3]), fd_pack2(h[4], h[5]),
                                                fd_pack2(h[6], h[7]));
        }
        }
      }
    }
    __syncthreads();

    // ---------------- rotation: warp per 128-channel group, 2 pairs per lane per layer ----------------
    for (int gi = warp; gi < ng; gi += kPrepThreads / 32) {
      const int gcol = c0 + gi * kRotGroup;
      float th[kRotLayers][2];
      int ij[kRotLayers][2];
#pragma unroll
      for (int r = 0; r < kRotLayers; ++r) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
          const int pr = lane + 32 * p;
          th[r][p] = __half2float(st.theta[(size_t)r * (K / 2) + gcol / 2 + pr]);
          ij[r][p] = *reinterpret_cast<const int*>(st.pairs + (size_t)r * K + gcol + 2 * pr);
        }
      }
      const int gl = gi * kRotGroup;      // span-local first channel of the group
#pragma unroll
      for (int r = 0; r < kRotLayers; ++r) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
          const int i = gl + (ij[r][p] & 0xFFFF), j = gl + (ij[r][p] >> 16);
          float s_, c_;
          __sincosf(th[r][p], &s_, &c_);
#pragma unroll
          for (int ch = 0; ch < kChunks; ++ch) {
            const uint4 vi = xs4[slot(i, ch)];
            const uint4 vj = xs4[slot(j, ch)];
            auto rot_i = [&](int k) {
              const float2 xi = __half22float2(fd_u2h(fd_u4(vi, k)));
              const float2 xj = __half22float2(fd_u2h(fd_u4(vj, k)));
              return fd_h2u(__floats2half2_rn(fmaf(c_, xi.x, s_ * xj.x), fmaf(c_, xi.y, s_ * xj.y)));
            };
            auto rot_j = [&](int k) {
              const float2 xi = __half22float2(fd_u2h(fd_u4(vi, k)));
              const float2 xj = __half22float2(fd_u2h(fd_u4(vj, k)));
              return fd_h2u(__floats2half2_rn(fmaf(c_, xj.x, -s_ * xi.x), fmaf(c_, xj.y, -s_ * xi.y)));
            };
            xs4[slot(i, ch)] = make_uint4(rot_i(0), rot_i(1), rot_i(2), rot_i(3));
            xs4[slot(j, ch)] = make_uint4(rot_j(0), rot_j(1), rot_j(2), rot_j(3));
          }
        }
        __syncwarp();
      }
    }
    __syncthreads();

    // ---------------- row absmax over the span ----------------
    {
      float mx[R];
#pragma unroll
      for (int m = 0; m < R; ++m) mx[m] = 0.0f;
      for (int cb = tid; cb < nblk; cb += kPrepThreads) {
#pragma unroll
        for (int ch = 0; ch < kChunks; ++ch) {
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const uint4 v = xs4[slot(8 * cb + q, ch)];
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const float2 f = __half22float2(fd_u2h(fd_u4(v, k)));
            mx[8 * ch + 2 * k] = fmaxf(mx[8 * ch + 2 * k], fabsf(f.x));
            mx[8 * ch + 2 * k + 1] = fmaxf(mx[8 * ch + 2 * k + 1], fabsf(f.y));
          }
        }
        }
      }
#pragma unroll
      for (int m = 0; m < R; ++m) {
        float v = mx[m];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
        if (lane == 0) wmax[warp][m] = v;
      }
      __syncthreads();
      if (tid < R) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) v = fmaxf(v, wmax[w][tid]);
        rmax[tid] = v;
      }
      if constexpr (kCluster) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        cluster.sync();
        if (tid < R) {
          float v = 0.0f;
          for (unsigned rk = 0; rk < cluster.num_blocks(); ++rk) v = fmaxf(v, *cluster.map_shared_rank(&rmax[tid], rk));
          rall[tid] = v;
        }
        cluster.sync();   // peers are done reading rmax before the next set rewrites it
#endif
      } else {
        __syncthreads();
        if (tid < R) rall[tid] = rmax[tid];
        __syncthreads();
      }
    }

    // ---------------- INT8 quantization straight from shared memory ----------------
    {
      float inv[R], amax[R];
#pragma unroll
      for (int m = 0; m < R; ++m) {
        amax[m] = fmaxf(rall[m], 1e-10f);
        inv[m] = div_full(127.0f, amax[m]);
      }
      for (int cb = tid; cb < nblk; cb += kPrepThreads) {
        const int col = c0 + 8 * cb;
#pragma unroll
        for (int ch = 0; ch < kChunks; ++ch) {
        uint4 v[8];
#pragma unroll
        for (int q = 0; q < 8; ++q) v[q] = xs4[slot(8 * cb + q, ch)];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
          const int m = 8 * ch + r;
          if (m < nrow) {
            unsigned lo = 0u, hi = 0u;
#pragma unroll
            for (int q = 0; q < 8; ++q) {
              const unsigned u = fd_u4(v[q], r >> 1);
              const float f = __half2float(r & 1 ? fd_hi(u) : fd_lo(u));
              const unsigned b = (unsigned)(static_cast<int>(round_half_away_triton(__fmul_rn(inv[m], f))) & 0xFF);
              if (q < 4) lo |= b << (8 * q); else hi |= b << (8 * (q - 4));
            }
            *reinterpret_cast<uint2*>(st.xq + (size_t)(row0 + m) * K + col) = make_uint2(lo, hi);
          }
        }
        }
      }
      if (blockIdx.y == 0 && tid < nrow) st.a_scale[row0 + tid] = __fmul_rn(div_full(amax[tid], 127.0f), *st.igs);
    }
    __syncthreads();   // xs is rewritten by the next set's producer
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

int flashdrive_w4a8_rotate(const void* x, int ld_x, int rows, int K, int n_sets,
                           const void* const* theta, const void* const* pairs, const void* const* scales,
                           void* const* xr, float* const* part, cudaStream_t stream) {
  if (!x || rows <= 0 || K <= 0 || n_sets < 1 || n_sets > 3) return 2;
  if ((K % kRotGroup) != 0 || ld_x < K || (ld_x % 2) != 0) return 2;
  if ((reinterpret_cast<uintptr_t>(x) & 3u) != 0) return 5;
  ActPrepSets sets{};
  for (int i = 0; i < n_sets; ++i) {
    if (!theta[i] || !pairs[i] || !scales[i] || !xr[i] || !part[i]) return 1;
    if ((reinterpret_cast<uintptr_t>(pairs[i]) & 3u) != 0 || (reinterpret_cast<uintptr_t>(scales[i]) & 3u) != 0 ||
        (reinterpret_cast<uintptr_t>(xr[i]) & 3u) != 0)
      return 5;
    sets.s[i] = ActPrepSet{static_cast<const __half*>(theta[i]), static_cast<const int16_t*>(pairs[i]),
                           static_cast<const __half*>(scales[i]), nullptr, static_cast<__half*>(xr[i]), part[i], nullptr, nullptr};
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
                               const int* sg, void* out, int M, int N, int K, cudaStream_t stream, int ld_out, int out_bf16) {
  if (ld_out <= 0) ld_out = N;
  if (ld_out < N || (ld_out % 2) != 0) return 2;
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
        static_cast<const unsigned char*>(zp), sg, out, ld_out, out_bf16, M, N, K);
  } else {
    fd_launch(fd_w4a8_small_kernel<2>, dim3(grid), dim3(kSmWarps * 32), (size_t)(kRingBytes), stream, 
        static_cast<const int8_t*>(xq), a_scale, static_cast<const uint4*>(w_pack),
        static_cast<const unsigned char*>(zp), sg, out, ld_out, out_bf16, M, N, K);
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

int alpamayo15_layernorm_bf16(const void* x, const void* w, const void* b, void* out, int rows, int dim, float eps,
                              cudaStream_t stream) {
  if (!x || !w || !b || !out) return 1;
  if (rows <= 0 || dim <= 0 || (dim % 8) != 0 || dim > 16384) return 2;
  if (!aligned8(x) || !aligned8(w) || !aligned8(b) || !aligned8(out)) return 5;
  const auto* xb = static_cast<const __nv_bfloat16*>(x);
  const auto* wb = static_cast<const __nv_bfloat16*>(w);
  const auto* bb = static_cast<const __nv_bfloat16*>(b);
  auto* ob = static_cast<__nv_bfloat16*>(out);
  const dim3 wgrid((rows + 7) / 8);
  switch (dim) {        // warp-per-row forms for the vision widths; one CTA per row otherwise
    case 1024: fd_launch(fd_layernorm_warp_kernel<8>, wgrid, dim3(256), (size_t)0, stream, xb, wb, bb, ob, rows, eps); break;
    case 1152: fd_launch(fd_layernorm_warp_kernel<9>, wgrid, dim3(256), (size_t)0, stream, xb, wb, bb, ob, rows, eps); break;
    case 2048: fd_launch(fd_layernorm_warp_kernel<16>, wgrid, dim3(256), (size_t)0, stream, xb, wb, bb, ob, rows, eps); break;
    default: fd_launch(fd_layernorm_kernel, dim3(rows), dim3(256), (size_t)(0), stream, xb, wb, bb, ob, dim, eps); break;
  }
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int alpamayo15_layernorm_fp8_rows_bf16(const void* x, const void* w, const void* b, void* q, float* scale, int rows, int dim,
                                       float eps, cudaStream_t stream) {
  if (!x || !w || !b || !q || !scale) return 1;
  if (rows <= 0 || dim <= 0 || (dim % 8) != 0 || dim > 16384) return 2;
  if (!aligned8(x) || !aligned8(w) || !aligned8(b) || (reinterpret_cast<uintptr_t>(q) & 3u) != 0) return 5;
  const auto* xb = static_cast<const __nv_bfloat16*>(x);
  const auto* wb = static_cast<const __nv_bfloat16*>(w);
  const auto* bb = static_cast<const __nv_bfloat16*>(b);
  auto* qb = static_cast<uint8_t*>(q);
  const dim3 wgrid((rows + 7) / 8);
  switch (dim) {        // the warp-per-row forms of alpamayo15_layernorm_bf16 (same statistics) for the vision widths
    case 1024: fd_launch(fd_layernorm_fp8_rows_warp_kernel<8>, wgrid, dim3(256), (size_t)0, stream, xb, wb, bb, qb, scale, rows, eps); break;
    case 1152: fd_launch(fd_layernorm_fp8_rows_warp_kernel<9>, wgrid, dim3(256), (size_t)0, stream, xb, wb, bb, qb, scale, rows, eps); break;
    case 2048: fd_launch(fd_layernorm_fp8_rows_warp_kernel<16>, wgrid, dim3(256), (size_t)0, stream, xb, wb, bb, qb, scale, rows, eps); break;
    default: fd_launch(fd_layernorm_fp8_rows_kernel, dim3(rows), dim3(256), (size_t)(0), stream, xb, wb, bb, qb, scale, dim, eps); break;
  }
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int alpamayo15_silu_mul_split_bf16(const void* gate, const void* up, int ld, void* out, int rows, int F,
                                   cudaStream_t stream) {
  if (!gate || !up || !out) return 1;
  if (rows <= 0 || F <= 0 || (F % 4) != 0 || ld < F || (ld % 4) != 0) return 2;
  if (!aligned8(gate) || !aligned8(up) || !aligned8(out)) return 5;
  const long long n4 = (long long)rows * (F / 4);
  const long long blocks = (n4 + 255) / 256;
  fd_launch(fd_silu_mul_split_kernel, dim3((unsigned)(blocks > 4096 ? 4096 : blocks)), dim3(256), (size_t)(0), stream,
      static_cast<const __nv_bfloat16*>(gate), static_cast<const __nv_bfloat16*>(up), ld,
      static_cast<__nv_bfloat16*>(out), rows, F);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int alpamayo15_quantize_fp8_rows_bf16(const void* x, int ld_x, void* q, float* scale, int rows, int K, cudaStream_t stream) {
  if (!x || !q || !scale) return 1;
  if (rows <= 0 || K <= 0 || (K % 8) != 0 || ld_x < K || (ld_x % 4) != 0) return 2;
  if (!aligned8(x) || (reinterpret_cast<uintptr_t>(q) & 3u) != 0) return 5;
  fd_launch(fd_quantize_fp8_rows_kernel, dim3(rows), dim3(256), (size_t)(0), stream, static_cast<const __nv_bfloat16*>(x), ld_x, static_cast<uint8_t*>(q),
                                                        scale, K);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

void alpamayo15_set_pdl(bool on) { g_fd_pdl = on; }
bool alpamayo15_pdl() { return g_fd_pdl; }

namespace {
// K splits per tile so that at least ~128 CTAs stream concurrently (and K stays divisible).
int tile_ks(int NT, int K, int kunit) {
  int ks = 1;
  while (ks < 4 && NT * ks / 4 < 128 && (K / kunit) % (ks * 2) == 0) ks *= 2;
  return ks;
}
}  // namespace

int flashdrive_w4a8_gemm_tile(const void* xq, const float* a_scale, const void* w_pack, const void* zp, const int* sg,
                              void* out, int M, int N, int K, cudaStream_t stream, int ld_out, int out_bf16) {
  if (ld_out <= 0) ld_out = N;
  if (ld_out < N || (ld_out % 2) != 0) return 2;
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
#define FD_TILE(MT, KS) fd_launch(fd_w4a8_tile_kernel<MT, KS>, grid, dim3(128), (size_t)0, stream, x8, a_scale, wp, z8, sg, out, ld_out, out_bf16, M, N, K)
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

namespace {
template <int R, bool kCluster>
int fd_prep_tile_launch(const ActPrepTileArgs& a, int tiles, int C, size_t smem, cudaStream_t stream) {
  static size_t configured = 0;
  if (smem > configured) {
    if (cudaFuncSetAttribute(fd_w4a8_act_prep_tile_kernel<R, kCluster>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)kPrepSmemMax) != cudaSuccess)
      return 6;
    configured = kPrepSmemMax;
  }
  return fd_launch_cluster(fd_w4a8_act_prep_tile_kernel<R, kCluster>, dim3(tiles, C), dim3(kPrepThreads), smem, stream,
                           (unsigned)C, a) == cudaSuccess ? 0 : 4;
}
}  // namespace

int flashdrive_w4a8_act_prep_tile(int mode, const void* src, const void* src2, int src_ld, void* residual, int res_ld,
                                  const void* norm_w, float eps, int rows, int K, int n_sets, const void* const* theta,
                                  const void* const* pairs, const void* const* scales, const float* const* igs,
                                  void* const* xq, float* const* a_scale, int tile_rows, int cluster,
                                  cudaStream_t stream) {
  if (rows <= 0 || K <= 0 || (K % kRotGroup) != 0 || n_sets < 1 || n_sets > 3) return 2;
  if (mode < 0 || mode > 2 || (mode != 1 && !src) || (mode == 1 && (!residual || !norm_w)) || (mode == 2 && !src2)) return 1;
  if (mode == 1 ? (res_ld < K || (src && src_ld < K)) : src_ld < K) return 2;
  if (tile_rows != 0 && tile_rows != 8 && tile_rows != 16) return 2;
  if (cluster < 0 || cluster > 8) return 2;
  ActPrepTileArgs a{};
  for (int i = 0; i < n_sets; ++i) {
    if (!theta[i] || !pairs[i] || !scales[i] || !igs[i] || !xq[i] || !a_scale[i]) return 1;
    if ((reinterpret_cast<uintptr_t>(pairs[i]) & 3u) != 0) return 5;
    a.sets.s[i] = ActPrepRowSet{static_cast<const __half*>(theta[i]), static_cast<const int16_t*>(pairs[i]),
                                static_cast<const __half*>(scales[i]), igs[i], static_cast<int8_t*>(xq[i]), a_scale[i]};
  }
  const int ng = K / kRotGroup;
  const int R = tile_rows ? tile_rows : 8;
  auto smem_of = [&](int C) { return fd_prep_smem_bytes(((ng + C - 1) / C) * kRotGroup, R); };
  auto covers = [&](int C) { return (C - 1) * ((ng + C - 1) / C) < ng; };   // every CTA of the cluster owns a group
  int C = cluster;
  if (C == 0) {
    if (rows <= R) {                 // one tile: spread it over as many SMs as the groups allow
      for (C = 8; C > 1; --C)
        if (covers(C) && smem_of(C) <= kPrepSmemMax) break;
    } else {                         // many tiles: the fewest CTAs per tile whose span fits
      for (C = 1; C <= 8; ++C)
        if (covers(C) && smem_of(C) <= kPrepSmemMax) break;
      if (C > 8) return 2;
    }
  }
  if (!covers(C)) return 2;
  const size_t smem = smem_of(C);
  if (smem > kPrepSmemMax) return 2;
  if (((reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(src2) | reinterpret_cast<uintptr_t>(residual) |
        reinterpret_cast<uintptr_t>(norm_w)) & 15u) != 0 || (src_ld % 8) != 0 || (res_ld % 8) != 0)
    return 5;                                   // 16-byte loads of 8 channels
  a.mode = mode; a.src = src; a.src2 = src2; a.src_ld = src_ld;
  a.residual = static_cast<__nv_bfloat16*>(residual); a.res_ld = res_ld;
  a.norm_w = static_cast<const __nv_bfloat16*>(norm_w); a.eps = eps;
  a.n_sets = n_sets; a.rows = rows; a.K = K; a.span = ((ng + C - 1) / C) * kRotGroup;
  const int tiles = (rows + R - 1) / R;
  int rc;
  if (R == 16) rc = C > 1 ? fd_prep_tile_launch<16, true>(a, tiles, C, smem, stream) : fd_prep_tile_launch<16, false>(a, tiles, C, smem, stream);
  else         rc = C > 1 ? fd_prep_tile_launch<8, true>(a, tiles, C, smem, stream) : fd_prep_tile_launch<8, false>(a, tiles, C, smem, stream);
  if (rc) return rc;
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

}  // namespace kernels
}  // namespace flash_rt
