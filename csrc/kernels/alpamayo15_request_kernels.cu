// SPDX-License-Identifier: Apache-2.0
// See alpamayo15_request_kernels.cuh.

#include "alpamayo15_request_kernels.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cmath>

namespace flash_rt {
namespace kernels {
namespace {

constexpr int kSampleThreads = 1024;
constexpr int kSampleHistBytes = (kSampleThreads / 32) * 256 * 8;   // one 256-bin uint64 histogram per warp

struct alignas(16) Bf8 { __nv_bfloat16 v[8]; };
struct alignas(8) Bf4o { __nv_bfloat16 v[4]; };

inline bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

__device__ __forceinline__ float bf(const __nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ __nv_bfloat16 to_bf(float v) { return __float2bfloat16_rn(v); }

// ---------------------------------------------------------------------------------------------------------------------
// gathers and scatters of bf16 rows (16-byte vectors)
// ---------------------------------------------------------------------------------------------------------------------
__global__ void gather_rows_kernel(const unsigned char* __restrict__ table, long long ld_t, const long long* __restrict__ idx,
                                   const long long* __restrict__ base, unsigned char* __restrict__ out, long long ld_o, int rows,
                                   int vec) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)rows * vec) return;
  const int r = (int)(i / vec), c = (int)(i - (long long)r * vec);
  const long long row = idx[r] + (base ? *base : 0ll);
  const uint4 v = *reinterpret_cast<const uint4*>(table + row * ld_t + 16 * c);
  *reinterpret_cast<uint4*>(out + (long long)r * ld_o + 16 * c) = v;
}

__global__ void cast_f16_bf16_kernel(const __half* __restrict__ x, __nv_bfloat16* __restrict__ out, long long n4) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n4) return;
  const __half2* h2 = reinterpret_cast<const __half2*>(x) + 2 * i;
  const float2 a = __half22float2(h2[0]), b = __half22float2(h2[1]);
  Bf4o o;
  o.v[0] = to_bf(a.x); o.v[1] = to_bf(a.y); o.v[2] = to_bf(b.x); o.v[3] = to_bf(b.y);
  reinterpret_cast<Bf4o*>(out)[i] = o;
}

// DFlash round bookkeeping (FlashDrive's block_step_fn decisions, one thread): the proposal block from the draft's
// greedy tokens (a proposal after a proposed stop token becomes the trajectory-start token) and, after the verify
// pass, the acceptance length, the first stop position and the period hit.
__global__ void dflash_propose_kernel(const long long* __restrict__ proposed, const long long* __restrict__ first, long long stop_id,
                                      long long traj_id, long long* __restrict__ block, int B) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  block[0] = first[0];
  for (int i = 1; i < B; ++i) {
    const long long p = proposed[i - 1];
    block[i] = (i >= 2 && proposed[i - 2] == stop_id) ? traj_id : p;
  }
}

__global__ void dflash_decide_kernel(const long long* __restrict__ block, const long long* __restrict__ posterior, long long stop_id,
                                     long long period_id, int B, long long* __restrict__ stats) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  long long acceptance = 0;
  for (int i = 0; i < B - 1; ++i) {                     // matches = block[1:] == posterior[:-1], cumprod, sum
    if (block[i + 1] == posterior[i]) ++acceptance; else break;
  }
  const long long bonus = posterior[acceptance];
  long long first_stop = -1;                            // emission = block with the bonus at slot acceptance + 1
  for (int i = 0; i <= acceptance + 1 && i <= B; ++i) {
    const long long e = (i == acceptance + 1) ? bonus : block[i];
    if (e == stop_id) { first_stop = i; break; }
  }
  stats[0] = acceptance; stats[1] = first_stop; stats[2] = (bonus == period_id) ? 1 : 0;
}

__global__ void cast_f32_bf16_kernel(const float* __restrict__ x, __nv_bfloat16* __restrict__ out, long long n4) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n4) return;
  const float4 v = reinterpret_cast<const float4*>(x)[i];
  Bf4o o;
  o.v[0] = to_bf(v.x); o.v[1] = to_bf(v.y); o.v[2] = to_bf(v.z); o.v[3] = to_bf(v.w);
  reinterpret_cast<Bf4o*>(out)[i] = o;
}

template <bool kAdd>
__global__ void scatter_rows_kernel(__nv_bfloat16* __restrict__ dst, int ld_d, const long long* __restrict__ idx,
                                    const __nv_bfloat16* __restrict__ src, int ld_s, int rows, int dim8) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)rows * dim8) return;
  const int r = (int)(i / dim8), c = (int)(i - (long long)r * dim8);
  Bf8* d = reinterpret_cast<Bf8*>(dst + idx[r] * ld_d + 8 * c);
  const Bf8 s = *reinterpret_cast<const Bf8*>(src + (size_t)r * ld_s + 8 * c);
  if (kAdd) {
    Bf8 o = *d;
#pragma unroll
    for (int e = 0; e < 8; ++e) o.v[e] = to_bf(bf(o.v[e]) + bf(s.v[e]));
    *d = o;
  } else {
    *d = s;
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// interleaved M-RoPE tables
// ---------------------------------------------------------------------------------------------------------------------
template <class T>
__device__ __forceinline__ T rope_out(float v);
template <> __device__ __forceinline__ float rope_out<float>(float v) { return v; }
template <> __device__ __forceinline__ __nv_bfloat16 rope_out<__nv_bfloat16>(float v) { return to_bf(v); }

template <class T>
__global__ void mrope_tables_kernel(const int* __restrict__ pos3, const float* __restrict__ inv_freq, int sec_h, int sec_w,
                                    float scaling, T* __restrict__ cos, T* __restrict__ sin, int P, int half) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)P * half) return;
  const int p = (int)(i / half), j = (int)(i - (long long)p * half);
  int dim = 0;
  if (j % 3 == 1 && j < 3 * sec_h) dim = 1;
  else if (j % 3 == 2 && j < 3 * sec_w) dim = 2;
  const float f = inv_freq[j] * (float)pos3[(size_t)dim * P + p];
  const T c = rope_out<T>(cosf(f) * scaling), s = rope_out<T>(sinf(f) * scaling);
  cos[(size_t)p * 2 * half + j] = c;
  cos[(size_t)p * 2 * half + half + j] = c;
  sin[(size_t)p * 2 * half + j] = s;
  sin[(size_t)p * 2 * half + half + j] = s;
}

// ---------------------------------------------------------------------------------------------------------------------
// sampler
// ---------------------------------------------------------------------------------------------------------------------
// Philox4x32-10 (curand's generator, the one torch's CUDA distributions use): the four 32-bit words of counter
// (c0, c1, c2, c3) under key (seed lo, seed hi).
__device__ __forceinline__ void philox4x32_10(unsigned c0, unsigned c1, unsigned c2, unsigned c3, unsigned long long seed,
                                              unsigned out[4]) {
  unsigned k0 = (unsigned)seed, k1 = (unsigned)(seed >> 32);
#pragma unroll
  for (int r = 0; r < 10; ++r) {
    const unsigned long long m0 = (unsigned long long)0xD2511F53u * c0, m1 = (unsigned long long)0xCD9E8D57u * c2;
    const unsigned n0 = (unsigned)(m1 >> 32) ^ c1 ^ k0, n1 = (unsigned)m1, n2 = (unsigned)(m0 >> 32) ^ c3 ^ k1, n3 = (unsigned)m0;
    c0 = n0; c1 = n1; c2 = n2; c3 = n3;
    k0 += 0x9E3779B9u; k1 += 0xBB67AE85u;
  }
  out[0] = c0; out[1] = c1; out[2] = c2; out[3] = c3;
}

// The Philox word torch's CUDA distribution kernel (distribution_nullary_kernel, four words per thread per round)
// assigns to element li of a tensor drawn with the generator at (seed, offset): thread li mod TT, round
// (li / TT) / 4 -> curand counter (offset / 4 + round, thread), word (li / TT) mod 4. TT = the kernel's thread count.
__device__ __forceinline__ unsigned torch_philox_word(unsigned long long seed, unsigned long long offset, unsigned li, unsigned TT) {
  const unsigned idx = li % TT, j = li / TT;
  const unsigned long long ctr = (offset >> 2) + (unsigned long long)(j >> 2);
  unsigned w[4];
  philox4x32_10((unsigned)ctr, (unsigned)(ctr >> 32), idx, 0u, seed, w);
  return w[j & 3u];
}

// Deterministic block reductions (fixed order): warp shuffles, then warp 0 over the warps' values.
__device__ __forceinline__ float block_max_f(float v, float* smem) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  if (warp == 0) {
    float t = lane < (blockDim.x >> 5) ? smem[lane] : -INFINITY;
    for (int o = 16; o > 0; o >>= 1) t = fmaxf(t, __shfl_xor_sync(0xffffffffu, t, o));
    if (lane == 0) smem[0] = t;
  }
  __syncthreads();
  const float r = smem[0];
  __syncthreads();
  return r;
}

__device__ __forceinline__ float block_sum_f(float v, float* smem) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  if (warp == 0) {
    float t = lane < (blockDim.x >> 5) ? smem[lane] : 0.0f;
    for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
    if (lane == 0) smem[0] = t;
  }
  __syncthreads();
  const float r = smem[0];
  __syncthreads();
  return r;
}

// Exclusive block scan of one uint64 per thread (fixed order): returns the exclusive prefix; total in *total.
__device__ __forceinline__ unsigned long long block_exscan_u64(unsigned long long v, unsigned long long* smem,
                                                               unsigned long long* total) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
  unsigned long long x = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const unsigned long long y = __shfl_up_sync(0xffffffffu, x, o);
    if (lane >= o) x += y;
  }
  if (lane == 31) smem[warp] = x;                 // inclusive warp total
  __syncthreads();
  if (warp == 0) {
    unsigned long long w = lane < nw ? smem[lane] : 0ull;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const unsigned long long y = __shfl_up_sync(0xffffffffu, w, o);
      if (lane >= o) w += y;
    }
    if (lane < nw) smem[32 + lane] = w;           // inclusive prefix of the warps
  }
  __syncthreads();
  const unsigned long long before = (warp > 0 ? smem[32 + warp - 1] : 0ull);
  *total = smem[32 + nw - 1];
  __syncthreads();
  return before + x - v;
}

struct SampleArgs {
  const __nv_bfloat16* logits; int ld, V, mask_lo, mask_n;
  float temperature, top_p; int rule;
  unsigned long long* rng;                 // [0] seed, [1] torch's Philox offset (advanced as torch's exponential_ does), [2] ticket
  unsigned rng_blocks;                     // torch's launch cap of its distribution kernel: SMs x blocks per SM
  float* scratch;                          // [n, V] fp32: the row's exp(score - max), computed once
  long long* tok; const long long* unfinished; long long pad_id;
  long long* slot; long long* rpos; int* len; int* glen;
};

// The bookkeeping of one step after every row's draw: the last block to finish (ticket) advances the draw counter,
// the cache slot and the rope positions; every row's row counters are advanced by its own block.
__device__ __forceinline__ void sample_finish(const SampleArgs& a, int row, int rows, long long t, unsigned long long rng_inc) {
  a.tok[row] = (a.unfinished && a.unfinished[row] == 0) ? a.pad_id : t;
  if (a.len) a.len[row] += 1;
  if (a.glen) a.glen[row] += 1;
  if (a.rpos) a.rpos[row] += 1;
  __threadfence();
  const unsigned finished = atomicAdd(reinterpret_cast<unsigned*>(&a.rng[2]), 1u) + 1u;
  if (finished == (unsigned)rows) {
    a.rng[1] += rng_inc;
    if (a.slot) *a.slot += 1;
    *reinterpret_cast<unsigned*>(&a.rng[2]) = 0u;
    __threadfence();
  }
}

// Probability mass of e = exp(score - max) as a fixed-point integer: e * 2^62 / S (S = the row's sum of e, so the masses
// of a row total about 2^62 and never overflow); integer sums are order-independent, the mass is monotone in e and
// equal for equal e.
__device__ __forceinline__ unsigned long long mass_of(float e, float scale) { return __float2ull_rn(e * scale); }

// Workspace of one row shared by its P cooperating blocks (all co-resident; the row's blocks meet at a counter barrier
// after every pass and read each other's partial results in a fixed order, so the result does not depend on timing).
// Every field is written once per launch and read by other SMs only through L2 (__ldcg): an SM's L1 is not coherent
// with the other SMs' writes, so a re-read of an address it had cached could return a stale line. The four radix
// passes therefore use four histogram buffers instead of reusing two.
constexpr int kSampleMaxParts = 8;        // blocks per row
constexpr int kSampleTieCap = 2048;       // ties at the threshold handled by rank (more fall back to the index-order path)
struct SampleRowWs {
  unsigned barrier, exit_ticket, tie_n, pad0;
  float fmax[kSampleMaxParts]; int imax[kSampleMaxParts];
  float fsum[kSampleMaxParts]; float fabove[kSampleMaxParts];
  float best[kSampleMaxParts]; int besti[kSampleMaxParts];
  int tie_cnt[kSampleMaxParts];           // index-order fallback: ties per block
  int tie_idx[kSampleTieCap];
  unsigned long long hist[4][kSampleMaxParts][256];
};

__device__ __forceinline__ void row_barrier(unsigned* ctr, unsigned target) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    atomicAdd(ctr, 1u);
    while (*reinterpret_cast<volatile unsigned*>(ctr) < target) __nanosleep(64);
    __threadfence();
  }
  __syncthreads();
}

// One row per P blocks of 1024 threads: the top-p decision by a radix select over the ascending bit patterns of
// e = exp(score - max) (non-negative floats order as their bits) with fixed-point mass histograms (one per warp, the
// blocks' partials summed in order), then torch's multinomial draw over the kept tokens. Every pass reads the row
// coalesced, eight tokens per thread.
__global__ void __launch_bounds__(kSampleThreads) sample_tokens_kernel(SampleArgs a, SampleRowWs* ws_all, int P) {
  extern __shared__ unsigned long long hist_w[];          // [warps][256] mass histograms, one per warp (no inter-warp conflicts)
  __shared__ float red_f[32];
  __shared__ unsigned long long red_u[64];
  __shared__ unsigned sh_u32[4];
  __shared__ int red_i[32];
  __shared__ int tie_idx[kSampleTieCap];
  const int row = blockIdx.x / P, part = blockIdx.x - row * P, V = a.V, tid = threadIdx.x, nt = blockDim.x;
  const int warp = tid >> 5, lane = tid & 31, nw = nt >> 5;
  const int g = part * nt + tid, G = P * nt;               // this thread's index and the row's thread count
  SampleRowWs& ws = ws_all[row];
  unsigned phase = 0u;
  auto barrier = [&]() { ++phase; row_barrier(&ws.barrier, phase * (unsigned)P); };
  const __nv_bfloat16* __restrict__ lg = a.logits + (size_t)row * a.ld;
  const unsigned long long seed = a.rng[0], offset = a.rng[1];
  auto score = [&](int i) -> float {
    if (i >= a.mask_lo && i < a.mask_lo + a.mask_n) return -INFINITY;
    const float s = bf(lg[i]);
    return a.temperature > 0.0f ? __fdiv_rn(s, a.temperature) : s;
  };
  // Vector form: 8 consecutive tokens per thread per iteration (independent loads in flight) when the row is 16-byte
  // aligned and V % 8 == 0 (Qwen3's vocabulary is); the scalar loops serve the rest.
  const bool vec8 = (V & 7) == 0 && ((reinterpret_cast<uintptr_t>(lg) & 15u) == 0);
  auto score8 = [&](int i, float* out) {                      // scores of tokens i .. i + 7
    const uint4 raw = *reinterpret_cast<const uint4*>(lg + i);
    const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = __bfloat1622float2(h2[j]);
      out[2 * j] = f.x; out[2 * j + 1] = f.y;
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int t = i + j;
      if (t >= a.mask_lo && t < a.mask_lo + a.mask_n) out[j] = -INFINITY;
      else if (a.temperature > 0.0f) out[j] = __fdiv_rn(out[j], a.temperature);
    }
  };
  auto load8 = [&](const float* src, int i, float* out) {
    const float4 lo4 = *reinterpret_cast<const float4*>(src + i), hi4 = *reinterpret_cast<const float4*>(src + i + 4);
    out[0] = lo4.x; out[1] = lo4.y; out[2] = lo4.z; out[3] = lo4.w; out[4] = hi4.x; out[5] = hi4.y; out[6] = hi4.z; out[7] = hi4.w;
  };
  // --- maximum (and its lowest index: the greedy token)
  float mx = -INFINITY; int mi = V;
  if (vec8) {
    for (int i = g * 8; i < V; i += G * 8) {
      float sc[8]; score8(i, sc);
#pragma unroll
      for (int j = 0; j < 8; ++j) if (sc[j] > mx) { mx = sc[j]; mi = i + j; }
    }
  } else {
    for (int i = g; i < V; i += G) {
      const float s = score(i);
      if (s > mx) { mx = s; mi = i; }
    }
  }
  const float Mb = block_max_f(mx, red_f);
  int am = (mx == Mb) ? mi : V;
  for (int o = 16; o > 0; o >>= 1) am = min(am, __shfl_xor_sync(0xffffffffu, am, o));
  if (tid == 0) sh_u32[1] = (unsigned)V;
  __syncthreads();
  if (lane == 0) atomicMin(&sh_u32[1], (unsigned)am);
  __syncthreads();
  if (tid == 0) { ws.fmax[part] = Mb; ws.imax[part] = (int)sh_u32[1]; }
  barrier();
  float M = -INFINITY; int argmax_i = V;
  for (int q = 0; q < P; ++q) {                             // the row's maximum and its lowest index, in a fixed order
    const float m = __ldcg(&ws.fmax[q]); const int im = __ldcg(&ws.imax[q]);
    if (m > M || (m == M && im < argmax_i)) { M = m; argmax_i = im; }
  }
  if (a.temperature <= 0.0f) {
    if (part == 0 && tid == 0) sample_finish(a, row, (int)(gridDim.x / P), (long long)argmax_i, 0ull);     // greedy draws nothing
    if (tid == 0 && atomicAdd(&ws.exit_ticket, 1u) + 1u == (unsigned)P) { ws.barrier = 0u; ws.exit_ticket = 0u; ws.tie_n = 0u; __threadfence(); }
    return;
  }
  // --- e_i = exp(score - M) once (fp32 scratch) and the softmax denominator
  float* __restrict__ e = a.scratch + (size_t)row * V;
  float sum = 0.0f;
  if (vec8) {
    for (int i = g * 8; i < V; i += G * 8) {
      float sc[8]; score8(i, sc);
      float4 lo4, hi4;
      lo4.x = expf(sc[0] - M); lo4.y = expf(sc[1] - M); lo4.z = expf(sc[2] - M); lo4.w = expf(sc[3] - M);
      hi4.x = expf(sc[4] - M); hi4.y = expf(sc[5] - M); hi4.z = expf(sc[6] - M); hi4.w = expf(sc[7] - M);
      *reinterpret_cast<float4*>(e + i) = lo4;
      *reinterpret_cast<float4*>(e + i + 4) = hi4;
      sum += (lo4.x + lo4.y) + (lo4.z + lo4.w) + (hi4.x + hi4.y) + (hi4.z + hi4.w);
    }
  } else {
    for (int i = g; i < V; i += G) {
      const float v = expf(score(i) - M);
      e[i] = v;
      sum += v;
    }
  }
  const float Sb = block_sum_f(sum, red_f);
  if (tid == 0) ws.fsum[part] = Sb;
  barrier();
  float S = 0.0f;
  for (int q = 0; q < P; ++q) S += __ldcg(&ws.fsum[q]);
  const float mscale = __fdiv_rn(4611686018427387904.0f, S);           // S >= 1: the maximum's e is 1
  // rule 0: drop tokens whose ascending cumulative mass (incl. themselves) <= T, T = (1 - top_p) * total
  // rule 1: drop tokens whose descending cumulative mass before them > top_p, i.e. ascending cumulative < T,
  //         T = (1 - top_p) * total   (both: the ascending prefix of mass up to T goes; among equal scores torch's
  //         stable sort removes the lowest indices first under rule 0 and the highest first under rule 1)
  // --- radix select of the key tau (bits of e) with mass(key < tau) <= T < mass(key <= tau); the first pass's bins
  // sum to the total mass
  unsigned long long* my_hist = hist_w + warp * 256;
  unsigned prefix = 0u; unsigned long long below = 0ull, T = 0ull;
  for (int d = 3; d >= 0; --d) {
    for (int b = tid; b < nw * 256; b += nt) hist_w[b] = 0ull;
    __syncthreads();
    const int shift = 8 * d;
    const unsigned pmask = d == 3 ? 0u : (0xFFFFFFFFu << (shift + 8));
    if (vec8) {
      for (int i = g * 8; i < V; i += G * 8) {
        float v[8]; load8(e, i, v);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const unsigned k = __float_as_uint(v[j]);
          if ((k & pmask) == prefix) atomicAdd(&my_hist[(k >> shift) & 0xFFu], mass_of(v[j], mscale));
        }
      }
    } else {
      for (int i = g; i < V; i += G) {
        const float v = e[i];
        const unsigned k = __float_as_uint(v);
        if ((k & pmask) == prefix) atomicAdd(&my_hist[(k >> shift) & 0xFFu], mass_of(v, mscale));
      }
    }
    __syncthreads();
    unsigned long long* part_hist = ws.hist[d][part];
    for (int b = tid; b < 256; b += nt) {                   // the warps' bins summed, this block's partial published
      unsigned long long t = 0ull;
      for (int w = 0; w < nw; ++w) t += hist_w[w * 256 + b];
      part_hist[b] = t;
    }
    barrier();
    for (int b = tid; b < 256; b += nt) {                   // the row's bins: the blocks' partials in a fixed order
      unsigned long long t = 0ull;
      for (int q = 0; q < P; ++q) t += __ldcg(&ws.hist[d][q][b]);
      hist_w[b] = t;
    }
    __syncthreads();
    if (d == 3) {
      unsigned long long total = 0ull;
      for (int b = 0; b < 256; ++b) total += hist_w[b];
      T = (unsigned long long)((double)(1.0f - a.top_p) * (double)total);
    }
    unsigned long long acc = below; int chosen = 255;
    for (int b = 0; b < 256; ++b) {                          // every thread scans the 256 bins (redundant, no sync)
      if (acc + hist_w[b] > T) { chosen = b; break; }
      acc += hist_w[b];
    }
    prefix |= ((unsigned)chosen << shift); below = acc;
    __syncthreads();
  }
  const unsigned tau = prefix;
  const unsigned long long tie_q = mass_of(__uint_as_float(tau), mscale);   // every tie token has this mass
  // --- ties (key == tau) collected by index, and the kept sum above tau
  float s_above = 0.0f;
  if (vec8) {
    for (int i = g * 8; i < V; i += G * 8) {
      float v[8]; load8(e, i, v);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const unsigned k = __float_as_uint(v[j]);
        if (k > tau) s_above += v[j];
        else if (k == tau) { const unsigned pos = atomicAdd(&ws.tie_n, 1u); if (pos < (unsigned)kSampleTieCap) ws.tie_idx[pos] = i + j; }
      }
    }
  } else {
    for (int i = g; i < V; i += G) {
      const unsigned k = __float_as_uint(e[i]);
      if (k > tau) s_above += e[i];
      else if (k == tau) { const unsigned pos = atomicAdd(&ws.tie_n, 1u); if (pos < (unsigned)kSampleTieCap) ws.tie_idx[pos] = i; }
    }
  }
  const float ab = block_sum_f(s_above, red_f);
  if (tid == 0) ws.fabove[part] = ab;
  barrier();
  const int tie_total = (int)__ldcg(&ws.tie_n);
  unsigned long long k_rem = 0ull;                          // ties removed (rule 0: the lowest indices, rule 1: the highest)
  if (tie_q > 0ull) {
    const unsigned long long room = T > below ? T - below : 0ull;
    // rule 0 removes the k-th tie when below + (k + 1) * tie_q <= T, rule 1 when it is < T
    k_rem = a.rule == 0 ? room / tie_q : (room == 0ull ? 0ull : (room - 1ull) / tie_q);
    if (k_rem > (unsigned long long)tie_total) k_rem = (unsigned long long)tie_total;
  }
  const bool ranked = tie_total <= kSampleTieCap;
  if (ranked) {                                             // the tie indices sorted ascending (bitonic, in every block)
    int n2 = 1; while (n2 < tie_total) n2 <<= 1;
    for (int i = tid; i < n2; i += nt) tie_idx[i] = i < tie_total ? __ldcg(&ws.tie_idx[i]) : 0x7FFFFFFF;
    __syncthreads();
    if (tie_total > 1)
      for (int size = 2; size <= n2; size <<= 1)
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
          for (int i = tid; i < n2; i += nt) {
            const int partner = i ^ stride;
            if (partner > i) {
              const bool up = (i & size) == 0;
              const int x = tie_idx[i], y = tie_idx[partner];
              if ((x > y) == up) { tie_idx[i] = y; tie_idx[partner] = x; }
            }
          }
          __syncthreads();
        }
  }
  // kept: above tau, the maximum, and the ties by rank (rule 0 keeps ranks >= k_rem, rule 1 ranks < total - k_rem)
  auto tie_rank_of = [&](int i) -> int {                    // position of index i in the sorted tie list
    int lo = 0, hi = tie_total - 1;
    while (lo < hi) { const int mid = (lo + hi) >> 1; if (tie_idx[mid] < i) lo = mid + 1; else hi = mid; }
    return lo;
  };
  auto rank_kept = [&](unsigned long long r) -> bool {
    return a.rule == 0 ? r >= k_rem : r < (unsigned long long)tie_total - k_rem;
  };
  float Sn = S;
  if (a.rule == 0) {                                        // the kept sum: tokens above tau plus the kept ties (equal values)
    Sn = 0.0f;
    for (int q = 0; q < P; ++q) Sn += __ldcg(&ws.fabove[q]);
    Sn += (float)((unsigned long long)tie_total - k_rem) * __uint_as_float(tau);
  }
  // --- the draw: torch.multinomial(probs, 1) == argmax_i probs_i / q_i with q_i ~ Exp(1) drawn as torch's
  // exponential_ draws them over the [n, V] tensor (its Philox word per element, curand's (0, 1] uniform, -__logf with
  // the near-1 guard; dropped tokens have probability 0, and their words are skipped since a zero ratio never wins).
  // probs: rule 0 renormalizes over the kept tokens (transformers' softmax after top-p), rule 1 keeps FlashDrive's
  // probabilities as they are (the scale does not change the argmax). Equal ratios: the lowest index, as torch.argmax.
  const unsigned long long numel = (unsigned long long)(gridDim.x / P) * (unsigned long long)V;
  const unsigned long long blocks = (numel + 255ull) / 256ull;
  const unsigned TT = (unsigned)(blocks < a.rng_blocks ? blocks : a.rng_blocks) * 256u;
  const unsigned long long rng_inc = 4ull * ((numel - 1ull) / (4ull * TT) + 1ull);
  const unsigned row_base = (unsigned)row * (unsigned)V;
  float best = -1.0f; int besti = V;
  auto draw = [&](int i, float v) {
    const float p = __fdiv_rn(v, Sn);
    const unsigned x = torch_philox_word(seed, offset, row_base + (unsigned)i, TT);
    const float u = x * 2.3283064365386963e-10f + (2.3283064365386963e-10f / 2.0f);     // curand_uniform: (0, 1]
    const float q = u >= 1.0f - 1.1920928955078125e-07f / 2.0f ? 1.1920928955078125e-07f / 2.0f : -__logf(u);
    const float r = __fdiv_rn(p, q);
    if (r > best || (r == best && i < besti)) { best = r; besti = i; }
  };
  if (ranked) {
    if (vec8) {
      for (int i = g * 8; i < V; i += G * 8) {
        float v[8]; load8(e, i, v);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const unsigned k = __float_as_uint(v[j]);
          const int t = i + j;
          if (k > tau || t == argmax_i || (k == tau && rank_kept((unsigned long long)tie_rank_of(t)))) draw(t, v[j]);
        }
      }
    } else {
      for (int i = g; i < V; i += G) {
        const unsigned k = __float_as_uint(e[i]);
        if (k > tau || i == argmax_i || (k == tau && rank_kept((unsigned long long)tie_rank_of(i)))) draw(i, e[i]);
      }
    }
  } else {
    // more ties than the list holds: contiguous chunks in index order, ranks by scans of the chunks' tie counts
    // (the chunks cover entries other blocks wrote: read them through L2)
    const int chunk = (V + G - 1) / G, i0 = min(V, g * chunk), i1 = min(V, i0 + chunk);
    int local_ties = 0;
    for (int i = i0; i < i1; ++i) local_ties += (__float_as_uint(__ldcg(&e[i])) == tau);
    unsigned long long block_ties;
    int tr = (int)block_exscan_u64((unsigned long long)local_ties, red_u, &block_ties);
    if (tid == 0) ws.tie_cnt[part] = (int)block_ties;
    barrier();
    for (int q = 0; q < part; ++q) tr += __ldcg(&ws.tie_cnt[q]);
    for (int i = i0; i < i1; ++i) {
      const float v = __ldcg(&e[i]);
      const unsigned k = __float_as_uint(v);
      bool kept = k > tau || i == argmax_i;
      if (k == tau) kept = kept || rank_kept((unsigned long long)(tr++));
      if (kept) draw(i, v);
    }
  }
  // block argmax of (best, besti): the larger ratio, the lower index among equal ratios; then the row's blocks
  for (int o = 16; o > 0; o >>= 1) {
    const float ob = __shfl_xor_sync(0xffffffffu, best, o);
    const int oi = __shfl_xor_sync(0xffffffffu, besti, o);
    if (ob > best || (ob == best && oi < besti)) { best = ob; besti = oi; }
  }
  if (lane == 0) { red_f[warp] = best; red_i[warp] = besti; }
  __syncthreads();
  if (warp == 0) {
    float bv = lane < nw ? red_f[lane] : -1.0f;
    int bi = lane < nw ? red_i[lane] : V;
    for (int o = 16; o > 0; o >>= 1) {
      const float ob = __shfl_xor_sync(0xffffffffu, bv, o);
      const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
      if (ob > bv || (ob == bv && oi < bi)) { bv = ob; bi = oi; }
    }
    if (lane == 0) { ws.best[part] = bv; ws.besti[part] = bi; }
  }
  barrier();
  if (tid == 0) {
    if (part == 0) {
      float bv = -1.0f; int bi = V;
      for (int q = 0; q < P; ++q) {
        const float ob = __ldcg(&ws.best[q]); const int oi = __ldcg(&ws.besti[q]);
        if (ob > bv || (ob == bv && oi < bi)) { bv = ob; bi = oi; }
      }
      sample_finish(a, row, (int)(gridDim.x / P), (long long)bi, rng_inc);
    }
    if (atomicAdd(&ws.exit_ticket, 1u) + 1u == (unsigned)P) {   // the last block out resets the row's counters
      ws.barrier = 0u; ws.exit_ticket = 0u; ws.tie_n = 0u;
      __threadfence();
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// action expert: Fourier features, LayerNorm with fp32 output, output projection + Euler step
// ---------------------------------------------------------------------------------------------------------------------
template <bool kXBf16>
__global__ void action_fourier_kernel(const void* __restrict__ xp, const void* __restrict__ tp, const __nv_bfloat16* __restrict__ freqs,
                                      int rows, int rows_per_sample, int act_dim, int nf, __nv_bfloat16* __restrict__ out, int ld_out,
                                      int width) {
  const int row = blockIdx.x, half = nf / 2;
  const float sqrt2 = 1.41421356237309504880f, two_pi_step = 3.14159274101257324219f;
  for (int c = threadIdx.x; c < ld_out; c += blockDim.x) {
    float v = 0.0f;
    if (c < width) {
      const int grp = c / nf, k = c - grp * nf;          // group grp: action dims, then the timestep
      const int f = k < half ? k : k - half;
      if (kXBf16) {
        // BF16 state and timesteps: every elementwise op of x * freq * 2 * pi, sin / cos, * sqrt(2) rounds to BF16
        const float xv = grp < act_dim ? bf(static_cast<const __nv_bfloat16*>(xp)[(size_t)row * act_dim + grp])
                                       : bf(static_cast<const __nv_bfloat16*>(tp)[row / rows_per_sample]);
        const float a1 = bf(to_bf(xv * bf(freqs[f]))), a2 = bf(to_bf(a1 * 2.0f)), arg = bf(to_bf(a2 * two_pi_step));
        const float tr = bf(to_bf(k < half ? sinf(arg) : cosf(arg)));
        v = bf(to_bf(tr * sqrt2));
      } else {
        const float src = grp < act_dim ? static_cast<const float*>(xp)[(size_t)row * act_dim + grp]
                                        : static_cast<const float*>(tp)[row / rows_per_sample];
        const float arg = ((src * bf(freqs[f])) * 2.0f) * two_pi_step;   // x * freq * 2 * pi, left to right, fp32
        v = (k < half ? sinf(arg) : cosf(arg)) * sqrt2;
      }
    }
    out[(size_t)row * ld_out + c] = to_bf(v);
  }
}

// FlashDrive's BF16 Euler update: x = bf16(x + bf16(dt * v)).
__global__ void euler_bf16_kernel(__nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ v, float dt, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = to_bf(bf(x[i]) + bf(to_bf(dt * bf(v[i]))));
}

__global__ void __launch_bounds__(256) layernorm_f32out_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                                                               const __nv_bfloat16* __restrict__ b, float* __restrict__ out, int dim,
                                                               float eps) {
  __shared__ float smem[32];
  const int row = blockIdx.x;
  const __nv_bfloat16* xr = x + (size_t)row * dim;
  float s = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) s += bf(xr[i]);
  const float mean = block_sum_f(s, smem) / (float)dim;
  float v = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) { const float d = bf(xr[i]) - mean; v += d * d; }
  const float rstd = rsqrtf(block_sum_f(v, smem) / (float)dim + eps);
  for (int i = threadIdx.x; i < dim; i += blockDim.x)
    out[(size_t)row * dim + i] = (bf(xr[i]) - mean) * rstd * bf(w[i]) + bf(b[i]);
}

// one warp per (row, j): v = bf16(dot + bias) (stored when v_out is given); x += dt * v when x is given
__global__ void action_out_euler_kernel(const __nv_bfloat16* __restrict__ hn, const __nv_bfloat16* __restrict__ w,
                                        const __nv_bfloat16* __restrict__ b, float dt, float* __restrict__ x,
                                        __nv_bfloat16* __restrict__ v_out, int rows, int E, int act_dim) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (warp >= rows * act_dim) return;
  const int row = warp / act_dim, j = warp - row * act_dim;
  const Bf8* h8 = reinterpret_cast<const Bf8*>(hn + (size_t)row * E);
  const Bf8* w8 = reinterpret_cast<const Bf8*>(w + (size_t)j * E);
  float acc = 0.0f;
  for (int c = lane; c < E / 8; c += 32) {
    const Bf8 h = h8[c], ww = w8[c];
#pragma unroll
    for (int e = 0; e < 8; ++e) acc += bf(h.v[e]) * bf(ww.v[e]);
  }
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if (lane == 0) {
    const __nv_bfloat16 vb = to_bf(acc + bf(b[j]));
    if (v_out) v_out[(size_t)row * act_dim + j] = vb;
    if (x) x[(size_t)row * act_dim + j] += dt * bf(vb);
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// unicycle action -> trajectory (one thread per sample)
// ---------------------------------------------------------------------------------------------------------------------
__global__ void action_to_traj_kernel(const float* __restrict__ action, const float* __restrict__ v0, const float* __restrict__ z,
                                      int n, int T, float dt, float a_mean, float a_std, float k_mean, float k_std,
                                      float* __restrict__ xyz, float* __restrict__ rot) {
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= n) return;
  const float dt2 = 0.5f * (dt * dt), hdt = 0.5f * dt;
  const float* act = action + (size_t)s * T * 2;
  float vel_prev = v0[s], th_prev = 0.0f;                  // velocity[0] = v0, theta[0] = 0
  float cv = 0.0f, ct1 = 0.0f, ct2 = 0.0f, cx1 = 0.0f, cx2 = 0.0f, cy1 = 0.0f, cy2 = 0.0f;
  for (int i = 0; i < T; ++i) {
    const float accel = act[2 * i] * a_std + a_mean, kappa = act[2 * i + 1] * k_std + k_mean;
    cv += accel * dt;
    const float vel = v0[s] + cv;                           // velocity[i + 1]
    ct1 += kappa * vel_prev * dt;
    ct2 += kappa * accel * dt2;
    const float th = 0.0f + ct1 + ct2;                      // theta[i + 1]
    cx1 += vel_prev * cosf(th_prev) * hdt;
    cx2 += vel * cosf(th) * hdt;
    cy1 += vel_prev * sinf(th_prev) * hdt;
    cy2 += vel * sinf(th) * hdt;
    float* p = xyz + ((size_t)s * T + i) * 3;
    p[0] = 0.0f + cx1 + cx2;
    p[1] = 0.0f + cy1 + cy2;
    p[2] = z[s];
    float* r = rot + ((size_t)s * T + i) * 9;
    const float c = cosf(th), sn = sinf(th);
    r[0] = c; r[1] = -sn; r[2] = 0.0f;
    r[3] = sn; r[4] = c; r[5] = 0.0f;
    r[6] = 0.0f; r[7] = 0.0f; r[8] = 1.0f;
    vel_prev = vel; th_prev = th;
  }
}

}  // namespace

int alpamayo15_gather_rows_bytes(const void* table, long long ld_table, const long long* idx, const long long* base, void* out,
                                 long long ld_out, int rows, long long row_bytes, cudaStream_t stream) {
  if (!table || !idx || !out) return 1;
  if (rows <= 0 || row_bytes <= 0 || (row_bytes % 16) != 0 || ld_table < row_bytes || ld_out < row_bytes || (ld_table % 16) != 0 ||
      (ld_out % 16) != 0)
    return 2;
  if (!aligned16(table) || !aligned16(out)) return 5;
  const long long n = (long long)rows * (row_bytes / 16);
  gather_rows_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(static_cast<const unsigned char*>(table), ld_table, idx, base,
                                                                     static_cast<unsigned char*>(out), ld_out, rows, (int)(row_bytes / 16));
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int alpamayo15_gather_rows_bf16(const void* table, int ld_table, const long long* idx, void* out, int ld_out, int rows, int dim,
                                cudaStream_t stream) {
  if (dim <= 0 || (dim % 8) != 0 || ld_table < dim || ld_out < dim) return 2;
  return alpamayo15_gather_rows_bytes(table, 2LL * ld_table, idx, nullptr, out, 2LL * ld_out, rows, 2LL * dim, stream);
}

int alpamayo15_cast_f16_bf16(const void* x, void* out, long long n, cudaStream_t stream) {
  if (!x || !out) return 1;
  if (n <= 0 || (n % 4) != 0) return 2;
  if ((reinterpret_cast<uintptr_t>(x) & 7u) || (reinterpret_cast<uintptr_t>(out) & 7u)) return 5;
  cast_f16_bf16_kernel<<<(unsigned)((n / 4 + 255) / 256), 256, 0, stream>>>(static_cast<const __half*>(x), static_cast<__nv_bfloat16*>(out),
                                                                            n / 4);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int alpamayo15_dflash_propose(const long long* proposed, const long long* first, long long stop_id, long long traj_id, long long* block,
                              int B, cudaStream_t stream) {
  if (!proposed || !first || !block) return 1;
  if (B < 2) return 2;
  dflash_propose_kernel<<<1, 32, 0, stream>>>(proposed, first, stop_id, traj_id, block, B);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int alpamayo15_dflash_decide(const long long* block, const long long* posterior, long long stop_id, long long period_id, int B,
                             long long* stats, cudaStream_t stream) {
  if (!block || !posterior || !stats) return 1;
  if (B < 1) return 2;
  dflash_decide_kernel<<<1, 32, 0, stream>>>(block, posterior, stop_id, period_id, B, stats);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int alpamayo15_cast_f32_bf16(const float* x, void* out, long long n, cudaStream_t stream) {
  if (!x || !out) return 1;
  if (n <= 0 || (n % 4) != 0) return 2;
  if (!aligned16(x) || (reinterpret_cast<uintptr_t>(out) & 7u)) return 5;
  cast_f32_bf16_kernel<<<(unsigned)((n / 4 + 255) / 256), 256, 0, stream>>>(x, static_cast<__nv_bfloat16*>(out), n / 4);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int alpamayo15_scatter_rows_bf16(void* dst, int ld_dst, const long long* idx, const void* src, int ld_src, int rows, int dim, int add,
                                 cudaStream_t stream) {
  if (!dst || !idx || !src) return 1;
  if (rows <= 0 || dim <= 0 || (dim % 8) != 0 || ld_dst < dim || ld_src < dim || (ld_dst % 8) != 0 || (ld_src % 8) != 0) return 2;
  if (!aligned16(dst) || !aligned16(src)) return 5;
  const long long n = (long long)rows * (dim / 8);
  const unsigned blocks = (unsigned)((n + 255) / 256);
  if (add)
    scatter_rows_kernel<true><<<blocks, 256, 0, stream>>>(static_cast<__nv_bfloat16*>(dst), ld_dst, idx,
                                                           static_cast<const __nv_bfloat16*>(src), ld_src, rows, dim / 8);
  else
    scatter_rows_kernel<false><<<blocks, 256, 0, stream>>>(static_cast<__nv_bfloat16*>(dst), ld_dst, idx,
                                                            static_cast<const __nv_bfloat16*>(src), ld_src, rows, dim / 8);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int alpamayo15_mrope_tables(const int* pos3, const float* inv_freq, int sec_h, int sec_w, float scaling, void* cos, void* sin, int P,
                            int D, int out_f32, cudaStream_t stream) {
  if (!pos3 || !inv_freq || !cos || !sin) return 1;
  if (P <= 0 || D <= 0 || (D % 16) != 0 || sec_h < 0 || sec_w < 0 || 3 * sec_h > D / 2 || 3 * sec_w > D / 2) return 2;
  const long long n = (long long)P * (D / 2);
  const unsigned blocks = (unsigned)((n + 255) / 256);
  if (out_f32)
    mrope_tables_kernel<float><<<blocks, 256, 0, stream>>>(pos3, inv_freq, sec_h, sec_w, scaling, static_cast<float*>(cos),
                                                           static_cast<float*>(sin), P, D / 2);
  else
    mrope_tables_kernel<__nv_bfloat16><<<blocks, 256, 0, stream>>>(pos3, inv_freq, sec_h, sec_w, scaling, static_cast<__nv_bfloat16*>(cos),
                                                                   static_cast<__nv_bfloat16*>(sin), P, D / 2);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int alpamayo15_sample_workspace_bytes(int n) { return n > 0 ? n * (int)sizeof(SampleRowWs) : 0; }

int alpamayo15_sample_tokens_bf16(const void* logits, int ld, int n, int V, int mask_lo, int mask_n, float temperature, float top_p,
                                  int top_p_rule, unsigned long long* rng, int rng_blocks, float* scratch, void* workspace,
                                  long long workspace_bytes, long long* tok, const long long* unfinished, long long pad_id,
                                  long long* slot, long long* rpos, int* len, int* glen, cudaStream_t stream) {
  if (!logits || !tok || !rng || !workspace || (!scratch && temperature > 0.0f)) return 1;
  if (n <= 0 || V <= 0 || (ld != 0 && ld < V) || mask_lo < 0 || mask_n < 0 || mask_lo + mask_n > V || top_p <= 0.0f || top_p > 1.0f ||
      (top_p_rule != 0 && top_p_rule != 1) || mask_n >= V || rng_blocks <= 0 || workspace_bytes < alpamayo15_sample_workspace_bytes(n))
    return 2;
  if ((reinterpret_cast<uintptr_t>(workspace) & 15u) != 0) return 5;
  // Blocks per row: as many as stay co-resident for n rows (the row's blocks wait for each other), at most 8.
  static int max_resident = 0;
  if (max_resident == 0) {
    cudaFuncSetAttribute(sample_tokens_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSampleHistBytes);
    int per_sm = 0, dev = 0; cudaDeviceProp prop;
    cudaGetDevice(&dev); cudaGetDeviceProperties(&prop, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, sample_tokens_kernel, kSampleThreads, kSampleHistBytes);
    max_resident = per_sm > 0 ? per_sm * prop.multiProcessorCount : 1;
  }
  int P = max_resident / n;
  if (P > kSampleMaxParts) P = kSampleMaxParts;
  if (P < 1) P = 1;
  SampleArgs a{static_cast<const __nv_bfloat16*>(logits), ld, V, mask_lo, mask_n, temperature, top_p, top_p_rule, rng,
               (unsigned)rng_blocks, scratch, tok, unfinished, pad_id, slot, rpos, len, glen};
  sample_tokens_kernel<<<n * P, kSampleThreads, kSampleHistBytes, stream>>>(a, static_cast<SampleRowWs*>(workspace), P);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int alpamayo15_action_fourier_bf16(const void* x, int x_bf16, const void* t, const void* freqs, int rows, int rows_per_sample, int act_dim,
                                   int nf, void* out, int ld_out, cudaStream_t stream) {
  if (!x || !t || !freqs || !out) return 1;
  const int width = (act_dim + 1) * nf;
  if (rows <= 0 || rows_per_sample <= 0 || act_dim <= 0 || nf <= 0 || (nf % 2) != 0 || ld_out < width) return 2;
  if (x_bf16)
    action_fourier_kernel<true><<<rows, 128, 0, stream>>>(x, t, static_cast<const __nv_bfloat16*>(freqs), rows, rows_per_sample, act_dim,
                                                          nf, static_cast<__nv_bfloat16*>(out), ld_out, width);
  else
    action_fourier_kernel<false><<<rows, 128, 0, stream>>>(x, t, static_cast<const __nv_bfloat16*>(freqs), rows, rows_per_sample, act_dim,
                                                           nf, static_cast<__nv_bfloat16*>(out), ld_out, width);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int alpamayo15_euler_bf16(void* x, const void* v, float dt, int n, cudaStream_t stream) {
  if (!x || !v) return 1;
  if (n <= 0) return 2;
  euler_bf16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(static_cast<__nv_bfloat16*>(x), static_cast<const __nv_bfloat16*>(v), dt, n);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int alpamayo15_layernorm_f32out_bf16(const void* x, const void* w, const void* b, float* out, int rows, int dim, float eps,
                                     cudaStream_t stream) {
  if (!x || !w || !b || !out) return 1;
  if (rows <= 0 || dim <= 0 || (dim % 4) != 0) return 2;
  layernorm_f32out_kernel<<<rows, 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(x), static_cast<const __nv_bfloat16*>(w),
                                                     static_cast<const __nv_bfloat16*>(b), out, dim, eps);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int alpamayo15_action_out_euler(const void* hn, const void* w, const void* b, float dt, float* x, void* v_out, int rows, int E,
                                int act_dim, cudaStream_t stream) {
  if (!hn || !w || !b || (!x && !v_out)) return 1;
  if (rows <= 0 || E <= 0 || (E % 8) != 0 || act_dim <= 0) return 2;
  if (!aligned16(hn) || !aligned16(w)) return 5;
  const int warps = rows * act_dim;
  action_out_euler_kernel<<<(warps + 7) / 8, 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(hn), static_cast<const __nv_bfloat16*>(w),
                                                                static_cast<const __nv_bfloat16*>(b), dt, x,
                                                                static_cast<__nv_bfloat16*>(v_out), rows, E, act_dim);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int alpamayo15_action_to_traj(const float* action, const float* v0, const float* z, int n, int T, float dt, float accel_mean,
                              float accel_std, float kappa_mean, float kappa_std, float* xyz, float* rot, cudaStream_t stream) {
  if (!action || !v0 || !z || !xyz || !rot) return 1;
  if (n <= 0 || T <= 0) return 2;
  action_to_traj_kernel<<<(n + 63) / 64, 64, 0, stream>>>(action, v0, z, n, T, dt, accel_mean, accel_std, kappa_mean, kappa_std, xyz, rot);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

}  // namespace kernels
}  // namespace flash_rt
