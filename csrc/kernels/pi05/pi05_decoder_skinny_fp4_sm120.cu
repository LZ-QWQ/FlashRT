// SPDX-License-Identifier: Apache-2.0
// Skinny NVFP4 (W4A4) decoder family for small-row action decoders on sm_120a. See the header.
#include "pi05_decoder_skinny_fp4_sm120.cuh"

#include <cuda_fp8.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace flash_rt {
namespace pi05_dec_skinny_fp4 {
namespace {

// D = A(e2m1, 16x64) * B(e2m1, 64x8) with per-16 UE4M3 scales, FP32 accumulate.
__device__ __forceinline__ void mma_nvf4(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb) {
    const uint16_t z = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "h"(z), "h"(z), "r"(sfb), "h"(z),
          "h"(z));
}

__device__ __forceinline__ void cp_async_16(void* dst, const void* src, bool valid) {
    const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    const int n = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(n));
}

__device__ __forceinline__ uint4 ldg_stream(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ uint2 ldg_stream8(const void* p) {
    uint2 v;
    asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];\n" : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}

__device__ __forceinline__ void pdl_prologue() {
    asm volatile("griddepcontrol.launch_dependents;\n" ::);
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
}

__device__ __forceinline__ float tanh_gelu(float g) {
    return g / (1.0f + expf(-1.5957691216057308f * g * (1.0f + 0.044715f * g * g)));
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Smallest UE4M3 value >= v (v >= 0), saturating at 448.
__device__ __forceinline__ uint8_t ue4m3_ceil(float v) {
    if (!(v > 0.f)) return 0;
    if (v >= 448.f) return 0x7E;
    const __nv_fp8_e4m3 q(v);
    uint8_t b = q.__x;
    if (static_cast<float>(q) < v) ++b;
    return b;
}

__device__ __forceinline__ float ue4m3_value(uint8_t b) {
    __nv_fp8_e4m3 q;
    q.__x = b;
    return static_cast<float>(q);
}

// Four floats -> four e2m1 (round to nearest even, saturating), element 0 in the low nibble.
__device__ __forceinline__ uint16_t e2m1x4(float v0, float v1, float v2, float v3) {
    uint16_t r;
    asm("{\n"
        ".reg .b8 b0, b1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
        "mov.b16 %0, {b0, b1};\n"
        "}"
        : "=h"(r)
        : "f"(v0), "f"(v1), "f"(v2), "f"(v3));
    return r;
}

// Quantize the 16 values v[] (already divided by the global scale) of one block.
__device__ __forceinline__ void quant_block16(const float* v, uint8_t* out8, uint8_t* sf) {
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(v[i]));
    const uint8_t s = ue4m3_ceil(amax * (1.0f / 6.0f));
    const float d = ue4m3_value(s);
    const float inv = d > 0.f ? 1.0f / d : 0.f;
    uint16_t h[4];
#pragma unroll
    for (int q = 0; q < 4; ++q)
        h[q] = e2m1x4(v[4 * q] * inv, v[4 * q + 1] * inv, v[4 * q + 2] * inv, v[4 * q + 3] * inv);
    *reinterpret_cast<uint2*>(out8) = make_uint2(h[0] | (static_cast<uint32_t>(h[1]) << 16),
                                                 h[2] | (static_cast<uint32_t>(h[3]) << 16));
    *sf = s;
}

// Quad-cooperative quantization: four lanes (lane ^ 1, lane ^ 2 partners) own
// four consecutive values each of one 16-element block. Lane (q & 3) == 0
// writes the scale; every lane writes its two bytes.
__device__ __forceinline__ void quant_quad4(float v0, float v1, float v2, float v3, uint8_t* out2,
                                            uint8_t* sf, bool write_sf) {
    float amax = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    const uint8_t s = ue4m3_ceil(amax * (1.0f / 6.0f));
    const float d = ue4m3_value(s);
    const float inv = d > 0.f ? 1.0f / d : 0.f;
    *reinterpret_cast<uint16_t*>(out2) = e2m1x4(v0 * inv, v1 * inv, v2 * inv, v3 * inv);
    if (write_sf) *sf = s;
}

// One CTA: BN weight rows x KC of K for one 16-row tile (blockIdx.z picks the tile).
template <int BN, int KC, int NW, bool ABF16, bool PDL>
__global__ void __launch_bounds__(NW * 32)
gemm_kernel(const void* __restrict__ A_, const uint8_t* __restrict__ SFA, const float* __restrict__ a_gs,
            const uint8_t* __restrict__ W, const uint8_t* __restrict__ SFB, float* __restrict__ partials, int M,
            int N, int K) {
    constexpr int THREADS = NW * 32;
    constexpr int NT = BN / 8;
    constexpr int NTW = NT / NW;
    constexpr int KCH = KC / 128;
    constexpr int AB = KC / 2;        // e2m1 bytes per row
    constexpr int ASTR = AB + 16;
    constexpr int SB = KC / 16;       // scale bytes per row
    constexpr int SSTR = SB + 4;
    static_assert(NT % NW == 0, "n8 tiles must split across warps");
    static_assert(KC % 128 == 0, "K chunk must be a multiple of 128");
    __shared__ __align__(16) uint8_t sA[16 * ASTR];
    __shared__ __align__(16) uint8_t sS[16 * SSTR];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, t0 = lane & 3, t1 = lane >> 2;
    const int n0 = blockIdx.x * BN;
    const int kb = blockIdx.y * KC;
    const int m0 = blockIdx.z * 16;
    const int kchunks = K / 128;

    uint4 w[KCH][NTW];
    uint2 sb[KCH][NTW];
#pragma unroll
    for (int c = 0; c < KCH; ++c)
#pragma unroll
        for (int t = 0; t < NTW; ++t) {
            const size_t tile = static_cast<size_t>(n0 / 8 + warp * NTW + t) * kchunks + kb / 128 + c;
            w[c][t] = ldg_stream(W + (tile * 32 + lane) * 16);
            sb[c][t] = ldg_stream8(SFB + (tile * 8 + t1) * 8);
        }

    if constexpr (PDL) pdl_prologue();

    if constexpr (!ABF16) {
        const uint8_t* A = static_cast<const uint8_t*>(A_);
        for (int c = tid; c < 16 * (AB / 16); c += THREADS) {
            const int row = c / (AB / 16);
            const int off = (c % (AB / 16)) * 16;
            const bool v = m0 + row < M;
            cp_async_16(sA + row * ASTR + off, A + static_cast<size_t>(v ? m0 + row : 0) * (K / 2) + kb / 2 + off, v);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        for (int c = tid; c < 16 * (SB / 4); c += THREADS) {
            const int row = c / (SB / 4);
            const int off = (c % (SB / 4)) * 4;
            uint32_t s = 0;
            if (m0 + row < M)
                s = *reinterpret_cast<const uint32_t*>(SFA + static_cast<size_t>(m0 + row) * (K / 16) + kb / 16 + off);
            *reinterpret_cast<uint32_t*>(sS + row * SSTR + off) = s;
        }
        asm volatile("cp.async.wait_group 0;\n" ::);
    } else {
        const __nv_bfloat16* A = static_cast<const __nv_bfloat16*>(A_);
        const float inv_gs = 1.0f / (*a_gs);
        for (int b = tid; b < 16 * SB; b += THREADS) {
            const int row = b / SB;
            const int blk = b % SB;
            uint8_t* dst = sA + row * ASTR + blk * 8;
            if (m0 + row < M) {
                const __nv_bfloat16* src = A + static_cast<size_t>(m0 + row) * K + kb + blk * 16;
                const uint4 p0 = *reinterpret_cast<const uint4*>(src);
                const uint4 p1 = *reinterpret_cast<const uint4*>(src + 8);
                const __nv_bfloat162* h0 = reinterpret_cast<const __nv_bfloat162*>(&p0);
                const __nv_bfloat162* h1 = reinterpret_cast<const __nv_bfloat162*>(&p1);
                float v[16];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float2 f0 = __bfloat1622float2(h0[i]);
                    const float2 f1 = __bfloat1622float2(h1[i]);
                    v[2 * i] = f0.x * inv_gs; v[2 * i + 1] = f0.y * inv_gs;
                    v[8 + 2 * i] = f1.x * inv_gs; v[8 + 2 * i + 1] = f1.y * inv_gs;
                }
                quant_block16(v, dst, sS + row * SSTR + blk);
            } else {
                *reinterpret_cast<uint2*>(dst) = make_uint2(0, 0);
                sS[row * SSTR + blk] = 0;
            }
        }
    }
    __syncthreads();

    float acc[NTW][4];
#pragma unroll
    for (int t = 0; t < NTW; ++t)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[t][i] = 0.f;

    const int sau = (lane & 1) * 8 + (lane >> 2);   // scale-A row owned by this lane
#pragma unroll
    for (int c = 0; c < KCH; ++c) {
#pragma unroll
        for (int s = 0; s < 2; ++s) {
            const uint8_t* ra = sA + t1 * ASTR + c * 64 + s * 32 + t0 * 4;
            const uint32_t a0 = *reinterpret_cast<const uint32_t*>(ra);
            const uint32_t a1 = *reinterpret_cast<const uint32_t*>(ra + 8 * ASTR);
            const uint32_t a2 = *reinterpret_cast<const uint32_t*>(ra + 16);
            const uint32_t a3 = *reinterpret_cast<const uint32_t*>(ra + 8 * ASTR + 16);
            const uint32_t sfa = *reinterpret_cast<const uint32_t*>(sS + sau * SSTR + c * 8 + s * 4);
#pragma unroll
            for (int t = 0; t < NTW; ++t) {
                const uint32_t b0 = s == 0 ? w[c][t].x : w[c][t].z;
                const uint32_t b1 = s == 0 ? w[c][t].y : w[c][t].w;
                const uint32_t sfb = s == 0 ? sb[c][t].x : sb[c][t].y;
                mma_nvf4(acc[t], a0, a1, a2, a3, b0, b1, sfa, sfb);
            }
        }
    }

    float* base = partials + static_cast<size_t>(blockIdx.y) * M * N;
#pragma unroll
    for (int t = 0; t < NTW; ++t) {
        const int col = n0 + (warp * NTW + t) * 8 + 2 * t0;
        const int row0 = m0 + t1;
        if (row0 < M)
            *reinterpret_cast<float2*>(base + static_cast<size_t>(row0) * N + col) = make_float2(acc[t][0], acc[t][1]);
        if (row0 + 8 < M)
            *reinterpret_cast<float2*>(base + static_cast<size_t>(row0 + 8) * N + col) =
                make_float2(acc[t][2], acc[t][3]);
    }
}

// Sum of the split partials for one element pair, scaled and rounded to BF16
// (identical to the FP8 family's consumer arithmetic).
__device__ __forceinline__ float2 sum_pair(const float* __restrict__ partials, int splits, size_t elements,
                                           size_t pair, float alpha) {
    float2 s0 = make_float2(0.f, 0.f), s1 = make_float2(0.f, 0.f);
    int s = 0;
    for (; s + 1 < splits; s += 2) {
        const float2 p0 = *reinterpret_cast<const float2*>(partials + s * elements + 2 * pair);
        const float2 p1 = *reinterpret_cast<const float2*>(partials + (s + 1) * elements + 2 * pair);
        s0.x += p0.x; s0.y += p0.y; s1.x += p1.x; s1.y += p1.y;
    }
    if (s < splits) {
        const float2 p0 = *reinterpret_cast<const float2*>(partials + s * elements + 2 * pair);
        s0.x += p0.x; s0.y += p0.y;
    }
    return __bfloat1622float2(__floats2bfloat162_rn((s0.x + s1.x) * alpha, (s0.y + s1.y) * alpha));
}

// One 256-thread block per row of 1024; thread t owns elements [4t, 4t + 4).
template <bool PDL>
__global__ void __launch_bounds__(256)
residual_ada_norm_kernel(const float* __restrict__ partials, int splits, const float* __restrict__ a_gs,
                         const float* __restrict__ w_gs, __nv_bfloat16* __restrict__ residual,
                         const __nv_bfloat16* __restrict__ gate, const __nv_bfloat16* __restrict__ weight,
                         const __nv_bfloat16* __restrict__ style, uint8_t* __restrict__ out,
                         uint8_t* __restrict__ out_sf, const float* __restrict__ out_gs,
                         __nv_bfloat16* __restrict__ gate_out, int rows, float eps) {
    constexpr int DIM = 1024;
    __shared__ float warp_partial[8];
    const int row = blockIdx.x, tid = threadIdx.x, n0 = tid * 4;
    const size_t base = static_cast<size_t>(row) * DIM;
    const size_t elements = static_cast<size_t>(rows) * DIM;
    const __nv_bfloat16* style_row = style + static_cast<size_t>(row) * 3 * DIM;
    const uint2 wv = *reinterpret_cast<const uint2*>(weight + n0);
    const uint2 sc = *reinterpret_cast<const uint2*>(style_row + n0);
    const uint2 sh = *reinterpret_cast<const uint2*>(style_row + DIM + n0);
    const uint2 gt = *reinterpret_cast<const uint2*>(style_row + 2 * DIM + n0);
    if constexpr (PDL) pdl_prologue();
    const float alpha = (*a_gs) * (*w_gs);
    const float2 x0 = sum_pair(partials, splits, elements, base / 2 + 2 * tid, alpha);
    const float2 x1 = sum_pair(partials, splits, elements, base / 2 + 2 * tid + 1, alpha);
    const uint2 rr = *reinterpret_cast<const uint2*>(residual + base + n0);
    const uint2 gg = *reinterpret_cast<const uint2*>(gate + base + n0);
    const float2 r0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&rr.x));
    const float2 r1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&rr.y));
    const float2 g0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gg.x));
    const float2 g1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gg.y));
    const float v[4] = {r0.x + x0.x * g0.x, r0.y + x0.y * g0.y, r1.x + x1.x * g1.x, r1.y + x1.y * g1.y};
    const __nv_bfloat162 k0 = __floats2bfloat162_rn(v[0], v[1]);
    const __nv_bfloat162 k1 = __floats2bfloat162_rn(v[2], v[3]);
    uint2 kept;
    kept.x = *reinterpret_cast<const uint32_t*>(&k0);
    kept.y = *reinterpret_cast<const uint32_t*>(&k1);
    *reinterpret_cast<uint2*>(residual + base + n0) = kept;
    float sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
    sq = warp_sum(sq);
    if ((tid & 31) == 0) warp_partial[tid >> 5] = sq;
    __syncthreads();
    float total = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) total += warp_partial[w];
    const float rms = rsqrtf(total / DIM + eps);
    const float inv_gs = 1.0f / (*out_gs);
    const float2 kf0 = __bfloat1622float2(k0), kf1 = __bfloat1622float2(k1);
    const float2 w0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wv.x));
    const float2 w1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wv.y));
    const float2 s0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&sc.x));
    const float2 s1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&sc.y));
    const float2 h0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&sh.x));
    const float2 h1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&sh.y));
    quant_quad4((kf0.x * rms * w0.x * (1.0f + s0.x) + h0.x) * inv_gs,
                (kf0.y * rms * w0.y * (1.0f + s0.y) + h0.y) * inv_gs,
                (kf1.x * rms * w1.x * (1.0f + s1.x) + h1.x) * inv_gs,
                (kf1.y * rms * w1.y * (1.0f + s1.y) + h1.y) * inv_gs,
                out + base / 2 + 2 * tid, out_sf + static_cast<size_t>(row) * (DIM / 16) + (tid >> 2),
                (tid & 3) == 0);
    *reinterpret_cast<uint2*>(gate_out + base + n0) = gt;
}

// GeGLU over merged [gate | up] partials; thread owns 4 consecutive outputs.
template <bool PDL>
__global__ void __launch_bounds__(256)
gate_gelu_kernel(const float* __restrict__ partials, int splits, const float* __restrict__ a_gs,
                 const float* __restrict__ w_gs, uint8_t* __restrict__ out, uint8_t* __restrict__ out_sf,
                 int rows, int half, const float* __restrict__ out_gs) {
    if constexpr (PDL) pdl_prologue();
    const int quads = half >> 2;
    // lanes past the end compute a clamped duplicate so every quad shuffles with its full membership
    const int raw = blockIdx.x * blockDim.x + threadIdx.x;
    const bool valid = raw < rows * quads;
    const int index = valid ? raw : rows * quads - 4 + (raw & 3);
    const int row = index / quads;
    const int q = index % quads;
    const size_t elements = static_cast<size_t>(rows) * 2 * half;
    const float alpha = (*a_gs) * (*w_gs);
    const size_t row_pairs = static_cast<size_t>(row) * half;
    const float2 ga = sum_pair(partials, splits, elements, row_pairs + 2 * q, alpha);
    const float2 gb = sum_pair(partials, splits, elements, row_pairs + 2 * q + 1, alpha);
    const float2 ua = sum_pair(partials, splits, elements, row_pairs + (half >> 1) + 2 * q, alpha);
    const float2 ub = sum_pair(partials, splits, elements, row_pairs + (half >> 1) + 2 * q + 1, alpha);
    const float inv_gs = 1.0f / (*out_gs);
    uint16_t scratch = 0;
    quant_quad4(tanh_gelu(ga.x) * ua.x * inv_gs, tanh_gelu(ga.y) * ua.y * inv_gs,
                tanh_gelu(gb.x) * ub.x * inv_gs, tanh_gelu(gb.y) * ub.y * inv_gs,
                valid ? out + (static_cast<size_t>(row) * half + 4 * q) / 2 : reinterpret_cast<uint8_t*>(&scratch),
                out_sf + static_cast<size_t>(row) * (half / 16) + (q >> 2), valid && (q & 3) == 0);
}

template <bool PDL>
__global__ void __launch_bounds__(256)
action_in_norm_kernel(const __nv_bfloat16* __restrict__ noise, const __nv_bfloat16* __restrict__ w_in,
                      const __nv_bfloat16* __restrict__ b_in, __nv_bfloat16* __restrict__ x,
                      const __nv_bfloat16* __restrict__ weight, const __nv_bfloat16* __restrict__ style,
                      uint8_t* __restrict__ out, uint8_t* __restrict__ out_sf,
                      __nv_bfloat16* __restrict__ gate_out, const float* __restrict__ out_gs, float eps) {
    constexpr int DIM = 1024, KDIM = 32;
    __shared__ float warp_partial[8];
    const int row = blockIdx.x, tid = threadIdx.x;
    const int n0 = tid * 4;
    const __nv_bfloat16* style_row = style + static_cast<size_t>(row) * 3 * DIM;
    float2 wq[KDIM][2];
#pragma unroll
    for (int k = 0; k < KDIM; ++k) {
        const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(w_in + k * DIM + n0);
        wq[k][0] = __bfloat1622float2(wp[0]);
        wq[k][1] = __bfloat1622float2(wp[1]);
    }
    const float2 bv0 = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(b_in + n0)[0]);
    const float2 bv1 = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(b_in + n0)[1]);
    const float2 wv0 = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(weight + n0)[0]);
    const float2 wv1 = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(weight + n0)[1]);
    const float2 s0 = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(style_row + n0)[0]);
    const float2 s1 = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(style_row + n0)[1]);
    const float2 h0 = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(style_row + DIM + n0)[0]);
    const float2 h1 = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(style_row + DIM + n0)[1]);
    const uint2 gt = *reinterpret_cast<const uint2*>(style_row + 2 * DIM + n0);
    if constexpr (PDL) pdl_prologue();
    float nv[KDIM];
#pragma unroll
    for (int k = 0; k < KDIM; k += 2) {
        const float2 v = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(noise + static_cast<size_t>(row) * KDIM + k));
        nv[k] = v.x; nv[k + 1] = v.y;
    }
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int k = 0; k < KDIM; ++k) {
        acc[0] += nv[k] * wq[k][0].x; acc[1] += nv[k] * wq[k][0].y;
        acc[2] += nv[k] * wq[k][1].x; acc[3] += nv[k] * wq[k][1].y;
    }
    const float2 g0 = __bfloat1622float2(__floats2bfloat162_rn(acc[0], acc[1]));
    const float2 g1 = __bfloat1622float2(__floats2bfloat162_rn(acc[2], acc[3]));
    const __nv_bfloat162 x0 = __floats2bfloat162_rn(g0.x + bv0.x, g0.y + bv0.y);
    const __nv_bfloat162 x1 = __floats2bfloat162_rn(g1.x + bv1.x, g1.y + bv1.y);
    __nv_bfloat162* xrow = reinterpret_cast<__nv_bfloat162*>(x + static_cast<size_t>(row) * DIM + n0);
    xrow[0] = x0; xrow[1] = x1;
    const float2 xf0 = __bfloat1622float2(x0), xf1 = __bfloat1622float2(x1);
    float sq = xf0.x * xf0.x + xf0.y * xf0.y + xf1.x * xf1.x + xf1.y * xf1.y;
    sq = warp_sum(sq);
    if ((tid & 31) == 0) warp_partial[tid >> 5] = sq;
    __syncthreads();
    float total = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) total += warp_partial[w];
    const float rms = rsqrtf(total / DIM + eps);
    const float inv_gs = 1.0f / (*out_gs);
    quant_quad4((xf0.x * rms * wv0.x * (1.0f + s0.x) + h0.x) * inv_gs,
                (xf0.y * rms * wv0.y * (1.0f + s0.y) + h0.y) * inv_gs,
                (xf1.x * rms * wv1.x * (1.0f + s1.x) + h1.x) * inv_gs,
                (xf1.y * rms * wv1.y * (1.0f + s1.y) + h1.y) * inv_gs,
                out + static_cast<size_t>(row) * (DIM / 2) + 2 * tid,
                out_sf + static_cast<size_t>(row) * (DIM / 16) + (tid >> 2), (tid & 3) == 0);
    *reinterpret_cast<uint2*>(gate_out + static_cast<size_t>(row) * DIM + n0) = gt;
}

// One thread per (8-column tile, 128-wide K chunk, lane): gathers the lane's
// 16 bytes; lanes 0..7 also gather the column scales.
__global__ void pack_weight_kernel(const uint8_t* __restrict__ w, const uint8_t* __restrict__ sf,
                                   uint8_t* __restrict__ wp, uint8_t* __restrict__ sfp, int N, int K) {
    const int kchunks = K / 128;
    const size_t total = static_cast<size_t>(N / 8) * kchunks * 32;
    const size_t g = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= total) return;
    const int lane = static_cast<int>(g & 31);
    const size_t tile = g >> 5;
    const int kc = static_cast<int>(tile % kchunks);
    const int nt = static_cast<int>(tile / kchunks);
    const int t0 = lane & 3, t1 = lane >> 2;
    const int col = nt * 8 + t1;
    const uint8_t* src = w + static_cast<size_t>(col) * (K / 2) + kc * 64;
    uint32_t piece[4];
#pragma unroll
    for (int s = 0; s < 2; ++s)
#pragma unroll
        for (int r = 0; r < 2; ++r)
            piece[2 * s + r] = *reinterpret_cast<const uint32_t*>(src + s * 32 + r * 16 + t0 * 4);
    *reinterpret_cast<uint4*>(wp + g * 16) = make_uint4(piece[0], piece[1], piece[2], piece[3]);
    if (lane < 8) {
        const uint8_t* ssrc = sf + static_cast<size_t>(nt * 8 + lane) * (K / 16) + kc * 8;
        *reinterpret_cast<uint2*>(sfp + (tile * 8 + lane) * 8) = *reinterpret_cast<const uint2*>(ssrc);
    }
}

// One thread per 16-element block of a BF16 [N, K] weight.
__global__ void quantize_weight_kernel(const __nv_bfloat16* __restrict__ w, const float* __restrict__ gs,
                                       uint8_t* __restrict__ out, uint8_t* __restrict__ out_sf, int N, int K) {
    const size_t blocks = static_cast<size_t>(N) * (K / 16);
    const size_t b = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (b >= blocks) return;
    const float inv_gs = 1.0f / (*gs);
    const __nv_bfloat16* src = w + b * 16;
    float v[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) v[i] = __bfloat162float(src[i]) * inv_gs;
    quant_block16(v, out + b * 8, out_sf + b);
}

template <typename... Args>
int launch_ex(const void* fn, dim3 grid, dim3 block, cudaStream_t stream, bool pdl, Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = pdl ? 1 : 0;
    void* pargs[] = {(void*)&args...};
    return static_cast<int>(cudaLaunchKernelExC(&cfg, fn, pargs));
}

struct Cfg { int bn, kc, nw; };
constexpr Cfg kCfgs[] = {
    {64, 256, 4},  {64, 512, 4},  {128, 256, 4}, {128, 512, 4},
    {64, 1024, 4}, {128, 1024, 4}, {128, 256, 8}, {256, 256, 8},
    {64, 128, 4},  {32, 512, 4},  {32, 1024, 4}, {256, 512, 8},
};
constexpr int kNumCfgs = sizeof(kCfgs) / sizeof(kCfgs[0]);

template <int BN, int KC, int NW, bool ABF16>
int launch_gemm(const void* A, const void* SFA, const float* a_gs, const void* W, const void* SFB,
                float* partials, int M, int N, int K, bool pdl, cudaStream_t stream) {
    if (M < 1 || N % BN != 0 || K % KC != 0) return static_cast<int>(cudaErrorInvalidValue);
    const dim3 grid(N / BN, K / KC, (M + 15) / 16);
    const uint8_t* sfa = static_cast<const uint8_t*>(SFA);
    const uint8_t* w = static_cast<const uint8_t*>(W);
    const uint8_t* sfb = static_cast<const uint8_t*>(SFB);
    if (pdl)
        return launch_ex((const void*)gemm_kernel<BN, KC, NW, ABF16, true>, grid, dim3(NW * 32), stream, true, A,
                         sfa, a_gs, w, sfb, partials, M, N, K);
    return launch_ex((const void*)gemm_kernel<BN, KC, NW, ABF16, false>, grid, dim3(NW * 32), stream, false, A,
                     sfa, a_gs, w, sfb, partials, M, N, K);
}

template <bool ABF16>
int dispatch_gemm(const void* A, const void* SFA, const float* a_gs, const void* W, const void* SFB,
                  float* partials, int M, int N, int K, int cfg, bool pdl, cudaStream_t stream) {
#define CASE(I, BN, KC, NW) \
    case I: return launch_gemm<BN, KC, NW, ABF16>(A, SFA, a_gs, W, SFB, partials, M, N, K, pdl, stream)
    switch (cfg) {
        CASE(0, 64, 256, 4);
        CASE(1, 64, 512, 4);
        CASE(2, 128, 256, 4);
        CASE(3, 128, 512, 4);
        CASE(4, 64, 1024, 4);
        CASE(5, 128, 1024, 4);
        CASE(6, 128, 256, 8);
        CASE(7, 256, 256, 8);
        CASE(8, 64, 128, 4);
        CASE(9, 32, 512, 4);
        CASE(10, 32, 1024, 4);
        CASE(11, 256, 512, 8);
        default: return static_cast<int>(cudaErrorInvalidValue);
    }
#undef CASE
}

}  // namespace

int config_count() { return kNumCfgs; }

int config_k_chunk(int cfg) { return (cfg >= 0 && cfg < kNumCfgs) ? kCfgs[cfg].kc : 0; }

bool config_supports(int cfg, int N, int K) {
    return cfg >= 0 && cfg < kNumCfgs && N % kCfgs[cfg].bn == 0 && K % kCfgs[cfg].kc == 0;
}

int gemm(const void* A, const void* SFA, const void* W, const void* SFB, float* partials, int M, int N, int K,
         int cfg, bool pdl, cudaStream_t stream) {
    return dispatch_gemm<false>(A, SFA, nullptr, W, SFB, partials, M, N, K, cfg, pdl, stream);
}

int gemm_bf16_act(const __nv_bfloat16* A, const float* a_gs, const void* W, const void* SFB, float* partials,
                  int M, int N, int K, int cfg, bool pdl, cudaStream_t stream) {
    return dispatch_gemm<true>(A, nullptr, a_gs, W, SFB, partials, M, N, K, cfg, pdl, stream);
}

int residual_ada_norm(const float* partials, int splits, const float* a_gs, const float* w_gs,
                      __nv_bfloat16* residual, const __nv_bfloat16* gate, const __nv_bfloat16* weight,
                      const __nv_bfloat16* style, uint8_t* out, uint8_t* out_sf, const float* out_gs,
                      __nv_bfloat16* gate_out, int rows, int dim, float eps, bool pdl, cudaStream_t stream) {
    if (dim != 1024 || rows < 1) return static_cast<int>(cudaErrorInvalidValue);
    const void* fn = pdl ? (const void*)residual_ada_norm_kernel<true> : (const void*)residual_ada_norm_kernel<false>;
    return launch_ex(fn, dim3(rows), dim3(256), stream, pdl, partials, splits, a_gs, w_gs, residual, gate, weight,
                     style, out, out_sf, out_gs, gate_out, rows, eps);
}

int gate_gelu(const float* partials, int splits, const float* a_gs, const float* w_gs, uint8_t* out,
              uint8_t* out_sf, int rows, int half, const float* out_gs, bool pdl, cudaStream_t stream) {
    if (rows < 1 || half % 16 != 0) return static_cast<int>(cudaErrorInvalidValue);
    const int total = rows * (half >> 2);
    const dim3 grid((total + 255) / 256);
    const void* fn = pdl ? (const void*)gate_gelu_kernel<true> : (const void*)gate_gelu_kernel<false>;
    return launch_ex(fn, grid, dim3(256), stream, pdl, partials, splits, a_gs, w_gs, out, out_sf, rows, half, out_gs);
}

int action_in_norm(const __nv_bfloat16* noise, const __nv_bfloat16* w_in, const __nv_bfloat16* b_in,
                   __nv_bfloat16* x, const __nv_bfloat16* weight, const __nv_bfloat16* style, uint8_t* out,
                   uint8_t* out_sf, __nv_bfloat16* gate_out, const float* out_gs, int rows, float eps, bool pdl,
                   cudaStream_t stream) {
    if (rows < 1) return static_cast<int>(cudaErrorInvalidValue);
    const void* fn = pdl ? (const void*)action_in_norm_kernel<true> : (const void*)action_in_norm_kernel<false>;
    return launch_ex(fn, dim3(rows), dim3(256), stream, pdl, noise, w_in, b_in, x, weight, style, out, out_sf,
                     gate_out, out_gs, eps);
}

int pack_weight(const uint8_t* w, const uint8_t* sf, uint8_t* w_packed, uint8_t* sf_packed, int N, int K,
                cudaStream_t stream) {
    if (N % 8 != 0 || K % 128 != 0 || N < 8) return static_cast<int>(cudaErrorInvalidValue);
    const size_t total = static_cast<size_t>(N / 8) * (K / 128) * 32;
    pack_weight_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream>>>(w, sf, w_packed, sf_packed, N, K);
    return static_cast<int>(cudaGetLastError());
}

int quantize_weight(const __nv_bfloat16* w, const float* gs, uint8_t* out, uint8_t* out_sf, int N, int K,
                    cudaStream_t stream) {
    if (N < 1 || K % 16 != 0) return static_cast<int>(cudaErrorInvalidValue);
    const size_t blocks = static_cast<size_t>(N) * (K / 16);
    quantize_weight_kernel<<<static_cast<unsigned>((blocks + 255) / 256), 256, 0, stream>>>(w, gs, out, out_sf, N, K);
    return static_cast<int>(cudaGetLastError());
}

}  // namespace pi05_dec_skinny_fp4
}  // namespace flash_rt
