// SPDX-License-Identifier: Apache-2.0
#pragma once
// Pi0.5 skinny NVFP4 (W4A4) decoder family on sm_120a.
//
// The NVFP4 twin of the skinny FP8 family (pi05_decoder_skinny_fp8_sm120.cuh):
// the same K-split structure (each CTA streams BN weight rows x KC of K into
// mma fragments before the programmatic-dependent-launch wait, FP32 partials
// per K split) with block-scaled e2m1 operands
// (mma.sync kind::mxf4nvf4 m16n8k64, UE4M3 scale per 16 elements). The
// activations are never quantized by a standalone kernel: the consumers that
// already sum the partials (gated residual + adaptive RMS norm, GeGLU, the
// action input projection) emit e2m1 + block scales directly, and the
// attention-output GEMM quantizes its BF16 operand on load.
//
// Numerics: x_q = e2m1(x / (gs * sf)), sf = ue4m3_ceil(amax16(x / gs) / 6),
// with gs a per-tensor global scale (device pointer). The GEMM partials hold
// sum(e2m1 * e2m1 * sf_a * sf_b); consumers apply alpha = (*a_gs) * (*w_gs),
// exactly as the FP8 consumers apply (*a_scale) * (*w_scale), so the FP8
// family's sum_rope / attention / final-norm consumers are reused unchanged.
//
// Layouts:
//   activation  e2m1 [M, K/2] row-major, scales ue4m3 [M, K/16] row-major.
//   weight      fragment-major: for every (8-column tile nt, 128-wide K chunk
//               kc) 512 bytes = 32 lanes x 16 bytes, lane (t0 = lane & 3,
//               t1 = lane >> 2) holding column nt*8+t1, K elements
//               kc*128 + s*64 + r*32 + 8*t0 + [0, 8) at byte 8*s + 4*r
//               (s, r in {0, 1}); scales: for every (nt, kc) 8 columns x 8
//               bytes (4 block scales of each 64-wide half). Produced by
//               pack_weight (setup time) from a natural [N, K] quantization.
//   partials    fp32 [K / KC][M][N].
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace pi05_dec_skinny_fp4 {

// Tile configurations (BN x KC x warps).
int config_count();
int config_k_chunk(int cfg);
bool config_supports(int cfg, int N, int K);

// partials[s] = A(M, Ks) @ W(N, Ks)^T for NVFP4 A (+ SFA) and packed W (+ SFB).
int gemm(const void* A, const void* SFA, const void* W, const void* SFB, float* partials, int M, int N,
         int K, int cfg, bool pdl, cudaStream_t stream);

// Same with a BF16 activation quantized to NVFP4 on load (global scale *a_gs).
int gemm_bf16_act(const __nv_bfloat16* A, const float* a_gs, const void* W, const void* SFB, float* partials,
                  int M, int N, int K, int cfg, bool pdl, cudaStream_t stream);

// Gated residual + adaptive RMS norm (as the FP8 family's residual_ada_norm)
// emitting the normed row as NVFP4 with global scale *out_gs. dim == 1024.
int residual_ada_norm(const float* partials, int splits, const float* a_gs, const float* w_gs,
                      __nv_bfloat16* residual, const __nv_bfloat16* gate, const __nv_bfloat16* weight,
                      const __nv_bfloat16* style, uint8_t* out, uint8_t* out_sf, const float* out_gs,
                      __nv_bfloat16* gate_out, int rows, int dim, float eps, bool pdl, cudaStream_t stream);

// GeGLU over merged [gate | up] partials straight to NVFP4 (global scale *out_gs).
int gate_gelu(const float* partials, int splits, const float* a_gs, const float* w_gs, uint8_t* out,
              uint8_t* out_sf, int rows, int half, const float* out_gs, bool pdl, cudaStream_t stream);

// Action input projection (K = 32, N = 1024) + bias into the residual stream,
// then the first layer's adaptive RMS norm straight to NVFP4.
int action_in_norm(const __nv_bfloat16* noise, const __nv_bfloat16* w_in, const __nv_bfloat16* b_in,
                   __nv_bfloat16* x, const __nv_bfloat16* weight, const __nv_bfloat16* style, uint8_t* out,
                   uint8_t* out_sf, __nv_bfloat16* gate_out, const float* out_gs, int rows, float eps, bool pdl,
                   cudaStream_t stream);

// Setup-time weight packing: natural NVFP4 [N, K/2] e2m1 + [N, K/16] ue4m3
// into the fragment-major layout above. N % 8 == 0, K % 128 == 0.
int pack_weight(const uint8_t* w, const uint8_t* sf, uint8_t* w_packed, uint8_t* sf_packed, int N, int K,
                cudaStream_t stream);

// Setup-time weight quantization: BF16 [N, K] -> natural NVFP4 with a
// per-tensor global scale *gs (scales ue4m3_ceil(amax16 / (6 gs))).
int quantize_weight(const __nv_bfloat16* w, const float* gs, uint8_t* out, uint8_t* out_sf, int N, int K,
                    cudaStream_t stream);

}  // namespace pi05_dec_skinny_fp4
}  // namespace flash_rt
