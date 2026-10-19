// SPDX-License-Identifier: Apache-2.0
//
// Alpamayo 1.5 (Jetson Thor) request-level kernels: the operations of one full request that are neither GEMMs nor
// attention -- token / position gathers and scatters, the M-RoPE tables of the prompt, the sampler with its decode
// bookkeeping, the action expert's input encoding and output projection with the Euler update, and the unicycle
// trajectory integration -- so that every GPU operation of a request after host tokenization is a FlashRT kernel.
// Every entry returns 0 on success, 1 for a null pointer, 2 for an unsupported shape, 5 for misaligned operands and
// 4 for a launch failure. Pointers are device pointers; the caller's stream is used as given.
#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

// out[r, :] = table[idx[r], :] for r < rows: rows of row_bytes bytes (a multiple of 16, 16-byte aligned, leading
// dimensions in bytes), or bf16 rows of dim elements (dim % 8 == 0, leading dimensions in elements).
// base (optional, device) is added to every index.
int alpamayo15_gather_rows_bytes(const void* table, long long ld_table, const long long* idx, const long long* base, void* out,
                                 long long ld_out, int rows, long long row_bytes, cudaStream_t stream);
int alpamayo15_gather_rows_bf16(const void* table, int ld_table, const long long* idx, void* out, int ld_out, int rows,
                                int dim, cudaStream_t stream);

// out[i] = bf16(x[i]) for n fp32 values (n % 4 == 0; the pixel values of a request as the tower's BF16 input).
int alpamayo15_cast_f32_bf16(const float* x, void* out, long long n, cudaStream_t stream);
// out[i] = bf16(x[i]) for n fp16 values (n % 4 == 0; a W4A8 GEMM's fp16 output as the BF16 residual delta).
int alpamayo15_cast_f16_bf16(const void* x, void* out, long long n, cudaStream_t stream);

// DFlash round bookkeeping (FlashDrive's block_step_fn, int64 tokens). propose: block[0] = first[0], block[i] =
// proposed[i - 1], except that a proposal following a proposed stop token becomes traj_id (B block tokens, B - 1
// proposals). decide: stats = [acceptance length of block[1:] against posterior[:-1], position of the first stop
// token among the emitted tokens (block up to the acceptance, then the bonus posterior token) or -1, whether the
// bonus token is period_id].
int alpamayo15_dflash_propose(const long long* proposed, const long long* first, long long stop_id, long long traj_id,
                              long long* block, int B, cudaStream_t stream);
int alpamayo15_dflash_decide(const long long* block, const long long* posterior, long long stop_id, long long period_id, int B,
                             long long* stats, cudaStream_t stream);

// dst[idx[r], :] = src[r, :] (add = 0) or bf16(float(dst[idx[r], :]) + float(src[r, :])) (add = 1); dim % 8 == 0.
int alpamayo15_scatter_rows_bf16(void* dst, int ld_dst, const long long* idx, const void* src, int ld_src, int rows, int dim,
                                 int add, cudaStream_t stream);

// Qwen3-VL text rotary tables of P positions: pos3 [3, P] int32 (temporal, height, width position of each row),
// inv_freq [D/2] fp32; frequency j takes the height position for j in [1, 3 * sec_h) step 3, the width position for
// j in [2, 3 * sec_w) step 3 and the temporal one otherwise (the interleaved M-RoPE); cos / sin [P, D] =
// cosf(inv_freq[j] * pos) * scaling as fp32 (out_f32 = 1) or bf16, with the two halves of a row equal. D % 16 == 0.
int alpamayo15_mrope_tables(const int* pos3, const float* inv_freq, int sec_h, int sec_w, float scaling, void* cos, void* sin,
                            int P, int D, int out_f32, cudaStream_t stream);

// Token sampling of n rows of bf16 logits [n, V] (rows ld apart; ld = 0 reads one row for every output row) with
// the official decoding rule: the trajectory token span [mask_lo, mask_lo + mask_n) masked to -inf, scores divided by
// temperature, top-p filtering, softmax, one multinomial draw per row. top_p_rule 0 = transformers' TopPLogitsWarper
// (tokens whose ascending cumulative probability is <= 1 - top_p are dropped, the largest token always kept); 1 =
// FlashDrive's rule (tokens whose descending cumulative probability before them exceeds top_p are dropped).
// temperature <= 0 selects the greedy token (lowest index among ties) and draws nothing.
// The draw reproduces torch.multinomial(probs, 1) on the CUDA generator at rng = [seed, Philox offset, ticket]:
// torch draws q ~ Exp(1) per element of the [n, V] probabilities (its distribution kernel's Philox word per element,
// whose thread count is min(rng_blocks, ceil(n * V / 256)) * 256 with rng_blocks = the device's SM count times its
// blocks of 256 threads per SM) and takes argmax(probs / q); the kernel computes the same words and ratios for the
// kept tokens, so with the generator's seed and offset it returns torch's token (up to float rounding of the
// probabilities), and advances rng[1] by torch's increment (4 * ceil(n * V / (4 * threads))) afterwards; rng[2] is
// its block ticket, 0 between launches. scratch [n, V] fp32 holds each row's exp(score - max) (not needed for greedy);
// workspace (16-byte aligned, alpamayo15_sample_workspace_bytes(n) bytes, zero-initialized once, left zeroed by every
// launch) is the exchange area of the row's cooperating blocks: up to 8 co-resident blocks per row share a row.
// Writes tok[r] (int64); with unfinished != nullptr, tok[r] = pad_id where unfinished[r] == 0. The optional counters
// are the bookkeeping of one reasoning step, advanced after the draws: slot (int64, one), rpos / len / glen (one per
// row: int64, int32, int32) by one.
int alpamayo15_sample_workspace_bytes(int n);
int alpamayo15_sample_tokens_bf16(const void* logits, int ld, int n, int V, int mask_lo, int mask_n, float temperature,
                                  float top_p, int top_p_rule, unsigned long long* rng, int rng_blocks, float* scratch,
                                  void* workspace, long long workspace_bytes, long long* tok, const long long* unfinished,
                                  long long pad_id, long long* slot, long long* rpos, int* len, int* glen, cudaStream_t stream);

// Action expert input features (PerWaypointActionInProjV2 before its MLP): per row the Fourier features
// [sin(arg), cos(arg)] * sqrt(2) with arg = x * freq * 2 * pi of every action dimension (x [rows, act_dim]) and of
// the row's timestep (t [rows / rows_per_sample]); x and t fp32 (x_bf16 = 0: the official fp32 state) or bf16
// (x_bf16 = 1: a BF16 state and time grid, every elementwise step rounding to bf16); freqs [nf / 2] bf16 as the
// module keeps them. Written as bf16 into out [rows, ld_out] (the MLP's bf16 input under autocast), columns beyond
// act_dim * nf + nf zero.
int alpamayo15_action_fourier_bf16(const void* x, int x_bf16, const void* t, const void* freqs, int rows, int rows_per_sample,
                                   int act_dim, int nf, void* out, int ld_out, cudaStream_t stream);

// FlashDrive's BF16 Euler step over n values: x = bf16(x + bf16(dt * v)).
int alpamayo15_euler_bf16(void* x, const void* v, float dt, int n, cudaStream_t stream);

// LayerNorm of bf16 rows with FP32 statistics and output (torch.nn.LayerNorm under autocast): out = (x - mean) /
// sqrt(var + eps) * w + b in fp32. dim % 4 == 0.
int alpamayo15_layernorm_f32out_bf16(const void* x, const void* w, const void* b, float* out, int rows, int dim, float eps,
                                     cudaStream_t stream);

// Action output projection and Euler step: v[r, j] = bf16(sum_k hn[r, k] * w[j, k] + b[j]) (the expert's final
// BF16 linear), stored to v_out [rows, act_dim] bf16 when given; x[r, j] += dt * float(v[r, j]) in fp32 when x
// [rows, act_dim] fp32 is given. hn [rows, E] bf16, w [act_dim, E] bf16, b [act_dim] bf16; E % 8 == 0.
int alpamayo15_action_out_euler(const void* hn, const void* w, const void* b, float dt, float* x, void* v_out, int rows, int E,
                                int act_dim, cudaStream_t stream);

// Unicycle acceleration / curvature action space, action -> trajectory of n samples: action [n, T, 2] fp32
// (normalized accel, kappa), v0 [n] fp32 (the t0 velocity estimate), z [n] fp32 (the history's last z); writes
// xyz [n, T, 3] and rot [n, T, 3, 3] fp32, the official integration (trapezoidal velocity / heading cumsums).
int alpamayo15_action_to_traj(const float* action, const float* v0, const float* z, int n, int T, float dt, float accel_mean,
                              float accel_std, float kappa_mean, float kappa_std, float* xyz, float* rot, cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
