"""Alpamayo 1.5 inference on Jetson AGX Thor (SM110), BF16.

The engine wraps one official ``alpamayo1_5.models.alpamayo1_5.Alpamayo1_5``
instance (its weights are shared, never copied) and executes the request path
of the official ``sample_trajectories_from_data_with_vlm_rollout`` with
static buffers and CUDA graphs:

1. vision + prefill   batched vision tower and a 36-layer prefill on fused
                      kernels that write K/V straight into a static cache. The
                      prompt is encoded once and its K/V shared by every
                      trajectory sample (the official path repeats the whole
                      prompt, images included, per sample).
2. reasoning decode   one CUDA graph replayed per token for all samples: the
                      decode step over the static cache plus the Hugging Face
                      sampling chain (trajectory-token mask -> temperature ->
                      top-p -> softmax -> multinomial, finished rows padded).
                      The host only reads the sampled tokens to evaluate the
                      official stop criteria.
3. diffusion expert   every Euler step of the flow-matching expert in one CUDA
                      graph. Expert attention reads each sample's cache with a
                      device-side length, so one graph serves every reasoning
                      length.
4. trajectory         official unicycle integration.

Static cache layout per layer and sample: ``[prompt | generated | action]``.
During diffusion each sample's 64 action-token K/V rows are written right after
its visible prefix (later rows are masked for that sample anyway), so for one
sample expert attention is a single non-causal pass over a contiguous prefix.
With several samples the prompt K/V lives once (sample 0) and attention is a
shared-prompt pass for all samples plus a per-sample suffix pass, merged by
their log-sum-exp.

Numerics follow the official module graph under ``torch.autocast(bfloat16)``:
the VLM runs in BF16; the expert keeps an FP32 residual stream, applies RoPE in
FP32 and feeds BF16 operands to attention (autocast casts SDPA inputs).

Every model computation of a request runs on FlashRT kernels launched from C++: the GEMMs (tcgen05 CUTLASS
instantiations with bias, GELU and residual fused into the epilogues, or cuBLASLt where faster at the shape), FA4
attention, and the model-local kernels of this package (normalizations, rotary tables and rotations, the sampler,
the action projections, the Euler steps, the trajectory integration). torch operations remain outside that
computation: copies of each request's inputs and host-computed indices into the static buffers, the per-request
device-state resets of the decode and expert graphs, the official noise draw of the diffusion expert, and (several
trajectory samples only) the log-sum-exp merge of the shared-prompt and per-sample attention passes. Each frame
grid's vision tower, each decode step and the whole expert are CUDA graphs.

Layering: ``runtime_thor.Alpamayo15ThorRuntime`` validates the options, prepares the weights, owns every persistent
buffer and captures the CUDA graphs; this module holds the computation that runs on them (``Alpamayo15Thor``) and
the request orchestration. ``flash_rt.frontends.torch.alpamayo15_thor`` builds both.

Precision tiers (opt-in, numerics changes outside that contract):
``decode_weights="fp8"`` / ``"int8"`` read per-row FP8 E4M3 or symmetric INT8
copies of the reasoning-decode projections (q/k/v, o, gate/up, down and the LM
head) with BF16 activations and FP32 accumulation. Decode is weight-streaming
bound, so this halves its bytes; INT8 carries about 4x less weight error than
E4M3 for the same bytes. The expert still attends over the reasoning-token K/V
written by those projections, so trajectories can move even when the sampled
tokens are identical. ``decode_weights="w4a8"`` instead runs the decode
projections as ParoQuant W4A8 linears from the ``z-lab/Alpamayo-1.5-10B-PARO``
checkpoint (``paro_checkpoint``): learned pairwise rotations of the activations,
per-token INT8 activations, UINT4 weights with zero points and integer group
scales (Marlin's arithmetic on FlashRT's few-row kernels), the LM head as INT8;
about a quarter of the BF16 weight stream. ``expert_weights="fp8"`` runs the diffusion expert's
projections as W8A8 (per-row FP8 activations and weights, scales applied in
the GEMM epilogue), halving its weight stream; ``prefill_weights="fp8"`` does
the same for the 36 prefill layers on 2-SM tcgen05 tiles.
"""
from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from flash_rt.models.alpamayo15.runtime_thor import Alpamayo15ThorRuntime

BF16 = torch.bfloat16
DEV = "cuda"
logger = logging.getLogger(__name__)

VISION_TILE = 3                              # tcgen05 tile of the vision GEMMs: 256 x 256 on a 2-SM UMMA (2 = 256 x 128)
NVFP4_TILE = 2              # NVFP4 prefill GEMMs: 256 x 256 x 256 tiles on a 2-SM UMMA (500-575 TOPS on Thor)
PREFILL_FP8_TILE = 3        # FP8 prefill GEMMs: 256 x 256 x 128 tiles on a 2-SM UMMA (360-375 TOPS on Thor; 256 x 128: 250)
# BF16 prefill projections (preset bf16): q|k|v and o on the 2-SM 256 x 256 tcgen05 tile, gate / up / down on cuBLASLt
# (called from C++), whichever is faster at the prompt's shapes on Thor.
BF16_PREFILL_CUBLASLT = "gu+down"   # projections ("+"-separated) on cuBLASLt; the others on the 256 x 256 tile
PREFILL_DOWN_SPLIT = 4      # the K=12288 down projection as 4 column blocks (weight blocks that fit the L2: 1.47 vs 3.9 ms)


def update_stop_state(last: list[int], traj_found: list[bool], unfinished: list[bool], n_generated: int, *,
                      traj_start_id: int, max_new_tokens: int, eos_ids: set[int]) -> None:
    """Official stop criteria after appending ``last`` (one token per sample).

    Mirrors ``generate``: the token budget and the generation-config EOS finish
    individual rows, and ``StopAfterEOS`` finishes every row one token after all
    rows have produced ``<|traj_future_start|>`` (so its K/V reaches the cache).
    ``traj_found`` and ``unfinished`` are updated in place.
    """
    stop_all = all(traj_found)
    for b, t in enumerate(last):
        traj_found[b] = traj_found[b] or t == traj_start_id
        if stop_all or n_generated >= max_new_tokens or t in eos_ids:
            unfinished[b] = False


def expert_attention_spans(tokens: list[list[int]], prompt_len: int, traj_start_id: int) -> tuple[list[int], list[int]]:
    """Per-sample ``(offset, visible)`` for the expert, as the official rollout derives them.

    ``offset`` is one past the first ``<|traj_future_start|>`` (``_find_eos_offset``; the
    sequence length when absent) and anchors the action-token positions. The expert
    sees cache rows ``[0, min(offset, cache_len))`` plus its own action tokens, where
    ``cache_len = prompt_len + n_generated - 1`` (the last sampled token is never
    forwarded); ``_build_expert_pos_ids_and_attn_mask`` masks the rest.
    """
    cache_len = prompt_len + len(tokens[0]) - 1
    offsets = []
    for row in tokens:
        if traj_start_id in row:
            offsets.append(prompt_len + row.index(traj_start_id) + 1)
        else:
            logger.warning("No <|traj_future_start|> in %d generated tokens", len(row))
            offsets.append(prompt_len + len(row))
    return offsets, [min(o, cache_len) for o in offsets]


def merge_attention(o1: torch.Tensor, lse1: torch.Tensor, o2: torch.Tensor, lse2: torch.Tensor) -> torch.Tensor:
    """Combine two attention passes over disjoint key sets by their log-sum-exp.

    ``o*`` are [rows, heads, D], ``lse*`` [heads, rows] (natural log). Partial
    outputs arrive in BF16, so this adds one rounding relative to a single pass.
    """
    m = torch.maximum(lse1, lse2)
    w1, w2 = torch.exp(lse1 - m), torch.exp(lse2 - m)
    den = w1 + w2
    w1, w2 = (w1 / den).t().unsqueeze(-1), (w2 / den).t().unsqueeze(-1)
    return (o1.float() * w1 + o2.float() * w2).to(BF16)


def _autocast():
    return torch.autocast("cuda", dtype=BF16)


def torch_philox_increment(numel: int, blocks: int) -> int:
    """The Philox offset torch's CUDA generator advances by when it draws ``numel`` values (``exponential_`` of
    ``torch.multinomial``): four words per thread per round of its distribution kernel."""
    threads = min(blocks, (numel + 255) // 256) * 256
    return 4 * ((numel - 1) // (4 * threads) + 1)


def torch_rng_state(gen) -> torch.Tensor:
    """``[seed, Philox offset, 0]`` (int64, CPU) of a CUDA generator: the sampler kernel's generator state."""
    seed = gen.initial_seed() & 0xFFFFFFFFFFFFFFFF
    if seed >= 2 ** 63:
        seed -= 2 ** 64
    return torch.tensor([seed, int(gen.get_offset()), 0], dtype=torch.int64)


def mrope_positions(ids: list[int], grids: list[tuple[int, int, int]], *, image_token_id: int, vision_start_id: int,
                    spatial_merge: int) -> tuple[np.ndarray, int]:
    """Qwen3-VL ``get_rope_index`` for one unpadded prompt: the [3, P] (temporal, height, width) positions of its
    tokens and the rope delta, computed on the host from the token ids and the image grids."""
    P = len(ids)
    pos = np.zeros((3, P), dtype=np.int64)
    st, next_idx, gi = 0, 0, 0
    n_images = sum(1 for i in range(P - 1) if ids[i] == vision_start_id and ids[i + 1] == image_token_id)
    for _ in range(n_images):
        ed = ids.index(image_token_id, st)
        t, h, w = grids[gi]; gi += 1
        h, w = h // spatial_merge, w // spatial_merge
        text_len = ed - st
        pos[:, st:ed] = np.arange(text_len) + next_idx
        next_idx = next_idx + text_len
        n = t * h * w
        pos[0, ed:ed + n] = np.repeat(np.arange(t), h * w) + next_idx
        pos[1, ed:ed + n] = np.tile(np.repeat(np.arange(h), w), t) + next_idx
        pos[2, ed:ed + n] = np.tile(np.arange(w), t * h) + next_idx
        st = ed + n
        next_idx = int(pos[:, :st].max()) + 1
    if st < P:
        pos[:, st:] = np.arange(P - st) + next_idx
    return pos.astype(np.int32), int(pos.max()) + 1 - P


class _Gemm:
    """BF16 tcgen05 GEMM (``alpamayo15_bf16_gemm_sm100``): ``D = act(A @ W.T + bias) (+ C)`` with a workspace per shape."""

    def __init__(self, fvk) -> None:
        self._fvk = fvk
        self._ws: dict = {}

    def __call__(self, A: torch.Tensor, W: torch.Tensor, D: torch.Tensor, bias: torch.Tensor | None = None,
                 C: torch.Tensor | None = None, act: int = 0, tile: int = 0) -> torch.Tensor:
        M, K = A.shape
        N = W.shape[0]
        key = (M, N, K, act, tile, C is not None)
        ws = self._ws.get(key)
        if ws is None:
            nbytes = self._fvk.alpamayo15_bf16_gemm_sm100_workspace(M, N, K, act, tile, int(C is not None))
            ws = torch.empty(max(int(nbytes), 16), dtype=torch.uint8, device=DEV)
            self._ws[key] = ws
        self._fvk.alpamayo15_bf16_gemm_sm100(A.data_ptr(), W.data_ptr(), bias.data_ptr() if bias is not None else 0,
                                             C.data_ptr() if C is not None else 0, D.data_ptr(), M, N, K, act, tile,
                                             ws.data_ptr(), ws.numel(), torch.cuda.current_stream().cuda_stream)
        return D

    def fp8(self, A8: torch.Tensor, a_s: torch.Tensor, W8: torch.Tensor, w_s: torch.Tensor, D: torch.Tensor,
            tile: int = 1) -> torch.Tensor:
        """``D = bf16((A8 @ W8.T) * a_s[m] * w_s[n])`` on the FP8 tcgen05 GEMM (A8 [M, K], W8 [N, K] e4m3 bytes; D a
        row-major [M, N] block of a wider row-major tensor; tile 1 = 64-row, 0 = 128-row, 2 = 256-row 2-SM)."""
        M, K = A8.shape
        N = W8.shape[0]
        key = ("fp8", M, N, K, tile)
        ws = self._ws.get(key)
        if ws is None:
            nbytes = self._fvk.alpamayo15_fp8_gemm_sm100_workspace(M, N, K, tile)
            ws = torch.empty(max(int(nbytes), 16), dtype=torch.uint8, device=DEV)
            self._ws[key] = ws
        self._fvk.alpamayo15_fp8_gemm_sm100(A8.data_ptr(), W8.data_ptr(), a_s.data_ptr(), w_s.data_ptr(), D.data_ptr(), M, N, K,
                                            ws.data_ptr(), ws.numel(), torch.cuda.current_stream().cuda_stream, tile, D.stride(0))
        return D

    def fp8_act(self, A8: torch.Tensor, a_s: torch.Tensor, W8: torch.Tensor, w_s: torch.Tensor, D: torch.Tensor,
                bias: torch.Tensor | None = None, C: torch.Tensor | None = None, act: int = 0) -> torch.Tensor:
        """``D = act((A8 @ W8.T) * a_s[m] * w_s[n] + bias[n]) (+ C)`` on the 2-SM FP8 GEMM (vision tier); C / D row-major
        blocks of one leading dimension; act 0 identity, 1 GELU tanh, 2 GELU erf (residual with identity only)."""
        M, K = A8.shape
        N = W8.shape[0]
        key = ("fp8act", M, N, K, act, C is not None)
        ws = self._ws.get(key)
        if ws is None:
            nbytes = self._fvk.alpamayo15_fp8_gemm_act_sm100_workspace(M, N, K, act, int(C is not None))
            ws = torch.empty(max(int(nbytes), 16), dtype=torch.uint8, device=DEV)
            self._ws[key] = ws
        self._fvk.alpamayo15_fp8_gemm_act_sm100(A8.data_ptr(), W8.data_ptr(), a_s.data_ptr(), w_s.data_ptr(),
                                                bias.data_ptr() if bias is not None else 0, C.data_ptr() if C is not None else 0,
                                                D.data_ptr(), M, N, K, act, D.stride(0), ws.data_ptr(), ws.numel(),
                                                torch.cuda.current_stream().cuda_stream)
        return D


def vision_merger(rt, mod, x: torch.Tensor, rows: int, D_in: int) -> torch.Tensor:
    """Qwen3VLVisionPatchMerger on FlashRT kernels: LayerNorm (pre- or post-shuffle), fc1 + erf GELU, fc2."""
    fvk, g = rt.fvk, rt.gemm
    s = torch.cuda.current_stream().cuda_stream
    Hm = mod.hidden_size                                   # 4 * D_in
    n = rows // 4
    if mod.use_postshuffle_norm:
        h = torch.empty(n, Hm, device=DEV, dtype=BF16)
        fvk.alpamayo15_layernorm_bf16(x.data_ptr(), mod.norm.weight.data_ptr(), mod.norm.bias.data_ptr(), h.data_ptr(),
                                      n, Hm, mod.norm.eps, s)
    else:
        h = torch.empty(rows, D_in, device=DEV, dtype=BF16)
        fvk.alpamayo15_layernorm_bf16(x.data_ptr(), mod.norm.weight.data_ptr(), mod.norm.bias.data_ptr(), h.data_ptr(),
                                      rows, D_in, mod.norm.eps, s)
        h = h.view(n, Hm)
    f1 = g(h, mod.linear_fc1.weight, torch.empty(n, Hm, device=DEV, dtype=BF16), bias=mod.linear_fc1.bias, act=2,
           tile=VISION_TILE)
    return g(f1, mod.linear_fc2.weight, torch.empty(n, mod.linear_fc2.weight.shape[0], device=DEV, dtype=BF16),
             bias=mod.linear_fc2.bias, tile=VISION_TILE)


def vision_tower(rt, pixel_values: torch.Tensor, grid: torch.Tensor, px16: torch.Tensor | None = None):
    """The Qwen3-VL vision tower on FlashRT kernels: patch embedding (+ position embedding) and every block's
    LayerNorm, qkv / proj / fc1 / fc2 (bias, GELU and residual fused into the GEMM epilogues), FP32 RoPE and
    batched FA4 attention; mergers as ``vision_merger``. Same operation order as the official tower; GEMM reduction order and
    the fused epilogues (one rounding of bias + activation) differ from the cuBLAS + torch path."""
    V = rt.vlm.model.visual
    fvk, g = rt.fvk, rt.gemm
    (pos, cos, sin), key = rt.vision_tables(grid)
    n_img = len(key) // 3
    t, gh, gw = key[0:3]
    if not all(key[3 * i:3 * i + 3] == (t, gh, gw) for i in range(n_img)):
        raise ValueError("the CUTLASS vision path needs equal frame grids (batched attention)")
    pe = V.patch_embed.proj
    S, Din = pixel_values.shape[0], pe.weight.shape[0]
    s = torch.cuda.current_stream().cuda_stream
    if pixel_values.dtype == BF16:
        px = pixel_values
    else:                                             # the processor's fp32 pixels -> the tower's BF16 input
        px = px16 if px16 is not None else torch.empty(pixel_values.shape, device=DEV, dtype=BF16)
        fvk.alpamayo15_cast_f32_bf16(pixel_values.data_ptr(), px.data_ptr(), pixel_values.numel(), s)
    T = VISION_TILE
    x = g(px, pe.weight.view(Din, -1), torch.empty(S, Din, device=DEV, dtype=BF16), bias=pe.bias, C=pos, tile=T)
    nh = V.blocks[0].attn.num_heads
    hd = Din // nh
    L = t * gh * gw
    h = torch.empty(S, Din, device=DEV, dtype=BF16)
    qkv = torch.empty(S, 3 * Din, device=DEV, dtype=BF16)
    Ffc = V.blocks[0].mlp.linear_fc1.weight.shape[0]
    f1 = torch.empty(S, Ffc, device=DEV, dtype=BF16)
    x2 = torch.empty_like(x)
    v8 = rt.vision_fp8
    if v8 is not None:              # W8A8 blocks: FP8 rows of the LayerNorm outputs, the attention output and GELU(fc1)
        u8 = lambda *shape: torch.empty(*shape, device=DEV, dtype=torch.uint8)
        f32 = lambda n: torch.empty(n, device=DEV, dtype=torch.float32)
        h8, h_s, o8, o_s, f18, f1_s = u8(S, Din), f32(S), u8(S, Din), f32(S), u8(S, Ffc), f32(S)
    blk_gemm = lambda A, W, D, bias=None, C=None, act=0: g(A, W, D, bias=bias, C=C, act=act, tile=T)
    fc1_gemm = lambda A, W, D, bias: g(A, W, D, bias=bias, act=1, tile=T)   # faster in situ than cuBLASLt's GELU epilogue
    deep = []
    for li, blk in enumerate(V.blocks):
        a, m = blk.attn, blk.mlp
        if v8 is None:
            fvk.alpamayo15_layernorm_bf16(x.data_ptr(), blk.norm1.weight.data_ptr(), blk.norm1.bias.data_ptr(), h.data_ptr(),
                                          S, Din, blk.norm1.eps, s)
            blk_gemm(h, a.qkv.weight, qkv, bias=a.qkv.bias)
        else:
            fvk.alpamayo15_layernorm_fp8_rows_bf16(x.data_ptr(), blk.norm1.weight.data_ptr(), blk.norm1.bias.data_ptr(),
                                                   h8.data_ptr(), h_s.data_ptr(), S, Din, blk.norm1.eps, s)
            g.fp8_act(h8, h_s, *v8[li]["qkv"], qkv, bias=a.qkv.bias)
        fvk.alpamayo15_vision_rope_qk_bf16(qkv.data_ptr(), cos.data_ptr(), sin.data_ptr(), S, nh, hd, s)
        q, k, v = qkv.view(S, 3, nh, hd).unbind(1)
        o, _ = rt.fa4(*(z.view(n_img, L, nh, hd) for z in (q, k, v)), softmax_scale=a.scaling, causal=False)
        o = o.reshape(S, Din)
        if v8 is None:
            blk_gemm(o, a.proj.weight, x2, bias=a.proj.bias, C=x)                   # x2 = x + proj(o)
            fvk.alpamayo15_layernorm_bf16(x2.data_ptr(), blk.norm2.weight.data_ptr(), blk.norm2.bias.data_ptr(), h.data_ptr(),
                                          S, Din, blk.norm2.eps, s)
            fc1_gemm(h, m.linear_fc1.weight, f1, m.linear_fc1.bias)
            blk_gemm(f1, m.linear_fc2.weight, x, bias=m.linear_fc2.bias, C=x2)     # x = x2 + fc2(gelu(fc1(h)))
        else:
            fvk.alpamayo15_quantize_fp8_rows_bf16(o.data_ptr(), Din, o8.data_ptr(), o_s.data_ptr(), S, Din, s)
            g.fp8_act(o8, o_s, *v8[li]["proj"], x2, bias=a.proj.bias, C=x)
            fvk.alpamayo15_layernorm_fp8_rows_bf16(x2.data_ptr(), blk.norm2.weight.data_ptr(), blk.norm2.bias.data_ptr(),
                                                   h8.data_ptr(), h_s.data_ptr(), S, Din, blk.norm2.eps, s)
            g.fp8_act(h8, h_s, *v8[li]["fc1"], f1, bias=m.linear_fc1.bias, act=1)
            fvk.alpamayo15_quantize_fp8_rows_bf16(f1.data_ptr(), Ffc, f18.data_ptr(), f1_s.data_ptr(), S, Ffc, s)
            g.fp8_act(f18, f1_s, *v8[li]["fc2"], x, bias=m.linear_fc2.bias, C=x2)
        if li in V.deepstack_visual_indexes:
            deep.append(vision_merger(rt, V.deepstack_merger_list[V.deepstack_visual_indexes.index(li)], x, S, Din))
    return vision_merger(rt, V.merger, x, S, Din), deep


def action_in(rt, x: torch.Tensor, t: torch.Tensor, rows: int, res: torch.Tensor, s: int) -> None:
    """``res = action_in_proj(x, t)`` (fp32): Fourier features, the SiLU MLP on tcgen05 GEMMs with the activation in
    the epilogue and RMSNorms between, the LayerNorm with fp32 output."""
    fvk, b = rt.fvk, rt.expert_buffers()
    fvk.alpamayo15_action_fourier_bf16(x.data_ptr(), int(x.dtype == BF16), t.data_ptr(), rt.act.freqs.data_ptr(), rows, rows,
                                       rt.act_dim, rt.act.nf, b["af"].data_ptr(), rt.act.k0, s)
    src = b["af"][:rows]
    for li, (w, bias, act) in enumerate(rt.act.mlp):
        last = li == len(rt.act.mlp) - 1
        dst = (b["ae"] if last else b["ah0"])[:rows]
        rt.gemm(src, w, dst, bias=bias, act=act, tile=1)
        if not last:
            w_rms, eps = rt.act.rms[li]
            fvk.alpamayo15_rms_bf16(dst.data_ptr(), 0, w_rms.data_ptr(), b["ah1"].data_ptr(), rows, dst.shape[1], eps, s)
            src = b["ah1"][:rows]
    w_ln, b_ln, eps = rt.act.ln
    fvk.alpamayo15_layernorm_f32out_bf16(b["ae"].data_ptr(), w_ln.data_ptr(), b_ln.data_ptr(), res.data_ptr(), rows, rt.e_hidden,
                                         eps, s)


class Alpamayo15Thor:
    """The request computation of the Thor engine for one official Alpamayo 1.5 model, on the buffers, weights and
    CUDA graphs of ``runtime`` (``runtime_thor.Alpamayo15ThorRuntime``; the frontend
    ``flash_rt.frontends.torch.alpamayo15_thor`` builds both, ``build_engine`` from a loaded model)."""

    def __init__(self, runtime: Alpamayo15ThorRuntime) -> None:
        self.rt = runtime
        self.m = runtime.m
        self._mark = None

    _W4_NEXT = {"qkv": "o", "o": "gu", "gu": "down", "down": None}

    def _proj(self, x: torch.Tensor, w: torch.Tensor, key: str, li: int, out: torch.Tensor, stream: int) -> None:
        """``out = x @ w.T`` from the BF16 weight or, in the FP8 / INT8 / W4A8 / INT4 tiers, its quantized form."""
        rt = self.rt
        if li < 0 and rt.w.lm4 is not None:
            rt.w.lm4(x, out, stream)
            return
        if rt.w.q8 is None:                     # BF16 weights read in place
            M, (N, K) = x.shape[0], w.shape
            if x.stride(0) == K and (li >= 0 or M == 1):
                # cuBLASLt's few-row kernels stream the projections at 230-250 GB/s on Thor (the LM head for one row too)
                rt.fvk.alpamayo15_cublaslt_gemm_bf16(x.data_ptr(), w.data_ptr(), 0, 0, out.data_ptr(), M, N, K, 0, out.stride(0), 0,
                                                     stream)
            else:                                 # the LM head over several rows: FlashRT's few-row tensor-core GEMM
                rt.fvk.flashdrive_bf16_gemm_small(x.data_ptr(), x.stride(0), w.data_ptr(), out.data_ptr(), out.stride(0), 0,
                                                  M, N, K, stream)
            return
        if rt.w.w4 is not None and li >= 0:
            # the GEMM prefetches the rotation tables of the group that runs next (o, gate|up, down, next layer's q|k|v)
            nxt = self._W4_NEXT[key]
            nxt_group = rt.w.w4[li][nxt] if nxt else (rt.w.w4[li + 1]["qkv"] if li + 1 < rt.n_layers else None)
            rt.w.w4[li][key](x, out, stream, nxt_group.tables_region if nxt_group is not None else None)
            return
        frag, scale = rt.w.q8[key][li] if li >= 0 else rt.w.q8[key]
        if not (x.is_contiguous() and out.is_contiguous() and x.shape[0] == out.shape[0]):
            raise ValueError("quantized projection needs contiguous row-major x [n, K] and out [n, N]")
        rt.fvk.alpamayo15_w8a16_gemm_bf16(frag.data_ptr(), scale.data_ptr(), x.data_ptr(), out.data_ptr(),
                                          x.shape[0], out.shape[1], x.shape[1], stream, rt.w.q8["fmt"])

    # ------------------------------------------------------------------
    # vision
    # ------------------------------------------------------------------
    def _encode(self, pixel_values: torch.Tensor, grid: torch.Tensor):
        """(image embeddings, DeepStack features) of the request's frames: the CUTLASS tower as one CUDA graph per
        (pixel shape, frame grid); frames of different grids go through ``_encode_groups``."""
        rt = self.rt
        key_grid = tuple(grid.flatten().tolist())
        n_img = len(key_grid) // 3
        if not all(key_grid[3 * i:3 * i + 3] == key_grid[0:3] for i in range(n_img)):
            return self._encode_groups(pixel_values, grid, key_grid)
        ent = rt.vision_graphs.get(pixel_values, grid, partial(vision_tower, rt), key_grid)
        ent["px"].copy_(pixel_values)
        ent["graph"].replay()
        return ent["out"]["img"], ent["out"]["deep"]

    def _encode_groups(self, pixel_values: torch.Tensor, grid: torch.Tensor, key_grid: tuple):
        """Frames of different grids: each group of equal grids through its own tower graph, the features put back
        in frame order (the attention is batched per group, so this costs one graph replay per distinct grid)."""
        rt = self.rt
        fvk = rt.fvk
        s = torch.cuda.current_stream().cuda_stream
        n_img = len(key_grid) // 3
        merge = rt.spatial_merge ** 2
        patches = [key_grid[3 * i] * key_grid[3 * i + 1] * key_grid[3 * i + 2] for i in range(n_img)]
        starts = np.concatenate([[0], np.cumsum(patches)])
        groups: dict[tuple, list[int]] = {}
        for i in range(n_img):
            groups.setdefault(key_grid[3 * i:3 * i + 3], []).append(i)
        total = int(starts[-1]) // merge
        img = torch.empty(total, rt.hidden, device=DEV, dtype=BF16)
        deep = [torch.empty(total, rt.hidden, device=DEV, dtype=BF16) for _ in rt.vlm.model.visual.deepstack_merger_list]
        for g, frames in groups.items():
            rows = torch.tensor([r for i in frames for r in range(int(starts[i]), int(starts[i + 1]))], dtype=torch.long, device=DEV)
            px = torch.empty(rows.numel(), pixel_values.shape[1], device=DEV, dtype=pixel_values.dtype)
            es = pixel_values.element_size()
            fvk.alpamayo15_gather_rows_bytes(pixel_values.data_ptr(), pixel_values.stride(0) * es, rows.data_ptr(), 0, px.data_ptr(),
                                             px.stride(0) * es, rows.numel(), pixel_values.shape[1] * es, s)
            gi, gd = self._encode(px, torch.tensor([g] * len(frames), device=DEV, dtype=grid.dtype))
            out_rows = torch.tensor([r for i in frames for r in range(int(starts[i]) // merge, int(starts[i + 1]) // merge)],
                                    dtype=torch.long, device=DEV)
            fvk.alpamayo15_scatter_rows_bf16(img.data_ptr(), img.stride(0), out_rows.data_ptr(), gi.data_ptr(), gi.stride(0), gi.shape[0],
                                             rt.hidden, 0, s)
            for dst, src in zip(deep, gd):
                fvk.alpamayo15_scatter_rows_bf16(dst.data_ptr(), dst.stride(0), out_rows.data_ptr(), src.data_ptr(), src.stride(0),
                                                 src.shape[0], rt.hidden, 0, s)
        return img, deep


    # ------------------------------------------------------------------
    # prefill (writes sample 0 of the cache)
    # ------------------------------------------------------------------
    def _prefill_native(self, input_ids, tokenized, vis_idx: torch.Tensor, pos3: torch.Tensor):
        """Vision + 36-layer LM prefill on fused FlashRT kernels.

        K/V are normalized, rotated (interleaved M-RoPE) and written straight
        into the static cache; only the last row is normalized for the logits.
        """
        rt = self.rt
        vm, lm, fvk = rt.vlm.model, rt.lm, rt.fvk
        s = torch.cuda.current_stream().cuda_stream
        P = input_ids.shape[1]
        H, D, NH, NKV, F_ = rt.hidden, rt.head_dim, rt.n_heads, rt.n_kv, rt.ffn
        b = rt.prefill_buffers(P)
        x, h, qkv, q, attn, o, g, u, d = (b[k][:P] for k in ("x", "h", "qkv", "q", "attn", "o", "g", "u", "d"))
        grid = tokenized["image_grid_thw"]
        img, deepstack = self._encode(tokenized["pixel_values"], grid)
        if self._mark is not None:
            self._mark("vision_end")
        if vis_idx.numel() != img.shape[0]:
            raise ValueError(f"image tokens {vis_idx.numel()} != image features {img.shape[0]}")
        fvk.alpamayo15_gather_rows_bf16(lm.embed_tokens.weight.data_ptr(), H, input_ids[0].data_ptr(), x.data_ptr(), H, P, H, s)
        fvk.alpamayo15_scatter_rows_bf16(x.data_ptr(), H, vis_idx.data_ptr(), img.data_ptr(), img.stride(0), img.shape[0], H, 0, s)
        cos_f, sin_f = b["cos"][:P], b["sin"][:P]
        fvk.alpamayo15_mrope_tables(pos3.data_ptr(), rt.inv_freq.data_ptr(), rt.mrope_sec[1], rt.mrope_sec[2], rt.rope_scaling,
                                    cos_f.data_ptr(), sin_f.data_ptr(), P, D, 0, s)
        layers = lm.layers
        p8, p4 = rt.w.p8, rt.w.p4
        f4 = p4["module"] if p4 is not None else None
        # NVFP4 (W4A4) tier: every GEMM input is quantized once by its producer with per-row scales, and the producer
        # writes the per-row alphas (row scale x weight scale) of the input's consumers: q|k|v after the first norm,
        # gate and up after the second, o after the attention, down after the SiLU
        NVFP4_CONSUMERS = {"qkv": ("qkv",), "o": ("o",), "gate": ("gate", "up"), "down": ("down",)}

        def check(rc, what):
            if rc != 0:
                raise RuntimeError(f"NVFP4 {what} failed (rc={rc})")

        def norm(delta, w, group):
            """``h = RMSNorm(x += delta) * w``; the W8A8 tier's producer writes the FP8 rows of h directly, the W4A4
            tier's its NVFP4 rows and the alphas of their consumers ``group`` = (layer, "qkv" | "gate")."""
            dp = delta.data_ptr() if delta is not None else 0
            if p4 is not None:
                li, first = group
                check(f4.alpamayo15_nvfp4_rms_rows_bf16(x.data_ptr(), dp, w.data_ptr(), P, H, rt.eps, b["h4"].data_ptr(),
                                                        b["h_sf"].data_ptr(), p4["layers"][li]["gsw_" + first].data_ptr(),
                                                        len(NVFP4_CONSUMERS[first]), b["h_alpha"].data_ptr(), b["P"], s), "RMSNorm")
            elif p8 is None:
                fvk.alpamayo15_rms_bf16(x.data_ptr(), dp, w.data_ptr(), h.data_ptr(), P, H, rt.eps, s)
            else:
                fvk.alpamayo15_rms_fp8_rows_bf16(x.data_ptr(), dp, w.data_ptr(), b["h8"].data_ptr(), b["h_s"].data_ptr(), P, H,
                                                 rt.eps, s)

        def lin(src, akey, w, wname, li, out, col0=0, quantized=False):
            """``out = src @ w.T``: the BF16 weight ``w`` (tcgen05 or cuBLASLt), or (W8A8 tier) per-row FP8 rows of ``src``
            (buffers ``akey``, written by the producer when ``quantized``) times rows ``col0:`` of the FP8 copy
            ``wname`` of layer ``li``; the W4A4 tier always reads the producer's NVFP4 rows of ``akey``."""
            if p4 is not None:
                nvfp4_lin(akey, wname, li, out, col0)
                return
            if p8 is None:                               # BF16 weights: the faster of the 2-SM 256 x 256 tile and cuBLASLt
                if wname not in BF16_PREFILL_CUBLASLT.split("+"):
                    rt.gemm(src, w, out, tile=3)
                    return
                N, K = out.shape[1], src.shape[1]
                nb = PREFILL_DOWN_SPLIT if wname == "down" else 1    # down as column blocks whose weights fit the L2
                Nb = N // nb
                for j in range(nb):
                    fvk.alpamayo15_cublaslt_gemm_bf16(src.data_ptr(), w[j * Nb:(j + 1) * Nb].data_ptr(), 0, 0, out[:, j * Nb:].data_ptr(),
                                                      P, Nb, K, 0, out.stride(0), 0, s)
                return
            q8, q_s = b[akey + "8"][:P], b[akey + "_s"][:P]
            if col0 == 0 and not quantized:
                fvk.alpamayo15_quantize_fp8_rows_bf16(src.data_ptr(), src.stride(0), q8.data_ptr(), q_s.data_ptr(), P,
                                                      src.shape[1], s)
            w8, w_s = p8[li][wname]
            N = out.shape[1]
            if wname == "down" and PREFILL_DOWN_SPLIT > 1:          # column blocks: each weight block fits the L2
                Nb = N // PREFILL_DOWN_SPLIT
                for j in range(PREFILL_DOWN_SPLIT):
                    rt.gemm.fp8(q8, q_s, w8[j * Nb:(j + 1) * Nb], w_s[j * Nb:(j + 1) * Nb], out[:, j * Nb:(j + 1) * Nb],
                                tile=PREFILL_FP8_TILE)
            else:
                rt.gemm.fp8(q8, q_s, w8[col0:col0 + N], w_s[col0:col0 + N], out, tile=PREFILL_FP8_TILE)

        def nvfp4_lin(akey, wname, li, out, col0):
            """``out = rows(akey) @ W^T`` on the NVFP4 GEMM: the input's per-row NVFP4 rows (written by its producer)
            times the NVFP4 copy ``wname`` of layer ``li``, the rows' alphas applied in the epilogue."""
            wname = "gate" if wname == "gu" and col0 == 0 else ("up" if wname == "gu" else wname)
            first = "gate" if wname in ("gate", "up") else wname
            wp, wsf, _ = p4["layers"][li][wname]
            packed, sf = b[akey + "4"], b[akey + "_sf"]
            alpha = b[akey + "_alpha"][NVFP4_CONSUMERS[first].index(wname)]
            N, K = out.shape[1], packed.shape[1] * 2
            key = (P, N, K, NVFP4_TILE)
            ws = p4["ws"].get(key)
            if ws is None:
                ws = torch.empty(max(int(f4.alpamayo15_nvfp4_gemm_workspace(P, N, K, NVFP4_TILE)), 16), device=DEV, dtype=torch.uint8)
                p4["ws"][key] = ws
            check(f4.alpamayo15_nvfp4_gemm_bf16(packed.data_ptr(), sf.data_ptr(), wp.data_ptr(), wsf.data_ptr(), out.data_ptr(), P, N, K,
                                                NVFP4_TILE, ws.data_ptr(), ws.numel(), s, out.stride(0), alpha.data_ptr(), 1),
                  f"GEMM (layer {li}, {wname})")
        norm(None, layers[0].input_layernorm.weight, (0, "qkv"))
        for li, layer in enumerate(layers):
            a = layer.self_attn
            lin(h, "h", rt.w.qkv[li], "qkv", li, qkv, quantized=True)
            kc, vc = rt.k_cache[li, 0, :P], rt.v_cache[li, 0, :P]
            fvk.alpamayo15_vlm_qk_prep_bf16(
                qkv.data_ptr(), a.q_norm.weight.data_ptr(), a.k_norm.weight.data_ptr(),
                cos_f.data_ptr(), sin_f.data_ptr(), q.data_ptr(), kc.data_ptr(), vc.data_ptr(), 0,
                P, NH, NKV, NKV * D, P, 0, rt.eps, s)
            rt.fa4(q.view(1, P, NH, D), kc.unsqueeze(0), vc.unsqueeze(0), softmax_scale=rt.scale,
                   causal=True, pack_gqa=True, out=attn.view(1, P, NH, D))
            if p4 is not None:
                check(f4.alpamayo15_nvfp4_quantize_rows_rowscale_bf16(attn.data_ptr(), attn.stride(0), P, NH * D, b["attn4"].data_ptr(),
                                                                      b["attn_sf"].data_ptr(), p4["layers"][li]["gsw_o"].data_ptr(), 1,
                                                                      b["attn_alpha"].data_ptr(), b["P"], 0, s), "attention rows")
            lin(attn, "attn", a.o_proj.weight, "o", li, o)
            norm(o, layer.post_attention_layernorm.weight, (li, "gate"))
            lin(h, "h", rt.w.gu[li][:F_], "gu", li, g, quantized=True)   # cuBLAS is slower on the fused gate | up shape: two GEMMs
            lin(h, "h", rt.w.gu[li][F_:], "gu", li, u, col0=F_, quantized=True)
            if p4 is not None:
                check(f4.alpamayo15_nvfp4_silu_mul_rows_bf16(g.data_ptr(), u.data_ptr(), F_, P, F_, b["g4"].data_ptr(), b["g_sf"].data_ptr(),
                                                             p4["layers"][li]["gsw_down"].data_ptr(), 1, b["g_alpha"].data_ptr(), b["P"], s),
                      "SiLU rows")
            elif p8 is None:
                fvk.alpamayo15_silu_mul_split_bf16(g.data_ptr(), u.data_ptr(), F_, g.data_ptr(), P, F_, s)   # in place
            else:
                fvk.alpamayo15_silu_mul_fp8_rows_bf16(g.data_ptr(), u.data_ptr(), F_, b["g8"].data_ptr(), b["g_s"].data_ptr(), P, F_, s)
            lin(g, "g", layer.mlp.down_proj.weight, "down", li, d, quantized=True)
            if li < len(deepstack):
                fvk.residual_add(x.data_ptr(), d.data_ptr(), P * H, s)
                ds = deepstack[li]
                fvk.alpamayo15_scatter_rows_bf16(x.data_ptr(), H, vis_idx.data_ptr(), ds.data_ptr(), ds.stride(0), ds.shape[0], H, 1, s)
                if li + 1 < len(layers):
                    norm(None, layers[li + 1].input_layernorm.weight, (li + 1, "qkv"))
            elif li + 1 < len(layers):
                norm(d, layers[li + 1].input_layernorm.weight, (li + 1, "qkv"))
            else:
                fvk.residual_add(x.data_ptr(), d.data_ptr(), P * H, s)
        # The prompt's last-token logits stay on the BF16 LM head in every tier (the GEMV reads 1.2 GB, 5 ms on Thor):
        # sampling the first token from the INT8 head measured ADE +0.055 [+0.001, +0.161] on the nine clips.
        last, logits = b["last"], b["logits1"]
        fvk.alpamayo15_rms_bf16(x[P - 1:P].data_ptr(), 0, lm.norm.weight.data_ptr(), last.data_ptr(), 1, H, rt.eps, s)
        fvk.alpamayo15_cublaslt_gemm_bf16(last.data_ptr(), rt.vlm.lm_head.weight.data_ptr(), 0, 0, logits.data_ptr(), 1,
                                          logits.shape[1], H, 0, logits.shape[1], 0, s)
        return logits


    # ------------------------------------------------------------------
    # decode
    # ------------------------------------------------------------------
    def _attend(self, li: int, q: torch.Tensor, n: int, rows: int, single_len: torch.Tensor,
                suffix_len: torch.Tensor, splits: int) -> torch.Tensor:
        """Non-causal attention of ``q`` [n * rows, heads, D] over the cache.

        One sample: a single pass over rows [0, single_len). Several samples:
        the shared prompt (sample 0, rows [0, P)) once for all queries plus each
        sample's suffix (rows [P, P + suffix_len)), merged by log-sum-exp.
        """
        rt = self.rt
        kc, vc = rt.k_cache[li], rt.v_cache[li]
        heads, D = q.shape[1], q.shape[2]
        if n == 1:
            out, _ = rt.fa4(q.view(1, rows, heads, D), kc[:1], vc[:1], seqused_k=single_len[:1],
                            softmax_scale=rt.scale, causal=False, pack_gqa=True, num_splits=splits)
            return out.view(rows, -1)
        o1, l1 = rt.fa4(q.view(1, n * rows, heads, D), kc[:1], vc[:1], seqused_k=rt.d_plen,
                        softmax_scale=rt.scale, causal=False, pack_gqa=True, return_lse=True)
        S, KV = rt.max_seq, rt.n_kv
        o2, l2 = rt.fa4(q, kc[:n].view(n * S, KV, D), vc[:n].view(n * S, KV, D),
                        cu_seqlens_q=rt.cu_q(n, rows), cu_seqlens_k=rt.d_cuk[:n + 1],
                        seqused_k=suffix_len[:n], max_seqlen_q=rows, max_seqlen_k=rt.max_new + rt.n_act,
                        softmax_scale=rt.scale, causal=False, pack_gqa=True, return_lse=True)
        return merge_attention(o1.view(n * rows, heads, D), l1.view(heads, n * rows),
                               o2, l2.view(heads, n * rows)).view(n * rows, -1)

    def _decode_step(self, n: int) -> None:
        """One reasoning token for ``n`` samples: 36 layers (BF16, or FP8 projection weights) + on-device sampling."""
        rt = self.rt
        lm, fvk = rt.lm, rt.fvk
        s = torch.cuda.current_stream().cuda_stream
        H, D, NH, NKV, F_, S = rt.hidden, rt.head_dim, rt.n_heads, rt.n_kv, rt.ffn, rt.max_seq
        b = rt.decode_buffers()
        x, h, qkv, q, o, gu, act, d, logits = (b[k][:n] for k in ("x", "h", "qkv", "q", "o", "gu", "a", "d", "logits"))
        fvk.alpamayo15_gather_rows_bf16(lm.embed_tokens.weight.data_ptr(), H, rt.d_tok.data_ptr(), x.data_ptr(), H, n, H, s)
        cos, sin = b["cos"][:n], b["sin"][:n]
        fvk.alpamayo15_gather_rows_bf16(rt.cos_table.data_ptr(), D, rt.d_rpos.data_ptr(), cos.data_ptr(), D, n, D, s)
        fvk.alpamayo15_gather_rows_bf16(rt.sin_table.data_ptr(), D, rt.d_rpos.data_ptr(), sin.data_ptr(), D, n, D, s)
        layers = lm.layers
        fvk.alpamayo15_rms_bf16(x.data_ptr(), 0, layers[0].input_layernorm.weight.data_ptr(), h.data_ptr(), n, H, rt.eps, s)
        for li, layer in enumerate(layers):
            a = layer.self_attn
            self._proj(h, rt.w.qkv[li], "qkv", li, qkv, s)
            fvk.alpamayo15_vlm_qk_prep_bf16(
                qkv.data_ptr(), a.q_norm.weight.data_ptr(), a.k_norm.weight.data_ptr(),
                cos.data_ptr(), sin.data_ptr(), q.data_ptr(),
                rt.k_cache[li].data_ptr(), rt.v_cache[li].data_ptr(), rt.d_slot.data_ptr(),
                n, NH, NKV, NKV * D, 1, S, rt.eps, s)
            attn = self._attend(li, q.view(n, NH, D), n, 1, rt.d_len, rt.d_glen, rt.decode_splits)
            self._proj(attn, a.o_proj.weight, "o", li, o, s)
            fvk.alpamayo15_rms_bf16(x.data_ptr(), o.data_ptr(), layer.post_attention_layernorm.weight.data_ptr(),
                                    h.data_ptr(), n, H, rt.eps, s)
            self._proj(h, rt.w.gu[li], "gu", li, gu, s)
            fvk.alpamayo15_silu_mul_split_bf16(gu.data_ptr(), gu[:, F_:].data_ptr(), 2 * F_, act.data_ptr(), n, F_, s)
            self._proj(act, layer.mlp.down_proj.weight, "down", li, d, s)
            nxt = layers[li + 1].input_layernorm.weight if li + 1 < len(layers) else lm.norm.weight
            fvk.alpamayo15_rms_bf16(x.data_ptr(), d.data_ptr(), nxt.data_ptr(), h.data_ptr(), n, H, rt.eps, s)
        self._proj(h, rt.vlm.lm_head.weight, "lm_head", -1, logits, s)
        # the official sampler (trajectory-token mask, temperature, top-p, multinomial) and the step's bookkeeping
        fvk.alpamayo15_sample_tokens_bf16(logits.data_ptr(), logits.stride(0), n, logits.shape[1], rt.mask_lo, rt.mask_n,
                                          rt.temperature, rt.top_p, 0, rt.rng.data_ptr(), rt.rng_blocks, rt.samp_scratch.data_ptr(),
                                          rt.samp_ws.data_ptr(), rt.samp_ws.numel(), rt.d_tok.data_ptr(),
                                          rt.d_unfinished.data_ptr(), rt.pad_id, rt.d_slot.data_ptr(), rt.d_rpos.data_ptr(),
                                          rt.d_len.data_ptr(), rt.d_glen.data_ptr(), s)


    # ------------------------------------------------------------------
    # expert
    # ------------------------------------------------------------------
    def _expert_velocity(self, x: torch.Tensor, t: torch.Tensor, dt: float) -> None:
        """One Euler step of the expert over ``x`` [n, n_act, act_dim] (fp32, updated in place) at time ``t`` (device
        scalar), official autocast numerics."""
        rt = self.rt
        m, fvk = self.m, rt.fvk
        s = torch.cuda.current_stream().cuda_stream
        n = x.shape[0]
        A, E, D, HQ, KV, S = rt.n_act, rt.e_hidden, rt.head_dim, rt.e_heads, rt.n_kv, rt.max_seq
        rows = n * A
        b = rt.expert_buffers()
        res, hn, qkv, q, o, gu, act, d = (b[k][:rows] for k in ("res", "hn", "qkv", "q", "o", "gu", "a", "d"))
        ff = act.shape[1]
        eps = rt.e_eps
        layers = rt.expert.layers
        e8 = rt.w.e8

        def norm(delta, w):
            """``hn = RMSNorm(res += delta) * w`` on the FP32 residual; the W8A8 tier writes hn's FP8 rows directly."""
            if e8 is None:
                fvk.alpamayo15_res_rms_f32_bf16(res.data_ptr(), delta.data_ptr() if delta is not None else 0, w.data_ptr(),
                                                hn.data_ptr(), rows, E, eps, s)
            else:
                fvk.alpamayo15_res_rms_f32_fp8_rows_bf16(res.data_ptr(), delta.data_ptr() if delta is not None else 0,
                                                         w.data_ptr(), b["hn8"].data_ptr(), b["hn_s"].data_ptr(), rows, E, eps, s)

        def lin(src, akey, w, wname, li, out, quantized=False):
            """``out = src @ w.T``: tcgen05 / cuBLAS on the BF16 weight, or (W8A8 tier) per-row FP8 rows of ``src``
            (buffers ``akey``, written by the producer when ``quantized``) times the FP8 copy ``wname`` of layer
            ``li``, scales applied in the epilogue."""
            if e8 is not None:
                q8, q_s = b[akey + "8"][:rows], b[akey + "_s"][:rows]
                if not quantized:
                    fvk.alpamayo15_quantize_fp8_rows_bf16(src.data_ptr(), src.stride(0), q8.data_ptr(), q_s.data_ptr(), rows,
                                                          src.shape[1], s)
                w8, w_s = e8[li][wname]
                rt.gemm.fp8(q8, q_s, w8, w_s, out)
            else:
                rt.gemm(src, w, out, tile=1)
        action_in(rt, x, t, rows, res, s)                                  # fp32 residual stream (action_in_proj)
        norm(None, layers[0].input_layernorm.weight)
        for li, layer in enumerate(layers):
            a = layer.self_attn
            lin(hn, "hn", rt.w.e_qkv[li], "qkv", li, qkv, quantized=True)
            fvk.alpamayo15_expert_qk_prep_bf16(
                qkv.data_ptr(), a.q_norm.weight.data_ptr(), a.k_norm.weight.data_ptr(),
                rt.e_cos.data_ptr(), rt.e_sin.data_ptr(), q.data_ptr(),
                rt.k_cache[li].data_ptr(), rt.v_cache[li].data_ptr(), rt.d_aslot.data_ptr(),
                rows, HQ, KV, KV * D, A, S, eps, s)
            attn = self._attend(li, q.view(rows, HQ, D), n, A, rt.d_eseq, rt.d_esuf, rt.expert_splits)
            lin(attn, "attn", a.o_proj.weight, "o", li, o)
            norm(o, layer.post_attention_layernorm.weight)
            lin(hn, "hn", rt.w.e_gu[li], "gu", li, gu, quantized=True)
            if e8 is None:
                fvk.alpamayo15_silu_mul_split_bf16(gu.data_ptr(), gu[:, ff:].data_ptr(), 2 * ff, act.data_ptr(), rows, ff, s)
            else:
                fvk.alpamayo15_silu_mul_fp8_rows_bf16(gu.data_ptr(), gu[:, ff:].data_ptr(), 2 * ff, b["a8"].data_ptr(),
                                                      b["a_s"].data_ptr(), rows, ff, s)
            lin(act, "a", layer.mlp.down_proj.weight, "down", li, d, quantized=True)
            nxt = layers[li + 1].input_layernorm.weight if li + 1 < len(layers) else rt.expert.norm.weight
            if li + 1 < len(layers):
                norm(d, nxt)
            else:                                   # the final norm feeds the BF16 action_out_proj
                fvk.alpamayo15_res_rms_f32_bf16(res.data_ptr(), d.data_ptr(), nxt.data_ptr(), hn.data_ptr(), rows, E, eps, s)
        fvk.alpamayo15_action_out_euler(hn.data_ptr(), rt.act.out_w.data_ptr(), rt.act.out_b.data_ptr(), dt, x.data_ptr(), 0, rows, E,
                                        rt.act_dim, s)

    def _denoise(self, n: int) -> None:
        rt = self.rt
        ts, x = rt.time_steps, rt.x[:n]
        for i in range(rt.n_steps):
            self._expert_velocity(x, ts[i:i + 1], rt.step_dt[i])

    def capture(self, num_samples: tuple[int, ...] = (1,)) -> None:
        """Capture the decode and diffusion graphs for the given sample counts (the runtime captures this pipeline's
        step functions) and JIT-compile the prefill and vision attention kernels."""
        rt = self.rt
        rt.capture(num_samples, self._decode_step, self._denoise)


    # ------------------------------------------------------------------
    # request
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_trajectories(self, data: dict, *, num_traj_samples: int = 1, return_extra: bool = True,
                            stage_events: dict | None = None):
        """Official request semantics (``num_traj_sets=1``).

        ``data`` is the official model input dict (``tokenized_data`` plus
        ``ego_history_xyz``/``ego_history_rot`` on CUDA). The caller seeds the
        CUDA generator exactly as for the official entry point. Returns
        ``pred_xyz`` [1, 1, N, T, 3], ``pred_rot`` [1, 1, N, T, 3, 3] and the
        extracted text fields shaped [1, 1, N].
        """
        rt = self.rt
        def mark(name):
            if stage_events is not None:
                e = torch.cuda.Event(enable_timing=True)
                e.record()
                stage_events[name] = e

        m, N = self.m, int(num_traj_samples)
        self.capture((N,))
        self._mark = mark if stage_events is not None else None
        tokenized = dict(data["tokenized_data"])
        input_ids = tokenized.pop("input_ids")
        if tokenized.get("pixel_values_videos") is not None:
            raise ValueError("Alpamayo15Thor takes image frames; video inputs are not supported")
        hist_xyz, hist_rot = data["ego_history_xyz"], data["ego_history_rot"]
        if input_ids.shape[0] != 1 or hist_xyz.shape[1] != 1:
            raise ValueError("Alpamayo15Thor supports batch 1 / one trajectory group")
        # host-side preparation: the history tokens, the image-token rows, the M-RoPE positions, the t0 velocity
        hist_xyz_h, hist_rot_h = hist_xyz.detach().cpu(), hist_rot.detach().cpu()
        input_ids = m.fuse_traj_tokens(input_ids.cpu(), {"ego_history_xyz": hist_xyz_h, "ego_history_rot": hist_rot_h})
        P = input_ids.shape[1]
        if P > rt.max_prompt:
            raise ValueError(f"prompt length {P} exceeds max_prompt {rt.max_prompt}")
        ids = input_ids[0].tolist()
        vis_idx = torch.tensor([i for i, t in enumerate(ids) if t == rt.image_token_id], dtype=torch.long, device=DEV)
        grids = [tuple(int(v) for v in row) for row in tokenized["image_grid_thw"].tolist()]
        pos_h, rope_delta = mrope_positions(ids, grids, image_token_id=rt.image_token_id, vision_start_id=rt.vision_start_id,
                                            spatial_merge=rt.spatial_merge)
        pos3 = torch.from_numpy(pos_h).to(DEV)
        input_ids = input_ids.to(DEV)
        v0 = float(m.action_space.estimate_t0_states(hist_xyz_h[:, -1].float(), hist_rot_h[:, -1].float())["v"].reshape(-1)[0])
        z_last = float(hist_xyz_h[0, -1, -1, 2])
        gen = torch.cuda.default_generators[torch.cuda.current_device()]
        rng_offset = int(gen.get_offset())
        rt.rng.copy_(torch_rng_state(gen))             # the sampler draws torch's multinomial draws from here on

        # 1) vision + prefill once (sample 0 holds the shared prompt K/V); first tokens.
        mark("prefill_start")
        logits = self._prefill_native(input_ids, tokenized, vis_idx, pos3)
        rt.fvk.alpamayo15_sample_tokens_bf16(logits.data_ptr(), 0, N, logits.shape[1], rt.mask_lo, rt.mask_n, rt.temperature,
                                             rt.top_p, 0, rt.rng.data_ptr(), rt.rng_blocks, rt.samp_scratch.data_ptr(),
                                             rt.samp_ws.data_ptr(), rt.samp_ws.numel(), rt.d_tok.data_ptr(), 0, rt.pad_id, 0, 0, 0, 0,
                                             torch.cuda.current_stream().cuda_stream)
        first = rt.d_tok[:N]
        mark("prefill_end")

        # 2) decode: graph replays, host evaluates the official stop criteria.
        tokens = [[t] for t in first.tolist()]
        traj_found, unfinished = [False] * N, [True] * N
        update_stop_state([t[-1] for t in tokens], traj_found, unfinished, 1,
                          traj_start_id=rt.traj_start_id, max_new_tokens=rt.max_new, eos_ids=rt.eos_ids)
        graph = rt.decode_graphs[N]
        rt.d_unfinished[:N].copy_(torch.tensor(unfinished, dtype=torch.long))
        rt.d_slot.fill_(P)
        rt.d_len.fill_(P + 1)
        rt.d_glen.fill_(1)
        rt.d_plen.fill_(P)
        cuk = torch.arange(N + 1, dtype=torch.int32) * rt.max_seq
        cuk[:N] += P
        rt.d_cuk[:N + 1].copy_(cuk)
        rt.d_rpos.fill_(P + rope_delta)                # every sample's rope position (all advance together)
        while any(unfinished):
            graph.replay()
            last = rt.d_tok[:N].tolist()
            for row, t in zip(tokens, last):
                row.append(t)
            before = list(unfinished)
            update_stop_state(last, traj_found, unfinished, len(tokens[0]),
                              traj_start_id=rt.traj_start_id, max_new_tokens=rt.max_new, eos_ids=rt.eos_ids)
            if unfinished != before and any(unfinished):
                rt.d_unfinished[:N].copy_(torch.tensor(unfinished, dtype=torch.long))
        # torch's generator advances as the official sampling advanced it (one multinomial per step), so the noise
        # draw below reads the same random stream as the official model's
        gen.set_offset(rng_offset + torch_philox_increment(N * logits.shape[1], rt.rng_blocks) * len(tokens[0]))
        mark("decode_end")

        # 3) diffusion expert over each sample's cached prefix.
        offsets, visible = expert_attention_spans(tokens, P, rt.traj_start_id)
        epos = np.array([[rope_delta + o + k for k in range(rt.n_act)] for o in offsets], dtype=np.int32).reshape(-1)
        epos3 = torch.from_numpy(np.tile(epos, (3, 1)).copy()).to(DEV)
        s = torch.cuda.current_stream().cuda_stream
        rt.fvk.alpamayo15_mrope_tables(epos3.data_ptr(), rt.inv_freq.data_ptr(), rt.mrope_sec[1], rt.mrope_sec[2],
                                       rt.rope_scaling, rt.e_cos.data_ptr(), rt.e_sin.data_ptr(), N * rt.n_act, rt.head_dim,
                                       1, s)
        rt.d_aslot[:N].copy_(torch.tensor(visible, dtype=torch.long))
        rt.d_eseq[:N].copy_(torch.tensor([v + rt.n_act for v in visible], dtype=torch.int32))
        rt.d_esuf[:N].copy_(torch.tensor([v - P + rt.n_act for v in visible], dtype=torch.int32))
        torch.randn(N, rt.n_act, rt.act_dim, device=DEV, out=rt.x[:N])   # the official noise draw
        noise = rt.x[:N].clone() if return_extra else None
        rt.expert_graphs[N].replay()
        mark("diffusion_end")

        # 4) trajectory: the official unicycle integration from the sampled actions and the t0 velocity estimate.
        rt.a_v0[:N].copy_(torch.full((N,), v0))
        rt.a_z[:N].copy_(torch.full((N,), z_last))
        dt, a_mean, a_std, k_mean, k_std = rt.act.space
        rt.fvk.alpamayo15_action_to_traj(rt.x.data_ptr(), rt.a_v0.data_ptr(), rt.a_z.data_ptr(), N, rt.n_act, dt, a_mean,
                                         a_std, k_mean, k_std, rt.pred_xyz.data_ptr(), rt.pred_rot.data_ptr(), s)
        pred_xyz = rt.pred_xyz[:N].view(1, 1, N, rt.n_act, 3)
        pred_rot = rt.pred_rot[:N].view(1, 1, N, rt.n_act, 3, 3)
        mark("end")
        if not return_extra:
            return pred_xyz, pred_rot
        from alpamayo1_5.models.token_utils import extract_text_tokens, replace_padding_after_eos
        seq = torch.tensor([ids + row for row in tokens], dtype=torch.long)          # host: text extraction only
        seq = replace_padding_after_eos(seq, rt.traj_start_id, rt.pad_id)
        extra = extract_text_tokens(m.tokenizer, seq)
        for key in extra:
            extra[key] = np.array(extra[key]).reshape([1, 1, N])
        extra["generated_tokens"] = tokens[0] if N == 1 else tokens
        extra["noise"] = noise
        extra["rope_delta"] = rope_delta
        extra["prompt_len"] = P
        return pred_xyz, pred_rot, extra
