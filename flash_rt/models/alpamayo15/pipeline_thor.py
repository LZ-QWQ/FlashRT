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

Every GPU operation of a request after host tokenization is a FlashRT kernel (GEMMs on tcgen05 with the bias, GELU
and residual fused into the epilogues; FA4 attention; the model-local kernels of this package, including the
sampler, the position tables and the trajectory integration); the only torch operation left is the official
noise draw of the diffusion expert. Each frame grid's vision tower, each decode step and the whole expert are
CUDA graphs.

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

import numpy as np
import torch

from flash_rt.models.alpamayo15.weights import pack_action_modules, prepare_weights

BF16 = torch.bfloat16
DEV = "cuda"
logger = logging.getLogger(__name__)

ALPAMAYO15_KERNEL_SYMBOLS = (
    "alpamayo15_expert_qk_prep_bf16",
    "alpamayo15_res_rms_f32_bf16",
    "alpamayo15_rms_bf16",
    "alpamayo15_vision_rope_qk_bf16",
    "alpamayo15_vlm_qk_prep_bf16",
    "alpamayo15_w8a16_gemm_bf16",
    "alpamayo15_layernorm_bf16",
    "alpamayo15_silu_mul_split_bf16",
    "alpamayo15_quantize_fp8_rows_bf16",
    "alpamayo15_rms_fp8_rows_bf16",
    "alpamayo15_silu_mul_fp8_rows_bf16",
    "alpamayo15_res_rms_f32_fp8_rows_bf16",
    "alpamayo15_w4a8_gemm_grouped",
    "alpamayo15_w4a8_gemm_grouped_q",
    "alpamayo15_quantize_int8_rows_bf16",
    "alpamayo15_layernorm_fp8_rows_bf16",
    "alpamayo15_set_pdl",
    # request-level kernels (csrc/kernels/alpamayo15_request_kernels.cu), same build gate
    "alpamayo15_gather_rows_bf16",
    "alpamayo15_gather_rows_bytes",
    "alpamayo15_cast_f32_bf16",
    "alpamayo15_cast_f16_bf16",
    "alpamayo15_dflash_propose",
    "alpamayo15_dflash_decide",
    "alpamayo15_scatter_rows_bf16",
    "alpamayo15_mrope_tables",
    "alpamayo15_sample_tokens_bf16",
    "alpamayo15_sample_workspace_bytes",
    "alpamayo15_action_fourier_bf16",
    "alpamayo15_layernorm_f32out_bf16",
    "alpamayo15_action_out_euler",
    "alpamayo15_euler_bf16",
    "alpamayo15_action_to_traj",
    # few-row BF16 GEMM shared with the streaming backend (csrc/kernels/alpamayo15_flashdrive_kernels.cu)
    "flashdrive_bf16_gemm_small",
    # tcgen05 GEMMs (csrc/gemm/alpamayo15_bf16_gemm_sm100.cu), same build gate
    "alpamayo15_bf16_gemm_sm100",
    "alpamayo15_bf16_gemm_sm100_workspace",
    "alpamayo15_fp8_gemm_sm100",
    "alpamayo15_fp8_gemm_sm100_workspace",
    "alpamayo15_fp8_gemm_act_sm100",
    "alpamayo15_fp8_gemm_act_sm100_workspace",
    "alpamayo15_int8_gemm_sm100",
    "alpamayo15_int8_gemm_sm100_workspace",
)
VISION_TILE = 3                              # tcgen05 tile of the vision GEMMs: 256 x 256 on a 2-SM UMMA (2 = 256 x 128)
DECODE_WEIGHTS = ("bf16", "fp8", "int8", "w4a8")
EXPERT_WEIGHTS = ("bf16", "fp8")
PREFILL_WEIGHTS = ("bf16", "fp8", "nvfp4")
NVFP4_TILE = 2              # NVFP4 prefill GEMMs: 256 x 256 x 256 tiles on a 2-SM UMMA (500-575 TOPS on Thor)
VISION_WEIGHTS = ("bf16", "fp8")
LM_HEAD_WEIGHTS = ("decode", "int4")        # "decode" = the decode tier's format (INT8 with W4A8), "int4" = own INT4 linear
PREFILL_FP8_TILE = 2        # FP8 prefill GEMMs: 256 x 128 x 128 tiles on a 2-SM UMMA
PREFILL_DOWN_SPLIT = 4      # the K=12288 down projection as 4 column blocks (weight blocks that fit the L2: 1.47 vs 3.9 ms)
# Named option sets. "balanced" (the default) is the validated reduced-precision configuration (needs paro_checkpoint;
# neutral against the official model over three seeds of the nine-clip evaluation); "bf16" keeps the official weights
# everywhere (the bit-faithful reference); "fast" adds the 4-bit options measured on top of "balanced" (INT4 LM head,
# NVFP4 W4A4 prefill, which needs the flash_rt_fp4 module). Explicit weight options override the preset's.
PRESETS = {
    "bf16": {},
    "balanced": {"decode_weights": "w4a8", "expert_weights": "fp8", "prefill_weights": "fp8"},
    "fast": {"decode_weights": "w4a8", "expert_weights": "fp8", "prefill_weights": "nvfp4", "lm_head_weights": "int4"},
}
TIER_OPTIONS = ("decode_weights", "expert_weights", "prefill_weights", "vision_weights", "lm_head_weights")

TIER_DEFAULTS = {"decode_weights": "bf16", "expert_weights": "bf16", "prefill_weights": "bf16", "vision_weights": "bf16",
                 "lm_head_weights": "decode"}


def resolve_preset(preset: str, **explicit) -> dict:
    """The weight options (``TIER_OPTIONS``) of ``preset`` with the explicitly given ones (not ``None``) on top."""
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {tuple(PRESETS)}")
    unknown = set(explicit) - set(TIER_OPTIONS)
    if unknown:
        raise ValueError(f"not preset options: {sorted(unknown)}")
    opts = dict(TIER_DEFAULTS)
    opts.update(PRESETS[preset])
    opts.update({name: value for name, value in explicit.items() if value is not None})
    return opts


def require_alpamayo15_kernels(fvk: object) -> None:
    """Fail fast (before any model setup) when the Thor kernel group is missing."""
    missing = [name for name in ALPAMAYO15_KERNEL_SYMBOLS if not hasattr(fvk, name)]
    if missing:
        raise RuntimeError(
            "FlashRT was built without the Alpamayo 1.5 kernels "
            f"({', '.join(missing)}). Rebuild with -DGPU_ARCH=110 -DFLASHRT_ENABLE_ALPAMAYO15=ON.")


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


_SUPPORTED_SAMPLING = {"repetition_penalty": 1.0, "typical_p": 1.0, "epsilon_cutoff": 0.0, "eta_cutoff": 0.0,
                       "no_repeat_ngram_size": 0, "encoder_repetition_penalty": 1.0, "renormalize_logits": False}


def check_generation_config(gc) -> None:
    """The decode graph implements mask -> temperature -> top-p; reject configs that add other HF processors.

    Fields the official rollout sets on every call (top_p, temperature, top_k,
    do_sample, max_new_tokens, ...) are not inspected.
    """
    bad = [k for k, v in _SUPPORTED_SAMPLING.items() if getattr(gc, k, v) not in (v, None)]
    bad += [k for k in ("min_p", "bad_words_ids", "suppress_tokens", "begin_suppress_tokens",
                        "sequence_bias", "forced_bos_token_id", "forced_eos_token_id", "exponential_decay_length_penalty",
                        "guidance_scale", "watermarking_config") if getattr(gc, k, None)]
    if getattr(gc, "min_new_tokens", None) or getattr(gc, "min_length", 0):
        bad.append("min_new_tokens/min_length")
    if bad:
        raise ValueError(f"Alpamayo15Thor does not implement these generation settings: {sorted(bad)}")


def _autocast():
    return torch.autocast("cuda", dtype=BF16)


def torch_philox_blocks() -> int:
    """torch's launch cap of its CUDA distribution kernels on the current device: SMs times blocks of 256 threads per
    SM (the sampler kernel reproduces torch's per-element Philox words, which depend on this cap)."""
    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    return int(p.multi_processor_count) * (int(p.max_threads_per_multi_processor) // 256)


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


def _capture(fn, warmups: int = 2) -> torch.cuda.CUDAGraph:
    """Warm ``fn`` up on a side stream, then capture it (BF16 autocast, no autograd). The autocast cast cache is
    disabled: a cached cast would be freed when the caller's autocast region ends while the graph keeps its address."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.no_grad(), torch.autocast("cuda", dtype=BF16, cache_enabled=False):
        for _ in range(warmups):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    g = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.autocast("cuda", dtype=BF16, cache_enabled=False), torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


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


class Alpamayo15Thor:
    """Static-buffer CUDA-graph engine for one official Alpamayo 1.5 model.

    Args:
        model: official ``Alpamayo1_5`` on CUDA in BF16, eval mode.
        max_prompt: largest supported prompt length in tokens.
        max_new_tokens: generation budget (official ``max_generation_length``).
        max_samples: largest ``num_traj_samples`` served (cache is sized for it).
        top_p, temperature: sampling parameters, baked into the decode graphs.
        decode_splits, expert_splits: FA4 KV splits for decode and expert attention
            (one split streams the prompt cache fastest on Thor: 54 against 85 us per
            layer with four).
        preset: ``"balanced"`` (default: ``decode_weights="w4a8"``,
            ``expert_weights="fp8"``, ``prefill_weights="fp8"``; needs
            ``paro_checkpoint``), ``"bf16"`` (the official numerics up to kernel
            rounding, no extra weights) or ``"fast"`` (``"balanced"`` with
            ``prefill_weights="nvfp4"`` and ``lm_head_weights="int4"``). The
            weight options below default to the preset's and override it when
            given.
        decode_weights: ``"bf16"`` (official weights), ``"fp8"`` or ``"int8"``
            (opt-in per-row copies for the reasoning-decode projections and the
            LM head; about +7.6 GB, changes decode numerics - see module notes),
            or ``"w4a8"`` (the ParoQuant W4A8 linears of ``paro_checkpoint`` for
            the projections, INT8 for the LM head; about +4.9 GB).
        paro_checkpoint: directory of the ``z-lab/Alpamayo-1.5-10B-PARO``
            safetensors; required by ``decode_weights="w4a8"`` (the
            ``"balanced"`` and ``"fast"`` presets) and refused otherwise.
        pdl: launch the engine's kernels and GEMMs with programmatic dependent
            launch (a kernel's prologue overlaps its predecessor's tail inside
            the CUDA graphs). Process-wide switch shared with the FlashDrive
            backend; off by default.
        expert_weights: ``"bf16"`` (official weights) or ``"fp8"`` (opt-in W8A8
            expert projections; about +4.6 GB, changes expert numerics).
        prefill_weights: ``"bf16"`` (official weights on cuBLAS), ``"fp8"``
            (opt-in W8A8 prefill projections on the FP8 tcgen05 GEMM; about
            +7.6 GB, changes the prompt K/V and therefore every later token) or
            ``"nvfp4"`` (W4A4: e2m1 weights and activations with UE4M3 block-16
            scales, per-tensor weight scales and per-row activation scales
            written by the fused producers, on the 2-SM NVFP4 tcgen05 GEMM,
            about twice the FP8 rate; needs the ``flash_rt_fp4`` module; about
            +3.8 GB).
        vision_weights: ``"bf16"`` or ``"fp8"`` (opt-in W8A8 vision blocks: qkv,
            proj, fc1 and fc2 with per-row FP8 activations, bias / GELU /
            residual in the epilogue; patch embedding and mergers stay BF16;
            about +0.45 GB, changes the image features).
        lm_head_weights: ``"decode"`` (the LM head in the decode tier's format:
            BF16, FP8, INT8, or INT8 with W4A8) or ``"int4"`` (own INT4 linear,
            group 128 with zero points, per-token INT8 activations; the logits
            weight stream halves: 2.6 -> about 1.4 ms per token; changes the
            sampled tokens).
    """

    def __init__(
        self,
        model,
        *,
        max_prompt: int = 4096,
        max_new_tokens: int = 256,
        max_samples: int = 1,
        top_p: float = 0.98,
        temperature: float = 0.6,
        decode_splits: int = 1,
        expert_splits: int = 1,
        preset: str = "balanced",
        decode_weights: str | None = None,
        expert_weights: str | None = None,
        prefill_weights: str | None = None,
        vision_weights: str | None = None,
        lm_head_weights: str | None = None,
        paro_checkpoint: str | None = None,
        pdl: bool = False,
    ) -> None:
        tiers = resolve_preset(preset, decode_weights=decode_weights, expert_weights=expert_weights,
                               prefill_weights=prefill_weights, vision_weights=vision_weights, lm_head_weights=lm_head_weights)
        decode_weights, expert_weights = tiers["decode_weights"], tiers["expert_weights"]
        prefill_weights, vision_weights, lm_head_weights = tiers["prefill_weights"], tiers["vision_weights"], tiers["lm_head_weights"]
        if decode_weights not in DECODE_WEIGHTS:
            raise ValueError(f"unknown decode_weights {decode_weights!r}; choose from {DECODE_WEIGHTS}")
        if decode_weights != "bf16" and max_samples > 16:
            raise ValueError(f"decode_weights={decode_weights!r} supports max_samples <= 16 (the few-row kernels serve up to 16 rows)")
        if expert_weights not in EXPERT_WEIGHTS:
            raise ValueError(f"unknown expert_weights {expert_weights!r}; choose from {EXPERT_WEIGHTS}")
        if prefill_weights not in PREFILL_WEIGHTS:
            raise ValueError(f"unknown prefill_weights {prefill_weights!r}; choose from {PREFILL_WEIGHTS}")
        if vision_weights not in VISION_WEIGHTS:
            raise ValueError(f"unknown vision_weights {vision_weights!r}; choose from {VISION_WEIGHTS}")
        if lm_head_weights not in LM_HEAD_WEIGHTS:
            raise ValueError(f"unknown lm_head_weights {lm_head_weights!r}; choose from {LM_HEAD_WEIGHTS}")
        if lm_head_weights == "int4" and max_samples > 16:
            raise ValueError("lm_head_weights='int4' supports max_samples <= 16 (the few-row kernels serve up to 16 rows)")
        if (decode_weights == "w4a8") != (paro_checkpoint is not None):
            raise ValueError("decode_weights='w4a8' (the presets 'balanced' and 'fast') takes the ParoQuant checkpoint directory "
                             "as paro_checkpoint, and only then; preset='bf16' runs on the official weights alone")
        if max_samples < 1:
            raise ValueError("max_samples must be >= 1")
        import flash_rt.flash_rt_kernels as fvk
        require_alpamayo15_kernels(fvk)
        if model.training:
            raise ValueError("Alpamayo15Thor requires an eval-mode model")
        from flash_rt.hardware.thor import fa4_backend
        self._fa4 = fa4_backend.fa4_fwd()
        if self._fa4 is None:
            raise RuntimeError(f"FA4 unavailable: {fa4_backend.status()}")
        self._fvk = fvk
        self.m = model
        self.vlm = model.vlm
        self.lm = model.vlm.model.language_model
        self.expert = model.expert
        cfg = model.vlm.config.text_config
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.hidden = cfg.hidden_size
        self.ffn = cfg.intermediate_size
        self.eps = cfg.rms_norm_eps
        ecfg = model.expert.config
        if (ecfg.num_hidden_layers != self.n_layers or ecfg.num_key_value_heads != self.n_kv
                or ecfg.head_dim != self.head_dim):
            raise ValueError("expert attention geometry must match the VLM cache")
        if self.head_dim != 128:
            raise ValueError(f"Alpamayo 1.5 kernels require head_dim 128, got {self.head_dim}")
        check_generation_config(model.vlm.generation_config)
        self.e_heads = ecfg.num_attention_heads
        self.e_hidden = ecfg.hidden_size
        self.e_eps = ecfg.rms_norm_eps
        self.n_act, self.act_dim = model.action_space.get_action_space_dims()
        self.max_prompt = int(max_prompt)
        self.max_new = int(max_new_tokens)
        self.max_samples = int(max_samples)
        self.top_p = float(top_p)
        self.temperature = float(temperature)
        self.decode_splits = int(decode_splits)
        self.expert_splits = int(expert_splits)
        self.preset = preset
        self.decode_weights = decode_weights
        self.expert_weights, self.prefill_weights, self.vision_weights = expert_weights, prefill_weights, vision_weights
        self.lm_head_weights = lm_head_weights
        self._gemm = _Gemm(fvk)
        self.pdl = bool(pdl)
        fvk.alpamayo15_set_pdl(self.pdl)
        self.max_seq = self.max_prompt + self.max_new + self.n_act
        self.scale = self.head_dim ** -0.5

        tok = model.tokenizer
        self.traj_start_id = tok.convert_tokens_to_ids("<|traj_future_start|>")
        gen_eos = model.vlm.generation_config.eos_token_id
        self.eos_ids = set([gen_eos] if isinstance(gen_eos, int) else (gen_eos or []))
        self.pad_id = tok.pad_token_id

        B, L, S, KV, D = self.max_samples, self.n_layers, self.max_seq, self.n_kv, self.head_dim
        self.k_cache = torch.zeros(L, B, S, KV, D, device=DEV, dtype=BF16)
        self.v_cache = torch.zeros(L, B, S, KV, D, device=DEV, dtype=BF16)

        # Decode state: device scalars advanced inside the graph (all samples
        # sit at the same cache position), per-sample token / length / unfinished.
        self.d_tok = torch.zeros(B, dtype=torch.long, device=DEV)
        self.d_unfinished = torch.ones(B, dtype=torch.long, device=DEV)
        self.d_slot = torch.zeros(1, dtype=torch.long, device=DEV)     # cache row to write
        self.d_len = torch.zeros(B, dtype=torch.int32, device=DEV)     # attended prompt+generated rows
        self.d_glen = torch.zeros(B, dtype=torch.int32, device=DEV)    # attended generated rows
        self.d_plen = torch.zeros(1, dtype=torch.int32, device=DEV)    # prompt rows (sample 0)
        self.d_cuk = torch.zeros(B + 1, dtype=torch.int32, device=DEV) # per-sample suffix starts
        self.d_rpos = torch.zeros(B, dtype=torch.long, device=DEV)     # rope position per sample (all equal)
        # Rotary tables: the kernel computes the official module's fp32 products inv_freq * position with the
        # interleaved M-RoPE sections and the BF16 cast; the decode table covers every reasoning position.
        rot = self.lm.rotary_emb
        self.inv_freq = rot.inv_freq.detach().float().contiguous()
        self.mrope_sec = [int(v) for v in rot.mrope_section]
        self.rope_scaling = float(rot.attention_scaling)
        self.image_token_id = int(model.vlm.config.image_token_id)
        self.vision_start_id = int(model.vlm.config.vision_start_token_id)
        self.spatial_merge = int(model.vlm.config.vision_config.spatial_merge_size)
        max_rope = self.max_prompt + self.max_new + 8
        self.cos_table = torch.empty(max_rope, D, device=DEV, dtype=BF16)
        self.sin_table = torch.empty(max_rope, D, device=DEV, dtype=BF16)
        stream = torch.cuda.current_stream().cuda_stream
        pos3 = torch.arange(max_rope, device=DEV, dtype=torch.int32).repeat(3, 1).contiguous()
        fvk.alpamayo15_mrope_tables(pos3.data_ptr(), self.inv_freq.data_ptr(), self.mrope_sec[1], self.mrope_sec[2], self.rope_scaling,
                                    self.cos_table.data_ptr(), self.sin_table.data_ptr(), max_rope, D, 0, stream)
        # Sampler: the official ExpertLogitsProcessor's trajectory-token mask, temperature and top-p are kernel
        # arguments; the seed and the draw counter live on the device ([seed, counter, ticket]).
        self.mask_lo, self.mask_n = int(model.config.traj_token_start_idx), int(model.config.traj_vocab_size)
        self.rng = torch.zeros(3, dtype=torch.int64, device=DEV)
        self._rng_blocks = torch_philox_blocks()
        vocab = int(model.vlm.lm_head.weight.shape[0])
        self.samp_scratch = torch.empty(B, vocab, device=DEV, dtype=torch.float32)      # exp(score - max) per row
        self.samp_ws = torch.zeros(int(fvk.alpamayo15_sample_workspace_bytes(B)), device=DEV, dtype=torch.uint8)
        self._decode_graphs: dict[int, torch.cuda.CUDAGraph] = {}

        # Expert state (per sample).
        self.x = torch.zeros(B, self.n_act, self.act_dim, device=DEV, dtype=torch.float32)
        self.e_cos = torch.zeros(B, self.n_act, self.head_dim, device=DEV, dtype=torch.float32)
        self.e_sin = torch.zeros(B, self.n_act, self.head_dim, device=DEV, dtype=torch.float32)
        self.d_eseq = torch.zeros(B, dtype=torch.int32, device=DEV)    # single-sample expert key rows
        self.d_esuf = torch.zeros(B, dtype=torch.int32, device=DEV)    # per-sample suffix key rows
        self.d_aslot = torch.zeros(B, dtype=torch.long, device=DEV)    # action-row start per sample
        self._cuq = {}
        self.n_steps = model.diffusion.num_inference_steps
        self.time_steps = torch.linspace(0.0, 1.0, self.n_steps + 1, device=DEV)
        ts = np.array(self.time_steps.cpu().tolist(), dtype=np.float32)
        self.step_dt = [float(ts[i + 1] - ts[i]) for i in range(self.n_steps)]       # fp32, as the official step
        self._expert_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.a_v0 = torch.zeros(B, device=DEV, dtype=torch.float32)                 # t0 velocity estimate per sample
        self.a_z = torch.zeros(B, device=DEV, dtype=torch.float32)                  # the history's last z
        self.pred_xyz = torch.zeros(B, self.n_act, 3, device=DEV, dtype=torch.float32)
        self.pred_rot = torch.zeros(B, self.n_act, 3, 3, device=DEV, dtype=torch.float32)
        self.act = pack_action_modules(model, self.act_dim, self.n_act)
        self.w = prepare_weights(model, fvk, tiers, max_samples=self.max_samples, paro_checkpoint=paro_checkpoint)
        self._pf = None
        self._db = None
        self._eb = None
        self._vis_tables: dict = {}
        self._vis_graphs: dict = {}
        self._check_vision_modules()
        self._mark = None

    _W4_NEXT = {"qkv": "o", "o": "gu", "gu": "down", "down": None}

    def _proj(self, x: torch.Tensor, w: torch.Tensor, key: str, li: int, out: torch.Tensor, stream: int) -> None:
        """``out = x @ w.T`` from the BF16 weight or, in the FP8 / INT8 / W4A8 / INT4 tiers, its quantized form."""
        if li < 0 and self.w.lm4 is not None:
            self.w.lm4(x, out, stream)
            return
        if self.w.q8 is None:                     # BF16 weights read in place by the few-row tensor-core GEMM
            self._fvk.flashdrive_bf16_gemm_small(x.data_ptr(), x.stride(0), w.data_ptr(), out.data_ptr(), out.stride(0), 0,
                                                 x.shape[0], w.shape[0], w.shape[1], stream)
            return
        if self.w.w4 is not None and li >= 0:
            # the GEMM prefetches the rotation tables of the group that runs next (o, gate|up, down, next layer's q|k|v)
            nxt = self._W4_NEXT[key]
            nxt_group = self.w.w4[li][nxt] if nxt else (self.w.w4[li + 1]["qkv"] if li + 1 < self.n_layers else None)
            self.w.w4[li][key](x, out, stream, nxt_group.tables_region if nxt_group is not None else None)
            return
        frag, scale = self.w.q8[key][li] if li >= 0 else self.w.q8[key]
        if not (x.is_contiguous() and out.is_contiguous() and x.shape[0] == out.shape[0]):
            raise ValueError("quantized projection needs contiguous row-major x [n, K] and out [n, N]")
        self._fvk.alpamayo15_w8a16_gemm_bf16(frag.data_ptr(), scale.data_ptr(), x.data_ptr(), out.data_ptr(),
                                             x.shape[0], out.shape[1], x.shape[1], stream, self.w.q8["fmt"])

    # ------------------------------------------------------------------
    # vision
    # ------------------------------------------------------------------
    def _vision_tables(self, grid: torch.Tensor):
        """Per-grid position embeddings and rotary tables (official functions), cached."""
        key = tuple(grid.flatten().tolist())
        tab = self._vis_tables.pop(key, None)
        if tab is None:
            V = self.vlm.model.visual
            with _autocast():
                pos = V.fast_pos_embed_interpolate(grid)
            rot = V.rot_pos_emb(grid)
            emb = torch.cat((rot, rot), dim=-1)
            tab = (pos.to(BF16), emb.cos().float().contiguous(), emb.sin().float().contiguous())
            if len(self._vis_tables) >= 4:
                self._vis_tables.pop(next(iter(self._vis_tables)))
        self._vis_tables[key] = tab        # most recently used last
        return tab, key

    def _check_vision_modules(self) -> None:
        """The CUTLASS vision path assumes the Qwen3-VL tower layout of Alpamayo 1.5 (bias linears, LayerNorms,
        GELU-tanh block MLP, erf-GELU mergers, equal frame grids)."""
        V = self.vlm.model.visual
        blk = V.blocks[0]
        if any(getattr(lin, "bias", None) is None for lin in (V.patch_embed.proj, blk.attn.qkv, blk.attn.proj,
                                                               blk.mlp.linear_fc1, blk.mlp.linear_fc2, V.merger.linear_fc1,
                                                               V.merger.linear_fc2)):
            raise ValueError("the CUTLASS vision path expects bias linears in the Qwen3-VL vision tower")
        if type(blk.mlp.act_fn).__name__ not in ("PytorchGELUTanh", "GELUTanh") and getattr(blk.mlp.act_fn, "approximate", "") != "tanh":
            raise ValueError(f"unexpected vision MLP activation {type(blk.mlp.act_fn).__name__}")
        if getattr(V.merger.act_fn, "approximate", "none") != "none":
            raise ValueError("the vision merger is expected to use the erf GELU")
        if V.merger.use_postshuffle_norm or not all(m.use_postshuffle_norm for m in V.deepstack_merger_list):
            raise ValueError("unexpected merger normalization placement")

    def _merger(self, mod, x: torch.Tensor, rows: int, D_in: int) -> torch.Tensor:
        """Qwen3VLVisionPatchMerger on FlashRT kernels: LayerNorm (pre- or post-shuffle), fc1 + erf GELU, fc2."""
        fvk, g = self._fvk, self._gemm
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

    def _vision_cutlass(self, pixel_values: torch.Tensor, grid: torch.Tensor, px16: torch.Tensor | None = None):
        """The Qwen3-VL vision tower on FlashRT kernels: patch embedding (+ position embedding) and every block's
        LayerNorm, qkv / proj / fc1 / fc2 (bias, GELU and residual fused into the GEMM epilogues), FP32 RoPE and
        batched FA4 attention; mergers as ``_merger``. Same operation order as ``_vision``; GEMM reduction order and
        the fused epilogues (one rounding of bias + activation) differ from the cuBLAS + torch path."""
        V = self.vlm.model.visual
        fvk, g = self._fvk, self._gemm
        (pos, cos, sin), key = self._vision_tables(grid)
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
        v8 = getattr(self, "_v8", None)
        if v8 is not None:              # W8A8 blocks: FP8 rows of the LayerNorm outputs, the attention output and GELU(fc1)
            u8 = lambda *shape: torch.empty(*shape, device=DEV, dtype=torch.uint8)
            f32 = lambda n: torch.empty(n, device=DEV, dtype=torch.float32)
            h8, h_s, o8, o_s, f18, f1_s = u8(S, Din), f32(S), u8(S, Din), f32(S), u8(S, Ffc), f32(S)
        blk_gemm = lambda A, W, D, bias=None, C=None, act=0: g(A, W, D, bias=bias, C=C, act=act, tile=T)
        fc1_gemm = lambda A, W, D, bias: g(A, W, D, bias=bias, act=1, tile=T)
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
            o, _ = self._fa4(*(z.view(n_img, L, nh, hd) for z in (q, k, v)), softmax_scale=a.scaling, causal=False)
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
                deep.append(self._merger(V.deepstack_merger_list[V.deepstack_visual_indexes.index(li)], x, S, Din))
        return self._merger(V.merger, x, S, Din), deep

    def _encode(self, pixel_values: torch.Tensor, grid: torch.Tensor):
        """(image embeddings, DeepStack features) of the request's frames: the CUTLASS tower as one CUDA graph per
        (pixel shape, frame grid) when its conditions hold, otherwise the torch tower."""
        key_grid = tuple(grid.flatten().tolist())
        n_img = len(key_grid) // 3
        if not all(key_grid[3 * i:3 * i + 3] == key_grid[0:3] for i in range(n_img)):
            return self._encode_groups(pixel_values, grid, key_grid)
        key = (tuple(pixel_values.shape), key_grid)
        ent = self._vis_graphs.get(key)
        if ent is None:
            if len(self._vis_graphs) >= 2:
                self._vis_graphs.pop(next(iter(self._vis_graphs)))
            px = torch.empty_like(pixel_values)
            px.copy_(pixel_values)
            px16 = torch.empty(pixel_values.shape, device=DEV, dtype=BF16) if pixel_values.dtype != BF16 else None
            with _autocast():
                self._vision_cutlass(px, grid, px16)       # builds the per-grid tables outside capture
            grid_host = grid.cpu()                         # table lookup key only: no device sync while capturing
            tables = self._vis_tables[key_grid]            # captured by address: keep alive with the graph
            out = {}

            def run():
                out["img"], out["deep"] = self._vision_cutlass(px, grid_host, px16)
            ent = {"px": px, "px16": px16, "out": out, "tables": tables, "graph": _capture(run)}
            self._vis_graphs[key] = ent
        ent["px"].copy_(pixel_values)
        ent["graph"].replay()
        return ent["out"]["img"], ent["out"]["deep"]

    def _encode_groups(self, pixel_values: torch.Tensor, grid: torch.Tensor, key_grid: tuple):
        """Frames of different grids: each group of equal grids through its own tower graph, the features put back
        in frame order (the attention is batched per group, so this costs one graph replay per distinct grid)."""
        fvk = self._fvk
        s = torch.cuda.current_stream().cuda_stream
        n_img = len(key_grid) // 3
        merge = self.spatial_merge ** 2
        patches = [key_grid[3 * i] * key_grid[3 * i + 1] * key_grid[3 * i + 2] for i in range(n_img)]
        starts = np.concatenate([[0], np.cumsum(patches)])
        groups: dict[tuple, list[int]] = {}
        for i in range(n_img):
            groups.setdefault(key_grid[3 * i:3 * i + 3], []).append(i)
        total = int(starts[-1]) // merge
        img = torch.empty(total, self.hidden, device=DEV, dtype=BF16)
        deep = [torch.empty(total, self.hidden, device=DEV, dtype=BF16) for _ in self.vlm.model.visual.deepstack_merger_list]
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
                                             self.hidden, 0, s)
            for dst, src in zip(deep, gd):
                fvk.alpamayo15_scatter_rows_bf16(dst.data_ptr(), dst.stride(0), out_rows.data_ptr(), src.data_ptr(), src.stride(0),
                                                 src.shape[0], self.hidden, 0, s)
        return img, deep

    # ------------------------------------------------------------------
    # prefill (writes sample 0 of the cache)
    # ------------------------------------------------------------------
    def _prefill_buffers(self, P: int) -> dict:
        if self._pf is None or self._pf["P"] < P:
            n = (max(P, 1024) + 127) // 128 * 128
            H, Hq, Hkv, F_ = self.hidden, self.n_heads * self.head_dim, self.n_kv * self.head_dim, self.ffn
            mk = lambda *shape: torch.empty(*shape, device=DEV, dtype=BF16)
            self._pf = {"P": n, "x": mk(n, H), "h": mk(n, H), "qkv": mk(n, Hq + 2 * Hkv), "q": mk(n, Hq),
                        "attn": mk(n, Hq), "o": mk(n, H), "g": mk(n, F_), "u": mk(n, F_), "d": mk(n, H),
                        "cos": mk(n, self.head_dim), "sin": mk(n, self.head_dim), "last": torch.zeros(16, H, device=DEV, dtype=BF16),
                        "logits1": mk(1, self.vlm.lm_head.weight.shape[0])}
            if self.w.p8 is not None:       # W8A8 tier: FP8 rows + per-row scales of the three GEMM inputs
                for name, width in (("h", H), ("attn", Hq), ("g", F_)):
                    self._pf[name + "8"] = torch.empty(n, width, device=DEV, dtype=torch.uint8)
                    self._pf[name + "_s"] = torch.empty(n, device=DEV, dtype=torch.float32)
            if self.w.p4 is not None:       # W4A4 tier: NVFP4 rows, their scale layout and per-row GEMM alphas [consumer, row]
                f4 = self.w.p4["module"]
                for name, width in (("h", H), ("attn", Hq), ("g", F_)):
                    self._pf[name + "4"] = torch.empty(n, width // 2, device=DEV, dtype=torch.uint8)
                    self._pf[name + "_sf"] = torch.zeros(int(f4.alpamayo15_nvfp4_sf_bytes(n, width)), device=DEV, dtype=torch.uint8)
                    self._pf[name + "_alpha"] = torch.empty(2, n, device=DEV, dtype=torch.float32)
        return self._pf

    def _prefill_native(self, input_ids, tokenized, vis_idx: torch.Tensor, pos3: torch.Tensor):
        """Vision + 36-layer LM prefill on fused FlashRT kernels.

        K/V are normalized, rotated (interleaved M-RoPE) and written straight
        into the static cache; only the last row is normalized for the logits.
        """
        vm, lm, fvk = self.vlm.model, self.lm, self._fvk
        s = torch.cuda.current_stream().cuda_stream
        P = input_ids.shape[1]
        H, D, NH, NKV, F_ = self.hidden, self.head_dim, self.n_heads, self.n_kv, self.ffn
        b = self._prefill_buffers(P)
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
        fvk.alpamayo15_mrope_tables(pos3.data_ptr(), self.inv_freq.data_ptr(), self.mrope_sec[1], self.mrope_sec[2], self.rope_scaling,
                                    cos_f.data_ptr(), sin_f.data_ptr(), P, D, 0, s)
        layers = lm.layers
        p8, p4 = self.w.p8, self.w.p4
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
                check(f4.alpamayo15_nvfp4_rms_rows_bf16(x.data_ptr(), dp, w.data_ptr(), P, H, self.eps, b["h4"].data_ptr(),
                                                        b["h_sf"].data_ptr(), p4["layers"][li]["gsw_" + first].data_ptr(),
                                                        len(NVFP4_CONSUMERS[first]), b["h_alpha"].data_ptr(), b["P"], s), "RMSNorm")
            elif p8 is None:
                fvk.alpamayo15_rms_bf16(x.data_ptr(), dp, w.data_ptr(), h.data_ptr(), P, H, self.eps, s)
            else:
                fvk.alpamayo15_rms_fp8_rows_bf16(x.data_ptr(), dp, w.data_ptr(), b["h8"].data_ptr(), b["h_s"].data_ptr(), P, H,
                                                 self.eps, s)

        def lin(src, akey, w, wname, li, out, col0=0, quantized=False):
            """``out = src @ w.T``: cuBLAS on the BF16 weight ``w``, or (W8A8 tier) per-row FP8 rows of ``src``
            (buffers ``akey``, written by the producer when ``quantized``) times rows ``col0:`` of the FP8 copy
            ``wname`` of layer ``li``; the W4A4 tier always reads the producer's NVFP4 rows of ``akey``."""
            if p4 is not None:
                nvfp4_lin(akey, wname, li, out, col0)
                return
            if p8 is None:                               # BF16 weights on the 2-SM 256 x 256 tcgen05 tile
                self._gemm(src, w, out, tile=3)
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
                    self._gemm.fp8(q8, q_s, w8[j * Nb:(j + 1) * Nb], w_s[j * Nb:(j + 1) * Nb], out[:, j * Nb:(j + 1) * Nb],
                                   tile=PREFILL_FP8_TILE)
            else:
                self._gemm.fp8(q8, q_s, w8[col0:col0 + N], w_s[col0:col0 + N], out, tile=PREFILL_FP8_TILE)

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
            lin(h, "h", self.w.qkv[li], "qkv", li, qkv, quantized=True)
            kc, vc = self.k_cache[li, 0, :P], self.v_cache[li, 0, :P]
            fvk.alpamayo15_vlm_qk_prep_bf16(
                qkv.data_ptr(), a.q_norm.weight.data_ptr(), a.k_norm.weight.data_ptr(),
                cos_f.data_ptr(), sin_f.data_ptr(), q.data_ptr(), kc.data_ptr(), vc.data_ptr(), 0,
                P, NH, NKV, NKV * D, P, 0, self.eps, s)
            self._fa4(q.view(1, P, NH, D), kc.unsqueeze(0), vc.unsqueeze(0), softmax_scale=self.scale,
                      causal=True, pack_gqa=True, out=attn.view(1, P, NH, D))
            if p4 is not None:
                check(f4.alpamayo15_nvfp4_quantize_rows_rowscale_bf16(attn.data_ptr(), attn.stride(0), P, NH * D, b["attn4"].data_ptr(),
                                                                      b["attn_sf"].data_ptr(), p4["layers"][li]["gsw_o"].data_ptr(), 1,
                                                                      b["attn_alpha"].data_ptr(), b["P"], 0, s), "attention rows")
            lin(attn, "attn", a.o_proj.weight, "o", li, o)
            norm(o, layer.post_attention_layernorm.weight, (li, "gate"))
            lin(h, "h", self.w.gu[li][:F_], "gu", li, g, quantized=True)   # cuBLAS is slower on the fused gate | up shape: two GEMMs
            lin(h, "h", self.w.gu[li][F_:], "gu", li, u, col0=F_, quantized=True)
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
        fvk.alpamayo15_rms_bf16(x[P - 1:P].data_ptr(), 0, lm.norm.weight.data_ptr(), last.data_ptr(), 1, H, self.eps, s)
        fvk.flashdrive_bf16_gemm_small(last.data_ptr(), H, self.vlm.lm_head.weight.data_ptr(), logits.data_ptr(), logits.shape[1], 0, 1,
                                       logits.shape[1], H, s)
        return logits

    # ------------------------------------------------------------------
    # decode
    # ------------------------------------------------------------------
    def _decode_buffers(self):
        if self._db is None:
            B, H, D, F_ = max(self.max_samples, 16), self.hidden, self.head_dim, self.ffn
            mk = lambda *shape: torch.zeros(*shape, device=DEV, dtype=BF16)
            self._db = {"x": mk(B, H), "h": mk(B, H), "qkv": mk(B, (self.n_heads + 2 * self.n_kv) * D),
                        "q": mk(B, self.n_heads * D), "o": mk(B, H), "gu": mk(B, 2 * F_), "a": mk(B, F_),
                        "d": mk(B, H), "cos": mk(B, D), "sin": mk(B, D),
                        "logits": mk(B, self.vlm.lm_head.weight.shape[0])}
        return self._db

    def _cu_q(self, n: int, rows: int) -> torch.Tensor:
        key = (n, rows)
        if key not in self._cuq:
            self._cuq[key] = (torch.arange(n + 1, device=DEV, dtype=torch.int32) * rows).contiguous()
        return self._cuq[key]

    def _attend(self, li: int, q: torch.Tensor, n: int, rows: int, single_len: torch.Tensor,
                suffix_len: torch.Tensor, splits: int) -> torch.Tensor:
        """Non-causal attention of ``q`` [n * rows, heads, D] over the cache.

        One sample: a single pass over rows [0, single_len). Several samples:
        the shared prompt (sample 0, rows [0, P)) once for all queries plus each
        sample's suffix (rows [P, P + suffix_len)), merged by log-sum-exp.
        """
        kc, vc = self.k_cache[li], self.v_cache[li]
        heads, D = q.shape[1], q.shape[2]
        if n == 1:
            out, _ = self._fa4(q.view(1, rows, heads, D), kc[:1], vc[:1], seqused_k=single_len[:1],
                               softmax_scale=self.scale, causal=False, pack_gqa=True, num_splits=splits)
            return out.view(rows, -1)
        o1, l1 = self._fa4(q.view(1, n * rows, heads, D), kc[:1], vc[:1], seqused_k=self.d_plen,
                           softmax_scale=self.scale, causal=False, pack_gqa=True, return_lse=True)
        S, KV = self.max_seq, self.n_kv
        o2, l2 = self._fa4(q, kc[:n].view(n * S, KV, D), vc[:n].view(n * S, KV, D),
                           cu_seqlens_q=self._cu_q(n, rows), cu_seqlens_k=self.d_cuk[:n + 1],
                           seqused_k=suffix_len[:n], max_seqlen_q=rows, max_seqlen_k=self.max_new + self.n_act,
                           softmax_scale=self.scale, causal=False, pack_gqa=True, return_lse=True)
        return merge_attention(o1.view(n * rows, heads, D), l1.view(heads, n * rows),
                               o2, l2.view(heads, n * rows)).view(n * rows, -1)

    def _decode_step(self, n: int) -> None:
        """One reasoning token for ``n`` samples: 36 layers (BF16, or FP8 projection weights) + on-device sampling."""
        lm, fvk = self.lm, self._fvk
        s = torch.cuda.current_stream().cuda_stream
        H, D, NH, NKV, F_, S = self.hidden, self.head_dim, self.n_heads, self.n_kv, self.ffn, self.max_seq
        b = self._decode_buffers()
        x, h, qkv, q, o, gu, act, d, logits = (b[k][:n] for k in ("x", "h", "qkv", "q", "o", "gu", "a", "d", "logits"))
        fvk.alpamayo15_gather_rows_bf16(lm.embed_tokens.weight.data_ptr(), H, self.d_tok.data_ptr(), x.data_ptr(), H, n, H, s)
        cos, sin = b["cos"][:n], b["sin"][:n]
        fvk.alpamayo15_gather_rows_bf16(self.cos_table.data_ptr(), D, self.d_rpos.data_ptr(), cos.data_ptr(), D, n, D, s)
        fvk.alpamayo15_gather_rows_bf16(self.sin_table.data_ptr(), D, self.d_rpos.data_ptr(), sin.data_ptr(), D, n, D, s)
        layers = lm.layers
        fvk.alpamayo15_rms_bf16(x.data_ptr(), 0, layers[0].input_layernorm.weight.data_ptr(), h.data_ptr(), n, H, self.eps, s)
        for li, layer in enumerate(layers):
            a = layer.self_attn
            self._proj(h, self.w.qkv[li], "qkv", li, qkv, s)
            fvk.alpamayo15_vlm_qk_prep_bf16(
                qkv.data_ptr(), a.q_norm.weight.data_ptr(), a.k_norm.weight.data_ptr(),
                cos.data_ptr(), sin.data_ptr(), q.data_ptr(),
                self.k_cache[li].data_ptr(), self.v_cache[li].data_ptr(), self.d_slot.data_ptr(),
                n, NH, NKV, NKV * D, 1, S, self.eps, s)
            attn = self._attend(li, q.view(n, NH, D), n, 1, self.d_len, self.d_glen, self.decode_splits)
            self._proj(attn, a.o_proj.weight, "o", li, o, s)
            fvk.alpamayo15_rms_bf16(x.data_ptr(), o.data_ptr(), layer.post_attention_layernorm.weight.data_ptr(),
                                    h.data_ptr(), n, H, self.eps, s)
            self._proj(h, self.w.gu[li], "gu", li, gu, s)
            fvk.alpamayo15_silu_mul_split_bf16(gu.data_ptr(), gu[:, F_:].data_ptr(), 2 * F_, act.data_ptr(), n, F_, s)
            self._proj(act, layer.mlp.down_proj.weight, "down", li, d, s)
            nxt = layers[li + 1].input_layernorm.weight if li + 1 < len(layers) else lm.norm.weight
            fvk.alpamayo15_rms_bf16(x.data_ptr(), d.data_ptr(), nxt.data_ptr(), h.data_ptr(), n, H, self.eps, s)
        self._proj(h, self.vlm.lm_head.weight, "lm_head", -1, logits, s)
        # the official sampler (trajectory-token mask, temperature, top-p, multinomial) and the step's bookkeeping
        fvk.alpamayo15_sample_tokens_bf16(logits.data_ptr(), logits.stride(0), n, logits.shape[1], self.mask_lo, self.mask_n,
                                          self.temperature, self.top_p, 0, self.rng.data_ptr(), self._rng_blocks, self.samp_scratch.data_ptr(),
                                          self.samp_ws.data_ptr(), self.samp_ws.numel(), self.d_tok.data_ptr(),
                                          self.d_unfinished.data_ptr(), self.pad_id, self.d_slot.data_ptr(), self.d_rpos.data_ptr(),
                                          self.d_len.data_ptr(), self.d_glen.data_ptr(), s)

    def _ensure_decode_graph(self, n: int) -> torch.cuda.CUDAGraph:
        g = self._decode_graphs.get(n)
        if g is not None:
            return g
        # Warm up and capture on scratch state; restore state and the RNG so that
        # capture never consumes the caller's random stream or touches live rows.
        state = (self.d_tok, self.d_unfinished, self.d_slot, self.d_len, self.d_glen, self.d_plen, self.d_cuk,
                 self.d_rpos, self.rng)
        saved = [t.clone() for t in state]
        gen = torch.cuda.default_generators[torch.cuda.current_device()]
        rng = gen.get_state()

        def scratch():
            self.d_slot.fill_(self.max_seq - 1)
            self.d_len.fill_(1)
            self.d_glen.fill_(1)
            self.d_plen.fill_(1)
            self.d_cuk.copy_(torch.arange(self.max_samples + 1, device=DEV, dtype=torch.int32) * self.max_seq)
            self.d_rpos.zero_()
            self.d_unfinished.fill_(1)
        scratch()
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st), torch.no_grad(), _autocast():
            for _ in range(2):
                self._decode_step(n)
                scratch()
        torch.cuda.current_stream().wait_stream(st)
        scratch()
        g = torch.cuda.CUDAGraph()
        with torch.no_grad(), _autocast(), torch.cuda.graph(g):
            self._decode_step(n)
        torch.cuda.synchronize()
        for t, sv in zip(state, saved):
            t.copy_(sv)
        gen.set_state(rng)
        self._decode_graphs[n] = g
        return g

    # ------------------------------------------------------------------
    # expert
    # ------------------------------------------------------------------
    def _expert_buffers(self):
        if self._eb is None:
            rows, E, D = self.max_samples * self.n_act, self.e_hidden, self.head_dim
            mk = lambda *shape, dt=BF16: torch.empty(*shape, device=DEV, dtype=dt)
            ff = self.expert.config.intermediate_size
            hid = self.act.mlp[0][0].shape[0]
            self._eb = {"res": mk(rows, E, dt=torch.float32), "hn": mk(rows, E),
                        "qkv": mk(rows, (self.e_heads + 2 * self.n_kv) * D), "q": mk(rows, self.e_heads * D),
                        "o": mk(rows, E), "gu": mk(rows, 2 * ff), "a": mk(rows, ff), "d": mk(rows, E),
                        "af": mk(rows, self.act.k0), "ah0": mk(rows, hid), "ah1": mk(rows, hid), "ae": mk(rows, E)}
            if self.w.e8 is not None:       # W8A8 tier: FP8 rows + per-row scales of the three GEMM inputs
                for name, width in (("hn", E), ("attn", self.e_heads * D), ("a", ff)):
                    self._eb[name + "8"] = mk(rows, width, dt=torch.uint8)
                    self._eb[name + "_s"] = mk(rows, dt=torch.float32)
        return self._eb

    def _action_in(self, x: torch.Tensor, t: torch.Tensor, rows: int, res: torch.Tensor, s: int) -> None:
        """``res = action_in_proj(x, t)`` (fp32): Fourier features, the SiLU MLP on tcgen05 GEMMs with the activation in
        the epilogue and RMSNorms between, the LayerNorm with fp32 output."""
        fvk, b = self._fvk, self._eb
        fvk.alpamayo15_action_fourier_bf16(x.data_ptr(), int(x.dtype == BF16), t.data_ptr(), self.act.freqs.data_ptr(), rows, rows,
                                           self.act_dim, self.act.nf, b["af"].data_ptr(), self.act.k0, s)
        src = b["af"][:rows]
        for li, (w, bias, act) in enumerate(self.act.mlp):
            last = li == len(self.act.mlp) - 1
            dst = (b["ae"] if last else b["ah0"])[:rows]
            self._gemm(src, w, dst, bias=bias, act=act, tile=1)
            if not last:
                w_rms, eps = self.act.rms[li]
                fvk.alpamayo15_rms_bf16(dst.data_ptr(), 0, w_rms.data_ptr(), b["ah1"].data_ptr(), rows, dst.shape[1], eps, s)
                src = b["ah1"][:rows]
        w_ln, b_ln, eps = self.act.ln
        fvk.alpamayo15_layernorm_f32out_bf16(b["ae"].data_ptr(), w_ln.data_ptr(), b_ln.data_ptr(), res.data_ptr(), rows, self.e_hidden,
                                             eps, s)

    def _expert_velocity(self, x: torch.Tensor, t: torch.Tensor, dt: float) -> None:
        """One Euler step of the expert over ``x`` [n, n_act, act_dim] (fp32, updated in place) at time ``t`` (device
        scalar), official autocast numerics."""
        m, fvk = self.m, self._fvk
        s = torch.cuda.current_stream().cuda_stream
        n = x.shape[0]
        A, E, D, HQ, KV, S = self.n_act, self.e_hidden, self.head_dim, self.e_heads, self.n_kv, self.max_seq
        rows = n * A
        b = self._expert_buffers()
        res, hn, qkv, q, o, gu, act, d = (b[k][:rows] for k in ("res", "hn", "qkv", "q", "o", "gu", "a", "d"))
        ff = act.shape[1]
        eps = self.e_eps
        layers = self.expert.layers
        e8 = self.w.e8

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
                self._gemm.fp8(q8, q_s, w8, w_s, out)
            else:
                self._gemm(src, w, out, tile=1)
        self._action_in(x, t, rows, res, s)                                  # fp32 residual stream (action_in_proj)
        norm(None, layers[0].input_layernorm.weight)
        for li, layer in enumerate(layers):
            a = layer.self_attn
            lin(hn, "hn", self.w.e_qkv[li], "qkv", li, qkv, quantized=True)
            fvk.alpamayo15_expert_qk_prep_bf16(
                qkv.data_ptr(), a.q_norm.weight.data_ptr(), a.k_norm.weight.data_ptr(),
                self.e_cos.data_ptr(), self.e_sin.data_ptr(), q.data_ptr(),
                self.k_cache[li].data_ptr(), self.v_cache[li].data_ptr(), self.d_aslot.data_ptr(),
                rows, HQ, KV, KV * D, A, S, eps, s)
            attn = self._attend(li, q.view(rows, HQ, D), n, A, self.d_eseq, self.d_esuf, self.expert_splits)
            lin(attn, "attn", a.o_proj.weight, "o", li, o)
            norm(o, layer.post_attention_layernorm.weight)
            lin(hn, "hn", self.w.e_gu[li], "gu", li, gu, quantized=True)
            if e8 is None:
                fvk.alpamayo15_silu_mul_split_bf16(gu.data_ptr(), gu[:, ff:].data_ptr(), 2 * ff, act.data_ptr(), rows, ff, s)
            else:
                fvk.alpamayo15_silu_mul_fp8_rows_bf16(gu.data_ptr(), gu[:, ff:].data_ptr(), 2 * ff, b["a8"].data_ptr(),
                                                      b["a_s"].data_ptr(), rows, ff, s)
            lin(act, "a", layer.mlp.down_proj.weight, "down", li, d, quantized=True)
            nxt = layers[li + 1].input_layernorm.weight if li + 1 < len(layers) else self.expert.norm.weight
            if li + 1 < len(layers):
                norm(d, nxt)
            else:                                   # the final norm feeds the BF16 action_out_proj
                fvk.alpamayo15_res_rms_f32_bf16(res.data_ptr(), d.data_ptr(), nxt.data_ptr(), hn.data_ptr(), rows, E, eps, s)
        fvk.alpamayo15_action_out_euler(hn.data_ptr(), self.act.out_w.data_ptr(), self.act.out_b.data_ptr(), dt, x.data_ptr(), 0, rows, E,
                                        self.act_dim, s)

    def _denoise(self, n: int) -> None:
        ts, x = self.time_steps, self.x[:n]
        for i in range(self.n_steps):
            self._expert_velocity(x, ts[i:i + 1], self.step_dt[i])

    def _ensure_expert_graph(self, n: int) -> torch.cuda.CUDAGraph:
        g = self._expert_graphs.get(n)
        if g is not None:
            return g
        saved = [t.clone() for t in (self.x, self.d_eseq, self.d_esuf, self.d_aslot, self.d_plen, self.d_cuk)]
        self.d_eseq.fill_(self.n_act + 1)
        self.d_esuf.fill_(self.n_act)
        self.d_aslot.fill_(self.max_seq - self.n_act)   # scratch rows past any live prefix
        self.d_plen.fill_(1)
        self.d_cuk.copy_(torch.arange(self.max_samples + 1, device=DEV, dtype=torch.int32) * self.max_seq)
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st), torch.no_grad(), _autocast():
            for _ in range(2):
                self._denoise(n)
        torch.cuda.current_stream().wait_stream(st)
        g = torch.cuda.CUDAGraph()
        with torch.no_grad(), _autocast(), torch.cuda.graph(g):
            self._denoise(n)
        torch.cuda.synchronize()
        for t, sv in zip((self.x, self.d_eseq, self.d_esuf, self.d_aslot, self.d_plen, self.d_cuk), saved):
            t.copy_(sv)
        self._expert_graphs[n] = g
        return g

    def capture(self, num_samples: tuple[int, ...] = (1,)) -> None:
        """Capture the decode and diffusion graphs for the given sample counts and
        JIT-compile the prefill and vision attention kernels."""
        for n in num_samples:
            if not 1 <= n <= self.max_samples:
                raise ValueError(f"num_samples {n} outside [1, max_samples={self.max_samples}]")
            self._ensure_decode_graph(n)
            self._ensure_expert_graph(n)
        self._warm_attention()

    def _warm_attention(self) -> None:
        if getattr(self, "_attention_warm", False):
            return
        # FA4 compiles per configuration, which includes length-dependent tiling
        # choices: warm with prompt- and frame-sized sequences, not toy shapes.
        D, NH, NKV = self.head_dim, self.n_heads, self.n_kv
        q = torch.zeros(1, 1024, NH, D, device=DEV, dtype=BF16)
        kv = torch.zeros(1, 1024, NKV, D, device=DEV, dtype=BF16)
        self._fa4(q, kv, kv, softmax_scale=self.scale, causal=True, pack_gqa=True)
        if True:
            V = self.vlm.model.visual
            nh = V.blocks[0].attn.num_heads
            hd = V.blocks[0].attn.qkv.weight.shape[1] // nh
            qv = torch.zeros(2, 3, 720, nh, hd, device=DEV, dtype=BF16)   # interleaved q/k/v views
            self._fa4(qv[:, 0], qv[:, 1], qv[:, 2], softmax_scale=V.blocks[0].attn.scaling, causal=False)
        torch.cuda.synchronize()
        self._attention_warm = True

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
        if P > self.max_prompt:
            raise ValueError(f"prompt length {P} exceeds max_prompt {self.max_prompt}")
        ids = input_ids[0].tolist()
        vis_idx = torch.tensor([i for i, t in enumerate(ids) if t == self.image_token_id], dtype=torch.long, device=DEV)
        grids = [tuple(int(v) for v in row) for row in tokenized["image_grid_thw"].tolist()]
        pos_h, rope_delta = mrope_positions(ids, grids, image_token_id=self.image_token_id, vision_start_id=self.vision_start_id,
                                            spatial_merge=self.spatial_merge)
        pos3 = torch.from_numpy(pos_h).to(DEV)
        input_ids = input_ids.to(DEV)
        v0 = float(m.action_space.estimate_t0_states(hist_xyz_h[:, -1].float(), hist_rot_h[:, -1].float())["v"].reshape(-1)[0])
        z_last = float(hist_xyz_h[0, -1, -1, 2])
        gen = torch.cuda.default_generators[torch.cuda.current_device()]
        rng_offset = int(gen.get_offset())
        self.rng.copy_(torch_rng_state(gen))             # the sampler draws torch's multinomial draws from here on

        # 1) vision + prefill once (sample 0 holds the shared prompt K/V); first tokens.
        mark("prefill_start")
        logits = self._prefill_native(input_ids, tokenized, vis_idx, pos3)
        self._fvk.alpamayo15_sample_tokens_bf16(logits.data_ptr(), 0, N, logits.shape[1], self.mask_lo, self.mask_n, self.temperature,
                                                self.top_p, 0, self.rng.data_ptr(), self._rng_blocks, self.samp_scratch.data_ptr(),
                                                self.samp_ws.data_ptr(), self.samp_ws.numel(), self.d_tok.data_ptr(), 0, self.pad_id, 0, 0, 0, 0,
                                                torch.cuda.current_stream().cuda_stream)
        first = self.d_tok[:N]
        mark("prefill_end")

        # 2) decode: graph replays, host evaluates the official stop criteria.
        tokens = [[t] for t in first.tolist()]
        traj_found, unfinished = [False] * N, [True] * N
        update_stop_state([t[-1] for t in tokens], traj_found, unfinished, 1,
                          traj_start_id=self.traj_start_id, max_new_tokens=self.max_new, eos_ids=self.eos_ids)
        graph = self._decode_graphs[N]
        self.d_unfinished[:N].copy_(torch.tensor(unfinished, dtype=torch.long))
        self.d_slot.fill_(P)
        self.d_len.fill_(P + 1)
        self.d_glen.fill_(1)
        self.d_plen.fill_(P)
        cuk = torch.arange(N + 1, dtype=torch.int32) * self.max_seq
        cuk[:N] += P
        self.d_cuk[:N + 1].copy_(cuk)
        self.d_rpos.fill_(P + rope_delta)                # every sample's rope position (all advance together)
        while any(unfinished):
            graph.replay()
            last = self.d_tok[:N].tolist()
            for row, t in zip(tokens, last):
                row.append(t)
            before = list(unfinished)
            update_stop_state(last, traj_found, unfinished, len(tokens[0]),
                              traj_start_id=self.traj_start_id, max_new_tokens=self.max_new, eos_ids=self.eos_ids)
            if unfinished != before and any(unfinished):
                self.d_unfinished[:N].copy_(torch.tensor(unfinished, dtype=torch.long))
        # torch's generator advances as the official sampling advanced it (one multinomial per step), so the noise
        # draw below reads the same random stream as the official model's
        gen.set_offset(rng_offset + torch_philox_increment(N * logits.shape[1], self._rng_blocks) * len(tokens[0]))
        mark("decode_end")

        # 3) diffusion expert over each sample's cached prefix.
        offsets, visible = expert_attention_spans(tokens, P, self.traj_start_id)
        epos = np.array([[rope_delta + o + k for k in range(self.n_act)] for o in offsets], dtype=np.int32).reshape(-1)
        epos3 = torch.from_numpy(np.tile(epos, (3, 1)).copy()).to(DEV)
        s = torch.cuda.current_stream().cuda_stream
        self._fvk.alpamayo15_mrope_tables(epos3.data_ptr(), self.inv_freq.data_ptr(), self.mrope_sec[1], self.mrope_sec[2],
                                          self.rope_scaling, self.e_cos.data_ptr(), self.e_sin.data_ptr(), N * self.n_act, self.head_dim,
                                          1, s)
        self.d_aslot[:N].copy_(torch.tensor(visible, dtype=torch.long))
        self.d_eseq[:N].copy_(torch.tensor([v + self.n_act for v in visible], dtype=torch.int32))
        self.d_esuf[:N].copy_(torch.tensor([v - P + self.n_act for v in visible], dtype=torch.int32))
        torch.randn(N, self.n_act, self.act_dim, device=DEV, out=self.x[:N])   # the official noise draw
        noise = self.x[:N].clone() if return_extra else None
        self._expert_graphs[N].replay()
        mark("diffusion_end")

        # 4) trajectory: the official unicycle integration from the sampled actions and the t0 velocity estimate.
        self.a_v0[:N].copy_(torch.full((N,), v0))
        self.a_z[:N].copy_(torch.full((N,), z_last))
        dt, a_mean, a_std, k_mean, k_std = self.act.space
        self._fvk.alpamayo15_action_to_traj(self.x.data_ptr(), self.a_v0.data_ptr(), self.a_z.data_ptr(), N, self.n_act, dt, a_mean,
                                            a_std, k_mean, k_std, self.pred_xyz.data_ptr(), self.pred_rot.data_ptr(), s)
        pred_xyz = self.pred_xyz[:N].view(1, 1, N, self.n_act, 3)
        pred_rot = self.pred_rot[:N].view(1, 1, N, self.n_act, 3, 3)
        mark("end")
        if not return_extra:
            return pred_xyz, pred_rot
        from alpamayo1_5.models.token_utils import extract_text_tokens, replace_padding_after_eos
        seq = torch.tensor([ids + row for row in tokens], dtype=torch.long)          # host: text extraction only
        seq = replace_padding_after_eos(seq, self.traj_start_id, self.pad_id)
        extra = extract_text_tokens(m.tokenizer, seq)
        for key in extra:
            extra[key] = np.array(extra[key]).reshape([1, 1, N])
        extra["generated_tokens"] = tokens[0] if N == 1 else tokens
        extra["noise"] = noise
        extra["rope_delta"] = rope_delta
        extra["prompt_len"] = P
        return pred_xyz, pred_rot, extra
