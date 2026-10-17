"""FlashDrive's Alpamayo 1.5 rollout on Jetson AGX Thor (SM110) with FlashRT execution.

FlashDrive (https://github.com/z-lab/flashdrive) accelerates Alpamayo 1.5 by
changing the inference algorithm: streaming windows over a pre-RoPE static KV
cache, DFlash block speculation, a W4A8 (ParoQuant) language model, and
adaptive action caching with a fine-tuned action expert. This module keeps that
orchestration -- FlashDrive's own ``sample_trajectories_streaming``,
``_dflash_generate`` and ``_dflash_decode_loop`` drive every window -- and
replaces the execution of its primitives with FlashRT kernels and CUDA graphs.
Each group of primitives is selected independently (``backends``), so every
replacement can be validated and measured against FlashDrive's own execution.

expert   ``_action``: the Euler loop with cached velocities (``euler_with_cache``:
         fresh expert evaluations on the non-cached steps, the previous velocity
         reused on ``cache_steps``) as one CUDA graph on FlashRT kernels (FP32
         residual stream, fused head RMSNorm + FP32 RoPE, FA4 attention with a
         device-side key length) and FlashDrive's fused q|k|v and gate|up weights.
vision   ``_encode``: the Qwen3-VL vision tower of the FlashRT Alpamayo 1.5 engine
         (one GEMM patch embedding, FP32 RoPE kernel, batched FA4 attention), one
         CUDA graph per frame grid.
prefill  ``_dflash_prefill`` (streaming windows), ``_shift_streaming_kv_cache``
         and the cache bookkeeping of ``_first_prefill``: the 36-layer language
         model on FlashRT kernels with FlashDrive's W4A8 projections, FA4 causal
         attention by cache position, one CUDA graph per stream.
decode   ``_dflash_block_step`` (draft proposal, 8-row verification, sampling and
         acceptance decisions in one CUDA graph) and ``_dflash_traj_forward``.
         Requires ``prefill="flashrt"``.

Cache contract (kept from FlashDrive): FlashDrive's ``StaticCache`` stays the
store of record -- keys *un-rotated* and values, ``[1, Hk, S, 128]`` per layer,
fixed position table per stream. FlashDrive rotates the whole cache on every
forward; the backend instead keeps attention copies that always equal those
rotations: ``kvlm`` (VLM view, BF16 tables, each product and sum rounded to
BF16) for every cache row, updated wherever a row is written, and the expert's
view (FP32 tables, one BF16 rounding) rebuilt once per window. Values are read
in place through strided views. Masks become FA4 lengths:

- streaming prefill: FlashDrive's block mask equals causal attention by cache
  position for Alpamayo 1.5 (extended view ranges); each contiguous run of
  window rows attends ``[0, its last row]`` (one varlen call, shared key prefix);
- verification block: causal over ``[0, cur + 8)``;
- trajectory-start forward (one row, FlashDrive passes no mask): every cache row;
- expert: rows ``[0, action_start)`` plus its 64 action rows (contiguous).

Other semantics kept from FlashDrive under ``autocast(bfloat16)``: the expert's
Euler state, time grid and noise are BF16 (noise drawn with ``normal_`` at
FlashDrive's draw point); draft proposals are greedy, posterior sampling uses
FlashDrive's ``sample_tokens``; every cache row FlashDrive writes (including
rejected draft rows and the expert's action rows) is written with the same
contents, so later reads of stale rows see what FlashDrive would.
"""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

from flash_rt.models.alpamayo15.marlin_w4a8 import marlin_w4a8_unpack, pack_w4a8_small  # noqa: F401  (shared Marlin layout)
from flash_rt.models.alpamayo15.pipeline_thor import _Gemm, _capture  # noqa: F401  (shared tcgen05 GEMM / graph capture)

BF16 = torch.bfloat16
DEV = "cuda"
logger = logging.getLogger(__name__)

BACKEND_CHOICES = ("flashrt", "flashdrive")

# Kernels of the FlashDrive backend (csrc/kernels/alpamayo15_flashdrive_kernels.cuh and the W4A8 GEMM in
# csrc/gemm/alpamayo15_flashdrive_w4a8_sm100.cu), built with
# -DFLASHRT_ENABLE_ALPAMAYO15=ON next to the Alpamayo 1.5 engine kernels (whose LayerNorm, split SiLU-mul and
# per-row FP8 quantization kernels and tcgen05 GEMMs the backend shares).
FLASHDRIVE_KERNEL_SYMBOLS = ("flashdrive_qk_prep_bf16", "flashdrive_rope_rows", "flashdrive_w4a8_act_prep",
                             "flashdrive_shift_cache",
                             "flashdrive_rms_f16delta_bf16", "flashdrive_silu_mul_f16in_bf16",
                             "flashdrive_w4a8_gemm_sm100", "flashdrive_w4a8_gemm_sm100_sizes", "flashdrive_w4a8_gemm_small",
                             "flashdrive_bf16_gemm_small", "flashdrive_draft_qkv_prep_bf16", "flashdrive_w4a8_rotate_dense",
                             "flashdrive_w4a8_quant", "flashdrive_w4a8_gemm_tile",
                             "flashdrive_bf16_gemm_frag", "flashdrive_w4a8_act_prep_tile")
TILE_MIN_N = 2048           # few-row W4A8 GEMMs with at least this many output channels stream one warp per 16-row tile
ENGINE_GEMMS = ("cutlass", "torch")    # BF16 GEMMs of the vision tower and the action expert
DECODE_WEIGHTS = ("bf16", "int8", "fp8")   # LM head and DFlash draft linears of the decode graphs
PREFILL_WEIGHTS = ("w4a8", "int8")         # streaming-window GEMM operands: Marlin's group-scaled integers, or per-channel INT8
EXPERT_WEIGHTS = ("bf16", "fp8")       # action expert projections (FP8 = W8A8 with per-row activation scales)
W4A8_BACKENDS = ("flashdrive", "fused")
W4A8_GEMMS = ("tcgen05", "marlin")
ACT_ROTS = ("paro", "dense")
TCGEN05_MIN_ROWS = 256      # W4A8 GEMMs with at least this many rows (streaming-window prefill) run on tcgen05
INT8_PREFILL_TILE = 2       # per-channel INT8 prefill GEMM: 256 x 128 x 128 on a 2-SM UMMA (0: 128 x 128, 1: 64 x 128; identical integers)
SMALL_MAX_ROWS = 16         # ... and with at most this many (verification, trajectory start) on the few-row mma kernel


def euler_cache_schedule(inference_step: int, cache_steps) -> list[bool]:
    """Per Euler step: True when the step reuses the cached velocity (FlashDrive ``euler_with_cache``)."""
    steps = set(int(s) for s in cache_steps)
    if 0 in steps:
        raise ValueError("cache_steps cannot include step 0: no velocity is cached yet.")
    if any(s < 0 or s >= inference_step for s in steps):
        raise ValueError(f"cache_steps {sorted(steps)} outside [1, {inference_step})")
    return [i in steps for i in range(inference_step)]


def contiguous_runs(rows: list[int]) -> list[tuple[int, int]]:
    """Split strictly increasing cache rows into contiguous ``(start, end)`` runs (end exclusive)."""
    runs: list[tuple[int, int]] = []
    for r in rows:
        if runs and r == runs[-1][1]:
            runs[-1] = (runs[-1][0], r + 1)
        elif runs and r < runs[-1][1]:
            raise ValueError("cache rows must be strictly increasing")
        else:
            runs.append((r, r + 1))
    return runs


def rotate_keys_fp32(k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """FlashDrive's expert-side rotation of cached (un-rotated) BF16 keys with FP32 tables.

    ``(k * cos) + (rotate_half(k) * sin)`` with BF16 ``k`` and FP32 ``cos``/``sin``
    promotes to FP32 (each product and the sum rounded to FP32); attention under
    autocast then casts the result to BF16 once. ``k`` is ``[..., S, H, D]`` and
    the tables ``[S, 1, D]`` (broadcast over heads). Returns FP32. Reference for
    the ``flashdrive_rope_rows`` kernel (``fp32_rule=1``).
    """
    half = k.shape[-1] // 2
    rot = torch.cat((-k[..., half:], k[..., :half]), dim=-1)
    return k * cos + rot * sin


def reset_flashdrive_stream(model) -> None:
    """End FlashDrive's current stream so its next window starts a fresh one.

    FlashDrive treats ``_past_key_values is None`` as a new stream but keeps
    per-stream state elsewhere: the vision and language-model shape caches of
    the last (4-frame) streaming window, the streaming attention mask, and its
    compiled-step registry (input buffers sized for the old stream). All are
    cleared, as a fresh process would start; FlashDrive-executed compiled
    primitives are therefore compiled again for the next stream.
    """
    model._past_key_values = None
    model.vlm.model.visual.reset_shape_caches()
    model.vlm.model.language_model.reset_shape_caches()
    model._cached_streaming_attention_mask = None
    model._compiled_step_registry.clear()


def _require_autocast() -> None:
    if not torch.is_autocast_enabled("cuda") or torch.get_autocast_dtype("cuda") != BF16:
        raise ValueError("FlashDriveThor reproduces FlashDrive's rollout under torch.autocast('cuda', bfloat16); "
                         "call it inside that context (as FlashDrive's scripts/infer.py does)")


def load_flashdrive(base: str, paro: str, dflash: str, *, torch_compile: str | None = None) -> torch.nn.Module:
    """Load the full FlashDrive stack from explicit checkpoint locations.

    Mirrors ``flashdrive.from_pretrained`` for an optimized checkpoint (setup
    order: rollout, compile, streaming, ParoQuant, expert fusion, DFlash) but
    takes the W4A8 and draft companions explicitly instead of deriving them from
    a ``z-lab/`` path prefix, so local directories load the optimized stack too.
    ``torch_compile`` is FlashDrive's compile mode for the primitives FlashRT
    does not replace (``None`` = eager). The model is placed on the current CUDA device.
    """
    try:
        import flashdrive
        from flashdrive import FlashDriveMixin, resolve_model_class
    except ImportError as exc:
        raise ImportError("the FlashDrive backend needs the flashdrive package "
                          "(https://github.com/z-lab/flashdrive)") from exc
    model_cls = resolve_model_class(base)
    extended = getattr(flashdrive, "_EXTENDED_VISION_RANGES", {}).get(model_cls)
    if extended is None:
        raise ValueError(f"FlashDrive does not define streaming ranges for {model_cls.__name__}")
    model_cls._supports_sdpa = True
    config = model_cls.config_class.from_pretrained(base)
    config.attn_implementation = config._attn_implementation = "sdpa"
    model = model_cls.from_pretrained(base, config=config, dtype=BF16).to(DEV).eval()
    model.__class__ = type(f"FlashDrive{model_cls.__name__}", (FlashDriveMixin, model_cls), {})
    model.setup_rollout()
    model.setup_torch_compile(torch_compile)
    model.setup_streaming(extended_vision_ranges=extended)
    model.setup_paroquant(paro)
    model.setup_expert_fusion()
    model.setup_dflash(dflash)
    return model


def marlin_w4a8_to_int8(qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor, K: int, N: int,
                        maps: dict | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """``(w8, s)``: ``w8 [N, K]`` int8 = ``w - z`` and ``s [K/128, N]`` int32 (the tcgen05 prefill GEMM operands)."""
    w, z, s = marlin_w4a8_unpack(qweight, scales, qzeros, K, N, maps)
    G = K // 128
    w8 = (w.to(torch.int16) - z.to(torch.int16).repeat_interleave(128, 0)).to(torch.int8).t().contiguous()
    return w8, s


def w4a8_to_int8_channel(w8: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel INT8 requantization of Marlin's group-scaled integers: ``w8 [N, K]`` int8 (w - z) and ``s [K/128, N]``
    int32 group scales give the exact integer weight ``w8 * s[group]``; its per-channel symmetric INT8 codes
    (``round(v / c)``, ``c = absmax / 127``) and the fp32 channel scale ``c`` (1 for an all-zero channel) are returned
    for the per-channel INT8 prefill GEMM (``D = half(acc * a_scale[m] * c[n])``). A precision tier: the codes carry the
    group-scale ratios of a channel with at most half a step of error."""
    N, K = w8.shape
    v = w8.to(torch.int32) * s.t().repeat_interleave(128, 1)           # [N, K] exact integers (|w8| <= 15, s <= 4096)
    amax = v.abs().amax(dim=1).to(torch.float32)
    c = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
    codes = torch.round(v.to(torch.float32) / c[:, None]).clamp_(-127.0, 127.0).to(torch.int8).contiguous()
    return codes, c.contiguous()


def paro_pair_bank_order(theta: torch.Tensor, pairs: torch.Tensor, K: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Reorder the 64 pairs of every (layer, channel group) of a ParoQuant rotation so that the eight pairs a
    quarter-warp of the tile kernel handles together touch eight distinct shared-memory bank groups with their first
    channel (bank group = 16-byte chunk index ``c + c // 8`` mod 8 of the 8-row tile layout). The pairs of a layer are
    disjoint, so their order does not change the rotation; returns (theta, pairs) in the new order."""
    G = K // 128
    pr = pairs.view(8, G, 64, 2).to(torch.int64)
    th = theta.view(8, G, 64)
    i = pr[..., 0]
    res = (i + (i >> 3)) & 7
    ar = torch.arange(64, device=pairs.device)
    order = torch.argsort(res * 64 + ar, dim=-1)                       # pairs sorted by bank residue (stable)
    res_s = torch.gather(res, -1, order)
    counts = torch.nn.functional.one_hot(res, 8).sum(-2)                # [8, G, 8]
    starts = counts.cumsum(-1) - counts
    rank = ar - torch.gather(starts, -1, res_s)                         # rank within the residue class
    main = rank < 8
    pos = torch.where(main, rank * 8 + res_s, torch.zeros_like(rank))   # bucket = rank, lane-in-bucket = residue
    taken = torch.zeros_like(rank).scatter_add(-1, pos, main.long()) > 0
    free_order = torch.argsort(taken.long() * 64 + ar, dim=-1)           # free positions first, ascending
    ovf = (~main).long().cumsum(-1) - 1
    pos = torch.where(main, pos, torch.gather(free_order, -1, ovf.clamp(min=0)))
    new_pr = torch.empty_like(pr).scatter_(-2, pos.unsqueeze(-1).expand(-1, -1, -1, 2),
                                           torch.gather(pr, -2, order.unsqueeze(-1).expand(-1, -1, -1, 2)))
    new_th = torch.empty_like(th).scatter_(-1, pos, torch.gather(th, -1, order))
    return new_th.reshape(8, K // 2).contiguous(), new_pr.reshape(8, K).to(pairs.dtype).contiguous()


def paro_rotation_matrices(theta: torch.Tensor, pairs: torch.Tensor, K: int) -> torch.Tensor:
    """The 128 x 128 matrix of every channel group of a ParoQuant rotation: ``R[g][c][k]`` such that
    ``y[g*128 + c] = sum_k R[g][c][k] x[g*128 + k]`` equals the 8 Givens layers applied in order (FP32 products of
    the layer cosines/sines, returned as FP16 for the tensor-core kernel)."""
    G = K // 128
    R = torch.eye(128, device=theta.device, dtype=torch.float32).repeat(G, 1, 1)         # [G, 128, 128]
    th = theta.float().view(8, G, 64)
    pr = pairs.view(8, G, 64, 2).long()
    gidx = torch.arange(G, device=theta.device).unsqueeze(1)
    for r in range(8):
        c, s_ = torch.cos(th[r]).unsqueeze(-1), torch.sin(th[r]).unsqueeze(-1)       # [G, 64, 1]
        i, j = pr[r, :, :, 0], pr[r, :, :, 1]                                        # [G, 64]
        Ri, Rj = R[gidx, i], R[gidx, j]                                              # rows i, j: [G, 64, 128]
        R[gidx, i] = c * Ri + s_ * Rj
        R[gidx, j] = c * Rj - s_ * Ri
    return R.to(torch.float16).contiguous()


class _W4A8Group:
    """FlashDrive ParoQuant linears that share one input (q|k|v, o, gate|up, down) executed as one fused
    activation pass (``flashdrive_w4a8_act_prep``: FP16 cast, pairwise rotation, per-token INT8 quantization,
    bit-exact with paroquant + vLLM) and the W4A8 GEMM of each linear: vLLM's Marlin kernel, or (``gemm="tcgen05"``,
    groups of at least ``TCGEN05_MIN_ROWS`` rows) FlashRT's INT8 tensor-core GEMM on the unpacked weights, which
    computes Marlin's integer group-scaled sums exactly. Returns the FP16 GEMM outputs; FlashDrive's cast back to
    BF16 is applied by the consumer kernels on load."""

    def __init__(self, fvk, wrappers, scratch: dict, gemm: str = "marlin", maps: dict | None = None,
                 act_rot: str = "paro", ktile: int = 256, prefill_weights: str = "w4a8") -> None:
        from flashdrive.quantization import BF16RotateLinearWrapper
        from vllm.scalar_type import scalar_types
        mods = []
        for w in wrappers:
            if not isinstance(w, BF16RotateLinearWrapper):
                raise TypeError(f"expected FlashDrive's BF16RotateLinearWrapper, got {type(w).__name__}")
            mods.append(w.rotate_linear)
        self.K = mods[0].in_features
        for m in mods:
            q, r = m.qlinear, m.rotation
            if m.in_features != self.K or self.K % 128:
                raise ValueError("fused W4A8 group needs one input width, a multiple of 128")
            if (r.theta.shape != (8, self.K // 2) or r.pairs.shape != (8, self.K) or r.channel_scales.numel() != self.K
                    or r.theta.dtype != torch.float16 or r.channel_scales.dtype != torch.float16
                    or r.pairs.dtype != torch.int16):
                raise ValueError("unexpected ParoQuant rotation layout (8 layers, group 128, fp16 angles/scales)")
            if q.quant_type != scalar_types.uint4 or q.bias is not None or q.g_idx.numel() or q.scales.dtype != torch.float16:
                raise ValueError("unexpected Marlin W4A8 layout (uint4 + zero points, fp16 group scales, no act-order/bias)")
        self.mods = mods
        self.N = [m.out_features for m in mods]
        self._scratch = scratch     # shared by every group: consumed by this group's GEMMs before the next group runs
        self.act_rot = act_rot
        self.ktile = ktile
        self.rot = [paro_rotation_matrices(m.rotation.theta, m.rotation.pairs, self.K) for m in mods] if act_rot == "dense" else None
        tabs = [paro_pair_bank_order(m.rotation.theta, m.rotation.pairs, self.K) for m in mods]   # bank-ordered pair tables
        self.theta, self.pairs = [t for t, _ in tabs], [p for _, p in tabs]
        self.int8 = None
        self.int8pc = None
        self.small = None
        if gemm == "tcgen05":       # FlashRT GEMMs: tcgen05 INT8 ((w - z) int8 [N, K]) for prefill-size groups,
            self.int8, self.small = [], []   # mma.sync on packed nibbles + zero points for a few rows
            if prefill_weights == "int8":    # ... or per-channel INT8 codes on the plain INT8 GEMM (precision tier)
                self.int8pc = []
            for m, N in zip(mods, self.N):
                w, z, sg = marlin_w4a8_unpack(m.qlinear.qweight, m.qlinear.scales, m.qlinear.qzeros, self.K, N, maps)
                w8 = (w.to(torch.int16) - z.to(torch.int16).repeat_interleave(128, 0)).to(torch.int8).t().contiguous()
                if self.int8pc is not None:
                    self.int8pc.append(w4a8_to_int8_channel(w8, sg))
                    del w8
                else:
                    self.int8.append((w8, sg))
                self.small.append((pack_w4a8_small(w, self.K, N), z.contiguous(), sg))
                del w, z

    def _buffers(self, M: int) -> dict:
        n = len(self.mods)
        b = self._scratch.get((M, self.K, n))
        if b is None:
            Ma = (M + 31) // 32 * 32             # the GEMMs read whole token tiles (8 rows mma, 32 rows tcgen05): zeros
            b = {"xq": [torch.zeros(Ma, self.K, device=DEV, dtype=torch.int8) for _ in range(n)],
                 "as": [torch.zeros(Ma, 1, device=DEV, dtype=torch.float32) for _ in range(n)]}
            if self.rot is not None or M <= SMALL_MAX_ROWS:   # two-pass preparation: FP16 rows and per-group maxima
                b["xr"] = [torch.empty(M, self.K, device=DEV, dtype=torch.float16) for _ in range(n)]
                b["part"] = [torch.empty(M, self.K // 128, device=DEV, dtype=torch.float32) for _ in range(n)]
            self._scratch[(M, self.K, n)] = b
        return b

    def prep(self, fvk, mode: int, src: torch.Tensor | None, M: int, src2: torch.Tensor | None = None,
             residual: torch.Tensor | None = None, norm_w: torch.Tensor | None = None, eps: float = 0.0) -> None:
        """Fused producer + activation preparation of ``M`` rows (``flashdrive_w4a8_act_prep_tile``): mode 0 plain bf16
        rows, mode 1 residual (+= bf16(delta fp16) when ``src`` is given) RMSNorm, mode 2 SiLU-mul of fp16 gate | up."""
        b = self._buffers(M)
        ptrs = lambda ts: [t.data_ptr() for t in ts]
        fvk.flashdrive_w4a8_act_prep_tile(
            mode, src.data_ptr() if src is not None else 0, src2.data_ptr() if src2 is not None else 0,
            (src if src is not None else residual).stride(0), residual.data_ptr() if residual is not None else 0,
            residual.stride(0) if residual is not None else 0, norm_w.data_ptr() if norm_w is not None else 0, eps, M,
            self.K, ptrs(self.theta), ptrs(self.pairs),
            ptrs(m.rotation.channel_scales for m in self.mods), ptrs(m.qlinear.input_global_scale for m in self.mods),
            ptrs(b["xq"]), ptrs(b["as"]), 0, 0, torch.cuda.current_stream().cuda_stream)

    def gemm(self, fvk, M: int) -> list[torch.Tensor]:
        """The GEMMs of the group on the prepared INT8 activations of ``M`` rows."""
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.utils.marlin_utils import should_use_atomic_add_reduce
        b = self._buffers(M)
        outs = []
        if self.int8pc is not None and M >= TCGEN05_MIN_ROWS:
            for (codes, c), xq, a_s, N in zip(self.int8pc, b["xq"], b["as"], self.N):
                outs.append(self._gemm_int8pc(fvk, xq, codes, c, a_s, M, N))
            return outs
        if self.int8 is not None and M >= TCGEN05_MIN_ROWS:
            for (w8, sg), xq, a_s, N in zip(self.int8, b["xq"], b["as"], self.N):
                outs.append(self._gemm_int8(fvk, xq, w8, sg, a_s, M, N))
            return outs
        if self.small is not None and M <= SMALL_MAX_ROWS:
            for (wp, z, sg), xq, a_s, N in zip(self.small, b["xq"], b["as"], self.N):
                out = torch.empty(M, N, dtype=torch.float16, device=DEV)
                gemm = fvk.flashdrive_w4a8_gemm_tile if N >= TILE_MIN_N else fvk.flashdrive_w4a8_gemm_small
                gemm(xq.data_ptr(), a_s.data_ptr(), wp.data_ptr(), z.data_ptr(), sg.data_ptr(), out.data_ptr(), M, N, self.K,
                     torch.cuda.current_stream().cuda_stream)
                outs.append(out)
            return outs
        for m, xq, a_s, N in zip(self.mods, b["xq"], b["as"], self.N):
            q = m.qlinear
            out = ops.marlin_gemm(xq[:M], None, q.qweight, q.bias, q.scales, a_s, None, q.qzeros, q.g_idx,
                                  q.g_idx_sort_indices, q.workspace, q.quant_type, size_m=M, size_n=N, size_k=self.K,
                                  use_atomic_add=should_use_atomic_add_reduce(M, N, self.K, xq.device, torch.float16),
                                  use_fp32_reduce=True, is_zp_float=False)
            outs.append(out)
        return outs

    def __call__(self, fvk, x: torch.Tensor) -> list[torch.Tensor]:
        M = x.shape[0]
        b = self._buffers(M)
        ptrs = lambda ts: [t.data_ptr() for t in ts]
        stream = torch.cuda.current_stream().cuda_stream
        if self.rot is not None:          # dense rotation on tensor cores, then the shared quantization pass
            fvk.flashdrive_w4a8_rotate_dense(x.data_ptr(), x.stride(0), M, self.K, ptrs(self.rot),
                                             ptrs(m.rotation.channel_scales for m in self.mods), ptrs(b["xr"]),
                                             ptrs(b["part"]), stream)
            fvk.flashdrive_w4a8_quant(M, self.K, ptrs(b["xr"]), ptrs(b["part"]),
                                      ptrs(m.qlinear.input_global_scale for m in self.mods), ptrs(b["xq"]), ptrs(b["as"]),
                                      stream)
        elif M > SMALL_MAX_ROWS:          # row tiles: producer-fused preparation
            self.prep(fvk, 0, x, M)
        else:                             # a few rows: the two-pass kernels spread the groups over the SMs
            fvk.flashdrive_w4a8_act_prep(
                x.data_ptr(), x.stride(0), M, self.K, ptrs(self.theta), ptrs(self.pairs),
                ptrs(m.rotation.channel_scales for m in self.mods),
                ptrs(m.qlinear.input_global_scale for m in self.mods), ptrs(b["xr"]), ptrs(b["part"]), ptrs(b["xq"]),
                ptrs(b["as"]), stream)
        return self.gemm(fvk, M)

    def _gemm_int8(self, fvk, xq, w8, sg, a_s, M, N) -> torch.Tensor:
        key = ("int8", M, self.K, N)
        ws = self._scratch.get(key)
        if ws is None:
            nbytes, n_ones = fvk.flashdrive_w4a8_gemm_sm100_sizes(M, N, self.K)
            ws = (torch.empty(max(int(nbytes), 1), dtype=torch.uint8, device=DEV),     # padded: the last token tile
                  torch.ones(int(n_ones) + 1024, dtype=torch.int32, device=DEV))      # may read past the layout
            self._scratch[key] = ws
        out = torch.empty(M, N, dtype=torch.float16, device=DEV)
        fvk.flashdrive_w4a8_gemm_sm100(xq.data_ptr(), w8.data_ptr(), sg.data_ptr(), ws[1].data_ptr(), a_s.data_ptr(),
                                       out.data_ptr(), M, N, self.K, ws[0].data_ptr(), ws[0].numel(),
                                       torch.cuda.current_stream().cuda_stream, self.ktile)
        return out

    def _gemm_int8pc(self, fvk, xq, codes, c, a_s, M, N) -> torch.Tensor:
        key = ("int8pc", M, self.K, N)
        ws = self._scratch.get(key)
        if ws is None:
            nbytes = fvk.alpamayo15_int8_gemm_sm100_workspace(M, N, self.K, INT8_PREFILL_TILE)
            ws = torch.empty(max(int(nbytes), 16), dtype=torch.uint8, device=DEV)
            self._scratch[key] = ws
        out = torch.empty(M, N, dtype=torch.float16, device=DEV)
        fvk.alpamayo15_int8_gemm_sm100(xq.data_ptr(), codes.data_ptr(), a_s.data_ptr(), c.data_ptr(), out.data_ptr(), M, N,
                                       self.K, INT8_PREFILL_TILE, ws.data_ptr(), ws.numel(),
                                       torch.cuda.current_stream().cuda_stream)
        return out

    def check_against_marlin(self, fvk, rows: int = TCGEN05_MIN_ROWS) -> float:
        """Run Marlin and the FlashRT GEMMs (tcgen05 at ``rows`` rows, the few-row kernel at 8 and 16 rows) of every
        linear on random INT8 activations; returns the smallest fraction of identical FP16 outputs. Raises on a gross
        disagreement (unpacking or layout mismatch): an output off by more than one FP16 ulp plus Marlin's FP32
        slice-sum error bound, or fewer than 99% identical outputs."""
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.utils.marlin_utils import should_use_atomic_add_reduce
        g = torch.Generator(device=DEV).manual_seed(0)
        worst = 1.0
        for R in (rows, 8, 16):
            xq = torch.randint(-127, 128, (R, self.K), dtype=torch.int8, device=DEV, generator=g)
            u = torch.rand(R, 1, device=DEV, generator=g) * 2e-2 + 1e-3     # per-token absmax / 127, as act prep
            for m, (w8, sg), (wp, z, _), N in zip(self.mods, self.int8, self.small, self.N):
                q = m.qlinear
                a_s = (u * q.input_global_scale).contiguous()
                ref = ops.marlin_gemm(xq, None, q.qweight, q.bias, q.scales, a_s, None, q.qzeros, q.g_idx,
                                      q.g_idx_sort_indices, q.workspace, q.quant_type, size_m=R, size_n=N, size_k=self.K,
                                      use_atomic_add=should_use_atomic_add_reduce(R, N, self.K, xq.device, torch.float16),
                                      use_fp32_reduce=True, is_zp_float=False)
                if R == rows:
                    out = self._gemm_int8(fvk, xq, w8, sg, a_s, R, N)
                else:
                    out = torch.empty(R, N, dtype=torch.float16, device=DEV)
                    fvk.flashdrive_w4a8_gemm_small(xq.data_ptr(), a_s.data_ptr(), wp.data_ptr(), z.data_ptr(),
                                                   sg.data_ptr(), out.data_ptr(), R, N, self.K,
                                                   torch.cuda.current_stream().cuda_stream)
                r = ref.float()
                tol = (r.abs() * 2.0 ** -10).clamp_min(2.0 ** -24) + r.abs().amax(dim=1, keepdim=True) * 2.0 ** -20
                same = float((out == ref).float().mean())
                if bool(((out.float() - r).abs() > tol).any()) or same < 0.99:
                    raise RuntimeError(f"FlashRT W4A8 GEMM ({R} rows) disagrees with Marlin (weight layout?)")
                worst = min(worst, same)
        return worst


def bf16_fragment_layout(w: torch.Tensor) -> torch.Tensor:
    """Reorder a [N, K] bf16 weight into the ``flashdrive_bf16_gemm_frag`` operand: ``[ceil(N/16)][K/16][32 lanes][16 B]``,
    lane ``4g + c`` holding the four ``mma.m16n8k16`` A registers of one 16-wide k step (rows g, g+8; k 2c and 8+2c).
    Rows are zero-padded to a multiple of 16."""
    N, K = w.shape
    if K % 16:
        raise ValueError(f"K={K} must be a multiple of 16")
    T = (N + 15) // 16
    if T * 16 != N:
        w = torch.cat([w, w.new_zeros(T * 16 - N, K)])
    # n = 16T + 8rh + g ; k = 16kc + 8half + 2c + e  ->  [T, kc, g, c, half, rh, e]  (register a = 2half + rh)
    frag = w.view(T, 2, 8, K // 16, 2, 4, 2).permute(0, 3, 2, 5, 4, 1, 6).contiguous()
    return frag.view(-1)


class _Bf16Frag:
    """Fragment-ordered BF16 copy of a [N, K] linear weight for the streaming few-row GEMM."""

    def __init__(self, w: torch.Tensor) -> None:
        with torch.no_grad():
            self.frag = bf16_fragment_layout(w.detach())
        self.shape = tuple(w.shape)


class _Fp8Weight:
    """Per-row one-byte copy of a [N, K] linear weight (``fmt`` "fp8": E4M3 codes, "int8": symmetric INT8 codes; both
    with per-row FP32 scales) in the ``alpamayo15_w8a16_gemm_bf16`` fragment order."""

    def __init__(self, w: torch.Tensor, fmt: str = "fp8") -> None:
        from flash_rt.models.alpamayo15.pipeline_thor import fp8_fragment_layout, quantize_fp8_rows, quantize_int8_rows
        with torch.no_grad():
            codes, scale = (quantize_int8_rows if fmt == "int8" else quantize_fp8_rows)(w.detach())
            self.frag, self.scale = fp8_fragment_layout(codes, scale)
        self.shape = tuple(w.shape)
        self.fmt = 1 if fmt == "int8" else 0


def _gemm_small(fvk, x: torch.Tensor, w, y: torch.Tensor) -> torch.Tensor:
    """``y = x @ w.T`` for at most 16 rows: ``flashdrive_bf16_gemm_small`` on a BF16 weight, or the W8A16 kernel on
    an ``_Fp8Weight`` (``x`` must come from a buffer with whole 8-row tiles readable past its rows; ``y`` row-major,
    bf16, or fp32 for the BF16 path only)."""
    if isinstance(w, _Fp8Weight):
        N, K = w.shape
        if y.dtype != BF16 or y.stride(0) != N or x.stride(0) != K:
            raise ValueError("the W8A16 path needs contiguous bf16 operands")
        fvk.alpamayo15_w8a16_gemm_bf16(w.frag.data_ptr(), w.scale.data_ptr(), x.data_ptr(), y.data_ptr(), x.shape[0], N, K,
                                       torch.cuda.current_stream().cuda_stream, w.fmt)
        return y
    if isinstance(w, _Bf16Frag):
        N, K = w.shape
        fvk.flashdrive_bf16_gemm_frag(x.data_ptr(), x.stride(0), w.frag.data_ptr(), y.data_ptr(), y.stride(0),
                                      int(y.dtype == torch.float32), x.shape[0], N, K, torch.cuda.current_stream().cuda_stream)
        return y
    fvk.flashdrive_bf16_gemm_small(x.data_ptr(), x.stride(0), w.data_ptr(), y.data_ptr(), y.stride(0),
                                   int(y.dtype == torch.float32), x.shape[0], w.shape[0], w.shape[1],
                                   torch.cuda.current_stream().cuda_stream)
    return y


class FlashDriveThor:
    """FlashRT execution backend installed on a FlashDrive Alpamayo 1.5 model.

    Args:
        model: a FlashDrive model (``FlashDriveMixin``) after all setups, on CUDA.
        expert, vision, prefill, decode: ``"flashrt"`` (default) or
            ``"flashdrive"`` per primitive group (see the module docstring);
            ``decode="flashrt"`` needs ``prefill="flashrt"``.
        w4a8: execution of the W4A8 linears inside the FlashRT language model:
            ``"fused"`` (default with ``prefill="flashrt"``; one activation pass
            per shared input and Marlin outputs consumed by the next kernel) or
            ``"flashdrive"`` (FlashDrive's ParoQuant modules).
        w4a8_gemm: GEMM of fused W4A8 groups with at least ``TCGEN05_MIN_ROWS``
            rows (streaming-window prefill): ``"tcgen05"`` (default; INT8 tensor
            cores on unpacked ``w - z`` weights, about 7 GB more memory) or
            ``"marlin"``. Smaller groups always use Marlin. Only with
            ``w4a8="fused"``.
        act_rot: rotation of the W4A8 activations (``w4a8="fused"``): ``"paro"``
            (default; layer by layer, bit-exact with ParoQuant's kernel) or
            ``"dense"`` (one 128 x 128 FP16 matrix per channel group on tensor
            cores, FP32 accumulation, one FP16 rounding: a numerics change
            that can move INT8 codes by one step).
        vision_gemm, expert_gemm: ``"cutlass"`` (default; FlashRT's tcgen05 BF16
            GEMMs with fused bias / GELU / residual epilogues) or ``"torch"``
            (cuBLAS through torch) for the vision tower and the action expert.
        decode_weights: ``"bf16"`` (default; fragment-ordered BF16 copies of the LM
            head and the DFlash draft linears for the streaming few-row GEMM),
            ``"int8"`` (per-row symmetric INT8 copies on the W8A16 kernel) or
            ``"fp8"`` (per-row FP8 E4M3 copies): precision tiers, validate on your
            scenarios.
        expert_weights: ``"bf16"`` (default) or ``"fp8"``: the expert's projections
            as per-row FP8 E4M3 weights with per-row FP8 activations (W8A8 on the
            FP8 tcgen05 GEMM, scales applied in the epilogue); a precision tier.
        w4a8_ktile: K tile of the blockwise tcgen05 prefill GEMM: 256 (default;
            two groups per mainloop step) or 128; identical integer results.
        prefill_weights: ``"w4a8"`` (default; Marlin's group-scaled integers on the
            blockwise tcgen05 GEMM, bit-identical to Marlin) or ``"int8"``
            (per-channel INT8 requantization of the same integers on the plain
            INT8 tcgen05 GEMM; a precision tier).
        pdl: launch the backend's kernels and GEMMs with programmatic dependent
            launch (off by default: measured neutral on this bandwidth-bound
            pipeline). Process-wide switch.
        expert_splits: FA4 KV splits for expert attention.

    The backend serves one trajectory sample per window (``num_traj_samples=1``,
    ``num_traj_sets=1``) and FlashDrive's ``euler_with_cache`` integrator.
    """

    # The vision tower of the FlashRT Alpamayo 1.5 engine (same Qwen3-VL tower and numerics).
    from flash_rt.models.alpamayo15.pipeline_thor import Alpamayo15Thor as _Engine
    _vision_tables = _Engine._vision_tables
    _vision = _Engine._vision
    _vision_cutlass = _Engine._vision_cutlass
    _merger = _Engine._merger
    _check_vision_modules = _Engine._check_vision_modules
    del _Engine

    def __init__(self, model, *, expert: str = "flashrt", vision: str = "flashrt", prefill: str = "flashrt",
                 decode: str = "flashrt", w4a8: str | None = None, w4a8_gemm: str | None = None,
                 act_rot: str = "paro", vision_gemm: str = "cutlass", expert_gemm: str = "cutlass",
                 decode_weights: str = "bf16", expert_weights: str = "bf16", pdl: bool = False, w4a8_ktile: int = 256,
                 expert_splits: int = 1, prefill_weights: str = "w4a8") -> None:
        if w4a8_ktile not in (128, 256):
            raise ValueError("w4a8_ktile must be 128 or 256")
        if prefill_weights not in PREFILL_WEIGHTS:
            raise ValueError(f"unknown prefill_weights {prefill_weights!r}; choose from {PREFILL_WEIGHTS}")
        backends = {"expert": expert, "vision": vision, "prefill": prefill, "decode": decode}
        if act_rot not in ACT_ROTS:
            raise ValueError(f"unknown act_rot {act_rot!r}; choose from {ACT_ROTS}")
        if vision_gemm not in ENGINE_GEMMS or expert_gemm not in ENGINE_GEMMS:
            raise ValueError(f"vision_gemm / expert_gemm must be one of {ENGINE_GEMMS}")
        if decode_weights not in DECODE_WEIGHTS:
            raise ValueError(f"unknown decode_weights {decode_weights!r}; choose from {DECODE_WEIGHTS}")
        if expert_weights not in EXPERT_WEIGHTS:
            raise ValueError(f"unknown expert_weights {expert_weights!r}; choose from {EXPERT_WEIGHTS}")
        if expert_weights == "fp8" and expert_gemm != "cutlass":
            raise ValueError("expert_weights='fp8' runs on the FlashRT GEMMs (expert_gemm='cutlass')")
        self.vision_gemm, self.expert_gemm, self.decode_weights = vision_gemm, expert_gemm, decode_weights
        self.expert_weights = expert_weights
        if w4a8 is None:
            w4a8 = "fused" if prefill == "flashrt" else "flashdrive"
        for name, choice in backends.items():
            if choice not in BACKEND_CHOICES:
                raise ValueError(f"unknown {name} backend {choice!r}; choose from {BACKEND_CHOICES}")
        if w4a8 not in W4A8_BACKENDS:
            raise ValueError(f"unknown w4a8 execution {w4a8!r}; choose from {W4A8_BACKENDS}")
        if w4a8 == "fused" and prefill != "flashrt":
            raise ValueError("w4a8='fused' applies to the FlashRT language model (prefill='flashrt')")
        if w4a8_gemm is not None and w4a8 != "fused":
            raise ValueError("w4a8_gemm applies to w4a8='fused'")
        if w4a8 == "fused" and w4a8_gemm is None:
            w4a8_gemm = "tcgen05"
        if w4a8 == "fused" and w4a8_gemm not in W4A8_GEMMS:
            raise ValueError(f"unknown w4a8_gemm {w4a8_gemm!r}; choose from {W4A8_GEMMS}")
        backends["w4a8"] = w4a8
        if w4a8 == "fused":
            backends["w4a8_gemm"] = w4a8_gemm
            backends["act_rot"] = act_rot
        if decode == "flashrt" and prefill != "flashrt":
            raise ValueError("decode='flashrt' needs prefill='flashrt' (the rotated cache copy is kept by FlashRT writes)")
        try:
            from flashdrive import FlashDriveMixin
        except ImportError as exc:
            raise ImportError("the FlashDrive backend needs the flashdrive package") from exc
        if not isinstance(model, FlashDriveMixin):
            raise TypeError("FlashDriveThor needs a FlashDrive model (see load_flashdrive)")
        if model.training:
            raise ValueError("FlashDriveThor requires an eval-mode model")
        self.m = model
        self.backends = backends
        self.expert_splits = int(expert_splits)
        self._st = None                     # per-stream state (FlashDrive cache, tables, graphs)
        self.record_steps = False           # expert parity mode: keep per-step velocities
        self._steps_rec: dict = {}
        if "flashrt" not in backends.values():
            return
        import flash_rt.flash_rt_kernels as fvk
        from flash_rt.hardware.thor import fa4_backend
        from flash_rt.models.alpamayo15.pipeline_thor import require_alpamayo15_kernels
        require_alpamayo15_kernels(fvk)
        missing = [n for n in FLASHDRIVE_KERNEL_SYMBOLS if not hasattr(fvk, n)]
        if missing:
            raise RuntimeError(f"FlashRT was built without the FlashDrive backend kernels ({', '.join(missing)}); "
                               "rebuild with -DGPU_ARCH=110 -DFLASHRT_ENABLE_ALPAMAYO15=ON")
        self._fvk = fvk
        fvk.alpamayo15_set_pdl(bool(pdl))      # process-wide: programmatic dependent launch of the backend's kernels
        backends["pdl"] = bool(pdl)
        self._gemm = _Gemm(fvk)
        self._fa4 = fa4_backend.fa4_fwd()
        if self._fa4 is None:
            raise RuntimeError(f"FA4 unavailable: {fa4_backend.status()}")
        self._init_geometry()
        if expert == "flashrt":
            self._init_expert()
            model._action = self._action
        if vision == "flashrt":
            self.vlm = model.vlm
            self.vision_attention = "fa4"
            self._vis_tables: dict = {}
            self._vis_graphs: dict = {}
            backends["vision_gemm"] = vision_gemm
            if vision_gemm == "cutlass":
                self._check_vision_modules()
            model._encode = self._encode
        if expert == "flashrt":
            backends["expert_gemm"] = expert_gemm
            backends["expert_weights"] = expert_weights
        if w4a8 == "fused":
            self._w4, scratch, maps = [], {}, {}
            grp = lambda mods: _W4A8Group(fvk, mods, scratch, w4a8_gemm, maps, act_rot, w4a8_ktile, prefill_weights)
            backends["w4a8_ktile"] = w4a8_ktile
            backends["prefill_weights"] = prefill_weights
            for layer in model.vlm.model.language_model.layers:
                a, mlp = layer.self_attn, layer.mlp
                self._w4.append({"qkv": grp([a.q_proj, a.k_proj, a.v_proj]), "o": grp([a.o_proj]),
                                 "gu": grp([mlp.gate_proj, mlp.up_proj]), "down": grp([mlp.down_proj])})
            del maps, grp
            torch.cuda.empty_cache()        # index maps of the weight unpacking (about 1 GB, transient)
            if w4a8_gemm == "tcgen05":      # unpacked layout checked against Marlin itself, once per shape
                self.w4a8_gemm_agreement = min(g.check_against_marlin(fvk) for g in self._w4[0].values())
        if prefill == "flashrt" and expert == "flashdrive":
            fd_action = model._action

            def action_then_sync(**kw):     # FlashDrive's expert writes action-row K/V; mirror them into kvlm
                out = fd_action(**kw)
                st = self._stream()
                slot = kw["cache_position"][:1]
                for li in range(self.n_layers):
                    self._rope_rows(st, li, st["kvlm"][li], st["cos"], st["sin"], 0, self.n_act, slot=slot)
                return out
            model._action = action_then_sync
        if prefill == "flashrt":
            self._orig_first_prefill = model._first_prefill
            model._first_prefill = self._first_prefill
            model._dflash_prefill = self._dflash_prefill
            model._shift_streaming_kv_cache = self._shift
        if decode == "flashrt":
            backends["decode_weights"] = decode_weights
            self._dw = self._decode_linears(decode_weights)
            model._dflash_block_step = self._block_step
            model._dflash_traj_forward = self._traj_forward

    # ------------------------------------------------------------------
    # geometry and per-stream state
    # ------------------------------------------------------------------
    def _init_geometry(self) -> None:
        m = self.m
        tcfg = m.vlm.config.text_config
        ecfg = m.expert.config
        if ecfg.num_hidden_layers != tcfg.num_hidden_layers or ecfg.num_key_value_heads != tcfg.num_key_value_heads:
            raise ValueError("expert attention geometry must match the VLM cache")
        if tcfg.head_dim != 128 or ecfg.head_dim != 128:
            raise ValueError("Alpamayo 1.5 kernels require head_dim 128")
        self.n_layers = tcfg.num_hidden_layers
        self.n_kv = tcfg.num_key_value_heads
        self.head_dim = 128
        self.n_heads = tcfg.num_attention_heads
        self.hidden = tcfg.hidden_size
        self.ffn = tcfg.intermediate_size
        self.eps = tcfg.rms_norm_eps
        self.e_heads = ecfg.num_attention_heads
        self.e_hidden = ecfg.hidden_size
        self.e_ffn = ecfg.intermediate_size
        self.e_eps = ecfg.rms_norm_eps
        self.n_act, self.act_dim = m.action_space.get_action_space_dims()
        self.scale = self.head_dim ** -0.5

    def _stream(self) -> dict:
        """State bound to FlashDrive's current stream: its cache object and fixed position table."""
        m = self.m
        pkv = m._past_key_values
        pos = m.streaming_position_ids
        st = self._st
        if st is not None and st["pkv"] is pkv and st["pos"] is pos:
            return st
        if pkv is None or pos is None:
            raise RuntimeError("no active FlashDrive stream")
        S = pkv.max_cache_len
        for layer in pkv.layers:
            if not layer.is_initialized or layer.keys.shape[0] != 1 or layer.keys.shape[2] != S:
                raise ValueError("FlashDriveThor expects an initialized single-sample FlashDrive cache")
        st = {"pkv": pkv, "pos": pos, "S": S, "graphs": {}, "keep": {},
              "k": [layer.keys for layer in pkv.layers], "v": [layer.values for layer in pkv.layers]}
        if self.backends["prefill"] == "flashrt":
            # VLM rotary tables over the stream's fixed position table (BF16, as FlashDrive's BF16 forward computes).
            cos, sin = m.vlm.model.language_model.rotary_emb(torch.zeros(1, device=DEV, dtype=BF16), pos)
            st["cos"], st["sin"] = cos[0].contiguous(), sin[0].contiguous()
            st["kvlm"] = torch.empty(self.n_layers, S, self.n_kv, self.head_dim, device=DEV, dtype=BF16)
        if self.backends["expert"] == "flashrt":
            cos, sin = m.expert.rotary_emb(torch.zeros(1, device=DEV, dtype=torch.float32), pos)
            st["e_cos_tab"], st["e_sin_tab"] = cos[0].contiguous(), sin[0].contiguous()
            st["ke"] = torch.empty(self.n_layers, S, self.n_kv, self.head_dim, device=DEV, dtype=BF16)
        self._st = st
        if "kvlm" in st:
            self._sync_kvlm_all(st)          # rotated view of whatever FlashDrive's cache already holds
        return st

    def _v_view(self, st: dict, li: int) -> torch.Tensor:
        """FlashDrive's value cache of layer ``li`` as an attention view [1, S, Hk, D] (strided, no copy)."""
        return st["v"][li].permute(0, 2, 1, 3)

    def _rope_rows(self, st: dict, li: int, dst: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, fp32_rule: int,
                   rows: int, row_map: torch.Tensor | None = None, row0: int = 0, slot: torch.Tensor | None = None,
                   n_dev: torch.Tensor | None = None) -> None:
        S, D = st["S"], self.head_dim
        self._fvk.flashdrive_rope_rows(
            st["k"][li].data_ptr(), D, S * D, dst.data_ptr(), self.n_kv * D, D, cos.data_ptr(), sin.data_ptr(),
            fp32_rule, row_map.data_ptr() if row_map is not None else 0, row0,
            slot.data_ptr() if slot is not None else 0, n_dev.data_ptr() if n_dev is not None else 0,
            rows, self.n_kv, torch.cuda.current_stream().cuda_stream)

    def _sync_kvlm_all(self, st: dict) -> None:
        """Rotate every cache row of every layer into the VLM view (after FlashDrive wrote the cache)."""
        for li in range(self.n_layers):
            self._rope_rows(st, li, st["kvlm"][li], st["cos"], st["sin"], 0, st["S"])

    # ------------------------------------------------------------------
    # expert
    # ------------------------------------------------------------------
    def _init_expert(self) -> None:
        m = self.m
        if not all(getattr(layer.self_attn, "fuse_qkv", False) and getattr(layer.mlp, "fuse_gate_up", False)
                   for layer in m.expert.layers):
            raise ValueError("FlashDriveThor expects FlashDrive's fused expert projections (setup_expert_fusion)")
        if not m.config.expert_non_causal_attention:
            raise ValueError("FlashDriveThor's expert attention is non-causal (expert_non_causal_attention=True)")
        if getattr(m.expert.config, "hidden_act", None) != "silu":
            raise ValueError(f"FlashDriveThor's expert MLP is SiLU-gated, got {getattr(m.expert.config, 'hidden_act', None)!r}")
        for li, layer in enumerate(m.expert.layers):
            for name, lin in (("qkv_proj", layer.self_attn.qkv_proj), ("o_proj", layer.self_attn.o_proj),
                              ("gate_up_proj", layer.mlp.gate_up_proj), ("down_proj", layer.mlp.down_proj)):
                if getattr(lin, "bias", None) is not None:
                    raise ValueError(f"expert layer {li} {name} has a bias; FlashDriveThor's expert GEMMs have none")
        A, E, D = self.n_act, self.e_hidden, self.head_dim
        mk = lambda *shape, dt=BF16: torch.zeros(*shape, device=DEV, dtype=dt)
        self._eb = {"res": mk(A, E, dt=torch.float32), "hn": mk(A, E),
                    "qkv": mk(A, (self.e_heads + 2 * self.n_kv) * D), "q": mk(A, self.e_heads * D),
                    "o": mk(A, E), "d": mk(A, E), "gu": mk(A, 2 * self.e_ffn), "act": mk(A, self.e_ffn),
                    "e_cos": mk(A, D, dt=torch.float32), "e_sin": mk(A, D, dt=torch.float32)}
        self._e8 = None
        if self.expert_weights == "fp8":
            from flash_rt.models.alpamayo15.pipeline_thor import quantize_fp8_rows
            self._e8 = []
            with torch.no_grad():
                for layer in m.expert.layers:
                    q = {}
                    for name, lin in (("qkv", layer.self_attn.qkv_proj), ("o", layer.self_attn.o_proj),
                                      ("gu", layer.mlp.gate_up_proj), ("down", layer.mlp.down_proj)):
                        codes, scale = quantize_fp8_rows(lin.weight.detach())
                        q[name] = (codes.contiguous(), scale.contiguous())
                    self._e8.append(q)
            torch.cuda.empty_cache()
            u8 = lambda *shape: torch.empty(*shape, device=DEV, dtype=torch.uint8)
            f32 = lambda *shape: torch.empty(*shape, device=DEV, dtype=torch.float32)
            self._eb.update({"hn8": u8(A, E), "hn_s": f32(A), "att8": u8(A, self.e_heads * D), "att_s": f32(A),
                             "act8": u8(A, self.e_ffn), "act_s": f32(A)})
        self.noise = mk(1, A, self.act_dim)            # FlashDrive's BF16 action noise buffer
        self.x_out = mk(1, A, self.act_dim)
        self.v_cache = mk(1, A, self.act_dim)
        self.d_slot = torch.zeros(1, dtype=torch.long, device=DEV)    # action_start (first action cache row)
        self.d_seqk = torch.zeros(1, dtype=torch.int32, device=DEV)   # action_start + n_act keys attended
        self.d_ke_rows = torch.zeros(1, dtype=torch.int32, device=DEV)  # rows [0, action_start) to rotate

    def _velocity(self, x: torch.Tensor, t: torch.Tensor, st: dict, rec: list | None) -> torch.Tensor:
        """One FlashDrive expert evaluation (``step_fn``) on FlashRT kernels; returns v [1, A, act_dim] bf16."""
        m, fvk = self.m, self._fvk
        s = torch.cuda.current_stream().cuda_stream
        A, E, D, HQ, KV, F_, S = self.n_act, self.e_hidden, self.head_dim, self.e_heads, self.n_kv, self.e_ffn, st["S"]
        b = self._eb
        res, hn, qkv, q, o, d = b["res"], b["hn"], b["qkv"], b["q"], b["o"], b["d"]
        eps = self.e_eps
        layers = m.expert.layers
        emb = m.action_in_proj(x, t)
        res.copy_(emb.reshape(A, E))                                             # FP32 residual stream
        fvk.alpamayo15_res_rms_f32_bf16(res.data_ptr(), 0, layers[0].input_layernorm.weight.data_ptr(),
                                        hn.data_ptr(), A, E, eps, s)
        ld = qkv.shape[1]
        cut = self.expert_gemm == "cutlass"
        e8 = self._e8

        def lin8(x, name, li, y):          # W8A8: per-row FP8 activations x per-row FP8 weights, scales in the epilogue
            q8, q_s = {"hn": (b["hn8"], b["hn_s"]), "att": (b["att8"], b["att_s"]), "act": (b["act8"], b["act_s"])}[x]
            src = {"hn": hn, "att": b["q"], "act": b["act"]}[x]
            fvk.alpamayo15_quantize_fp8_rows_bf16(src.data_ptr(), src.stride(0), q8.data_ptr(), q_s.data_ptr(), A, src.shape[1], s)
            w8, w_s = e8[li][name]
            return self._gemm.fp8(q8, q_s, w8, w_s, y)

        for li, layer in enumerate(layers):
            a = layer.self_attn
            if e8 is not None:
                lin8("hn", "qkv", li, qkv)
            elif cut:
                self._gemm(hn, a.qkv_proj.weight, qkv, tile=1)
            else:
                torch.mm(hn, a.qkv_proj.weight.t(), out=qkv)
            # q rotated (FP32 rule); action-row K (normalized, un-rotated) and V written into FlashDrive's cache
            # at action_start, as FlashDrive's expert forward does, and K's rotation into the expert view.
            fvk.flashdrive_qk_prep_bf16(
                qkv.data_ptr(), qkv[:, HQ * D:].data_ptr(), qkv[:, (HQ + KV) * D:].data_ptr(), ld, ld, ld,
                a.q_norm.weight.data_ptr(), a.k_norm.weight.data_ptr(), b["e_cos"].data_ptr(), b["e_sin"].data_ptr(), 1,
                q.data_ptr(), st["k"][li].data_ptr(), D, S * D, st["ke"][li].data_ptr(), KV * D, D,
                st["v"][li].data_ptr(), D, S * D, 0, self.d_slot.data_ptr(), A, HQ, KV, eps, s)
            out, _ = self._fa4(q.view(1, A, HQ, D), st["ke"][li].unsqueeze(0), self._v_view(st, li),
                               seqused_k=self.d_seqk, softmax_scale=self.scale, causal=False, pack_gqa=True,
                               num_splits=self.expert_splits)
            if e8 is not None:
                b["q"].copy_(out.view(A, HQ * D))        # attention output rows (q buffer reused as the FP8 source)
                lin8("att", "o", li, o)
            elif cut:
                self._gemm(out.view(A, HQ * D), a.o_proj.weight, o, tile=1)
            else:
                torch.mm(out.view(A, HQ * D), a.o_proj.weight.t(), out=o)
            fvk.alpamayo15_res_rms_f32_bf16(res.data_ptr(), o.data_ptr(), layer.post_attention_layernorm.weight.data_ptr(),
                                            hn.data_ptr(), A, E, eps, s)
            if cut:
                gu, act = b["gu"], b["act"]
                if e8 is not None:
                    lin8("hn", "gu", li, gu)
                else:
                    self._gemm(hn, layer.mlp.gate_up_proj.weight, gu, tile=1)
                # gate | up are the halves of one row-major [A, 2F] GEMM output: split them for the SiLU-mul kernel
                fvk.alpamayo15_silu_mul_split_bf16(gu.data_ptr(), gu[:, F_:].data_ptr(), 2 * F_, act.data_ptr(), A, F_, s)
                if e8 is not None:
                    lin8("act", "down", li, d)
                else:
                    self._gemm(act, layer.mlp.down_proj.weight, d, tile=1)
            else:
                gu = torch.mm(hn, layer.mlp.gate_up_proj.weight.t())
                act = F.silu(gu[:, :F_]) * gu[:, F_:]
                torch.mm(act, layer.mlp.down_proj.weight.t(), out=d)
            nxt = layers[li + 1].input_layernorm.weight if li + 1 < len(layers) else m.expert.norm.weight
            fvk.alpamayo15_res_rms_f32_bf16(res.data_ptr(), d.data_ptr(), nxt.data_ptr(), hn.data_ptr(), A, E, eps, s)
        v = F.linear(hn, m.action_out_proj.weight, m.action_out_proj.bias).view(1, A, self.act_dim)
        if rec is not None:
            rec.append(v.clone())
        return v

    def _denoise(self, st: dict, schedule: list[bool], time_steps: torch.Tensor, rec: dict | None) -> None:
        # Expert view of every cached prompt/reasoning row (FP32 rule); the action rows are written per step.
        for li in range(self.n_layers):
            self._rope_rows(st, li, st["ke"][li], st["e_cos_tab"], st["e_sin_tab"], 1, st["S"], n_dev=self.d_ke_rows)
        b = self._eb
        idx = torch.arange(self.n_act, device=DEV) + self.d_slot
        torch.index_select(st["e_cos_tab"], 0, idx, out=b["e_cos"])
        torch.index_select(st["e_sin_tab"], 0, idx, out=b["e_sin"])
        x = self.noise
        for i, cached in enumerate(schedule):
            dt = time_steps[i + 1] - time_steps[i]
            t = time_steps[i].view(1, 1, 1).expand(1, 1, 1)
            if cached:
                v = self.v_cache
            else:
                v = self._velocity(x, t, st, None if rec is None else rec["v"])
                self.v_cache.copy_(v)
            x = x + dt * v
            if rec is not None:
                rec["x"].append(x.clone())
        self.x_out.copy_(x)
        if "kvlm" in st:  # keep the VLM view of the action rows FlashDrive's cache now holds
            for li in range(self.n_layers):
                self._rope_rows(st, li, st["kvlm"][li], st["cos"], st["sin"], 0, self.n_act, slot=self.d_slot)

    @torch.no_grad()
    def _action(self, total_samples: int, position_ids: torch.Tensor, attention_mask: torch.Tensor,
                cache_position: torch.Tensor, diffusion_kwargs: dict[str, Any] | None = None) -> torch.Tensor:
        """Drop-in for FlashDrive's ``_action`` (same arguments, same BF16 [1, A, act_dim] result)."""
        kw = dict(diffusion_kwargs or {})
        if kw.get("int_method") != "euler_with_cache":
            raise ValueError("FlashDriveThor's expert graph implements FlashDrive's euler_with_cache integrator; "
                             f"got int_method={kw.get('int_method')!r}")
        if total_samples != 1:
            raise ValueError(f"FlashDriveThor serves one trajectory sample per window, got {total_samples}")
        _require_autocast()
        n_steps = int(kw.get("inference_step", self.m.diffusion.num_inference_steps))
        schedule = tuple(euler_cache_schedule(n_steps, kw["cache_steps"]))
        st = self._stream()
        if position_ids is not st["pos"]:
            raise ValueError("expert position table differs from the stream's")
        action_start = int(cache_position[0])
        if action_start + self.n_act > st["S"]:
            raise ValueError(f"action rows [{action_start}, {action_start + self.n_act}) exceed the cache ({st['S']})")
        key = ("expert", schedule, n_steps, self.record_steps)
        g = st["graphs"].get(key)
        if g is None:
            time_steps = torch.linspace(0.0, 1.0, n_steps + 1, device=DEV, dtype=BF16)   # FlashDrive's BF16 grid
            st["keep"][key] = time_steps                   # captured by address: keep alive with the graph
            rec = {"v": [], "x": []} if self.record_steps else None
            # Warm up / capture on scratch rows past every live row; the touched cache rows are restored.
            scratch = st["S"] - self.n_act
            saved = [t.clone() for t in (self.noise, self.d_slot, self.d_seqk, self.d_ke_rows, self.v_cache)]
            rows = [(k[0, :, scratch:].clone(), v[0, :, scratch:].clone()) for k, v in zip(st["k"], st["v"])]
            kvlm_rows = st["kvlm"][:, scratch:].clone() if "kvlm" in st else None
            self.d_slot.fill_(scratch); self.d_seqk.fill_(st["S"]); self.d_ke_rows.fill_(scratch)
            self.noise.zero_()
            g = _capture(lambda: self._denoise(st, list(schedule), time_steps, rec))
            for t, sv in zip((self.noise, self.d_slot, self.d_seqk, self.d_ke_rows, self.v_cache), saved):
                t.copy_(sv)
            for (k, v), (ks, vs) in zip(zip(st["k"], st["v"]), rows):
                k[0, :, scratch:].copy_(ks); v[0, :, scratch:].copy_(vs)
            if kvlm_rows is not None:
                st["kvlm"][:, scratch:].copy_(kvlm_rows)
            if rec is not None:          # keep the captured pass only (the two warm-ups ran on scratch rows)
                for k_ in rec:
                    rec[k_][:] = rec[k_][len(rec[k_]) * 2 // 3:]
            st["keep"][("rec",) + key] = rec or {}
            st["graphs"][key] = g
        self._steps_rec = st["keep"][("rec",) + key]
        self.d_slot.fill_(action_start)
        self.d_seqk.fill_(action_start + self.n_act)
        self.d_ke_rows.fill_(action_start)
        self.noise.normal_()             # FlashDrive's draw: BF16 noise, sampled before the graph
        g.replay()
        return self.x_out.clone()

    # ------------------------------------------------------------------
    # vision
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _encode(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor):
        """Drop-in for FlashDrive's ``_encode``: (image embeddings, DeepStack features) of the window's frames."""
        _require_autocast()
        key = (tuple(pixel_values.shape), tuple(image_grid_thw.flatten().tolist()))
        ent = self._vis_graphs.get(key)
        if ent is None:
            px = torch.empty_like(pixel_values)
            px.copy_(pixel_values)
            vision = self._vision_cutlass if self.vision_gemm == "cutlass" else self._vision
            with torch.autocast("cuda", dtype=BF16):
                vision(px, image_grid_thw)                  # builds the per-grid tables outside capture
            grid_host = image_grid_thw.cpu()                # table lookup key only: no device sync while capturing
            tables = self._vis_tables[tuple(grid_host.flatten().tolist())]   # captured by address: keep alive
            out = {}

            def run():
                out["img"], out["deep"] = vision(px, grid_host)
            ent = {"px": px, "out": out, "tables": tables, "graph": _capture(run)}
            self._vis_graphs[key] = ent
        ent["px"].copy_(pixel_values)
        ent["graph"].replay()
        return ent["out"]["img"], ent["out"]["deep"]

    # ------------------------------------------------------------------
    # language model on FlashRT kernels (FlashDrive's W4A8 projections)
    # ------------------------------------------------------------------
    def _lm_buffers(self, st: dict, n: int, ctx_rows: int) -> dict:
        key = ("lmbuf", n, ctx_rows)
        b = st["keep"].get(key)
        if b is None:
            H, D, NH, F_ = self.hidden, self.head_dim, self.n_heads, self.ffn
            mk = lambda *shape, dt=BF16: torch.empty(*shape, device=DEV, dtype=dt)
            ncap = len(self.m.vlm.model.language_model._capture_layer_ids)
            b = {"x": mk(n, H), "h": torch.zeros(n + 8, H, device=DEV, dtype=BF16)[:n],   # few-row GEMMs read 8-row tiles of h
                 "q": mk(n, NH * D), "attn": mk(n, NH * D), "act": mk(n, F_),
                 "cos": mk(n, D), "sin": mk(n, D), "rows": torch.zeros(n, dtype=torch.long, device=DEV),
                 "ctx": mk(ctx_rows, ncap * H)}
            st["keep"][key] = b
        return b

    def _lm_layers(self, st: dict, b: dict, n: int, attend, row_map: torch.Tensor | None, slot: torch.Tensor | None,
                   deep: list | None = None, vis_idx: torch.Tensor | None = None, cap_from: int = 0) -> None:
        """36 decoder layers over ``b['x']`` [n, H]; leaves the final-norm output in ``b['h']`` and the
        DFlash context (captured layers, rows ``cap_from:``) in ``b['ctx']``."""
        m, fvk = self.m, self._fvk
        s = torch.cuda.current_stream().cuda_stream
        lm = m.vlm.model.language_model
        layers = lm.layers
        H, D, NH, KV, S, F_, eps = self.hidden, self.head_dim, self.n_heads, self.n_kv, st["S"], self.ffn, self.eps
        x, h, q, act = b["x"], b["h"], b["q"], b["act"]
        cap_ids = sorted(lm._capture_layer_ids)
        # FlashRT W4A8 path over many rows (streaming-window prefill): the producer of each W4A8 input (RMSNorm,
        # SiLU-mul, attention output) and its activation preparation run as one kernel over row tiles; the normalized
        # rows are never materialised (h only for the LM head). A few rows (verification, trajectory start) keep the
        # separate kernels, whose grids spread the rotation groups over the SMs.
        fused = self.backends["w4a8"] == "fused" and self._w4[0]["qkv"].rot is None and n > SMALL_MAX_ROWS
        if fused:
            self._w4[0]["qkv"].prep(fvk, 1, None, n, residual=x, norm_w=layers[0].input_layernorm.weight, eps=eps)
        else:
            fvk.alpamayo15_rms_bf16(x.data_ptr(), 0, layers[0].input_layernorm.weight.data_ptr(), h.data_ptr(), n, H, eps, s)
        for li, layer in enumerate(layers):
            a = layer.self_attn
            # fused: FP16 GEMM outputs consumed directly (cast to BF16 on load, as FlashDrive's wrapper casts)
            w4 = self._w4[li] if self.backends["w4a8"] == "fused" else None
            if fused:
                qp, kp, vp = w4["qkv"].gemm(fvk, n)
            elif w4 is not None:
                qp, kp, vp = w4["qkv"](fvk, h)
            else:
                qp, kp, vp = a.q_proj(h), a.k_proj(h), a.v_proj(h)    # FlashDrive's W4A8 (ParoQuant) projections
            fvk.flashdrive_qk_prep_bf16(
                qp.data_ptr(), kp.data_ptr(), vp.data_ptr(), qp.shape[1], kp.shape[1], vp.shape[1],
                a.q_norm.weight.data_ptr(), a.k_norm.weight.data_ptr(), b["cos"].data_ptr(), b["sin"].data_ptr(), 0,
                q.data_ptr(), st["k"][li].data_ptr(), D, S * D, st["kvlm"][li].data_ptr(), KV * D, D,
                st["v"][li].data_ptr(), D, S * D, row_map.data_ptr() if row_map is not None else 0,
                slot.data_ptr() if slot is not None else 0, n, NH, KV, eps, s, f16_in=int(w4 is not None))
            attend(st, li, q, b["attn"])
            post_w = layer.post_attention_layernorm.weight.data_ptr()
            nxt = layers[li + 1].input_layernorm.weight if li + 1 < len(layers) else lm.norm.weight
            if fused:
                w4["o"].prep(fvk, 0, b["attn"], n)
                o_out = w4["o"].gemm(fvk, n)[0]
                w4["gu"].prep(fvk, 1, o_out, n, residual=x, norm_w=layer.post_attention_layernorm.weight, eps=eps)
                g, u = w4["gu"].gemm(fvk, n)
                w4["down"].prep(fvk, 2, g, n, src2=u)
                d = w4["down"].gemm(fvk, n)[0]
                if deep is not None and li < len(deep):   # residual += down, DeepStack injection, next input norm
                    d16 = d.to(BF16)
                    fvk.residual_add(x.data_ptr(), d16.data_ptr(), n * H, s)
                    x.index_add_(0, vis_idx, deep[li])
                    if li + 1 < len(layers):
                        self._w4[li + 1]["qkv"].prep(fvk, 1, None, n, residual=x, norm_w=nxt, eps=eps)
                    else:
                        fvk.alpamayo15_rms_bf16(x.data_ptr(), 0, nxt.data_ptr(), h.data_ptr(), n, H, eps, s)
                elif li + 1 < len(layers):                # residual += down, next input norm, next layer's q|k|v preparation
                    self._w4[li + 1]["qkv"].prep(fvk, 1, d, n, residual=x, norm_w=nxt, eps=eps)
                else:
                    fvk.flashdrive_rms_f16delta_bf16(x.data_ptr(), d.data_ptr(), nxt.data_ptr(), h.data_ptr(), n, H, eps, s)
                if li in cap_ids:
                    j = cap_ids.index(li)
                    b["ctx"][:, j * H:(j + 1) * H].copy_(x[cap_from:])
                continue
            if w4 is not None:
                fvk.flashdrive_rms_f16delta_bf16(x.data_ptr(), w4["o"](fvk, b["attn"])[0].data_ptr(), post_w,
                                                 h.data_ptr(), n, H, eps, s)
            else:
                fvk.alpamayo15_rms_bf16(x.data_ptr(), a.o_proj(b["attn"]).data_ptr(), post_w, h.data_ptr(), n, H, eps, s)
            mlp = layer.mlp
            if w4 is not None:
                g, u = w4["gu"](fvk, h)
                fvk.flashdrive_silu_mul_f16in_bf16(g.data_ptr(), u.data_ptr(), act.data_ptr(), n * F_, s)
                d = w4["down"](fvk, act)[0]
            else:
                g, u = mlp.gate_proj(h), mlp.up_proj(h)
                fvk.silu_mul_qwen36_bf16(g.data_ptr(), u.data_ptr(), act.data_ptr(), n * F_, s)
                d = mlp.down_proj(act)
            nxt = layers[li + 1].input_layernorm.weight if li + 1 < len(layers) else lm.norm.weight
            if deep is not None and li < len(deep):
                if w4 is not None:
                    d = d.to(BF16)
                fvk.residual_add(x.data_ptr(), d.data_ptr(), n * H, s)
                x.index_add_(0, vis_idx, deep[li])                      # FlashDrive's DeepStack injection
                fvk.alpamayo15_rms_bf16(x.data_ptr(), 0, nxt.data_ptr(), h.data_ptr(), n, H, eps, s)
            elif w4 is not None:
                fvk.flashdrive_rms_f16delta_bf16(x.data_ptr(), d.data_ptr(), nxt.data_ptr(), h.data_ptr(), n, H, eps, s)
            else:
                fvk.alpamayo15_rms_bf16(x.data_ptr(), d.data_ptr(), nxt.data_ptr(), h.data_ptr(), n, H, eps, s)
            if li in cap_ids:                                           # decoder-layer output, as FlashDrive captures
                j = cap_ids.index(li)
                b["ctx"][:, j * H:(j + 1) * H].copy_(x[cap_from:])

    # streaming prefill ---------------------------------------------------
    def _first_prefill(self, input_ids, attention_mask, pixel_values, image_grid_thw) -> None:
        """FlashDrive's full first-window prefill; the stream state created next rotates every row it wrote."""
        self._orig_first_prefill(input_ids, attention_mask, pixel_values, image_grid_thw)
        self._stream()

    def _attend_prefill(self, st: dict, li: int, q: torch.Tensor, out: torch.Tensor) -> None:
        p = st["keep"]["prefill_meta"]
        o, _ = self._fa4(q.view(-1, self.n_heads, self.head_dim), st["kvlm"][li], self._v_view(st, li)[0],
                         cu_seqlens_q=p["cu_q"], cu_seqlens_k=p["cu_k"], seqused_k=p["used"],
                         max_seqlen_q=p["max_q"], max_seqlen_k=st["S"], softmax_scale=self.scale, causal=True,
                         pack_gqa=True)
        out.copy_(o.view(out.shape))

    @torch.no_grad()
    def _dflash_prefill(self, inputs_embeds, position_ids, cache_position, visual_pos_masks, deepstack_image_embeds,
                        streaming_attention_mask):
        """Drop-in for FlashDrive's ``_dflash_prefill``: (last-row logits [1, V], DFlash context [1, ctx, C])."""
        _require_autocast()
        m = self.m
        st = self._stream()
        n = inputs_embeds.shape[1]
        g = st["graphs"].get("prefill")
        if g is not None and n != st["keep"]["prefill_n"]:
            raise ValueError(f"streaming window has {n} rows; this stream's prefill graph was captured for "
                             f"{st['keep']['prefill_n']}")
        b = self._lm_buffers(st, n, m._dflash_context_len)
        if g is None:
            st["keep"]["prefill_n"] = n
            rows = cache_position.tolist()
            if not m._extended_vision_ranges:
                raise ValueError("causal-by-position streaming attention holds for extended vision ranges (Alpamayo 1.5)")
            runs = contiguous_runs(rows)
            lens = [e - s for s, e in runs]
            st["keep"]["prefill_meta"] = {
                "cu_q": torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32, device=DEV),
                "cu_k": torch.zeros(len(runs) + 1, dtype=torch.int32, device=DEV),
                "used": torch.tensor([e for _, e in runs], dtype=torch.int32, device=DEV), "max_q": max(lens)}
            b["rows"].copy_(cache_position)
            st["keep"]["prefill_rows_obj"] = cache_position   # FlashDrive's fixed per-stream tensor
            b["emb"] = torch.empty_like(inputs_embeds[0])
            b["logits"] = torch.zeros(1, m.vlm.lm_head.weight.shape[0], device=DEV, dtype=BF16)
            b["vis"] = visual_pos_masks[0].nonzero().squeeze(1)
            b["deep"] = [torch.empty_like(t) for t in deepstack_image_embeds]
            b["ctx_len"] = m._dflash_context_len
            b["emb"].copy_(inputs_embeds[0])
            for dst, src in zip(b["deep"], deepstack_image_embeds):
                dst.copy_(src)
            torch.index_select(st["cos"], 0, b["rows"], out=b["cos"])
            torch.index_select(st["sin"], 0, b["rows"], out=b["sin"])
            saved = [(k.clone(), v.clone()) for k, v in zip(st["k"], st["v"])]   # capture writes the window rows
            kvlm = st["kvlm"].clone()
            out = {}

            def run():
                b["x"].copy_(b["emb"])
                self._lm_layers(st, b, n, self._attend_prefill, b["rows"], None, b["deep"], b["vis"],
                                cap_from=n - b["ctx_len"])
                out["logits"] = _gemm_small(self._fvk, b["h"][n - 1:n], self._dw["lm_head"] if self.backends["decode"] == "flashrt"
                                            else m.vlm.lm_head.weight, b["logits"])
            g = _capture(run)
            for (k, v), (ks, vs) in zip(zip(st["k"], st["v"]), saved):
                k.copy_(ks); v.copy_(vs)
            st["kvlm"].copy_(kvlm)
            del saved, kvlm
            st["graphs"]["prefill"] = g
            st["keep"]["prefill_out"] = out
        elif cache_position is not st["keep"]["prefill_rows_obj"]:
            raise ValueError("streaming cache positions changed within a stream")
        b["emb"].copy_(inputs_embeds[0])
        for dst, src in zip(b["deep"], deepstack_image_embeds):
            dst.copy_(src)
        # Image-token rows are fixed per stream (FlashDrive caches them at the first streaming window too).
        g.replay()
        return st["keep"]["prefill_out"]["logits"], b["ctx"].unsqueeze(0)

    @torch.no_grad()
    def _shift(self) -> None:
        """Drop-in for FlashDrive's ``_shift_streaming_kv_cache`` (frame blocks f -> f-1 of every view),
        plus the VLM view of the moved rows: one kernel over every layer's keys and values."""
        m = self.m
        st = self._stream()
        sh = st["keep"].get("shift")
        if sh is None:
            # FlashDrive's per-view frame ranges define its shift index pair; the kernel takes the ranges themselves.
            frames = m._frame_ranges
            F = len(frames[0])
            if F < 2 or any(len(v) != F for v in frames):
                raise ValueError("FlashDrive streaming needs the same number (>= 2) of frames per view")
            src, dst = [], []
            for view in frames:
                for (d0, d1), (s0, s1) in zip(view[:-1], view[1:]):
                    if d1 - d0 != s1 - s0:
                        raise ValueError("FlashDrive frame blocks of one view must have equal lengths")
                    dst.append(torch.arange(d0, d1)); src.append(torch.arange(s0, s1))
            src, dst = torch.cat(src).to(DEV), torch.cat(dst).to(DEV)
            if not torch.equal(src, m._shift_src_index) or not torch.equal(dst, m._shift_dst_index):
                raise ValueError("FlashDrive shift indices do not match its frame ranges")
            sh = {"ranges": torch.tensor(frames, dtype=torch.int32, device=DEV).contiguous(), "F": F,
                  "n_views": len(frames),
                  "k_ptrs": torch.tensor([t.data_ptr() for t in st["k"]], dtype=torch.int64, device=DEV),
                  "v_ptrs": torch.tensor([t.data_ptr() for t in st["v"]], dtype=torch.int64, device=DEV)}
            st["keep"]["shift"] = sh
        S, D = st["S"], self.head_dim
        self._fvk.flashdrive_shift_cache(sh["k_ptrs"].data_ptr(), sh["v_ptrs"].data_ptr(), D, S * D,
                                         st["kvlm"].data_ptr(), st["cos"].data_ptr(), st["sin"].data_ptr(),
                                         sh["ranges"].data_ptr(), sh["n_views"], sh["F"], self.n_layers, S, self.n_kv,
                                         torch.cuda.current_stream().cuda_stream)

    # decode ----------------------------------------------------------------
    def _attend_verify(self, st: dict, li: int, q: torch.Tensor, out: torch.Tensor) -> None:
        n = q.shape[0]
        o, _ = self._fa4(q.view(1, n, self.n_heads, self.head_dim), st["kvlm"][li].unsqueeze(0), self._v_view(st, li),
                         seqused_k=st["keep"]["dec"]["used"], softmax_scale=self.scale, causal=True, pack_gqa=True)
        out.copy_(o.view(out.shape))

    def _attend_all_rows(self, st: dict, li: int, q: torch.Tensor, out: torch.Tensor) -> None:
        # FlashDrive's one-row forward passes no mask: SDPA attends every cache row.
        o, _ = self._fa4(q.view(1, 1, self.n_heads, self.head_dim), st["kvlm"][li].unsqueeze(0), self._v_view(st, li),
                         softmax_scale=self.scale, causal=False, pack_gqa=True)
        out.copy_(o.view(out.shape))

    def _dec_state(self, st: dict) -> dict:
        d = st["keep"].get("dec")
        if d is None:
            m = self.m
            B, C = m.dflash_block_size, m._dflash_context_len
            dm = m._draft_model
            H, D, V = self.hidden, self.head_dim, m.vlm.lm_head.weight.shape[0]
            dcfg = dm.config
            if (dcfg.hidden_size != H or dcfg.head_dim != D or dcfg.num_key_value_heads != self.n_kv
                    or dcfg.num_attention_heads != self.n_heads or getattr(dcfg, "hidden_act", "silu") != "silu"
                    or any(getattr(lin, "bias", None) is not None for layer in dm.layers for lin in
                           (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj, layer.self_attn.o_proj,
                            layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj))
                    or dm.fc.bias is not None or m.vlm.lm_head.bias is not None):
                raise ValueError("FlashDriveThor's draft kernels expect the VLM's geometry, SiLU MLP and bias-free linears")
            if B > 8 or C > 8:
                raise ValueError("FlashDriveThor's draft kernels take at most 8 block and 8 context rows")
            mk = lambda *shape, dt=BF16: torch.zeros(*shape, device=DEV, dtype=dt)
            cos, sin = dm.rotary_emb(torch.zeros(1, device=DEV, dtype=BF16), torch.arange(C + B, device=DEV).unsqueeze(0))
            d = {"slot": torch.zeros(1, dtype=torch.long, device=DEV), "used": torch.zeros(1, dtype=torch.int32, device=DEV),
                 "block_ids": torch.zeros(1, B, dtype=torch.long, device=DEV),
                 "th": torch.zeros(1, C, len(dm.target_layer_ids) * H, device=DEV, dtype=BF16),
                 "traj_id": torch.full((1,), m.traj_start_token_id, dtype=torch.long, device=DEV),
                 "neg1": torch.full((), -1, dtype=torch.long, device=DEV),
                 "draft_pos": torch.arange(C + B, device=DEV).unsqueeze(0),
                 # draft model on FlashRT kernels (rows padded to whole 8-row tiles for the few-row GEMMs)
                 "d_cos": cos[0].contiguous(), "d_sin": sin[0].contiguous(),
                 "dx": mk(8, H), "dh": mk(8, H), "dthn": mk(8, H), "dkin": mk(16, H), "dq": mk(8, self.n_heads * D),
                 "dk": mk(16, self.n_kv * D), "dv": mk(16, self.n_kv * D), "dqo": mk(8, self.n_heads, D),
                 "dko": mk(16, self.n_kv, D), "dvo": mk(16, self.n_kv, D), "dattn": mk(8, self.n_heads * D),
                 "do": mk(8, H), "dg": mk(8, self.ffn), "du": mk(8, self.ffn), "dact": mk(8, self.ffn), "dd": mk(8, H),
                 "dhid": mk(9, H), "dlogits": mk(8, V), "vlogits": mk(8, V)}
            st["keep"]["dec"] = d
        return d

    def _decode_linears(self, decode_weights: str) -> dict:
        """Weights of the decode graphs' few-row GEMMs: the LM head and the draft linears, BF16 (the model's tensors)
        or per-row FP8 E4M3 copies for the W8A16 kernel (an opt-in precision tier)."""
        m = self.m
        dm = m._draft_model
        lin = {"lm_head": m.vlm.lm_head.weight, "fc": dm.fc.weight}
        for li, layer in enumerate(dm.layers):
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                lin[(li, name)] = getattr(layer.self_attn, name).weight
            for name in ("gate_proj", "up_proj", "down_proj"):
                lin[(li, name)] = getattr(layer.mlp, name).weight
        lin = {k: (_Fp8Weight(w, decode_weights) if decode_weights in ("fp8", "int8") else _Bf16Frag(w))
               for k, w in lin.items()}
        torch.cuda.empty_cache()
        return lin

    def _draft(self, d: dict, B: int) -> torch.Tensor:
        """FlashDrive's DFlash draft (``DFlashDraftModel.forward`` + greedy LM head on the proposal rows) on FlashRT
        kernels: context rows = hidden_norm(fc(target_hidden)), noise rows = the block embeddings; every layer attends
        the B noise queries over the C context + B noise keys. Returns the draft logits of rows 1..B-1 (bf16)."""
        m, fvk = self.m, self._fvk
        s = torch.cuda.current_stream().cuda_stream
        dm = m._draft_model
        lm = m.vlm.model.language_model
        H, D, NH, KV, F_, C = self.hidden, self.head_dim, self.n_heads, self.n_kv, self.ffn, m._dflash_context_len
        eps = dm.config.rms_norm_eps
        x, h, kin, thn = d["dx"][:B], d["dh"], d["dkin"], d["dthn"]
        torch.index_select(lm.embed_tokens.weight, 0, d["block_ids"][0], out=x)
        dw = self._dw
        _gemm_small(fvk, d["th"][0], dw["fc"], thn[:C])
        fvk.alpamayo15_rms_bf16(thn.data_ptr(), 0, dm.hidden_norm.weight.data_ptr(), kin.data_ptr(), C, H, eps, s)
        fvk.alpamayo15_rms_bf16(x.data_ptr(), 0, dm.layers[0].input_layernorm.weight.data_ptr(), kin[C:C + B].data_ptr(),
                                B, H, eps, s)
        for li, layer in enumerate(dm.layers):
            a = layer.self_attn
            _gemm_small(fvk, kin[C:C + B], dw[(li, "q_proj")], d["dq"][:B])
            _gemm_small(fvk, kin[:C + B], dw[(li, "k_proj")], d["dk"][:C + B])
            _gemm_small(fvk, kin[:C + B], dw[(li, "v_proj")], d["dv"][:C + B])
            fvk.flashdrive_draft_qkv_prep_bf16(d["dq"].data_ptr(), d["dk"].data_ptr(), d["dv"].data_ptr(), NH * D, KV * D, KV * D,
                                               a.q_norm.weight.data_ptr(), a.k_norm.weight.data_ptr(), d["d_cos"].data_ptr(),
                                               d["d_sin"].data_ptr(), d["dqo"].data_ptr(), d["dko"].data_ptr(),
                                               d["dvo"].data_ptr(), B, C, NH, KV, eps, s)
            o, _ = self._fa4(d["dqo"][:B].unsqueeze(0), d["dko"][:C + B].unsqueeze(0), d["dvo"][:C + B].unsqueeze(0),
                             softmax_scale=self.scale, causal=False, pack_gqa=True)
            d["dattn"][:B].copy_(o.view(B, NH * D))
            _gemm_small(fvk, d["dattn"][:B], dw[(li, "o_proj")], d["do"][:B])
            fvk.alpamayo15_rms_bf16(x.data_ptr(), d["do"].data_ptr(), layer.post_attention_layernorm.weight.data_ptr(),
                                    h.data_ptr(), B, H, eps, s)
            _gemm_small(fvk, h[:B], dw[(li, "gate_proj")], d["dg"][:B])
            _gemm_small(fvk, h[:B], dw[(li, "up_proj")], d["du"][:B])
            fvk.silu_mul_qwen36_bf16(d["dg"].data_ptr(), d["du"].data_ptr(), d["dact"].data_ptr(), B * F_, s)
            _gemm_small(fvk, d["dact"][:B], dw[(li, "down_proj")], d["dd"][:B])
            last = li + 1 == len(dm.layers)      # residual += mlp, then the final norm or the next input norm
            fvk.alpamayo15_rms_bf16(x.data_ptr(), d["dd"].data_ptr(),
                                    (dm.norm if last else dm.layers[li + 1].input_layernorm).weight.data_ptr(),
                                    (d["dhid"] if last else kin[C:C + B]).data_ptr(), B, H, eps, s)
        _gemm_small(fvk, d["dhid"][1:B], dw["lm_head"], d["dlogits"][:B - 1])
        return d["dlogits"][:B - 1].unsqueeze(0)

    @torch.no_grad()
    def _block_step(self, block_output_ids, target_hidden, position_ids, cache_position, temperature, top_p):
        """Drop-in for FlashDrive's ``_dflash_block_step``: (block, posterior, context, stats) in one graph."""
        _require_autocast()
        from flashdrive.dflash import sample_tokens
        m = self.m
        st = self._stream()
        d = self._dec_state(st)
        B = m.dflash_block_size
        key = ("block", float(temperature), float(top_p))
        g = st["graphs"].get(key)
        if g is None:
            b = self._lm_buffers(st, B, B)
            out = {}
            stop_id, period_id = m._dflash_stop_token_id, m._dflash_period_token_id

            def run():
                lm = m.vlm.model.language_model
                draft_logits = self._draft(d, B)
                proposed = sample_tokens(draft_logits, m._dflash_traj_mask, 0.0)
                follows_stop = F.pad(proposed[:, :-1] == stop_id, (1, 0))
                proposed = torch.where(follows_stop, d["traj_id"], proposed)
                block = torch.cat([d["block_ids"][:, :1], proposed], dim=1)
                torch.index_select(lm.embed_tokens.weight, 0, block[0], out=b["x"])
                rows = torch.arange(B, device=DEV) + d["slot"]
                torch.index_select(st["cos"], 0, rows, out=b["cos"])
                torch.index_select(st["sin"], 0, rows, out=b["sin"])
                self._lm_layers(st, b, B, self._attend_verify, None, d["slot"], cap_from=0)
                verify_logits = _gemm_small(self._fvk, b["h"], self._dw["lm_head"], d["vlogits"][:B]).unsqueeze(0)
                posterior = sample_tokens(verify_logits, m._dflash_traj_mask, temperature, top_p)
                # FlashDrive's round decisions (dflash.py, block_step_fn), verbatim.
                matches = block[:, 1:] == posterior[:, :-1]
                acceptance = matches.cumprod(dim=1).sum(dim=1)[0]
                slot = acceptance.unsqueeze(0)
                bonus = posterior[0].gather(0, slot)
                emission = torch.cat([block[0], bonus]).scatter(0, slot + 1, bonus)
                slots = torch.arange(emission.shape[0], device=DEV)
                stop_hits = (emission == stop_id) & (slots <= acceptance + 1)
                first_stop = torch.where(stop_hits.any(), stop_hits.int().argmax(), d["neg1"])
                period_hit = (bonus[0] == period_id).long()
                out["block"], out["posterior"] = block, posterior
                out["ctx"] = b["ctx"].unsqueeze(0)
                out["stats"] = torch.stack([acceptance, first_stop, period_hit])
            saved = self._snapshot_rows(st, cache_position)
            d["block_ids"].copy_(block_output_ids); d["th"].copy_(target_hidden)
            d["slot"].copy_(cache_position[:1]); d["used"].copy_((cache_position[:1] + B).int())
            gen = torch.cuda.default_generators[torch.cuda.current_device()]
            rng = gen.get_state()
            g = _capture(run)
            gen.set_state(rng)                      # warm-up / capture draws must not consume the caller's stream
            self._restore_rows(st, cache_position, saved)
            st["graphs"][key] = g
            st["keep"][("block_out",) + key] = out
        out = st["keep"][("block_out",) + key]
        d["block_ids"].copy_(block_output_ids)
        d["th"].copy_(target_hidden)
        d["slot"].copy_(cache_position[:1])
        d["used"].copy_((cache_position[:1] + B).int())
        g.replay()
        return out["block"], out["posterior"], out["ctx"], out["stats"]

    @torch.no_grad()
    def _traj_forward(self, input_ids, position_ids, cache_position) -> None:
        """Drop-in for FlashDrive's ``_dflash_traj_forward``: the one-row trajectory-start forward (K/V only)."""
        _require_autocast()
        m = self.m
        st = self._stream()
        d = self._dec_state(st)
        g = st["graphs"].get("traj")
        if g is None:
            b = self._lm_buffers(st, 1, 1)
            ids = torch.zeros(1, dtype=torch.long, device=DEV)
            st["keep"]["traj_ids"] = ids

            def run():
                lm = m.vlm.model.language_model
                torch.index_select(lm.embed_tokens.weight, 0, ids, out=b["x"])
                torch.index_select(st["cos"], 0, d["slot"], out=b["cos"])
                torch.index_select(st["sin"], 0, d["slot"], out=b["sin"])
                self._lm_layers(st, b, 1, self._attend_all_rows, None, d["slot"], cap_from=0)
            saved = self._snapshot_rows(st, cache_position)
            ids.copy_(input_ids.view(-1)); d["slot"].copy_(cache_position[:1])
            g = _capture(run)
            self._restore_rows(st, cache_position, saved)
            st["graphs"]["traj"] = g
        st["keep"]["traj_ids"].copy_(input_ids.view(-1))
        d["slot"].copy_(cache_position[:1])
        g.replay()
        # FlashDrive warms this primitive once (first trajectory window after a fresh registry) and skips it after.
        m._compiled_step_registry.setdefault("dflash_traj_forward", None)

    def _snapshot_rows(self, st: dict, cache_position: torch.Tensor) -> tuple:
        """Cache rows a captured decode graph writes during warm-up/capture (restored afterwards)."""
        r0 = int(cache_position[0])
        r1 = min(st["S"], r0 + self.m.dflash_block_size)
        return r0, r1, [(k[0, :, r0:r1].clone(), v[0, :, r0:r1].clone()) for k, v in zip(st["k"], st["v"])], \
            st["kvlm"][:, r0:r1].clone()

    def _restore_rows(self, st: dict, cache_position: torch.Tensor, saved: tuple) -> None:
        r0, r1, rows, kvlm = saved
        for (k, v), (ks, vs) in zip(zip(st["k"], st["v"]), rows):
            k[0, :, r0:r1].copy_(ks); v[0, :, r0:r1].copy_(vs)
        st["kvlm"][:, r0:r1].copy_(kvlm)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Start a new stream (see ``reset_flashdrive_stream``); stream-bound buffers and graphs are
        rebuilt for it."""
        reset_flashdrive_stream(self.m)
        self._st = None
