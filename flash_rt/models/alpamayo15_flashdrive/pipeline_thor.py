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
prefill  ``_first_prefill`` (the stream's first window; its bookkeeping computed
         on the host from the token ids), ``_dflash_prefill`` (streaming windows)
         and ``_shift_streaming_kv_cache``: the 36-layer language model on
         FlashRT kernels with FlashDrive's W4A8 projections, FA4 causal attention
         by cache position, one CUDA graph per stream for streaming windows.
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
FlashDrive's draw point); draft proposals are greedy, posterior sampling follows
FlashDrive's ``sample_tokens`` rule on the FlashRT sampler; every cache row FlashDrive writes (including
rejected draft rows and the expert's action rows) is written with the same
contents, so later reads of stale rows see what FlashDrive would.

Layering: ``runtime_thor.FlashDriveThorRuntime`` validates the backend choices, prepares the weights, owns every
persistent buffer and the per-stream state, starts each stream and captures the CUDA graphs; this module holds the
primitives' computations that run on them (``FlashDriveThor``) and routes FlashDrive's calls to them (``install``).
``flash_rt.frontends.torch.alpamayo15_flashdrive_thor`` builds both.
"""
from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING, Any

import torch

from flash_rt.models.alpamayo15.pipeline_thor import action_in, torch_philox_increment, torch_rng_state, vision_tower
from flash_rt.models.alpamayo15_flashdrive.weights import SMALL_MAX_ROWS, Bf16Fragment, Fp8Fragment

if TYPE_CHECKING:
    from flash_rt.models.alpamayo15_flashdrive.runtime_thor import FlashDriveThorRuntime

BF16 = torch.bfloat16
DEV = "cuda"
logger = logging.getLogger(__name__)

def _mrope_sections(rot) -> tuple[int, int]:
    """(height, width) section widths of a rotary module for the table kernel: Qwen3-VL's interleaved M-RoPE, or
    (0, 0) for a one-dimensional rotary (``Qwen3RotaryEmbedding``, the DFlash draft), where every frequency takes
    the first position row."""
    sec = getattr(rot, "mrope_section", None)
    return (int(sec[1]), int(sec[2])) if sec is not None else (0, 0)


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


def _gemm_small(fvk, x: torch.Tensor, w, y: torch.Tensor) -> torch.Tensor:
    """``y = x @ w.T`` for at most 16 rows: ``flashdrive_bf16_gemm_small`` on a BF16 weight, or the W8A16 kernel on
    an ``Fp8Fragment`` (``x`` must come from a buffer with whole 8-row tiles readable past its rows; ``y`` row-major,
    bf16, or fp32 for the BF16 path only)."""
    if isinstance(w, Fp8Fragment):
        N, K = w.shape
        if y.dtype != BF16 or y.stride(0) != N or x.stride(0) != K:
            raise ValueError("the W8A16 path needs contiguous bf16 operands")
        fvk.alpamayo15_w8a16_gemm_bf16(w.frag.data_ptr(), w.scale.data_ptr(), x.data_ptr(), y.data_ptr(), x.shape[0], N, K,
                                       torch.cuda.current_stream().cuda_stream, w.fmt)
        return y
    if isinstance(w, Bf16Fragment):
        N, K = w.shape
        fvk.flashdrive_bf16_gemm_frag(x.data_ptr(), x.stride(0), w.frag.data_ptr(), y.data_ptr(), y.stride(0),
                                      int(y.dtype == torch.float32), x.shape[0], N, K, torch.cuda.current_stream().cuda_stream)
        return y
    fvk.flashdrive_bf16_gemm_small(x.data_ptr(), x.stride(0), w.data_ptr(), y.data_ptr(), y.stride(0),
                                   int(y.dtype == torch.float32), x.shape[0], w.shape[0], w.shape[1],
                                   torch.cuda.current_stream().cuda_stream)
    return y


def rope_rows(rt, st: dict, li: int, dst: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, fp32_rule: int, rows: int,
              row_map: torch.Tensor | None = None, row0: int = 0, slot: torch.Tensor | None = None,
              n_dev: torch.Tensor | None = None) -> None:
    """Rotated copy of FlashDrive's un-rotated cache keys of layer ``li`` into ``dst`` (``flashdrive_rope_rows``)."""
    S, D = st["S"], rt.head_dim
    rt.fvk.flashdrive_rope_rows(
        st["k"][li].data_ptr(), D, S * D, dst.data_ptr(), rt.n_kv * D, D, cos.data_ptr(), sin.data_ptr(),
        fp32_rule, row_map.data_ptr() if row_map is not None else 0, row0,
        slot.data_ptr() if slot is not None else 0, n_dev.data_ptr() if n_dev is not None else 0,
        rows, rt.n_kv, torch.cuda.current_stream().cuda_stream)


def sync_kvlm_all(rt, st: dict) -> None:
    """Rotate every cache row of every layer into the VLM view (after FlashDrive wrote the cache)."""
    for li in range(rt.n_layers):
        rope_rows(rt, st, li, st["kvlm"][li], st["cos"], st["sin"], 0, st["S"])


class FlashDriveThor:
    """The FlashRT execution of FlashDrive's primitives for one FlashDrive Alpamayo 1.5 model, on the weights, buffers,
    per-stream state and CUDA graphs of ``runtime`` (``runtime_thor.FlashDriveThorRuntime``, which takes the backend
    choices). ``install`` routes FlashDrive's calls to these primitives; the frontend
    ``flash_rt.frontends.torch.alpamayo15_flashdrive_thor`` builds both (``attach_flashdrive_backend``)."""

    def __init__(self, runtime: FlashDriveThorRuntime) -> None:
        self.rt = runtime
        self.m = runtime.m
        self.record_steps = False           # expert parity mode: keep per-step velocities
        self._steps_rec: dict = {}

    @property
    def backends(self) -> dict:
        rt = self.rt
        return rt.backends

    def install(self) -> FlashDriveThor:
        """Route FlashDrive's primitives of the groups set to ``"flashrt"`` to this computation (FlashDrive's
        orchestration keeps calling them by name); returns ``self``."""
        rt = self.rt
        model, backends = self.m, rt.backends
        if backends["expert"] == "flashrt":
            model._action = self._action
        if backends["vision"] == "flashrt":
            model._encode = self._encode
        if backends["prefill"] == "flashrt" and backends["expert"] == "flashdrive":
            fd_action = model._action

            def action_then_sync(**kw):     # FlashDrive's expert writes action-row K/V; mirror them into kvlm
                out = fd_action(**kw)
                st = rt.stream()
                slot = kw["cache_position"][:1]
                for li in range(rt.n_layers):
                    rope_rows(rt, st, li, st["kvlm"][li], st["cos"], st["sin"], 0, rt.n_act, slot=slot)
                return out
            model._action = action_then_sync
        if backends["prefill"] == "flashrt":
            model._first_prefill = self._first_prefill
            model._dflash_prefill = self._dflash_prefill
            model._shift_streaming_kv_cache = self._shift
        if backends["decode"] == "flashrt":
            model._dflash_block_step = self._block_step
            model._dflash_traj_forward = self._traj_forward
        return self

    def reset(self) -> None:
        """Start a new stream (``FlashDriveThorRuntime.reset``)."""
        rt = self.rt
        rt.reset()

    # ------------------------------------------------------------------
    # expert
    # ------------------------------------------------------------------
    def _v_view(self, st: dict, li: int) -> torch.Tensor:
        """FlashDrive's value cache of layer ``li`` as an attention view [1, S, Hk, D] (strided, no copy)."""
        return st["v"][li].permute(0, 2, 1, 3)

    def _velocity(self, x: torch.Tensor, t: torch.Tensor, st: dict, rec: list | None) -> torch.Tensor:
        """One FlashDrive expert evaluation (``step_fn``) on FlashRT kernels; returns v [1, A, act_dim] bf16."""
        rt = self.rt
        m, fvk = self.m, rt.fvk
        s = torch.cuda.current_stream().cuda_stream
        A, E, D, HQ, KV, F_, S = rt.n_act, rt.e_hidden, rt.head_dim, rt.e_heads, rt.n_kv, rt.e_ffn, st["S"]
        b = rt.expert_buffers()
        res, hn, qkv, q, o, d = b["res"], b["hn"], b["qkv"], b["q"], b["o"], b["d"]
        eps = rt.e_eps
        layers = m.expert.layers
        action_in(rt, x.view(A, rt.act_dim), t, A, res, s)                  # FP32 residual stream (action_in_proj)
        fvk.alpamayo15_res_rms_f32_bf16(res.data_ptr(), 0, layers[0].input_layernorm.weight.data_ptr(),
                                        hn.data_ptr(), A, E, eps, s)
        ld = qkv.shape[1]
        e8 = rt.e8

        def lin8(x, name, li, y):          # W8A8: per-row FP8 activations x per-row FP8 weights, scales in the epilogue
            q8, q_s = {"hn": (b["hn8"], b["hn_s"]), "att": (b["att8"], b["att_s"]), "act": (b["act8"], b["act_s"])}[x]
            src = {"hn": hn, "att": b["q"], "act": b["act"]}[x]
            fvk.alpamayo15_quantize_fp8_rows_bf16(src.data_ptr(), src.stride(0), q8.data_ptr(), q_s.data_ptr(), A, src.shape[1], s)
            w8, w_s = e8[li][name]
            return rt.gemm.fp8(q8, q_s, w8, w_s, y)

        for li, layer in enumerate(layers):
            a = layer.self_attn
            if e8 is not None:
                lin8("hn", "qkv", li, qkv)
            else:
                rt.gemm(hn, a.qkv_proj.weight, qkv, tile=1)
            # q rotated (FP32 rule); action-row K (normalized, un-rotated) and V written into FlashDrive's cache
            # at action_start, as FlashDrive's expert forward does, and K's rotation into the expert view.
            fvk.flashdrive_qk_prep_bf16(
                qkv.data_ptr(), qkv[:, HQ * D:].data_ptr(), qkv[:, (HQ + KV) * D:].data_ptr(), ld, ld, ld,
                a.q_norm.weight.data_ptr(), a.k_norm.weight.data_ptr(), b["e_cos"].data_ptr(), b["e_sin"].data_ptr(), 1,
                q.data_ptr(), st["k"][li].data_ptr(), D, S * D, st["ke"][li].data_ptr(), KV * D, D,
                st["v"][li].data_ptr(), D, S * D, 0, rt.d_slot.data_ptr(), A, HQ, KV, eps, s)
            out, _ = rt.fa4(q.view(1, A, HQ, D), st["ke"][li].unsqueeze(0), self._v_view(st, li),
                            seqused_k=rt.d_seqk, softmax_scale=rt.scale, causal=False, pack_gqa=True,
                            num_splits=rt.expert_splits)
            if e8 is not None:
                b["q"].copy_(out.view(A, HQ * D))        # attention output rows (q buffer reused as the FP8 source)
                lin8("att", "o", li, o)
            else:
                rt.gemm(out.view(A, HQ * D), a.o_proj.weight, o, tile=1)
            fvk.alpamayo15_res_rms_f32_bf16(res.data_ptr(), o.data_ptr(), layer.post_attention_layernorm.weight.data_ptr(),
                                            hn.data_ptr(), A, E, eps, s)
            gu, act = b["gu"], b["act"]
            if e8 is not None:
                lin8("hn", "gu", li, gu)
            else:
                rt.gemm(hn, layer.mlp.gate_up_proj.weight, gu, tile=1)
            # gate | up are the halves of one row-major [A, 2F] GEMM output: split them for the SiLU-mul kernel
            fvk.alpamayo15_silu_mul_split_bf16(gu.data_ptr(), gu[:, F_:].data_ptr(), 2 * F_, act.data_ptr(), A, F_, s)
            if e8 is not None:
                lin8("act", "down", li, d)
            else:
                rt.gemm(act, layer.mlp.down_proj.weight, d, tile=1)
            nxt = layers[li + 1].input_layernorm.weight if li + 1 < len(layers) else m.expert.norm.weight
            fvk.alpamayo15_res_rms_f32_bf16(res.data_ptr(), d.data_ptr(), nxt.data_ptr(), hn.data_ptr(), A, E, eps, s)
        fvk.alpamayo15_action_out_euler(hn.data_ptr(), rt.act.out_w.data_ptr(), rt.act.out_b.data_ptr(), 0.0, 0, b["v"].data_ptr(), A, E,
                                        rt.act_dim, s)
        v = b["v"].view(1, A, rt.act_dim)
        if rec is not None:
            rec.append(v.clone())
        return v

    def _denoise(self, st: dict, schedule: list[bool], time_steps: torch.Tensor, step_dt: list[float], rec: dict | None) -> None:
        # Expert view of every cached prompt/reasoning row (FP32 rule); the action rows are written per step.
        rt = self.rt
        fvk = rt.fvk
        s = torch.cuda.current_stream().cuda_stream
        for li in range(rt.n_layers):
            rope_rows(rt, st, li, st["ke"][li], st["e_cos_tab"], st["e_sin_tab"], 1, st["S"], n_dev=rt.d_ke_rows)
        b, D, A = rt.expert_buffers(), rt.head_dim, rt.n_act
        for name in ("e_cos", "e_sin"):
            fvk.alpamayo15_gather_rows_bytes(st[name + "_tab"].data_ptr(), 4 * D, rt.e_idx.data_ptr(), rt.d_slot.data_ptr(),
                                             b[name].data_ptr(), 4 * D, A, 4 * D, s)
        x = rt.x_out
        x.copy_(rt.noise)
        for i, cached in enumerate(schedule):
            if cached:
                v = rt.v_cache
            else:
                v = self._velocity(x, time_steps[i:i + 1], st, None if rec is None else rec["v"])
                rt.v_cache.copy_(v)
            fvk.alpamayo15_euler_bf16(x.data_ptr(), v.data_ptr(), step_dt[i], A * rt.act_dim, s)   # x = x + dt * v (BF16)
            if rec is not None:
                rec["x"].append(x.clone())
        if "kvlm" in st:  # keep the VLM view of the action rows FlashDrive's cache now holds
            for li in range(rt.n_layers):
                rope_rows(rt, st, li, st["kvlm"][li], st["cos"], st["sin"], 0, rt.n_act, slot=rt.d_slot)

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
        rt = self.rt
        _require_autocast()
        n_steps = int(kw.get("inference_step", self.m.diffusion.num_inference_steps))
        schedule = tuple(euler_cache_schedule(n_steps, kw["cache_steps"]))
        st = rt.stream()
        if position_ids is not st["pos"]:
            raise ValueError("expert position table differs from the stream's")
        action_start = int(cache_position[0])
        if action_start + rt.n_act > st["S"]:
            raise ValueError(f"action rows [{action_start}, {action_start + rt.n_act}) exceed the cache ({st['S']})")
        key = ("expert", schedule, n_steps, self.record_steps)
        g = st["graphs"].get(key)
        if g is None:
            g = rt.expert_graph(st, key, schedule, n_steps, self.record_steps, self._denoise)
        self._steps_rec = st["keep"][("rec",) + key]
        rt.d_slot.fill_(action_start)
        rt.d_seqk.fill_(action_start + rt.n_act)
        rt.d_ke_rows.fill_(action_start)
        rt.noise.normal_()             # FlashDrive's draw: BF16 noise, sampled before the graph
        g.replay()
        return rt.x_out.clone()

    # ------------------------------------------------------------------
    # vision
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _encode(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor):
        """Drop-in for FlashDrive's ``_encode``: (image embeddings, DeepStack features) of the window's frames."""
        rt = self.rt
        _require_autocast()
        ent = rt.vision_graphs.get(pixel_values, image_grid_thw, partial(vision_tower, rt))
        ent["px"].copy_(pixel_values)
        ent["graph"].replay()
        return ent["out"]["img"], ent["out"]["deep"]

    # ------------------------------------------------------------------
    # language model on FlashRT kernels (FlashDrive's W4A8 projections)
    # ------------------------------------------------------------------
    def _lm_layers(self, st: dict, b: dict, n: int, attend, row_map: torch.Tensor | None, slot: torch.Tensor | None,
                   deep: list | None = None, vis_idx: torch.Tensor | None = None, cap_from: int = 0) -> None:
        """36 decoder layers over ``b['x']`` [n, H]; leaves the final-norm output in ``b['h']`` and the
        DFlash context (captured layers, rows ``cap_from:``) in ``b['ctx']``."""
        rt = self.rt
        m, fvk = self.m, rt.fvk
        s = torch.cuda.current_stream().cuda_stream
        lm = m.vlm.model.language_model
        layers = lm.layers
        H, D, NH, KV, S, F_, eps = rt.hidden, rt.head_dim, rt.n_heads, rt.n_kv, st["S"], rt.ffn, rt.eps
        x, h, q, act = b["x"], b["h"], b["q"], b["act"]
        cap_ids = sorted(lm._capture_layer_ids)
        # FlashRT W4A8 path over many rows (streaming-window prefill): the producer of each W4A8 input (RMSNorm,
        # SiLU-mul, attention output) and its activation preparation run as one kernel over row tiles; the normalized
        # rows are never materialised (h only for the LM head). A few rows (verification, trajectory start) keep the
        # separate kernels, whose grids spread the rotation groups over the SMs.
        fused = rt.backends["w4a8"] == "fused" and rt.w4[0]["qkv"].rot is None and n > SMALL_MAX_ROWS
        if fused:
            rt.w4[0]["qkv"].prep(fvk, 1, None, n, residual=x, norm_w=layers[0].input_layernorm.weight, eps=eps)
        else:
            fvk.alpamayo15_rms_bf16(x.data_ptr(), 0, layers[0].input_layernorm.weight.data_ptr(), h.data_ptr(), n, H, eps, s)
        for li, layer in enumerate(layers):
            a = layer.self_attn
            # fused: FP16 GEMM outputs consumed directly (cast to BF16 on load, as FlashDrive's wrapper casts)
            w4 = rt.w4[li] if rt.backends["w4a8"] == "fused" else None
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
                    d16 = b["d16"]
                    fvk.alpamayo15_cast_f16_bf16(d.data_ptr(), d16.data_ptr(), n * H, s)
                    fvk.residual_add(x.data_ptr(), d16.data_ptr(), n * H, s)
                    fvk.alpamayo15_scatter_rows_bf16(x.data_ptr(), H, vis_idx.data_ptr(), deep[li].data_ptr(), deep[li].stride(0),
                                                     deep[li].shape[0], H, 1, s)
                    if li + 1 < len(layers):
                        rt.w4[li + 1]["qkv"].prep(fvk, 1, None, n, residual=x, norm_w=nxt, eps=eps)
                    else:
                        fvk.alpamayo15_rms_bf16(x.data_ptr(), 0, nxt.data_ptr(), h.data_ptr(), n, H, eps, s)
                elif li + 1 < len(layers):                # residual += down, next input norm, next layer's q|k|v preparation
                    rt.w4[li + 1]["qkv"].prep(fvk, 1, d, n, residual=x, norm_w=nxt, eps=eps)
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
                    fvk.alpamayo15_cast_f16_bf16(d.data_ptr(), b["d16"].data_ptr(), n * H, s)
                    d = b["d16"]
                fvk.residual_add(x.data_ptr(), d.data_ptr(), n * H, s)
                fvk.alpamayo15_scatter_rows_bf16(x.data_ptr(), H, vis_idx.data_ptr(), deep[li].data_ptr(), deep[li].stride(0),
                                                 deep[li].shape[0], H, 1, s)                      # FlashDrive's DeepStack injection
                fvk.alpamayo15_rms_bf16(x.data_ptr(), 0, nxt.data_ptr(), h.data_ptr(), n, H, eps, s)
            elif w4 is not None:
                fvk.flashdrive_rms_f16delta_bf16(x.data_ptr(), d.data_ptr(), nxt.data_ptr(), h.data_ptr(), n, H, eps, s)
            else:
                fvk.alpamayo15_rms_bf16(x.data_ptr(), d.data_ptr(), nxt.data_ptr(), h.data_ptr(), n, H, eps, s)
            if li in cap_ids:                                           # decoder-layer output, as FlashDrive captures
                j = cap_ids.index(li)
                b["ctx"][:, j * H:(j + 1) * H].copy_(x[cap_from:])

    # streaming prefill ---------------------------------------------------
    @torch.no_grad()
    def _first_prefill(self, input_ids, attention_mask, pixel_values, image_grid_thw) -> None:
        """Drop-in for FlashDrive's full first-window prefill: the stream started by the runtime (FlashDrive's host
        bookkeeping and cache allocation), then the 36 layers over the prompt on FlashRT kernels writing cache rows
        [0, P) and their rotated views."""
        rt = self.rt
        _require_autocast()
        m, fvk = self.m, rt.fvk
        s = torch.cuda.current_stream().cuda_stream
        if attention_mask is not None and not bool(attention_mask.all()):
            raise ValueError("FlashDriveThor's first-window prefill takes an unpadded prompt")
        ids_host = input_ids[0].tolist()
        st = rt.start_stream(input_ids, ids_host, image_grid_thw)
        P, H, D = len(ids_host), rt.hidden, rt.head_dim
        cfg = m.vlm.config
        # the vision tower as FlashDrive's first prefill runs it (its streaming ``_encode`` is a compiled step whose
        # buffers take the streaming window's shape): the FlashRT tower, or FlashDrive's module when vision stays with it
        if rt.backends["vision"] == "flashrt":
            img, deep = self._encode(pixel_values, image_grid_thw)
        else:
            img, deep = m.vlm.model.visual(pixel_values, grid_thw=image_grid_thw)
        # the prompt rows through the language model, cache rows [0, P)
        b = rt.lm_buffers(st, P, m._dflash_context_len)
        vis_idx = torch.tensor([i for i, t in enumerate(ids_host) if t == int(cfg.image_token_id)], dtype=torch.long, device=DEV)
        if vis_idx.numel() != img.shape[0]:
            raise ValueError(f"image tokens {vis_idx.numel()} != image features {img.shape[0]}")
        lm = m.vlm.model.language_model
        ids_dev = input_ids.to(DEV).reshape(-1)
        fvk.alpamayo15_gather_rows_bf16(lm.embed_tokens.weight.data_ptr(), H, ids_dev.data_ptr(), b["x"].data_ptr(), H, P, H, s)
        fvk.alpamayo15_scatter_rows_bf16(b["x"].data_ptr(), H, vis_idx.data_ptr(), img.data_ptr(), img.stride(0), img.shape[0], H, 0, s)
        rows = torch.arange(P, dtype=torch.long, device=DEV)
        fvk.alpamayo15_gather_rows_bf16(st["cos"].data_ptr(), D, rows.data_ptr(), b["cos"].data_ptr(), D, P, D, s)
        fvk.alpamayo15_gather_rows_bf16(st["sin"].data_ptr(), D, rows.data_ptr(), b["sin"].data_ptr(), D, P, D, s)
        rt.first_prefill_lengths(st, P)
        self._lm_layers(st, b, P, self._attend_first, None, None, list(deep), vis_idx, cap_from=P - m._dflash_context_len)
        m.vlm.model.visual.reset_shape_caches()
        lm.reset_shape_caches()

    def _attend_first(self, st: dict, li: int, q: torch.Tensor, out: torch.Tensor) -> None:
        self._attend_runs(st, li, q, out, st["keep"]["first_meta"])

    def _attend_prefill(self, st: dict, li: int, q: torch.Tensor, out: torch.Tensor) -> None:
        self._attend_runs(st, li, q, out, st["keep"]["prefill_meta"])

    def _attend_runs(self, st: dict, li: int, q: torch.Tensor, out: torch.Tensor, p: dict) -> None:
        rt = self.rt
        o, _ = rt.fa4(q.view(-1, rt.n_heads, rt.head_dim), st["kvlm"][li], self._v_view(st, li)[0],
                      cu_seqlens_q=p["cu_q"], cu_seqlens_k=p["cu_k"], seqused_k=p["used"],
                      max_seqlen_q=p["max_q"], max_seqlen_k=st["S"], softmax_scale=rt.scale, causal=True,
                      pack_gqa=True)
        out.copy_(o.view(out.shape))

    @torch.no_grad()
    def _dflash_prefill(self, inputs_embeds, position_ids, cache_position, visual_pos_masks, deepstack_image_embeds,
                        streaming_attention_mask):
        """Drop-in for FlashDrive's ``_dflash_prefill``: (last-row logits [1, V], DFlash context [1, ctx, C])."""
        rt = self.rt
        _require_autocast()
        m = self.m
        st = rt.stream()
        n = inputs_embeds.shape[1]
        g = st["graphs"].get("prefill")
        if g is not None and n != st["keep"]["prefill_n"]:
            raise ValueError(f"streaming window has {n} rows; this stream's prefill graph was captured for "
                             f"{st['keep']['prefill_n']}")
        b = rt.lm_buffers(st, n, m._dflash_context_len)
        if g is None:
            rt.prefill_state(st, b, n, cache_position, inputs_embeds, visual_pos_masks, deepstack_image_embeds)
            out = {}

            def run():
                b["x"].copy_(b["emb"])
                self._lm_layers(st, b, n, self._attend_prefill, b["rows"], None, b["deep"], b["vis"],
                                cap_from=n - b["ctx_len"])
                out["logits"] = _gemm_small(rt.fvk, b["h"][n - 1:n], rt.dw["lm_head"] if rt.backends["decode"] == "flashrt"
                                            else m.vlm.lm_head.weight, b["logits"])
            g = rt.capture_prefill(st, run, out)
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
        rt = self.rt
        st = rt.stream()
        sh = rt.shift_plan(st)
        S, D = st["S"], rt.head_dim
        rt.fvk.flashdrive_shift_cache(sh["k_ptrs"].data_ptr(), sh["v_ptrs"].data_ptr(), D, S * D,
                                      st["kvlm"].data_ptr(), st["cos"].data_ptr(), st["sin"].data_ptr(),
                                      sh["ranges"].data_ptr(), sh["n_views"], sh["F"], rt.n_layers, S, rt.n_kv,
                                      torch.cuda.current_stream().cuda_stream)

    # decode ----------------------------------------------------------------
    def _attend_verify(self, st: dict, li: int, q: torch.Tensor, out: torch.Tensor) -> None:
        rt = self.rt
        n = q.shape[0]
        o, _ = rt.fa4(q.view(1, n, rt.n_heads, rt.head_dim), st["kvlm"][li].unsqueeze(0), self._v_view(st, li),
                      seqused_k=st["keep"]["dec"]["used"], softmax_scale=rt.scale, causal=True, pack_gqa=True)
        out.copy_(o.view(out.shape))

    def _attend_all_rows(self, st: dict, li: int, q: torch.Tensor, out: torch.Tensor) -> None:
        # FlashDrive's one-row forward passes no mask: SDPA attends every cache row.
        rt = self.rt
        o, _ = rt.fa4(q.view(1, 1, rt.n_heads, rt.head_dim), st["kvlm"][li].unsqueeze(0), self._v_view(st, li),
                      softmax_scale=rt.scale, causal=False, pack_gqa=True)
        out.copy_(o.view(out.shape))

    def _draft(self, d: dict, B: int) -> torch.Tensor:
        """FlashDrive's DFlash draft (``DFlashDraftModel.forward`` + greedy LM head on the proposal rows) on FlashRT
        kernels: context rows = hidden_norm(fc(target_hidden)), noise rows = the block embeddings; every layer attends
        the B noise queries over the C context + B noise keys. Returns the draft logits of rows 1..B-1 (bf16)."""
        rt = self.rt
        m, fvk = self.m, rt.fvk
        s = torch.cuda.current_stream().cuda_stream
        dm = m._draft_model
        lm = m.vlm.model.language_model
        H, D, NH, KV, F_, C = rt.hidden, rt.head_dim, rt.n_heads, rt.n_kv, rt.ffn, m._dflash_context_len
        eps = dm.config.rms_norm_eps
        x, h, kin, thn = d["dx"][:B], d["dh"], d["dkin"], d["dthn"]
        fvk.alpamayo15_gather_rows_bf16(lm.embed_tokens.weight.data_ptr(), H, d["block_ids"].data_ptr(), x.data_ptr(), H, B, H, s)
        dw = rt.dw
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
            o, _ = rt.fa4(d["dqo"][:B].unsqueeze(0), d["dko"][:C + B].unsqueeze(0), d["dvo"][:C + B].unsqueeze(0),
                          softmax_scale=rt.scale, causal=False, pack_gqa=True)
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
        rt = self.rt
        _require_autocast()
        m = self.m
        st = rt.stream()
        d = rt.dec_state(st)
        B = m.dflash_block_size
        key = ("block", float(temperature), float(top_p))

        def set_inputs():
            d["block_ids"].copy_(block_output_ids)
            d["th"].copy_(target_hidden)
            d["slot"].copy_(cache_position[:1])
            d["used"].copy_((cache_position[:1] + B).int())
        g = st["graphs"].get(key)
        if g is None:
            b = rt.lm_buffers(st, B, B)
            out = {}
            stop_id, period_id = int(m._dflash_stop_token_id), int(m._dflash_period_token_id)
            mask_lo, mask_n = (int(v) for v in m._dflash_traj_mask)
            fvk, H, D = rt.fvk, rt.hidden, rt.head_dim
            V = m.vlm.lm_head.weight.shape[0]
            block, posterior, stats = d["block"], d["posterior"], d["stats"]

            def run():
                s = torch.cuda.current_stream().cuda_stream
                lm = m.vlm.model.language_model
                draft_logits = self._draft(d, B)[0]                            # [B - 1, V]: greedy proposals
                fvk.alpamayo15_sample_tokens_bf16(draft_logits.data_ptr(), draft_logits.stride(0), B - 1, V, mask_lo, mask_n, 0.0, 1.0, 1,
                                                  st["rng"].data_ptr(), rt.rng_blocks, 0, d["sws"].data_ptr(), d["sws"].numel(),
                                                  d["proposed"].data_ptr(), 0, 0, 0, 0, 0, 0, s)
                fvk.alpamayo15_dflash_propose(d["proposed"].data_ptr(), d["block_ids"].data_ptr(), stop_id, int(m.traj_start_token_id),
                                              block.data_ptr(), B, s)
                fvk.alpamayo15_gather_rows_bf16(lm.embed_tokens.weight.data_ptr(), H, block.data_ptr(), b["x"].data_ptr(), H, B, H, s)
                fvk.alpamayo15_gather_rows_bytes(st["cos"].data_ptr(), 2 * D, d["b_idx"].data_ptr(), d["slot"].data_ptr(),
                                                 b["cos"].data_ptr(), 2 * D, B, 2 * D, s)
                fvk.alpamayo15_gather_rows_bytes(st["sin"].data_ptr(), 2 * D, d["b_idx"].data_ptr(), d["slot"].data_ptr(),
                                                 b["sin"].data_ptr(), 2 * D, B, 2 * D, s)
                self._lm_layers(st, b, B, self._attend_verify, None, d["slot"], cap_from=0)
                verify_logits = _gemm_small(rt.fvk, b["h"], rt.dw["lm_head"], d["vlogits"][:B])
                # FlashDrive's sampler (mask, temperature, its top-p rule, multinomial) and its round decisions
                fvk.alpamayo15_sample_tokens_bf16(verify_logits.data_ptr(), verify_logits.stride(0), B, V, mask_lo, mask_n,
                                                  float(temperature), float(top_p), 1, st["rng"].data_ptr(), rt.rng_blocks,
                                                  d["sscratch"].data_ptr(), d["sws"].data_ptr(), d["sws"].numel(), posterior.data_ptr(),
                                                  0, 0, 0, 0, 0, 0, s)
                fvk.alpamayo15_dflash_decide(block.data_ptr(), posterior.data_ptr(), stop_id, period_id, B, stats.data_ptr(), s)
                out["block"], out["posterior"] = block.unsqueeze(0), posterior.unsqueeze(0)
                out["ctx"] = b["ctx"].unsqueeze(0)
                out["stats"] = stats
            g = rt.capture_block(st, key, run, out, cache_position, set_inputs)
        out = st["keep"][("block_out",) + key]
        set_inputs()
        # FlashDrive samples the posterior from torch's generator: the kernel reads the generator's (seed, offset) at
        # every round and the host advances torch's offset by the draw's increment, so the round is exactly
        # torch.multinomial's draw from the generator's current state (restoring the generator restores the round)
        gen = torch.cuda.default_generators[torch.cuda.current_device()]
        st["rng"].copy_(torch_rng_state(gen))
        g.replay()
        if temperature > 0:
            gen.set_offset(int(gen.get_offset()) + torch_philox_increment(B * m.vlm.lm_head.weight.shape[0], rt.rng_blocks))
        return out["block"], out["posterior"], out["ctx"], out["stats"]

    @torch.no_grad()
    def _traj_forward(self, input_ids, position_ids, cache_position) -> None:
        """Drop-in for FlashDrive's ``_dflash_traj_forward``: the one-row trajectory-start forward (K/V only)."""
        rt = self.rt
        _require_autocast()
        m = self.m
        st = rt.stream()
        d = rt.dec_state(st)

        def set_inputs():
            rt.traj_ids(st).copy_(input_ids.view(-1))
            d["slot"].copy_(cache_position[:1])
        g = st["graphs"].get("traj")
        if g is None:
            b = rt.lm_buffers(st, 1, 1)
            ids = rt.traj_ids(st)

            def run():
                s = torch.cuda.current_stream().cuda_stream
                fvk, H, D = rt.fvk, rt.hidden, rt.head_dim
                lm = m.vlm.model.language_model
                fvk.alpamayo15_gather_rows_bf16(lm.embed_tokens.weight.data_ptr(), H, ids.data_ptr(), b["x"].data_ptr(), H, 1, H, s)
                fvk.alpamayo15_gather_rows_bf16(st["cos"].data_ptr(), D, d["slot"].data_ptr(), b["cos"].data_ptr(), D, 1, D, s)
                fvk.alpamayo15_gather_rows_bf16(st["sin"].data_ptr(), D, d["slot"].data_ptr(), b["sin"].data_ptr(), D, 1, D, s)
                self._lm_layers(st, b, 1, self._attend_all_rows, None, d["slot"], cap_from=0)
            g = rt.capture_traj(st, run, cache_position, set_inputs)
        set_inputs()
        g.replay()
        # FlashDrive warms this primitive once (first trajectory window after a fresh registry) and skips it after.
        m._compiled_step_registry.setdefault("dflash_traj_forward", None)
