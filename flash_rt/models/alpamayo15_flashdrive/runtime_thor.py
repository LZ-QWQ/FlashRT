"""FlashDrive backend on Jetson AGX Thor: initialization and resources.

``FlashDriveThorRuntime`` owns what the backend's primitives (``pipeline_thor.FlashDriveThor``) run on: the validated
backend choices and precision tiers, the kernel handles, the model geometry, the prepared weights (fused W4A8 groups,
fragment copies of the decode linears, FP8 copies of the expert), the expert's persistent buffers, and per stream: the
start of the stream (FlashDrive's host bookkeeping, its cache allocation, the rotated cache views and rotary tables),
the shape-sized work buffers of each primitive, and the CUDA graphs (captured on scratch cache rows with the touched
rows and the random state restored, looked up per stream and shape). The pipeline adds no buffers of its own and
captures nothing itself; it passes its computations to the runtime's capture methods.
"""
from __future__ import annotations

import numpy as np
import torch

from flash_rt.models.alpamayo15.pipeline_thor import BF16, DEV, _Gemm, mrope_positions
from flash_rt.models.alpamayo15.runtime_thor import (VisionGraphs, VisionTables, capture_graph, check_vision_modules,
                                                    require_alpamayo15_kernels, torch_philox_blocks)
from flash_rt.models.alpamayo15.weights import pack_action_modules
from flash_rt.models.alpamayo15_flashdrive.pipeline_thor import (_mrope_sections, contiguous_runs, reset_flashdrive_stream,
                                                              sync_kvlm_all)
from flash_rt.models.alpamayo15_flashdrive.weights import Bf16Fragment, Fp8Fragment, W4A8Group


BACKEND_CHOICES = ("flashrt", "flashdrive")

# Kernels of the FlashDrive backend (csrc/kernels/alpamayo15_flashdrive_kernels.cuh and the W4A8 GEMM in
# csrc/gemm/alpamayo15_flashdrive_w4a8_sm100.cu), built with
# -DFLASHRT_ENABLE_ALPAMAYO15=ON next to the Alpamayo 1.5 engine kernels (whose LayerNorm, split SiLU-mul and
# per-row FP8 quantization kernels and tcgen05 GEMMs the backend shares).
FLASHDRIVE_KERNEL_SYMBOLS = ("flashdrive_qk_prep_bf16", "flashdrive_rope_rows", "flashdrive_w4a8_act_prep", "flashdrive_w4a8_rotate",
                             "flashdrive_shift_cache",
                             "flashdrive_rms_f16delta_bf16", "flashdrive_silu_mul_f16in_bf16",
                             "flashdrive_w4a8_gemm_sm100", "flashdrive_w4a8_gemm_sm100_sizes", "flashdrive_w4a8_gemm_small",
                             "flashdrive_bf16_gemm_small", "flashdrive_draft_qkv_prep_bf16", "flashdrive_w4a8_rotate_dense",
                             "flashdrive_w4a8_quant", "flashdrive_w4a8_gemm_tile",
                             "flashdrive_bf16_gemm_frag", "flashdrive_w4a8_act_prep_tile")
DECODE_WEIGHTS = ("bf16", "int8", "fp8")   # LM head and DFlash draft linears of the decode graphs
PREFILL_WEIGHTS = ("w4a8", "int8")         # streaming-window GEMM operands: Marlin's group-scaled integers, or per-channel INT8
EXPERT_WEIGHTS = ("bf16", "fp8")       # action expert projections (FP8 = W8A8 with per-row activation scales)
W4A8_BACKENDS = ("flashdrive", "fused")
ACT_ROTS = ("paro", "dense")

class FlashDriveThorRuntime:
    """Options, kernel handles, prepared weights, persistent buffers, per-stream state and CUDA graphs of the FlashRT
    execution backend for a FlashDrive Alpamayo 1.5 model (the computation: ``pipeline_thor.FlashDriveThor``).

    Args:
        model: a FlashDrive model (``FlashDriveMixin``) after all setups, on CUDA.
        expert, vision, prefill, decode: ``"flashrt"`` (default) or
            ``"flashdrive"`` per primitive group (see the module docstring);
            ``decode="flashrt"`` needs ``prefill="flashrt"``.
        w4a8: execution of the W4A8 linears inside the FlashRT language model:
            ``"fused"`` (default with ``prefill="flashrt"``; one activation pass
            per shared input and Marlin outputs consumed by the next kernel) or
            ``"flashdrive"`` (FlashDrive's ParoQuant modules).
        act_rot: rotation of the W4A8 activations (``w4a8="fused"``): ``"paro"``
            (default; layer by layer, bit-exact with ParoQuant's kernel) or
            ``"dense"`` (one 128 x 128 FP16 matrix per channel group on tensor
            cores, FP32 accumulation, one FP16 rounding: a numerics change
            that can move INT8 codes by one step).
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

    def __init__(self, model, *, expert: str = "flashrt", vision: str = "flashrt", prefill: str = "flashrt",
                 decode: str = "flashrt", w4a8: str | None = None,
                 act_rot: str = "paro", decode_weights: str = "bf16", expert_weights: str = "bf16", pdl: bool = False, w4a8_ktile: int = 256,
                 expert_splits: int = 1, prefill_weights: str = "w4a8") -> None:
        if w4a8_ktile not in (128, 256):
            raise ValueError("w4a8_ktile must be 128 or 256")
        if prefill_weights not in PREFILL_WEIGHTS:
            raise ValueError(f"unknown prefill_weights {prefill_weights!r}; choose from {PREFILL_WEIGHTS}")
        backends = {"expert": expert, "vision": vision, "prefill": prefill, "decode": decode}
        if act_rot not in ACT_ROTS:
            raise ValueError(f"unknown act_rot {act_rot!r}; choose from {ACT_ROTS}")
        if decode_weights not in DECODE_WEIGHTS:
            raise ValueError(f"unknown decode_weights {decode_weights!r}; choose from {DECODE_WEIGHTS}")
        if expert_weights not in EXPERT_WEIGHTS:
            raise ValueError(f"unknown expert_weights {expert_weights!r}; choose from {EXPERT_WEIGHTS}")
        self.decode_weights = decode_weights
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
        backends["w4a8"] = w4a8
        if w4a8 == "fused":
            backends["act_rot"] = act_rot
        if decode == "flashrt" and prefill != "flashrt":
            raise ValueError("decode='flashrt' needs prefill='flashrt' (the rotated cache copy is kept by FlashRT writes)")
        try:
            from flashdrive import FlashDriveMixin
        except ImportError as exc:
            raise ImportError("the FlashDrive backend needs the flashdrive package") from exc
        if not isinstance(model, FlashDriveMixin):
            raise TypeError("the FlashDrive backend needs a FlashDrive model "
                            "(see flash_rt.frontends.torch.alpamayo15_flashdrive_thor.load_flashdrive)")
        if model.training:
            raise ValueError("the FlashDrive backend requires an eval-mode model")
        self.m = model
        self.backends = backends
        self.expert_splits = int(expert_splits)
        self._st = None                     # per-stream state (FlashDrive cache, tables, buffers, graphs)
        if "flashrt" not in backends.values():
            return
        import flash_rt.flash_rt_kernels as fvk
        from flash_rt.hardware.thor import fa4_backend
        require_alpamayo15_kernels(fvk)
        missing = [n for n in FLASHDRIVE_KERNEL_SYMBOLS if not hasattr(fvk, n)]
        if missing:
            raise RuntimeError(f"FlashRT was built without the FlashDrive backend kernels ({', '.join(missing)}); "
                               "rebuild with -DGPU_ARCH=110 -DFLASHRT_ENABLE_ALPAMAYO15=ON")
        self.fvk = fvk
        fvk.alpamayo15_set_pdl(bool(pdl))      # process-wide: programmatic dependent launch of the backend's kernels
        backends["pdl"] = bool(pdl)
        self.gemm = _Gemm(fvk)
        self.rng_blocks = torch_philox_blocks()
        self.fa4 = fa4_backend.fa4_fwd()
        if self.fa4 is None:
            raise RuntimeError(f"FA4 unavailable: {fa4_backend.status()}")
        self._init_geometry()
        self.vision_fp8 = None             # the backend's vision tower runs the BF16 blocks
        if expert == "flashrt":
            self._init_expert()
        if vision == "flashrt":
            self.vlm = model.vlm
            self.vision_attention = "fa4"
            check_vision_modules(model.vlm.model.visual)
            self.vision_tables = VisionTables(model.vlm.model.visual, max_entries=4)
            self.vision_graphs = VisionGraphs(self.vision_tables, max_entries=None)
        if expert == "flashrt":
            backends["expert_weights"] = expert_weights
        if w4a8 == "fused":
            self.w4, scratch, maps = [], {}, {}
            grp = lambda mods: W4A8Group(fvk, mods, scratch, maps, act_rot, w4a8_ktile, prefill_weights)
            backends["w4a8_ktile"] = w4a8_ktile
            backends["prefill_weights"] = prefill_weights
            for layer in model.vlm.model.language_model.layers:
                a, mlp = layer.self_attn, layer.mlp
                self.w4.append({"qkv": grp([a.q_proj, a.k_proj, a.v_proj]), "o": grp([a.o_proj]),
                                "gu": grp([mlp.gate_proj, mlp.up_proj]), "down": grp([mlp.down_proj])})
            del maps, grp
            torch.cuda.empty_cache()        # index maps of the weight unpacking (about 1 GB, transient)
            # the unpacked layout checked against Marlin itself, once per shape
            self.w4a8_gemm_agreement = min(g.check_against_marlin(fvk) for g in self.w4[0].values())
        if decode == "flashrt":
            backends["decode_weights"] = decode_weights
            self.dw = self._decode_linears(decode_weights)

    # ------------------------------------------------------------------
    # geometry, expert state, per-stream state
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

    def stream(self) -> dict:
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
        D = self.head_dim
        pos3 = pos[:, 0, :].to(torch.int32).contiguous()       # the stream's fixed (temporal, height, width) positions
        s = torch.cuda.current_stream().cuda_stream
        if self.backends["prefill"] == "flashrt":
            # VLM rotary tables over the stream's fixed position table (BF16, as FlashDrive's BF16 forward computes).
            rot = m.vlm.model.language_model.rotary_emb
            st["cos"], st["sin"] = torch.empty(S, D, device=DEV, dtype=BF16), torch.empty(S, D, device=DEV, dtype=BF16)
            self.fvk.alpamayo15_mrope_tables(pos3.data_ptr(), rot.inv_freq.detach().float().contiguous().data_ptr(),
                                              *_mrope_sections(rot), float(rot.attention_scaling),
                                              st["cos"].data_ptr(), st["sin"].data_ptr(), S, D, 0, s)
            st["kvlm"] = torch.empty(self.n_layers, S, self.n_kv, self.head_dim, device=DEV, dtype=BF16)
            st["rng"] = torch.tensor([torch.cuda.initial_seed() & 0x7FFFFFFFFFFFFFFF, 0, 0], dtype=torch.int64, device=DEV)
        if self.backends["expert"] == "flashrt":
            rot = m.expert.rotary_emb
            st["e_cos_tab"] = torch.empty(S, D, device=DEV, dtype=torch.float32)
            st["e_sin_tab"] = torch.empty(S, D, device=DEV, dtype=torch.float32)
            self.fvk.alpamayo15_mrope_tables(pos3.data_ptr(), rot.inv_freq.detach().float().contiguous().data_ptr(),
                                              *_mrope_sections(rot), float(rot.attention_scaling),
                                              st["e_cos_tab"].data_ptr(), st["e_sin_tab"].data_ptr(), S, D, 1, s)
            st["ke"] = torch.empty(self.n_layers, S, self.n_kv, self.head_dim, device=DEV, dtype=BF16)
        self._st = st
        if "kvlm" in st:
            sync_kvlm_all(self, st)          # rotated view of whatever FlashDrive's cache already holds
        return st

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
        self.e8 = None
        if self.expert_weights == "fp8":
            from flash_rt.models.alpamayo15.weights import quantize_fp8_rows
            self.e8 = []
            with torch.no_grad():
                for layer in m.expert.layers:
                    q = {}
                    for name, lin in (("qkv", layer.self_attn.qkv_proj), ("o", layer.self_attn.o_proj),
                                      ("gu", layer.mlp.gate_up_proj), ("down", layer.mlp.down_proj)):
                        codes, scale = quantize_fp8_rows(lin.weight.detach())
                        q[name] = (codes.contiguous(), scale.contiguous())
                    self.e8.append(q)
            torch.cuda.empty_cache()
            u8 = lambda *shape: torch.empty(*shape, device=DEV, dtype=torch.uint8)
            f32 = lambda *shape: torch.empty(*shape, device=DEV, dtype=torch.float32)
            self._eb.update({"hn8": u8(A, E), "hn_s": f32(A), "att8": u8(A, self.e_heads * D), "att_s": f32(A),
                             "act8": u8(A, self.e_ffn), "act_s": f32(A)})
        self.noise = mk(1, A, self.act_dim)            # FlashDrive's BF16 action noise buffer
        self.x_out = mk(1, A, self.act_dim)
        self.v_cache = mk(1, A, self.act_dim)
        self.act = pack_action_modules(m, self.act_dim, self.n_act)
        self._eb.update({"af": mk(A, self.act.k0), "ah0": mk(A, self.act.mlp[0][0].shape[0]), "ah1": mk(A, self.act.mlp[0][0].shape[0]),
                         "ae": mk(A, E), "v": mk(A, self.act_dim)})
        self.e_idx = torch.arange(A, dtype=torch.long, device=DEV)   # action rows: e_idx + d_slot
        self.d_slot = torch.zeros(1, dtype=torch.long, device=DEV)    # action_start (first action cache row)
        self.d_seqk = torch.zeros(1, dtype=torch.int32, device=DEV)   # action_start + n_act keys attended
        self.d_ke_rows = torch.zeros(1, dtype=torch.int32, device=DEV)  # rows [0, action_start) to rotate

    def expert_buffers(self) -> dict:
        """The expert's work buffers (allocated with the runtime; the shared action input projection reads them)."""
        return self._eb

    # ------------------------------------------------------------------
    # stream lifecycle
    # ------------------------------------------------------------------
    def start_stream(self, input_ids: torch.Tensor, ids_host: list[int], image_grid_thw: torch.Tensor) -> dict:
        """The first window of a stream, before its prefill: FlashDrive's stream bookkeeping (padded position table,
        per-view frame ranges, shift indices, streaming cache positions) computed on the host from the token ids
        (``ids_host``: ``input_ids[0]`` as a list), its cache layers allocated as its lazy initialization does, and
        the stream's state (rotated views, rotary tables). Returns the stream state."""
        m = self.m
        P, S, D = len(ids_host), int(m.max_cache_len), self.head_dim
        # FlashDrive's stream bookkeeping (streaming.py, _first_prefill), computed on the host
        cfg = m.vlm.config
        grids = [tuple(int(v) for v in row) for row in image_grid_thw.tolist()]
        pos_h, _ = mrope_positions(ids_host, grids, image_token_id=int(cfg.image_token_id), vision_start_id=int(cfg.vision_start_token_id),
                                   spatial_merge=int(cfg.vision_config.spatial_merge_size))
        if S > P:
            pos_h = np.concatenate([pos_h, pos_h[:, -1:] + np.arange(1, S - P + 1, dtype=np.int32)[None]], axis=1)
        m.streaming_position_ids = torch.from_numpy(pos_h.astype(np.int64)).unsqueeze(1).to(DEV)
        m._frame_ranges, m._traj_text_range = m._locate_prompt_ranges(input_ids.cpu())
        src_index, dst_index = [], []
        for view_frames in m._frame_ranges:
            for (dst_start, dst_end), (src_start, src_end) in zip(view_frames[:-1], view_frames[1:], strict=True):
                dst_index.append(torch.arange(dst_start, dst_end))
                src_index.append(torch.arange(src_start, src_end))
        m._shift_src_index = torch.cat(src_index).to(DEV)
        m._shift_dst_index = torch.cat(dst_index).to(DEV)
        m.streaming_cache_position = m._create_cache_position().to(DEV)
        proto = torch.zeros(1, self.n_kv, 1, D, device=DEV, dtype=BF16)
        for layer in m._past_key_values.layers:
            if not layer.is_initialized:
                layer.lazy_initialization(proto)
        return self.stream()

    @property
    def active_stream(self) -> dict | None:
        """The current stream's state, ``None`` before a stream starts and after ``reset``."""
        return self._st

    def first_prefill_lengths(self, st: dict, P: int) -> None:
        """Attention lengths of the first prefill: one causal run over the prompt's cache rows [0, P)."""
        st["keep"]["first_meta"] = {"cu_q": torch.tensor([0, P], dtype=torch.int32, device=DEV),
                                   "cu_k": torch.zeros(2, dtype=torch.int32, device=DEV),
                                   "used": torch.tensor([P], dtype=torch.int32, device=DEV), "max_q": P}

    def reset(self) -> None:
        """Start a new stream (see ``reset_flashdrive_stream``); stream-bound buffers and graphs are rebuilt for it."""
        reset_flashdrive_stream(self.m)
        self._st = None

    # ------------------------------------------------------------------
    # per-stream inputs of the primitives
    # ------------------------------------------------------------------
    def prefill_state(self, st: dict, b: dict, n: int, cache_position: torch.Tensor, inputs_embeds: torch.Tensor,
                      visual_pos_masks: torch.Tensor, deepstack_image_embeds) -> None:
        """The streaming prefill's per-stream static inputs (first streaming window): attention lengths of the window's
        contiguous cache-row runs, the window's cache rows and rotary rows, the embedding / DeepStack input buffers
        (filled with this window's) and the logits buffer, in the work buffers ``b`` of ``n`` rows."""
        m = self.m
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
        D = self.head_dim
        s = torch.cuda.current_stream().cuda_stream
        self.fvk.alpamayo15_gather_rows_bf16(st["cos"].data_ptr(), D, b["rows"].data_ptr(), b["cos"].data_ptr(), D, n, D, s)
        self.fvk.alpamayo15_gather_rows_bf16(st["sin"].data_ptr(), D, b["rows"].data_ptr(), b["sin"].data_ptr(), D, n, D, s)

    def traj_ids(self, st: dict) -> torch.Tensor:
        """The trajectory-start forward's token buffer of the stream."""
        ids = st["keep"].get("traj_ids")
        if ids is None:
            ids = torch.zeros(1, dtype=torch.long, device=DEV)
            st["keep"]["traj_ids"] = ids
        return ids

    def shift_plan(self, st: dict) -> dict:
        """The cache shift of the stream: FlashDrive's per-view frame ranges (which define its shift index pair, checked
        against it) and the cache layers' pointers, for the one-kernel shift."""
        sh = st["keep"].get("shift")
        if sh is None:
            m = self.m
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
        return sh

    # ------------------------------------------------------------------
    # CUDA graphs (captured on first use; cache rows and random state the warm-up / capture touch are restored)
    # ------------------------------------------------------------------
    def capture_prefill(self, st: dict, run, out: dict) -> torch.cuda.CUDAGraph:
        """The streaming prefill graph of the stream: ``run`` captured with the whole cache and its rotated view
        restored afterwards (capture writes the window rows); ``out`` (the graph's outputs) kept with it."""
        saved = [(k.clone(), v.clone()) for k, v in zip(st["k"], st["v"])]   # capture writes the window rows
        kvlm = st["kvlm"].clone()
        g = capture_graph(run)
        for (k, v), (ks, vs) in zip(zip(st["k"], st["v"]), saved):
            k.copy_(ks); v.copy_(vs)
        st["kvlm"].copy_(kvlm)
        del saved, kvlm
        st["graphs"]["prefill"] = g
        st["keep"]["prefill_out"] = out
        return g

    def capture_block(self, st: dict, key, run, out: dict, cache_position: torch.Tensor, set_inputs) -> torch.cuda.CUDAGraph:
        """A DFlash round graph: ``run`` captured on the round's inputs (``set_inputs``), the block's cache rows and both
        the generator's and the device sampler's random state restored; ``out`` kept with it."""
        saved = self.snapshot_rows(st, cache_position)
        set_inputs()
        gen = torch.cuda.default_generators[torch.cuda.current_device()]
        rng, rng_dev = gen.get_state(), st["rng"].clone()
        g = capture_graph(run)
        gen.set_state(rng)                      # warm-up / capture draws must not consume the caller's stream
        st["rng"].copy_(rng_dev)
        self.restore_rows(st, cache_position, saved)
        st["graphs"][key] = g
        st["keep"][("block_out",) + key] = out
        return g

    def capture_traj(self, st: dict, run, cache_position: torch.Tensor, set_inputs) -> torch.cuda.CUDAGraph:
        """The trajectory-start forward graph: ``run`` captured on its inputs (``set_inputs``), its cache rows restored."""
        saved = self.snapshot_rows(st, cache_position)
        set_inputs()
        g = capture_graph(run)
        self.restore_rows(st, cache_position, saved)
        st["graphs"]["traj"] = g
        return g

    def expert_graph(self, st: dict, key, schedule: tuple, n_steps: int, record: bool, denoise) -> torch.cuda.CUDAGraph:
        """The expert graph of an Euler schedule: FlashDrive's BF16 time grid, ``denoise(st, schedule, time_steps,
        step_dt, rec)`` warmed up and captured on scratch rows past every live row (touched cache rows and the expert
        state restored); ``record`` keeps the captured pass's per-step velocities and states."""
        time_steps = torch.linspace(0.0, 1.0, n_steps + 1, device=DEV, dtype=BF16)   # FlashDrive's BF16 grid
        ts_host = torch.linspace(0.0, 1.0, n_steps + 1, dtype=BF16)
        step_dt = [float((ts_host[i + 1] - ts_host[i]).float()) for i in range(n_steps)]   # BF16 differences
        st["keep"][key] = time_steps                   # captured by address: keep alive with the graph
        rec = {"v": [], "x": []} if record else None
        # Warm up / capture on scratch rows past every live row; the touched cache rows are restored.
        scratch = st["S"] - self.n_act
        saved = [t.clone() for t in (self.noise, self.d_slot, self.d_seqk, self.d_ke_rows, self.v_cache)]
        rows = [(k[0, :, scratch:].clone(), v[0, :, scratch:].clone()) for k, v in zip(st["k"], st["v"])]
        kvlm_rows = st["kvlm"][:, scratch:].clone() if "kvlm" in st else None
        self.d_slot.fill_(scratch); self.d_seqk.fill_(st["S"]); self.d_ke_rows.fill_(scratch)
        self.noise.zero_()
        g = capture_graph(lambda: denoise(st, list(schedule), time_steps, step_dt, rec))
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
        return g

    # ------------------------------------------------------------------
    # work buffers and decode weights
    # ------------------------------------------------------------------
    def dec_state(self, st: dict) -> dict:
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
            rot = dm.rotary_emb
            cos = torch.empty(C + B, D, device=DEV, dtype=BF16); sin = torch.empty(C + B, D, device=DEV, dtype=BF16)
            dpos = torch.arange(C + B, dtype=torch.int32, device=DEV).repeat(3, 1).contiguous()
            self.fvk.alpamayo15_mrope_tables(dpos.data_ptr(), rot.inv_freq.detach().float().contiguous().data_ptr(),
                                              *_mrope_sections(rot), float(rot.attention_scaling),
                                              cos.data_ptr(), sin.data_ptr(), C + B, D, 0, torch.cuda.current_stream().cuda_stream)
            cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
            d = {"slot": torch.zeros(1, dtype=torch.long, device=DEV), "used": torch.zeros(1, dtype=torch.int32, device=DEV),
                 "block_ids": torch.zeros(1, B, dtype=torch.long, device=DEV),
                 "th": torch.zeros(1, C, len(dm.target_layer_ids) * H, device=DEV, dtype=BF16),
                 "proposed": torch.zeros(B, dtype=torch.long, device=DEV), "block": torch.zeros(B, dtype=torch.long, device=DEV),
                 "posterior": torch.zeros(B, dtype=torch.long, device=DEV), "stats": torch.zeros(3, dtype=torch.long, device=DEV),
                 "b_idx": torch.arange(B, dtype=torch.long, device=DEV), "one_idx": torch.zeros(1, dtype=torch.long, device=DEV),
                 "draft_pos": torch.arange(C + B, device=DEV).unsqueeze(0),
                 # the sampler's exp scratch and the exchange area of its cooperating blocks (B verification rows)
                 "sscratch": torch.empty(B, m.vlm.lm_head.weight.shape[0], device=DEV, dtype=torch.float32),
                 "sws": torch.zeros(int(self.fvk.alpamayo15_sample_workspace_bytes(B)), device=DEV, dtype=torch.uint8),
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
        lin = {k: (Fp8Fragment(w, decode_weights) if decode_weights in ("fp8", "int8") else Bf16Fragment(w))
               for k, w in lin.items()}
        torch.cuda.empty_cache()
        return lin

    def lm_buffers(self, st: dict, n: int, ctx_rows: int) -> dict:
        key = ("lmbuf", n, ctx_rows)
        b = st["keep"].get(key)
        if b is None:
            H, D, NH, F_ = self.hidden, self.head_dim, self.n_heads, self.ffn
            mk = lambda *shape, dt=BF16: torch.empty(*shape, device=DEV, dtype=dt)
            ncap = len(self.m.vlm.model.language_model._capture_layer_ids)
            b = {"x": mk(n, H), "h": torch.zeros(n + 8, H, device=DEV, dtype=BF16)[:n],   # few-row GEMMs read 8-row tiles of h
                 "d16": mk(n, H),
                 "q": mk(n, NH * D), "attn": mk(n, NH * D), "act": mk(n, F_),
                 "cos": mk(n, D), "sin": mk(n, D), "rows": torch.zeros(n, dtype=torch.long, device=DEV),
                 "ctx": mk(ctx_rows, ncap * H)}
            st["keep"][key] = b
        return b

    def snapshot_rows(self, st: dict, cache_position: torch.Tensor) -> tuple:
        """Cache rows a captured decode graph writes during warm-up/capture (restored afterwards)."""
        r0 = int(cache_position[0])
        r1 = min(st["S"], r0 + self.m.dflash_block_size)
        return r0, r1, [(k[0, :, r0:r1].clone(), v[0, :, r0:r1].clone()) for k, v in zip(st["k"], st["v"])], \
            st["kvlm"][:, r0:r1].clone()

    def restore_rows(self, st: dict, cache_position: torch.Tensor, saved: tuple) -> None:
        r0, r1, rows, kvlm = saved
        for (k, v), (ks, vs) in zip(zip(st["k"], st["v"]), rows):
            k[0, :, r0:r1].copy_(ks); v[0, :, r0:r1].copy_(vs)
        st["kvlm"][:, r0:r1].copy_(kvlm)
