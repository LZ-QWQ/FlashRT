"""Alpamayo 1.5 engine on Jetson AGX Thor: initialization and resources.

``Alpamayo15ThorRuntime`` owns everything the request computation of
``pipeline_thor.Alpamayo15Thor`` runs on: the validated options and precision
tiers, the kernel handles (FlashRT kernels, FA4, cuBLASLt, programmatic
dependent launch), the model geometry, the prepared weights, the persistent
device buffers (KV cache, decode and expert state, rotary tables, sampler
workspace) and the shape-sized work buffers, and the CUDA graphs (capture on
scratch state, lookup by shape). The pipeline adds no buffers of its own and
captures nothing itself; it calls the runtime with its step functions.
"""
from __future__ import annotations

import numpy as np
import torch

from flash_rt.models.alpamayo15.pipeline_thor import BF16, DEV, _Gemm, _autocast
from flash_rt.models.alpamayo15.weights import pack_action_modules, prepare_weights

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
    # cuBLASLt GEMMs (csrc/gemm/alpamayo15_cublaslt.cu), same build gate
    "alpamayo15_cublaslt_init",
    "alpamayo15_cublaslt_gemm_bf16",
    "alpamayo15_cublaslt_gemm_fp8",
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
DECODE_WEIGHTS = ("bf16", "fp8", "int8", "w4a8")
EXPERT_WEIGHTS = ("bf16", "fp8")
PREFILL_WEIGHTS = ("bf16", "fp8", "nvfp4")
VISION_WEIGHTS = ("bf16", "fp8")
LM_HEAD_WEIGHTS = ("decode", "int4")        # "decode" = the decode tier's format (INT8 with W4A8), "int4" = own INT4 linear
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


def torch_philox_blocks() -> int:
    """torch's launch cap of its CUDA distribution kernels on the current device: SMs times blocks of 256 threads per
    SM (the sampler kernel reproduces torch's per-element Philox words, which depend on this cap)."""
    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    return int(p.multi_processor_count) * (int(p.max_threads_per_multi_processor) // 256)


def capture_graph(fn, warmups: int = 2) -> torch.cuda.CUDAGraph:
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


def check_vision_modules(V) -> None:
    """The CUTLASS vision path assumes the Qwen3-VL tower layout of Alpamayo 1.5 (bias linears, LayerNorms,
    GELU-tanh block MLP, erf-GELU mergers, equal frame grids)."""
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


class VisionTables:
    """Per-grid position embeddings and rotary tables of the vision tower (official functions), most recently used
    last; at most ``max_entries`` grids are kept (``None``: unbounded)."""

    def __init__(self, visual, max_entries: int | None = 4) -> None:
        self.visual = visual
        self.max_entries = max_entries
        self.tables: dict = {}

    def __call__(self, grid: torch.Tensor):
        key = tuple(grid.flatten().tolist())
        tab = self.tables.pop(key, None)
        if tab is None:
            V = self.visual
            with _autocast():
                pos = V.fast_pos_embed_interpolate(grid)
            rot = V.rot_pos_emb(grid)
            emb = torch.cat((rot, rot), dim=-1)
            tab = (pos.to(BF16), emb.cos().float().contiguous(), emb.sin().float().contiguous())
            if self.max_entries is not None and len(self.tables) >= self.max_entries:
                self.tables.pop(next(iter(self.tables)))
        self.tables[key] = tab        # most recently used last
        return tab, key


class VisionGraphs:
    """The vision tower as one CUDA graph per (pixel shape, frame grid): static pixel buffers (and the BF16 copy the
    tower casts into), the captured outputs and the per-grid tables the graph reads by address, kept alive together;
    at most ``max_entries`` graphs (``None``: unbounded)."""

    def __init__(self, tables: VisionTables, max_entries: int | None = 2) -> None:
        self.tables = tables
        self.max_entries = max_entries
        self.graphs: dict = {}

    def __len__(self) -> int:
        return len(self.graphs)

    def get(self, pixel_values: torch.Tensor, grid: torch.Tensor, tower, key_grid: tuple | None = None) -> dict:
        """The entry of this shape and grid (``key_grid``: the grid as a flat tuple, when the caller has it), captured
        on first use from ``tower(px, grid, px16)``."""
        if key_grid is None:
            key_grid = tuple(grid.flatten().tolist())
        key = (tuple(pixel_values.shape), key_grid)
        ent = self.graphs.get(key)
        if ent is None:
            if self.max_entries is not None and len(self.graphs) >= self.max_entries:
                self.graphs.pop(next(iter(self.graphs)))
            px = torch.empty_like(pixel_values)
            px.copy_(pixel_values)
            px16 = torch.empty(pixel_values.shape, device=DEV, dtype=BF16) if pixel_values.dtype != BF16 else None
            with _autocast():
                tower(px, grid, px16)                      # builds the per-grid tables outside capture
            grid_host = grid.cpu()                         # table lookup key only: no device sync while capturing
            tables = self.tables.tables[key_grid]          # captured by address: keep alive with the graph
            out = {}

            def run():
                out["img"], out["deep"] = tower(px, grid_host, px16)
            ent = {"px": px, "px16": px16, "out": out, "tables": tables, "graph": capture_graph(run)}
            self.graphs[key] = ent
        return ent


class Alpamayo15ThorRuntime:
    """Options, kernel handles, prepared weights, persistent buffers and CUDA graphs of the Thor engine for one
    official Alpamayo 1.5 model (the computation: ``pipeline_thor.Alpamayo15Thor``).

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
        self.fa4 = fa4_backend.fa4_fwd()
        if self.fa4 is None:
            raise RuntimeError(f"FA4 unavailable: {fa4_backend.status()}")
        self.fvk = fvk
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
        self.gemm = _Gemm(fvk)
        self.pdl = bool(pdl)
        fvk.alpamayo15_set_pdl(self.pdl)
        fvk.alpamayo15_cublaslt_init()          # handle + workspace for the cuBLASLt GEMMs (outside any graph capture)
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
        self.rng_blocks = torch_philox_blocks()
        vocab = int(model.vlm.lm_head.weight.shape[0])
        self.samp_scratch = torch.empty(B, vocab, device=DEV, dtype=torch.float32)      # exp(score - max) per row
        self.samp_ws = torch.zeros(int(fvk.alpamayo15_sample_workspace_bytes(B)), device=DEV, dtype=torch.uint8)
        self.decode_graphs: dict[int, torch.cuda.CUDAGraph] = {}

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
        self.expert_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.a_v0 = torch.zeros(B, device=DEV, dtype=torch.float32)                 # t0 velocity estimate per sample
        self.a_z = torch.zeros(B, device=DEV, dtype=torch.float32)                  # the history's last z
        self.pred_xyz = torch.zeros(B, self.n_act, 3, device=DEV, dtype=torch.float32)
        self.pred_rot = torch.zeros(B, self.n_act, 3, 3, device=DEV, dtype=torch.float32)
        self.act = pack_action_modules(model, self.act_dim, self.n_act)
        self.w = prepare_weights(model, fvk, tiers, max_samples=self.max_samples, paro_checkpoint=paro_checkpoint)
        self.vision_fp8 = self.w.v8                    # vision_weights="fp8": per-row FP8 block weights
        self._pf = None
        self._db = None
        self._eb = None
        check_vision_modules(self.vlm.model.visual)
        self.vision_tables = VisionTables(self.vlm.model.visual, max_entries=4)
        self.vision_graphs = VisionGraphs(self.vision_tables, max_entries=2)
        self._attention_warm = False

    # ------------------------------------------------------------------
    # work buffers, sized on first use (persistent: CUDA graphs read them by address)
    # ------------------------------------------------------------------
    def prefill_buffers(self, P: int) -> dict:
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

    def decode_buffers(self) -> dict:
        if self._db is None:
            B, H, D, F_ = max(self.max_samples, 16), self.hidden, self.head_dim, self.ffn
            mk = lambda *shape: torch.zeros(*shape, device=DEV, dtype=BF16)
            self._db = {"x": mk(B, H), "h": mk(B, H), "qkv": mk(B, (self.n_heads + 2 * self.n_kv) * D),
                        "q": mk(B, self.n_heads * D), "o": mk(B, H), "gu": mk(B, 2 * F_), "a": mk(B, F_),
                        "d": mk(B, H), "cos": mk(B, D), "sin": mk(B, D),
                        "logits": mk(B, self.vlm.lm_head.weight.shape[0])}
        return self._db

    def cu_q(self, n: int, rows: int) -> torch.Tensor:
        key = (n, rows)
        if key not in self._cuq:
            self._cuq[key] = (torch.arange(n + 1, device=DEV, dtype=torch.int32) * rows).contiguous()
        return self._cuq[key]

    def expert_buffers(self) -> dict:
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

    # ------------------------------------------------------------------
    # CUDA graphs
    # ------------------------------------------------------------------
    def decode_graph(self, n: int, step) -> torch.cuda.CUDAGraph:
        """The decode-step graph of ``n`` samples, ``step(n)`` captured on first use (scratch cache rows and state, the
        caller's state and random stream restored)."""
        g = self.decode_graphs.get(n)
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
                step(n)
                scratch()
        torch.cuda.current_stream().wait_stream(st)
        scratch()
        g = torch.cuda.CUDAGraph()
        with torch.no_grad(), _autocast(), torch.cuda.graph(g):
            step(n)
        torch.cuda.synchronize()
        for t, sv in zip(state, saved):
            t.copy_(sv)
        gen.set_state(rng)
        self.decode_graphs[n] = g
        return g

    def expert_graph(self, n: int, denoise) -> torch.cuda.CUDAGraph:
        """The diffusion graph of ``n`` samples, ``denoise(n)`` captured on first use (scratch action rows, the
        caller's state restored)."""
        g = self.expert_graphs.get(n)
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
                denoise(n)
        torch.cuda.current_stream().wait_stream(st)
        g = torch.cuda.CUDAGraph()
        with torch.no_grad(), _autocast(), torch.cuda.graph(g):
            denoise(n)
        torch.cuda.synchronize()
        for t, sv in zip((self.x, self.d_eseq, self.d_esuf, self.d_aslot, self.d_plen, self.d_cuk), saved):
            t.copy_(sv)
        self.expert_graphs[n] = g
        return g

    def capture(self, num_samples: tuple[int, ...], step, denoise) -> None:
        """Capture the decode and diffusion graphs for the given sample counts and
        JIT-compile the prefill and vision attention kernels."""
        for n in num_samples:
            if not 1 <= n <= self.max_samples:
                raise ValueError(f"num_samples {n} outside [1, max_samples={self.max_samples}]")
            self.decode_graph(n, step)
            self.expert_graph(n, denoise)
        self.warm_attention()

    def warm_attention(self) -> None:
        """JIT-compile the prefill and vision attention kernels once (outside any request)."""
        if self._attention_warm:
            return
        # FA4 compiles per configuration, which includes length-dependent tiling
        # choices: warm with prompt- and frame-sized sequences, not toy shapes.
        D, NH, NKV = self.head_dim, self.n_heads, self.n_kv
        q = torch.zeros(1, 1024, NH, D, device=DEV, dtype=BF16)
        kv = torch.zeros(1, 1024, NKV, D, device=DEV, dtype=BF16)
        self.fa4(q, kv, kv, softmax_scale=self.scale, causal=True, pack_gqa=True)
        V = self.vlm.model.visual
        nh = V.blocks[0].attn.num_heads
        hd = V.blocks[0].attn.qkv.weight.shape[1] // nh
        qv = torch.zeros(2, 3, 720, nh, hd, device=DEV, dtype=BF16)   # interleaved q/k/v views
        self.fa4(qv[:, 0], qv[:, 1], qv[:, 2], softmax_scale=V.blocks[0].attn.scaling, causal=False)
        torch.cuda.synchronize()
        self._attention_warm = True
