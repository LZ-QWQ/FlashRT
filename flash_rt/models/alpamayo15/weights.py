"""Weight preparation for the Alpamayo 1.5 Thor engine.

Everything here runs once when the engine is built and produces the operands the
kernels read: the fused q/k/v and gate/up projections (row views of the official
parameters, no copies), the precision tiers' quantized copies (per-row FP8 / INT8
fragments for the few-row GEMM, ParoQuant W4A8 decode groups, the INT4 LM head,
NVFP4 prefill operands, FP8 vision blocks) and the action expert's projection
operands. The engine (``pipeline_thor``) owns the buffers, graphs and the request
path and only reads what is prepared here.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

BF16 = torch.bfloat16
DEV = "cuda"
FP8_E4M3_MAX = 448.0
NVFP4_PARTIALS = 1024       # per-block maxima of one weight tensor (its per-tensor scale)
# Grouped W4A8 GEMM launch forms "ksplit:prefetch:warps" (warps per tile, weight groups in flight per warp, warps per
# CTA) for the decode groups under 1024 weight tiles (q|k|v, o, down) and from 1024 on (gate|up, the INT4 LM head).
W4A8_GEMM_NARROW = "1:2:4"
W4A8_GEMM_WIDE = "2:4:4"


@torch.no_grad()
def quantize_fp8_rows(w: torch.Tensor, chunk_rows: int = 16384) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-row FP8 E4M3 quantization of a [N, K] weight.

    ``scale = absmax(row) / 448`` (1 for an all-zero row) and the codes are
    ``round_to_nearest_even(w / scale)`` as ``torch.float8_e4m3fn``. Returns
    (codes as uint8 [N, K], scale fp32 [N]); rows are processed in chunks to
    bound the FP32 temporary. Runs without autograd: model parameters usually
    require grad, and a recorded graph would keep every FP32 temporary alive.
    """
    N, K = w.shape
    codes = torch.empty(N, K, device=w.device, dtype=torch.uint8)
    scale = torch.empty(N, device=w.device, dtype=torch.float32)
    for r0 in range(0, N, chunk_rows):
        wf = w[r0:r0 + chunk_rows].float()
        amax = wf.abs().amax(dim=1)
        s = torch.where(amax > 0, amax / FP8_E4M3_MAX, torch.ones_like(amax))
        q = (wf / s[:, None]).clamp_(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
        codes[r0:r0 + chunk_rows] = q.view(torch.uint8)
        scale[r0:r0 + chunk_rows] = s
    return codes, scale


@torch.no_grad()
def quantize_int8_rows(w: torch.Tensor, chunk_rows: int = 16384) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-row symmetric INT8 quantization of a [N, K] weight: ``scale = absmax(row) / 127`` (1 for an all-zero
    row), codes ``round_to_nearest_even(w / scale)`` in [-127, 127] as int8 viewed uint8 (the W8A16 kernel's byte
    layout); returns (codes uint8 [N, K], scale fp32 [N])."""
    N, K = w.shape
    codes = torch.empty(N, K, device=w.device, dtype=torch.uint8)
    scale = torch.empty(N, device=w.device, dtype=torch.float32)
    for r0 in range(0, N, chunk_rows):
        wf = w[r0:r0 + chunk_rows].float()
        amax = wf.abs().amax(dim=1)
        s = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
        q = torch.round(wf / s[:, None]).clamp_(-127.0, 127.0).to(torch.int8)
        codes[r0:r0 + chunk_rows] = q.view(torch.uint8)
        scale[r0:r0 + chunk_rows] = s
    return codes, scale


@torch.no_grad()
def fp8_fragment_layout(codes: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reorder [N, K] one-byte codes (e4m3 or int8) into the ``alpamayo15_w8a16_gemm_bf16`` fragment order.

    Rows are zero-padded to a multiple of 16 (scale 0); per 16-row tile and
    32-column chunk, lane ``4g + c`` holds the 16 bytes of its two
    ``mma.m16n8k16`` A fragments (see ``alpamayo15_kernels.cuh``).
    """
    N, K = codes.shape
    if K % 32:
        raise ValueError(f"K={K} must be a multiple of 32")
    T = (N + 15) // 16
    if T * 16 != N:
        codes = torch.cat([codes, codes.new_zeros(T * 16 - N, K)])
        scale = torch.cat([scale, scale.new_zeros(T * 16 - N)])
    # [t, rh, g, kc, kt, kh, c, e] -> [t, kc, g, c, kt, kh, rh, e]
    frag = codes.view(T, 2, 8, K // 32, 2, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7).contiguous()
    return frag.view(-1), scale.contiguous()


def fuse_qkv(layers) -> list[torch.Tensor]:
    """One ``[q | k | v]`` weight per layer; the modules' q/k/v parameters become row views of it (identical values,
    no extra memory), so the official modules keep working but the model must not be moved or have these parameters
    replaced afterwards."""
    out = []
    for layer in layers:
        a = layer.self_attn
        fused = torch.cat([a.q_proj.weight.data, a.k_proj.weight.data, a.v_proj.weight.data], 0).contiguous()
        nq, nk = a.q_proj.weight.shape[0], a.k_proj.weight.shape[0]
        a.q_proj.weight.data = fused[:nq]
        a.k_proj.weight.data = fused[nq:nq + nk]
        a.v_proj.weight.data = fused[nq + nk:]
        out.append(fused)
    return out


def fuse_gate_up(layers) -> list[torch.Tensor]:
    """One ``[gate | up]`` weight per layer, the modules' parameters becoming row views (see ``fuse_qkv``)."""
    out = []
    for layer in layers:
        mlp = layer.mlp
        fused = torch.cat([mlp.gate_proj.weight.data, mlp.up_proj.weight.data], 0).contiguous()
        ng = mlp.gate_proj.weight.shape[0]
        mlp.gate_proj.weight.data = fused[:ng]
        mlp.up_proj.weight.data = fused[ng:]
        out.append(fused)
    return out


@torch.no_grad()
def quantize_decode_weights(layers, w_qkv: list, w_gu: list, lm_head_weight: torch.Tensor | None, fmt: str,
                            projections: bool) -> dict:
    """Per-row FP8 or INT8 fragment copies of the decode projections (the BF16 originals stay for prefill) and of the
    LM head (``lm_head_weight`` None = no LM head copy). With the W4A8 tier only the LM head is wanted
    (``projections=False``)."""
    quant = quantize_int8_rows if fmt == "int8" else quantize_fp8_rows
    q8 = {"qkv": [], "o": [], "gu": [], "down": [], "fmt": 1 if fmt == "int8" else 0}
    if projections:
        for li, layer in enumerate(layers):
            q8["qkv"].append(fp8_fragment_layout(*quant(w_qkv[li])))
            q8["o"].append(fp8_fragment_layout(*quant(layer.self_attn.o_proj.weight)))
            q8["gu"].append(fp8_fragment_layout(*quant(w_gu[li])))
            q8["down"].append(fp8_fragment_layout(*quant(layer.mlp.down_proj.weight)))
    if lm_head_weight is not None:
        q8["lm_head"] = fp8_fragment_layout(*quant(lm_head_weight))
    torch.cuda.empty_cache()
    return q8


@torch.no_grad()
def load_w4_lm_head(fvk, lm_head_weight: torch.Tensor, max_samples: int):
    """The LM head as an own INT4 linear on the grouped few-row GEMM (``lm_head_weights="int4"``)."""
    from flash_rt.models.alpamayo15.marlin_w4a8 import W4LmHead
    head = W4LmHead(fvk, lm_head_weight, max_samples, tuple(int(v) for v in W4A8_GEMM_WIDE.split(":")))
    torch.cuda.empty_cache()
    return head


@torch.no_grad()
def load_w4a8_decode(fvk, path: str, model, max_samples: int) -> list[dict]:
    """The ParoQuant W4A8 linears of every language-model layer as decode groups (q|k|v, o, gate|up, down)."""
    from flash_rt.models.alpamayo15.marlin_w4a8 import W4A8DecodeGroup, load_paro_language_model
    missing = [n for n in ("flashdrive_w4a8_act_prep", "flashdrive_w4a8_rotate") if not hasattr(fvk, n)]
    if missing:
        raise RuntimeError(f"FlashRT was built without the W4A8 activation preparation kernels ({', '.join(missing)})")
    cfg = model.vlm.config.text_config
    H, F_, D = cfg.hidden_size, cfg.intermediate_size, cfg.head_dim
    layers = load_paro_language_model(path, cfg.num_hidden_layers)
    groups, maps = [], {}
    narrow, wide = (tuple(int(v) for v in c.split(":")) for c in (W4A8_GEMM_NARROW, W4A8_GEMM_WIDE))
    for li, lin in enumerate(layers):
        if lin["q_proj"]["N"] != cfg.num_attention_heads * D or lin["k_proj"]["N"] != cfg.num_key_value_heads * D \
                or lin["q_proj"]["K"] != H or lin["gate_proj"]["N"] != F_ or lin["down_proj"]["K"] != F_:
            raise ValueError(f"layer {li}: ParoQuant linear shapes do not match the model")
        grp = lambda names: W4A8DecodeGroup(fvk, [lin[n] for n in names], max_samples, maps, narrow, wide)
        groups.append({"qkv": grp(("q_proj", "k_proj", "v_proj")), "o": grp(("o_proj",)),
                       "gu": grp(("gate_proj", "up_proj")), "down": grp(("down_proj",))})
    del maps
    torch.cuda.empty_cache()
    return groups


@torch.no_grad()
def quantize_fp8_layers(layers, w_qkv: list, w_gu: list) -> list[dict]:
    """Per-row FP8 E4M3 copies (row-major codes + fp32 scales) of a decoder stack's projections (q|k|v, o, gate|up,
    down) for the W8A8 tcgen05 GEMM: the expert's and the prefill tiers."""
    out = []
    for li, layer in enumerate(layers):
        q = {}
        for name, w in (("qkv", w_qkv[li]), ("o", layer.self_attn.o_proj.weight), ("gu", w_gu[li]),
                        ("down", layer.mlp.down_proj.weight)):
            codes, scale = quantize_fp8_rows(w)
            q[name] = (codes.contiguous(), scale.contiguous())
        out.append(q)
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def quantize_nvfp4_prefill(layers, w_qkv: list, w_gu: list, ffn: int) -> dict:
    """NVFP4 copies (e2m1 nibbles, UE4M3 block-16 scales in the GEMM's tile-interleaved layout, a per-tensor scale)
    of the prefill projections: q|k|v, o, gate, up and down of every layer, plus the module and the workspaces.
    The activations are quantized at run time by their producers with per-row scales (the ``gsw_*`` tensors
    hold the weights' scales per input group, for the producers' per-row GEMM alphas)."""
    try:
        import flash_rt.flash_rt_fp4 as f4
    except Exception as exc:
        raise RuntimeError(f"prefill_weights='nvfp4' needs the flash_rt_fp4 module (ENABLE_SM100_CUTLASS build): {exc}")
    for name in ("alpamayo15_nvfp4_gemm_bf16", "alpamayo15_nvfp4_quantize_rows_bf16", "alpamayo15_nvfp4_absmax_bf16",
                 "alpamayo15_nvfp4_quantize_rows_rowscale_bf16", "alpamayo15_nvfp4_rms_rows_bf16", "alpamayo15_nvfp4_silu_mul_rows_bf16"):
        if not hasattr(f4, name):
            raise RuntimeError(f"flash_rt_fp4 was built without {name}")
    s = torch.cuda.current_stream().cuda_stream
    partials = torch.empty(NVFP4_PARTIALS, device=DEV, dtype=torch.float32)
    gs_dev = torch.empty(1, device=DEV, dtype=torch.float32)
    out = []
    for li, layer in enumerate(layers):
        q = {}
        for name, w in (("qkv", w_qkv[li]), ("o", layer.self_attn.o_proj.weight), ("gate", w_gu[li][:ffn]),
                        ("up", w_gu[li][ffn:]), ("down", layer.mlp.down_proj.weight)):
            w = w.contiguous()
            N, K = w.shape
            packed = torch.empty(N, K // 2, device=DEV, dtype=torch.uint8)
            sf = torch.zeros(int(f4.alpamayo15_nvfp4_sf_bytes(N, K)), device=DEV, dtype=torch.uint8)
            n = f4.alpamayo15_nvfp4_absmax_bf16(w.data_ptr(), K, N, K, partials.data_ptr(), NVFP4_PARTIALS, s)
            if n <= 0:
                raise RuntimeError(f"NVFP4 weight maxima failed (rc={n})")
            rc = f4.alpamayo15_nvfp4_quantize_rows_bf16(w.data_ptr(), K, N, K, partials.data_ptr(), n, packed.data_ptr(), sf.data_ptr(),
                                                        1, gs_dev.data_ptr(), 0, 0, 0, s)
            if rc != 0:
                raise RuntimeError(f"NVFP4 weight quantization failed (rc={rc})")
            torch.cuda.synchronize()
            q[name] = (packed, sf, float(gs_dev.item()))
        # the weights' per-tensor scales per input group (device, for the activation quantizer's alpha products)
        for group in (("qkv",), ("o",), ("gate", "up"), ("down",)):
            q["gsw_" + group[0]] = torch.tensor([q[g][2] for g in group], device=DEV, dtype=torch.float32)
        out.append(q)
    torch.cuda.empty_cache()
    return {"module": f4, "layers": out, "ws": {}}


@torch.no_grad()
def quantize_fp8_vision(blocks) -> list[dict]:
    """Per-row FP8 E4M3 copies of every vision block's qkv / proj / fc1 / fc2 weights."""
    v8 = []
    for blk in blocks:
        q = {}
        for name, lin in (("qkv", blk.attn.qkv), ("proj", blk.attn.proj), ("fc1", blk.mlp.linear_fc1), ("fc2", blk.mlp.linear_fc2)):
            codes, scale = quantize_fp8_rows(lin.weight)
            q[name] = (codes.contiguous(), scale.contiguous())
        v8.append(q)
    torch.cuda.empty_cache()
    return v8


@dataclass
class ActionModules:
    """Kernel operands of the action expert's input projection (``PerWaypointActionInProjV2``: Fourier features of
    every action dimension and of the timestep, a SiLU MLP with RMSNorms, a LayerNorm), of its output projection
    and the unicycle action space's constants."""

    freqs: torch.Tensor                                     # [nf / 2] bf16, shared by every Fourier encoder
    nf: int                                                 # features per encoder
    width: int                                              # (act_dim + 1) * nf
    k0: int                                                 # the first linear's input width padded to the GEMM alignment
    mlp: list[tuple[torch.Tensor, torch.Tensor, int]]       # (weight, bias, epilogue activation) per linear
    rms: list[tuple[torch.Tensor, float]]                   # (weight, eps) per RMSNorm between the linears
    ln: tuple[torch.Tensor, torch.Tensor, float]            # the final LayerNorm (weight, bias, eps)
    out_w: torch.Tensor                                     # output projection [act_dim, E] bf16
    out_b: torch.Tensor                                     # output projection bias [act_dim] bf16
    space: tuple[float, float, float, float, float]         # (dt, accel_mean, accel_std, kappa_mean, kappa_std)


@torch.no_grad()
def pack_action_modules(model, act_dim: int, n_act: int) -> ActionModules:
    """The action projection modules of an official model as kernel operands (BF16 copies of the small weights)."""
    ap, out, sp = model.action_in_proj, model.action_out_proj, model.action_space
    trunk = list(ap.encoder.trunk)
    lins = [m for m in trunk if isinstance(m, torch.nn.Linear)]
    rms = [m for m in trunk if type(m).__name__ == "RMSNorm"]
    if len(lins) != len(rms) + 1 or len(ap.sinus) != act_dim or not isinstance(out, torch.nn.Linear):
        raise ValueError("unexpected action projection modules")
    nf = int(ap.timestep_fourier_encoder.out_dim)
    for enc in list(ap.sinus) + [ap.timestep_fourier_encoder]:
        if int(enc.out_dim) != nf or not torch.equal(enc.freqs, ap.sinus[0].freqs):
            raise ValueError("the Fourier encoders are expected to share one frequency table")
    width = (act_dim + 1) * nf
    k0 = (width + 63) // 64 * 64
    w0 = torch.zeros(lins[0].out_features, k0, device=DEV, dtype=BF16)
    w0[:, :width] = lins[0].weight.detach().to(BF16)
    mlp = [(w0, lins[0].bias.detach().to(BF16).contiguous(), 3)]
    for li in range(1, len(lins)):
        mlp.append((lins[li].weight.detach().to(BF16).contiguous(), lins[li].bias.detach().to(BF16).contiguous(),
                    3 if li < len(lins) - 1 else 0))
    if int(sp.n_waypoints) != n_act:
        raise ValueError("action space waypoints differ from the expert's action tokens")
    return ActionModules(
        freqs=ap.sinus[0].freqs.detach().reshape(-1).to(BF16).contiguous(), nf=nf, width=width, k0=k0, mlp=mlp,
        rms=[(m.weight.detach().to(BF16).contiguous(), float(m.eps)) for m in rms],
        ln=(ap.norm.weight.detach().to(BF16).contiguous(), ap.norm.bias.detach().to(BF16).contiguous(), float(ap.norm.eps)),
        out_w=out.weight.detach().to(BF16).contiguous(), out_b=out.bias.detach().to(BF16).contiguous(),
        space=(float(sp.dt), float(sp.accel_mean), float(sp.accel_std), float(sp.curvature_mean), float(sp.curvature_std)))


@dataclass
class Alpamayo15Weights:
    """The engine's weight operands: the fused BF16 projections (always) and the tiers' copies (``None`` when the
    tier keeps BF16)."""

    qkv: list[torch.Tensor]                 # language model [q | k | v] per layer
    gu: list[torch.Tensor]                  # language model [gate | up] per layer
    e_qkv: list[torch.Tensor]               # expert [q | k | v] per layer
    e_gu: list[torch.Tensor]                # expert [gate | up] per layer
    q8: dict | None = None                  # decode_weights fp8 / int8 (and the INT8 LM head of w4a8)
    w4: list[dict] | None = None            # decode_weights w4a8: ParoQuant decode groups per layer
    lm4: object | None = None               # lm_head_weights int4
    e8: list[dict] | None = None            # expert_weights fp8
    p8: list[dict] | None = None            # prefill_weights fp8
    p4: dict | None = None                  # prefill_weights nvfp4
    v8: list[dict] | None = None            # vision_weights fp8


@torch.no_grad()
def prepare_weights(model, fvk, tiers: dict, *, max_samples: int, paro_checkpoint: str | None) -> Alpamayo15Weights:
    """Fuse the official projections and build the copies the resolved tiers (``resolve_preset``) need."""
    lm, expert = model.vlm.model.language_model, model.expert
    w = Alpamayo15Weights(qkv=fuse_qkv(lm.layers), gu=fuse_gate_up(lm.layers), e_qkv=fuse_qkv(expert.layers),
                          e_gu=fuse_gate_up(expert.layers))
    torch.cuda.empty_cache()
    decode, lm_head = tiers["decode_weights"], tiers["lm_head_weights"]
    if decode != "bf16":
        w.q8 = quantize_decode_weights(lm.layers, w.qkv, w.gu, model.vlm.lm_head.weight if lm_head == "decode" else None,
                                       "int8" if decode == "w4a8" else decode, projections=decode != "w4a8")
    if decode == "w4a8":
        w.w4 = load_w4a8_decode(fvk, paro_checkpoint, model, max_samples)
    if lm_head == "int4":
        w.lm4 = load_w4_lm_head(fvk, model.vlm.lm_head.weight, max_samples)
    if tiers["expert_weights"] == "fp8":
        w.e8 = quantize_fp8_layers(expert.layers, w.e_qkv, w.e_gu)
    if tiers["prefill_weights"] == "fp8":
        w.p8 = quantize_fp8_layers(lm.layers, w.qkv, w.gu)
    if tiers["prefill_weights"] == "nvfp4":
        w.p4 = quantize_nvfp4_prefill(lm.layers, w.qkv, w.gu, int(model.vlm.config.text_config.intermediate_size))
    if tiers["vision_weights"] == "fp8":
        w.v8 = quantize_fp8_vision(model.vlm.model.visual.blocks)
    return w
