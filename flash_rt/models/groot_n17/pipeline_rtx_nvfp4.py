"""GROOT N1.7 DiT action head on the skinny NVFP4 family (RTX SM120).

The 32-layer ``AlternateVLDiT`` of ``pipeline_thor.dit_forward`` with every
GEMM (fused self-attention QKV, cross-attention Q, output projection, FFN up
and down) as a block-scaled W4A4 skinny GEMM (``nvfp4_skinny_gemm``): each
CTA streams its weight slice before the programmatic-dependent-launch wait
and writes FP32 K-split partials. The consumers that sum those partials
finish the block step in the same launch and emit the next GEMM's NVFP4
operand directly, so no standalone quantize, bias, activation, residual or
copy launch remains:

    AdaLN(h) -> fp4                       (layer 0 only; later layers get it
                                           from the previous layer's consumer)
    QKV / Q GEMM -> bias, split into the attention Q | K | V slots
    attention (RTX FA2 backend)
    O GEMM (BF16 attention output quantized on load)
      -> bias + residual into h, pre-FF LayerNorm -> fp4
    FFN up GEMM -> bias + tanh-GELU -> fp4
    FFN down GEMM -> bias + residual into h, then the next layer's AdaLN
      -> fp4 (last layer: the output AdaLN -> BF16 for the output projection)

Activations use NVFP4 with per-16 dynamic block scales and global scale 1
(no calibration); weights carry a per-tensor global scale.
"""

from __future__ import annotations

D = 1536
FF = 6144
LAYERS = 32
EPS = 1e-5

# Tile configuration per (N, K) at M = 41 (RTX 5090, cold weights, measured).
SKINNY_CFG = {
    (3 * D, D): 3,     # fused self-attn QKV  128x512
    (D, D): 12,        # cross Q, O            32x256
    (FF, D): 6,        # FFN up               128x256, 8 warps
    (D, FF): 6,        # FFN down             128x256, 8 warps
}


def partial_floats(fvk, rows: int, cfg: dict | None = None) -> int:
    """FP32 partial scratch needed by the DiT GEMMs at ``rows`` rows."""
    cfg = cfg or SKINNY_CFG
    return max(k // fvk.nvfp4_skinny_config_k_chunk(c) * rows * n for (n, k), c in cfg.items())


def dit_forward_nvfp4(fvk, bufs: dict, weights: dict, dims: dict, *, attn,
                      stream: int = 0, pdl: bool = True, cfg: dict | None = None) -> None:
    """One denoising step of the DiT on NVFP4.

    Args:
        bufs: ``h`` (bf16 [Sa, D], residual stream, read and updated),
            ``xn`` / ``xn_sf`` (NVFP4 [Sa, D] block input), ``hid`` /
            ``hid_sf`` (NVFP4 [Sa, FF]), ``partials`` (fp32 scratch of
            ``partial_floats``), ``one`` (fp32 device scalar 1.0), ``hmod``
            (bf16 [Sa, D], output-AdaLN result for the output projection).
        weights: per-step pointer table — ``scale_msa`` / ``shift_msa``
            (32 bf16 [D] AdaLN vectors), ``out_scale`` / ``out_shift`` (output
            AdaLN), and the step-invariant NVFP4 sites ``qkv`` (16 self
            layers), ``q`` (16 cross layers), ``o``, ``ff_up``, ``ff_down``
            (32 layers), each entry ``(w_packed, sf_packed, w_gs, bias)``.
        dims: ``Sa``, ``Skv_text``, ``Skv_image``.
    """
    cfg = cfg or SKINNY_CFG
    Sa = int(dims["Sa"])
    Skv_text = int(dims["Skv_text"])
    Skv_image = int(dims["Skv_image"])
    h = int(bufs["h"])
    xn, xn_sf = int(bufs["xn"]), int(bufs["xn_sf"])
    hid, hid_sf = int(bufs["hid"]), int(bufs["hid_sf"])
    part = int(bufs["partials"])
    one = int(bufs["one"])
    hmod = int(bufs["hmod"])
    scale, shift = weights["scale_msa"], weights["shift_msa"]

    def _ck(rc: int, what: str, li: int) -> None:
        if rc != 0:
            raise RuntimeError(f"N1.7 NVFP4 DiT {what} layer {li} failed: cudaError {rc}")

    def _gemm(a, a_sf, site, n, k, li, what):
        w, sf, _, _ = site
        c = cfg[(n, k)]
        _ck(fvk.nvfp4_skinny_gemm(a, a_sf, w, sf, part, Sa, n, k, c, pdl, stream), what, li)
        return k // fvk.nvfp4_skinny_config_k_chunk(c)

    _ck(fvk.nvfp4_skinny_layer_norm(h, int(scale[0]), int(shift[0]), xn, xn_sf, one, 0,
                                    Sa, D, EPS, pdl, stream), "adaln", 0)
    for li in range(LAYERS):
        is_self = li % 2 == 1
        j = (li - 1) // 2 if is_self else li // 2
        slots = attn.get_slot_ptrs("dit_self" if is_self else "dit_cross", j)
        if is_self:
            site = weights["qkv"][j]
            splits = _gemm(xn, xn_sf, site, 3 * D, D, li, "qkv gemm")
            _ck(fvk.nvfp4_skinny_sum_bias_store(part, splits, one, site[2], site[3], slots["Q"], slots["K"],
                                                slots["V"], D, Sa, 3 * D, pdl, stream), "qkv store", li)
            attn.run("dit_self", j, q_seq=Sa, kv_seq=Sa, stream=stream)
        else:
            site = weights["q"][j]
            splits = _gemm(xn, xn_sf, site, D, D, li, "q gemm")
            _ck(fvk.nvfp4_skinny_sum_bias_store(part, splits, one, site[2], site[3], slots["Q"], 0, 0,
                                                D, Sa, D, pdl, stream), "q store", li)
            attn.run("dit_cross", j, q_seq=Sa, kv_seq=Skv_text if li % 4 == 0 else Skv_image, stream=stream)

        # O projection (attention output quantized on load) -> bias + residual
        # -> pre-FF LayerNorm (no affine) -> fp4.
        w, sf, gs, b = weights["o"][li]
        c = cfg[(D, D)]
        _ck(fvk.nvfp4_skinny_gemm_bf16_act(slots["O"], one, w, sf, part, Sa, D, D, c, pdl, stream), "o gemm", li)
        splits = D // fvk.nvfp4_skinny_config_k_chunk(c)
        _ck(fvk.nvfp4_skinny_sum_bias_residual_norm(part, splits, one, gs, b, h, 0, 0, xn, xn_sf, one, 0,
                                                    Sa, D, EPS, pdl, stream), "o residual norm", li)

        # FFN up -> bias + GELU -> fp4.
        site = weights["ff_up"][li]
        splits = _gemm(xn, xn_sf, site, FF, D, li, "ffn up gemm")
        _ck(fvk.nvfp4_skinny_sum_bias_gelu(part, splits, one, site[2], site[3], hid, hid_sf, one,
                                           Sa, FF, pdl, stream), "ffn gelu", li)

        # FFN down -> bias + residual -> next AdaLN (fp4) or output AdaLN (bf16).
        site = weights["ff_down"][li]
        splits = _gemm(hid, hid_sf, site, D, FF, li, "ffn down gemm")
        if li < LAYERS - 1:
            _ck(fvk.nvfp4_skinny_sum_bias_residual_norm(
                part, splits, one, site[2], site[3], h, int(scale[li + 1]), int(shift[li + 1]),
                xn, xn_sf, one, 0, Sa, D, EPS, pdl, stream), "ffn residual adaln", li)
        else:
            _ck(fvk.nvfp4_skinny_sum_bias_residual_norm(
                part, splits, one, site[2], site[3], h, int(weights["out_scale"]), int(weights["out_shift"]),
                0, 0, 0, hmod, Sa, D, EPS, pdl, stream), "ffn residual out-norm", li)
