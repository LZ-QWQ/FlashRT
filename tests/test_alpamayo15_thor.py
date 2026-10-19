"""Alpamayo 1.5 Thor: request contracts and kernel parity with the official op sequence."""

from __future__ import annotations

import importlib
import os
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers")

from flash_rt.models.alpamayo15.pipeline_thor import (  # noqa: E402
    ALPAMAYO15_KERNEL_SYMBOLS,
    Alpamayo15Thor,
    _Gemm,
    check_generation_config,
    expert_attention_spans,
    merge_attention,
    require_alpamayo15_kernels,
    resolve_preset,
    update_stop_state,
)
from flash_rt.models.alpamayo15.weights import (  # noqa: E402
    FP8_E4M3_MAX,
    fp8_fragment_layout,
    quantize_fp8_rows,
    quantize_int8_rows,
)

BF16 = torch.bfloat16


def test_required_symbols_fail_with_build_hint():
    with pytest.raises(RuntimeError, match=r"FLASHRT_ENABLE_ALPAMAYO15=ON") as exc:
        require_alpamayo15_kernels(SimpleNamespace())
    for name in ALPAMAYO15_KERNEL_SYMBOLS:
        assert name in str(exc.value)


def test_symbol_inventory_is_atomic():
    try:
        fvk = importlib.import_module("flash_rt.flash_rt_kernels")
    except Exception as exc:  # pragma: no cover - build/env dependent
        pytest.skip(f"flash_rt_kernels not importable: {exc}")
    found = {name for name in dir(fvk) if name.startswith("alpamayo15_")}
    gated = {name for name in ALPAMAYO15_KERNEL_SYMBOLS if name.startswith("alpamayo15_")}
    assert found in (set(), gated)
    if os.environ.get("FLASHRT_TEST_ALPAMAYO15_REQUIRED") == "1":
        assert found == gated
        assert all(hasattr(fvk, name) for name in ALPAMAYO15_KERNEL_SYMBOLS)


def _run_stop(seqs, max_new=6, traj=7, eos=frozenset({9})):
    """Feed per-step tokens (one list per step) and return the step after which generation ends."""
    n = len(seqs[0])
    traj_found, unfinished = [False] * n, [True] * n
    for step, last in enumerate(seqs, start=1):
        update_stop_state(list(last), traj_found, unfinished, step, traj_start_id=traj,
                          max_new_tokens=max_new, eos_ids=set(eos))
        if not any(unfinished):
            return step, unfinished
    return None, unfinished


def test_stop_criteria_match_official_sequence():
    # StopAfterEOS: one more token after <|traj_future_start|> (7), then stop.
    assert _run_stop([[1], [7], [3], [4]])[0] == 3
    # Generation-config EOS stops a row immediately; the budget bounds the loop.
    assert _run_stop([[1], [9]])[0] == 2
    assert _run_stop([[1]] * 6)[0] == 6
    # Batched: all rows must have produced 7 before the extra step; EOS rows finish alone.
    assert _run_stop([[7, 1], [2, 1], [3, 7], [4, 5], [5, 5]])[0] == 4
    step, unfinished = _run_stop([[9, 1], [0, 2], [0, 3]])
    assert step is None and unfinished == [False, True]


def test_expert_spans_match_official_offsets_and_mask():
    off_mod = pytest.importorskip("alpamayo1_5.models.alpamayo1_5")
    official = off_mod.Alpamayo1_5
    P, traj, pad = 5, 7, 0
    # rows: found early (later tokens masked), found at the next-to-last step, never found
    tokens = [[1, 7, 3, 4, 2], [1, 2, 3, 7, 4], [1, 2, 3, 4, 5]]
    offsets, visible = expert_attention_spans(tokens, P, traj)
    seq = torch.tensor([[9] * P + row for row in tokens])
    ref_off = official._find_eos_offset(seq, traj, torch.device("cpu"), warn=False)
    assert offsets == ref_off.tolist()
    cache_len = P + len(tokens[0]) - 1
    n_act = 4
    _, mask = official._build_expert_pos_ids_and_attn_mask(
        offset=ref_off, rope_deltas=torch.zeros(3, 1, dtype=torch.long), kv_cache_seq_len=cache_len,
        n_diffusion_tokens=n_act, b_star=3, device=torch.device("cpu"), prefix_mask=None)
    for b in range(3):  # visible cache rows of the official mask == our visible prefix
        seen = (mask[b, 0, 0, :cache_len] == 0).nonzero().squeeze(1).tolist()
        assert seen == list(range(visible[b]))
        assert (mask[b, 0, :, cache_len:] == 0).all()
    del pad


def test_merge_attention_equals_single_pass():
    torch.manual_seed(1)
    rows, heads, D, n1, n2 = 6, 4, 16, 9, 5
    q = torch.randn(rows, heads, D)
    k = torch.randn(n1 + n2, heads, D)
    v = torch.randn(n1 + n2, heads, D)

    def attend(kk, vv):
        s = torch.einsum("rhd,khd->hrk", q, kk) / D ** 0.5
        return torch.einsum("hrk,khd->rhd", s.softmax(-1), vv), torch.logsumexp(s, -1)

    full, _ = attend(k, v)
    o1, l1 = attend(k[:n1], v[:n1])
    o2, l2 = attend(k[n1:], v[n1:])
    merged = merge_attention(o1.to(BF16), l1, o2.to(BF16), l2)
    assert torch.allclose(merged.float(), full, atol=2e-2, rtol=2e-2)


def test_generation_config_guard():
    ok = SimpleNamespace(repetition_penalty=1.0, typical_p=1.0, top_k=50, min_length=0, min_new_tokens=None)
    check_generation_config(ok)
    with pytest.raises(ValueError, match="repetition_penalty"):
        check_generation_config(SimpleNamespace(repetition_penalty=1.1))
    with pytest.raises(ValueError, match="min_new_tokens"):
        check_generation_config(SimpleNamespace(min_new_tokens=4))


def test_resolver_entry_and_load_model_redirect():
    from flash_rt.hardware import _PIPELINE_MAP
    assert _PIPELINE_MAP[("alpamayo15", "torch", "thor")] == (
        "flash_rt.frontends.torch.alpamayo15_thor", "Alpamayo15TorchFrontendThor")
    import flash_rt
    with pytest.raises(NotImplementedError, match="Alpamayo15TorchFrontendThor"):
        flash_rt.load_model("unused", config="alpamayo15")


def test_fp8_row_quantization():
    w = torch.randn(5, 64) * torch.tensor([1e-3, 1.0, 30.0, 0.0, 2.0])[:, None]
    codes, scale = quantize_fp8_rows(w, chunk_rows=2)
    assert codes.dtype == torch.uint8 and scale.dtype == torch.float32
    assert scale[3] == 1.0 and not codes[3].view(torch.float8_e4m3fn).float().any()   # +-0 codes
    deq = codes.view(torch.float8_e4m3fn).float() * scale[:, None]
    nz = [0, 1, 2, 4]
    torch.testing.assert_close(deq[nz].abs().amax(1), w[nz].abs().amax(1), rtol=4e-7, atol=0)  # row max -> 448
    rel = ((deq - w).abs() / w.abs().clamp_min(1e-30))[nz]
    normal = w[nz].abs() >= scale[nz, None] * 2.0 ** -6
    assert float(rel[normal].max()) <= 2.0 ** -4 + 1e-6                     # 3 mantissa bits, RNE
    assert float(deq.abs().max() / scale.max()) <= FP8_E4M3_MAX
    # Parameters require grad: the quantized copies must not carry an autograd graph (it would pin FP32 temporaries).
    codes_g, scale_g = quantize_fp8_rows(torch.nn.Parameter(w.clone()))
    frag_g, _ = fp8_fragment_layout(codes_g, scale_g)
    assert all(t.grad_fn is None and not t.requires_grad for t in (codes_g, scale_g, frag_g))


def test_fp8_fragment_layout_matches_mma_a_fragments():
    N, K = 21, 64
    codes = torch.randint(0, 256, (N, K), dtype=torch.uint8)
    frag, scale = fp8_fragment_layout(codes, torch.arange(N, dtype=torch.float32))
    T = 2
    assert frag.numel() == T * 16 * K and scale.shape == (T * 16,) and not scale[N:].any()
    f = frag.view(T, K // 32, 32, 16)
    for t in range(T):
        for kc in range(K // 32):
            for lane in range(32):
                g, c = lane // 4, lane % 4
                for j in range(16):
                    kt, ri, e = j // 8, (j % 8) // 2, j % 2
                    row = 16 * t + g + 8 * (ri & 1)
                    col = 32 * kc + 16 * kt + 8 * (ri >> 1) + 2 * c + e
                    want = int(codes[row, col]) if row < N else 0
                    assert int(f[t, kc, lane, j]) == want


def _kernels():
    if not torch.cuda.is_available():
        pytest.skip("Alpamayo 1.5 kernel parity requires CUDA")
    try:
        import flash_rt.flash_rt_kernels as fvk
    except Exception as exc:  # pragma: no cover - build dependent
        pytest.skip(f"flash_rt_kernels is unavailable: {exc}")
    if not all(hasattr(fvk, n) for n in ALPAMAYO15_KERNEL_SYMBOLS):
        pytest.skip("built without -DFLASHRT_ENABLE_ALPAMAYO15=ON")
    return fvk


def _stream() -> int:
    return torch.cuda.current_stream().cuda_stream


def _rot(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _assert_bf16_close(actual: torch.Tensor, expected: torch.Tensor, max_frac: float = 0.01):
    """Equal up to one BF16 ulp on a small fraction of elements (FP32 reduction order)."""
    a, e = actual.float(), expected.float()
    ulp = torch.where(e == 0, torch.full_like(e, 2.0 ** -133), e.abs() * 2.0 ** -7)
    diff = (a - e).abs()
    assert bool((diff <= ulp * 1.01).all()), f"max diff {diff.max().item()}"
    assert float((diff > 0).float().mean()) <= max_frac


def _qwen3_head_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    n1 = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(BF16)
    return w * n1


def test_vision_rope_matches_official_rotary():
    fvk = _kernels()
    torch.manual_seed(3)
    S, H, D = 37, 16, 72
    qkv = (torch.randn(S, 3, H, D, device="cuda") * 0.5).to(BF16)
    angles = torch.randn(S, D // 2, device="cuda") * 3
    emb = torch.cat((angles, angles), -1)
    cos, sin = emb.cos().contiguous(), emb.sin().contiguous()
    ref = qkv.clone()
    for i in (0, 1):  # apply_rotary_pos_emb_vision: FP32 math, one BF16 rounding
        x = ref[:, i].float()
        ref[:, i] = (x * cos[:, None] + _rot(x) * sin[:, None]).to(BF16)
    out = qkv.clone()
    fvk.alpamayo15_vision_rope_qk_bf16(out.data_ptr(), cos.data_ptr(), sin.data_ptr(), S, H, D, _stream())
    torch.cuda.synchronize()
    assert torch.equal(out, ref)


def _fp4_module():
    _kernels()
    try:
        import flash_rt.flash_rt_fp4 as f4
    except Exception as exc:  # pragma: no cover - build dependent
        pytest.skip(f"flash_rt_fp4 is unavailable: {exc}")
    if not hasattr(f4, "alpamayo15_nvfp4_gemm_bf16"):
        pytest.skip("flash_rt_fp4 built without the Alpamayo 1.5 NVFP4 forms")
    return f4


def _nvfp4_emulate(x: torch.Tensor, per_row: bool = False):
    """The quantizer's arithmetic in torch: gs = amax / 2688 per tensor (or per row), block-16 scales
    e4m3(block_amax / 6 / gs) (round to nearest even), e2m1 codes with ties to even; returns the dequantized tensor
    (fp32) and gs (a float, or a [rows] tensor)."""
    x = x.float()
    if per_row:
        amax = x.abs().amax(dim=1)
        gs = torch.where(amax > 0, amax / 2688.0, torch.ones_like(amax))
        gs_b = gs.view(-1, 1)
    else:
        amax = float(x.abs().max())
        gs = gs_b = amax / 2688.0 if amax > 0 else 1.0
    blocks = x.view(x.shape[0], -1, 16)
    bs = (blocks.abs().amax(dim=2) / (6.0 * gs_b)).clamp(max=448.0).to(torch.float8_e4m3fn).float()    # [rows, nb]
    scaled = blocks / (bs * gs_b).unsqueeze(2).clamp_min(1e-30)
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x.device)
    mag = scaled.abs().clamp(max=6.0)
    dist = (mag.unsqueeze(-1) - grid).abs()
    idx = dist.argmin(-1)                                                       # nearest; the lower code on a tie
    up = (idx + 1).clamp(max=7)
    tie = (dist.gather(-1, up.unsqueeze(-1)).squeeze(-1) == dist.gather(-1, idx.unsqueeze(-1)).squeeze(-1)) & (up != idx)
    idx = torch.where(tie & (idx % 2 == 1), up, idx)                           # ties to the even code
    q = grid[idx] * torch.sign(scaled)
    deq = (q * (bs * gs_b).unsqueeze(2)).view_as(x)
    deq[(bs == 0).unsqueeze(2).expand_as(blocks).reshape(x.shape)] = 0.0
    return deq, gs


def _nvfp4_weight(f4, w, s):
    """A weight through the per-tensor quantizer (the engine's static path): packed, SFB layout, gs."""
    N, K = w.shape
    part = torch.empty(1024, device="cuda"); gs_dev = torch.empty(1, device="cuda")
    wp = torch.empty(N, K // 2, device="cuda", dtype=torch.uint8); wsf = torch.zeros(int(f4.alpamayo15_nvfp4_sf_bytes(N, K)), device="cuda", dtype=torch.uint8)
    n = f4.alpamayo15_nvfp4_absmax_bf16(w.data_ptr(), K, N, K, part.data_ptr(), 1024, s)
    assert n > 0
    assert f4.alpamayo15_nvfp4_quantize_rows_bf16(w.data_ptr(), K, N, K, part.data_ptr(), n, wp.data_ptr(), wsf.data_ptr(), 1, gs_dev.data_ptr(), 0, 0, 0, s) == 0
    torch.cuda.synchronize()
    return wp, wsf, gs_dev.clone()


@pytest.mark.parametrize("M,N,K,tile", [(3086, 4096, 4096, 2), (300, 2048, 12288, 2), (257, 1024, 1024, 1), (70, 2048, 4096, 0)])
def test_nvfp4_rowscale_quantizer_and_per_row_alpha_gemm(M, N, K, tile):
    """The per-row quantizer (the W4A4 tier's activation path) equals the torch emulation with per-row scales, writes
    the rows' alphas gs_row * gs_weight, and the GEMM's per-row alpha epilogue applies them: the product equals the
    emulated dequantized product within the kernel's midpoint rounding; exactly representable operands with per-row
    power-of-two factors come out exact up to the BF16 output rounding."""
    f4 = _fp4_module()
    torch.manual_seed(M + 2 * N)
    s = _stream()
    x = (torch.randn(M, K, device="cuda") * (torch.rand(M, 1, device="cuda") * 4 + 0.1)).to(BF16)   # rows of very different magnitude
    x[3] = 0
    w = (torch.randn(N, K, device="cuda") * 0.02).to(BF16)
    wp, wsf, gsw = _nvfp4_weight(f4, w, s)
    gsw2 = torch.stack([gsw[0], gsw[0] * 3])
    ld_a = (M + 127) // 128 * 128
    alpha = torch.zeros(2, ld_a, device="cuda"); gs_rows = torch.zeros(M, device="cuda")
    xp = torch.empty(M, K // 2, device="cuda", dtype=torch.uint8); xsf = torch.zeros(int(f4.alpamayo15_nvfp4_sf_bytes(M, K)), device="cuda", dtype=torch.uint8)
    assert f4.alpamayo15_nvfp4_quantize_rows_rowscale_bf16(x.data_ptr(), K, M, K, xp.data_ptr(), xsf.data_ptr(), gsw2.data_ptr(), 2,
                                                            alpha.data_ptr(), ld_a, gs_rows.data_ptr(), s) == 0
    torch.cuda.synchronize()
    xd, gx = _nvfp4_emulate(x, per_row=True); wd, gw = _nvfp4_emulate(w)
    assert torch.allclose(gs_rows, gx, rtol=1e-6) and float(gs_rows[3]) == 1.0
    assert torch.allclose(alpha[0, :M], gx * gw, rtol=1e-5) and torch.allclose(alpha[1, :M], gx * gw * 3, rtol=1e-5)
    ws = torch.empty(max(int(f4.alpamayo15_nvfp4_gemm_workspace(M, N, K, tile)), 16), device="cuda", dtype=torch.uint8)
    out = torch.empty(M, N, device="cuda", dtype=BF16)
    assert f4.alpamayo15_nvfp4_gemm_bf16(xp.data_ptr(), xsf.data_ptr(), wp.data_ptr(), wsf.data_ptr(), out.data_ptr(), M, N, K, tile,
                                         ws.data_ptr(), ws.numel(), s, N, alpha[0].data_ptr(), 1) == 0
    torch.cuda.synchronize()
    ref = xd.double() @ wd.double().t()
    err = (out.double() - ref).abs()
    assert float(err.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()) < 3e-2, "GEMM / layout / per-row alpha mismatch"
    assert bool((out[3] == 0).all())
    exact = x.float() @ w.float().t()
    rel = float((out.float() - exact).pow(2).mean().sqrt() / exact.pow(2).mean().sqrt())
    assert 0.05 < rel < 0.25, rel
    # per-row scales keep every row at the 4-bit level (a per-tensor scale would crush the small rows): the relative
    # error of the smallest-magnitude rows is no worse than that of the largest
    rr = (out.float() - exact).pow(2).mean(1).sqrt() / exact.pow(2).mean(1).sqrt().clamp_min(1e-20)
    mag = x.float().abs().amax(1)
    small, large = rr[mag < mag.median()], rr[mag >= mag.median()]
    assert float(small[torch.isfinite(small)].mean()) < 1.5 * float(large.mean())
    # the second consumer's alphas (3 x) scale the output by 3
    out3 = torch.empty_like(out)
    assert f4.alpamayo15_nvfp4_gemm_bf16(xp.data_ptr(), xsf.data_ptr(), wp.data_ptr(), wsf.data_ptr(), out3.data_ptr(), M, N, K, tile,
                                         ws.data_ptr(), ws.numel(), s, N, alpha[1].data_ptr(), 1) == 0
    torch.cuda.synchronize()
    assert bool(((out3.float() - 3 * out.float()).abs() <= 3 * out.float().abs() * 2 ** -7 + 1e-3).all())
    # exactly representable operands times per-row powers of two
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="cuda")
    def representable(rows, cols, row_pow):
        g = grid[torch.randint(0, 8, (rows, cols), device="cuda")] * torch.where(torch.rand(rows, cols, device="cuda") < 0.5, -1.0, 1.0) * 0.125
        g = g.view(rows, -1, 16)
        g[:, :, 0] = 0.75
        g = g * (2.0 ** (torch.arange(g.shape[1], device="cuda") % 5)).view(1, -1, 1)
        if row_pow:
            g = g * (2.0 ** (torch.arange(rows, device="cuda") % 4)).view(-1, 1, 1)
        return g.view(rows, cols).to(BF16)
    xg, wg = representable(M, K, True), representable(N, K, False)
    wp, wsf, gsw = _nvfp4_weight(f4, wg, s)
    assert f4.alpamayo15_nvfp4_quantize_rows_rowscale_bf16(xg.data_ptr(), K, M, K, xp.data_ptr(), xsf.data_ptr(), gsw.data_ptr(), 1,
                                                            alpha.data_ptr(), ld_a, 0, s) == 0
    assert f4.alpamayo15_nvfp4_gemm_bf16(xp.data_ptr(), xsf.data_ptr(), wp.data_ptr(), wsf.data_ptr(), out.data_ptr(), M, N, K, tile,
                                         ws.data_ptr(), ws.numel(), s, N, alpha[0].data_ptr(), 1) == 0
    torch.cuda.synchronize()
    exact = xg.float() @ wg.float().t()
    assert float((out != exact.to(BF16)).float().mean()) < 0.02 and bool(((out.float() - exact).abs() <= exact.abs() * 2 ** -7 + 2 ** -4).all())


@pytest.mark.parametrize("rows,dim,with_delta", [(300, 4096, True), (129, 1024, False), (64, 12288, True)])
def test_nvfp4_fused_producers_match_rowscale_quantizer(rows, dim, with_delta):
    """RMSNorm -> NVFP4 rows and SiLU-mul -> NVFP4 rows give the rows, scales, alphas and residual of the separate
    kernels followed by the per-row quantizer; the rowscale quantizer honours ld and rejects bad arguments."""
    f4 = _fp4_module()
    fvk = _kernels()
    torch.manual_seed(37)
    s = _stream()
    gsw = torch.tensor([0.0123, 0.0456], device="cuda")
    ld_a = (rows + 127) // 128 * 128
    nb = int(f4.alpamayo15_nvfp4_sf_bytes(rows, dim))
    mk = lambda: (torch.full((rows, dim // 2), 7, device="cuda", dtype=torch.uint8), torch.full((nb,), 9, device="cuda", dtype=torch.uint8),
                  torch.zeros(2, ld_a, device="cuda"))
    x = (torch.randn(rows, dim, device="cuda") * 1.5).to(BF16)
    delta = (torch.randn(rows, dim, device="cuda") * 0.3).to(BF16) if with_delta else None
    dp = delta.data_ptr() if with_delta else 0
    w = (torch.rand(dim, device="cuda") + 0.5).to(BF16)
    x_ref, x_fused = x.clone(), x.clone()
    h = torch.empty_like(x)
    fvk.alpamayo15_rms_bf16(x_ref.data_ptr(), dp, w.data_ptr(), h.data_ptr(), rows, dim, 1e-6, s)
    p_ref, sf_ref, a_ref = mk()
    gs_ref = torch.zeros(rows, device="cuda")
    assert f4.alpamayo15_nvfp4_quantize_rows_rowscale_bf16(h.data_ptr(), dim, rows, dim, p_ref.data_ptr(), sf_ref.data_ptr(), gsw.data_ptr(), 2,
                                                            a_ref.data_ptr(), ld_a, gs_ref.data_ptr(), s) == 0
    p, sf, a = mk()
    assert f4.alpamayo15_nvfp4_rms_rows_bf16(x_fused.data_ptr(), dp, w.data_ptr(), rows, dim, 1e-6, p.data_ptr(), sf.data_ptr(), gsw.data_ptr(), 2,
                                             a.data_ptr(), ld_a, s) == 0
    torch.cuda.synchronize()
    assert torch.equal(x_fused, x_ref)                                       # the residual update is element-wise
    # the fused producer accumulates the sum of squares over 4-channel vectors: the normalized row can differ in the
    # last BF16 bit where the scale rounding differs, so the row maxima / alphas within one BF16 ulp and a few codes off
    assert bool((torch.abs(a - a_ref) <= a_ref * 2 ** -7).all()) and bool((a[:, rows:] == 0).all())
    assert float((p != p_ref).float().mean()) < 0.03 and float((sf != sf_ref).float().mean()) < 0.03
    hd, _ = _nvfp4_emulate(h, per_row=True)

    def deq(pk, sfb, gs):
        """Dequantized rows and their block scales (per element) from the packed codes, the per-row gs and the scales
        read through the SFA layout: 128-row x 4-block atoms ((32, 4), 4) : ((16, 4), 1), K-blocks of 4 tiled first,
        then row blocks."""
        lo = (pk & 15).int(); hi = (pk >> 4).int()
        codes = torch.stack([lo, hi], -1).view(rows, dim)
        grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], device="cuda")
        vals = grid[codes]
        r = torch.arange(rows, device="cuda").view(-1, 1); bidx = torch.arange(dim // 16, device="cuda").view(1, -1)
        nbk = (dim // 16 + 3) // 4
        byte = (r // 128) * (512 * nbk) + (bidx // 4) * 512 + (r % 32) * 16 + ((r // 32) % 4) * 4 + (bidx % 4)
        bs = sfb[byte].view(torch.float8_e4m3fn).float()
        gs = gs.view(-1, 1)
        scale = (bs * gs).unsqueeze(2).expand(rows, dim // 16, 16).reshape(rows, dim)
        return (vals.view(rows, -1, 16) * (bs * gs).unsqueeze(2)).view(rows, dim), scale
    d_ref, sc_ref = deq(p_ref, sf_ref, gs_ref)
    # the separate path dequantized through the layout equals the emulation up to the kernel's midpoint rounding (a
    # few per mille of the codes one e2m1 step off; the largest step, 4 -> 6, is two block-scale units)
    assert float((d_ref != hd).float().mean()) < 0.01 and bool(((d_ref - hd).abs() <= sc_ref * 2.0 + 1e-6).all())
    # the fused path (its gs recovered from the alphas, so compared with a tolerance): at most one e2m1 step (plus one
    # e4m3 step of the block scale) away from the separate path, and a few per cent of the codes different
    d_f, _ = deq(p, sf, a[0, :rows] / gsw[0])
    assert bool(((d_f - d_ref).abs() <= sc_ref * 2.5 + 1e-6).all())
    assert float(((d_f - d_ref).abs() > sc_ref * 0.01 + 1e-9).float().mean()) < 0.03
    # SiLU-mul: identical inputs and arithmetic -> identical rows, scales and alphas
    F = dim
    g = (torch.randn(rows, F, device="cuda") * 2).to(BF16); u = (torch.randn(rows, F, device="cuda") * 2).to(BF16)
    act = torch.empty_like(g)
    fvk.alpamayo15_silu_mul_split_bf16(g.data_ptr(), u.data_ptr(), F, act.data_ptr(), rows, F, s)
    assert f4.alpamayo15_nvfp4_quantize_rows_rowscale_bf16(act.data_ptr(), F, rows, F, p_ref.data_ptr(), sf_ref.data_ptr(), gsw.data_ptr(), 1,
                                                            a_ref.data_ptr(), ld_a, 0, s) == 0
    assert f4.alpamayo15_nvfp4_silu_mul_rows_bf16(g.data_ptr(), u.data_ptr(), F, rows, F, p.data_ptr(), sf.data_ptr(), gsw.data_ptr(), 1,
                                                  a.data_ptr(), ld_a, s) == 0
    torch.cuda.synchronize()
    assert torch.equal(p, p_ref) and torch.equal(sf, sf_ref) and torch.equal(a[0], a_ref[0])
    # the gate | up form of the engine (rows 2F apart) equals the split form
    gu = torch.cat([g, u], 1)
    assert f4.alpamayo15_nvfp4_silu_mul_rows_bf16(gu.data_ptr(), gu[:, F:].data_ptr(), 2 * F, rows, F, p.data_ptr(), sf.data_ptr(), gsw.data_ptr(), 1,
                                                  a.data_ptr(), ld_a, s) == 0
    torch.cuda.synchronize()
    assert torch.equal(p, p_ref) and torch.equal(sf, sf_ref)
    # a column block of a wider tensor (ld > K) quantizes the same rows
    wide = torch.cat([act, torch.randn(rows, 64, device="cuda").to(BF16)], 1)
    assert f4.alpamayo15_nvfp4_quantize_rows_rowscale_bf16(wide.data_ptr(), F + 64, rows, F, p.data_ptr(), sf.data_ptr(), gsw.data_ptr(), 1,
                                                            a.data_ptr(), ld_a, 0, s) == 0
    torch.cuda.synchronize()
    assert torch.equal(p, p_ref) and torch.equal(sf, sf_ref)
    # rejections: a row beyond the shared-memory bound, alphas without room for every row, an unaligned per-row alpha
    assert f4.alpamayo15_nvfp4_quantize_rows_rowscale_bf16(act.data_ptr(), F, rows, 16384 + 16, p.data_ptr(), sf.data_ptr(), 0, 0, 0, 0, 0, s) == -2
    assert f4.alpamayo15_nvfp4_rms_rows_bf16(x_fused.data_ptr(), 0, w.data_ptr(), rows, dim, 1e-6, p.data_ptr(), sf.data_ptr(), gsw.data_ptr(), 2,
                                             a.data_ptr(), rows - 1, s) == -1
    ws = torch.empty(16, device="cuda", dtype=torch.uint8)
    assert f4.alpamayo15_nvfp4_gemm_bf16(p.data_ptr(), sf.data_ptr(), p.data_ptr(), sf.data_ptr(), act.data_ptr(), rows, 64, dim, 2,
                                         ws.data_ptr(), ws.numel(), s, 0, a.data_ptr() + 4, 1) == 3


@pytest.mark.parametrize("M,N,K,tile,ldd_extra", [(3086, 6144, 4096, 2, 0), (300, 4096, 12288, 2, 8), (257, 1024, 1024, 1, 0),
                                                  (64, 2048, 4096, 0, 16)])
def test_nvfp4_quantize_and_gemm_match_emulation(M, N, K, tile, ldd_extra):
    """Activations and weights through the NVFP4 quantizer (per-tensor + block-16 scales) and the 2-SM / 1-SM GEMM
    forms equal the torch emulation of the same quantization within BF16 output rounding; the W4A4 error against the
    exact product stays at the 4-bit level."""
    f4 = _fp4_module()
    torch.manual_seed(M + N)
    s = _stream()
    x = (torch.randn(M, K, device="cuda") * 1.5).to(BF16)
    w = (torch.randn(N, K, device="cuda") * 0.02).to(BF16)
    part = torch.empty(1024, device="cuda"); gs_dev = torch.empty(1, device="cuda")
    xp = torch.empty(M, K // 2, device="cuda", dtype=torch.uint8); xsf = torch.zeros(int(f4.alpamayo15_nvfp4_sf_bytes(M, K)), device="cuda", dtype=torch.uint8)
    wp = torch.empty(N, K // 2, device="cuda", dtype=torch.uint8); wsf = torch.zeros(int(f4.alpamayo15_nvfp4_sf_bytes(N, K)), device="cuda", dtype=torch.uint8)
    n = f4.alpamayo15_nvfp4_absmax_bf16(w.data_ptr(), K, N, K, part.data_ptr(), 1024, s)
    assert n > 0
    assert f4.alpamayo15_nvfp4_quantize_rows_bf16(w.data_ptr(), K, N, K, part.data_ptr(), n, wp.data_ptr(), wsf.data_ptr(), 1, gs_dev.data_ptr(), 0, 0, 0, s) == 0
    torch.cuda.synchronize()
    gsw = gs_dev.clone()
    alpha = torch.zeros(2, device="cuda"); gsw2 = torch.stack([gsw[0], gsw[0] * 3])
    n = f4.alpamayo15_nvfp4_absmax_bf16(x.data_ptr(), K, M, K, part.data_ptr(), 1024, s)
    assert f4.alpamayo15_nvfp4_quantize_rows_bf16(x.data_ptr(), K, M, K, part.data_ptr(), n, xp.data_ptr(), xsf.data_ptr(), 0, gs_dev.data_ptr(),
                                                   gsw2.data_ptr(), 2, alpha.data_ptr(), s) == 0
    torch.cuda.synchronize()
    xd, gx = _nvfp4_emulate(x); wd, gw = _nvfp4_emulate(w)
    assert abs(float(gs_dev) - gx) <= 1e-6 * gx and abs(float(gsw) - gw) <= 1e-6 * gw
    assert torch.allclose(alpha, torch.tensor([gx * gw, gx * gw * 3], device="cuda"), rtol=1e-5)
    ws = torch.empty(max(int(f4.alpamayo15_nvfp4_gemm_workspace(M, N, K, tile)), 16), device="cuda", dtype=torch.uint8)
    out = torch.full((M, N + ldd_extra), 7.0, device="cuda", dtype=BF16)
    assert f4.alpamayo15_nvfp4_gemm_bf16(xp.data_ptr(), xsf.data_ptr(), wp.data_ptr(), wsf.data_ptr(), out.data_ptr(), M, N, K, tile,
                                         ws.data_ptr(), ws.numel(), s, N + ldd_extra, alpha.data_ptr()) == 0
    torch.cuda.synchronize()
    ref = xd.double() @ wd.double().t()                                          # the emulated dequantized product
    err = (out[:, :N].double() - ref).abs()
    # the kernel scales by a rounded reciprocal, so exact midpoints (about 0.1% of the elements) can round the other
    # way than the emulation's division: a few per-mille of RMS, not the tens of per cent of a layout or scale error
    assert float(err.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()) < 3e-2, "GEMM / layout / scale mismatch"
    if ldd_extra:
        assert torch.equal(out[:, N:], torch.full((M, ldd_extra), 7.0, device="cuda", dtype=BF16))
    exact = x.float() @ w.float().t()
    rel = float((out[:, :N].float() - exact).pow(2).mean().sqrt() / exact.pow(2).mean().sqrt())
    assert 0.05 < rel < 0.25, rel                                                # the 4-bit level (FP8 W8A8: ~0.04)
    # exactly representable operands (e2m1 grid values times block scales that are powers of two): the GEMM equals
    # the exact product up to the BF16 output rounding, which pins the scale-to-block mapping of both layouts
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="cuda")
    def representable(rows, cols):
        g = grid[torch.randint(0, 8, (rows, cols), device="cuda")] * torch.where(torch.rand(rows, cols, device="cuda") < 0.5, -1.0, 1.0) * 0.125
        g = g.view(rows, -1, 16)
        g[:, :, 0] = 0.75
        return (g * (2.0 ** (torch.arange(g.shape[1], device="cuda") % 5)).view(1, -1, 1)).view(rows, cols).to(BF16)
    xg, wg = representable(M, K), representable(N, K)
    for src, pk, sfb_, flag in ((xg, xp, xsf, 0), (wg, wp, wsf, 1)):
        n = f4.alpamayo15_nvfp4_absmax_bf16(src.data_ptr(), K, src.shape[0], K, part.data_ptr(), 1024, s)
        assert f4.alpamayo15_nvfp4_quantize_rows_bf16(src.data_ptr(), K, src.shape[0], K, part.data_ptr(), n, pk.data_ptr(), sfb_.data_ptr(), flag,
                                                       gs_dev.data_ptr(), 0, 0, 0, s) == 0
        torch.cuda.synchronize()
        alpha[flag] = gs_dev[0]
    alpha[0] = alpha[0] * alpha[1]
    assert f4.alpamayo15_nvfp4_gemm_bf16(xp.data_ptr(), xsf.data_ptr(), wp.data_ptr(), wsf.data_ptr(), out.data_ptr(), M, N, K, tile,
                                         ws.data_ptr(), ws.numel(), s, N + ldd_extra, alpha.data_ptr()) == 0
    torch.cuda.synchronize()
    exact = xg.float() @ wg.float().t()
    assert float((out[:, :N] != exact.to(BF16)).float().mean()) < 0.02 and bool(((out[:, :N].float() - exact).abs() <= exact.abs() * 2 ** -7 + 2 ** -4).all())


def test_w4a8_grouped_gemm_rejects_unsupported_arguments():
    fvk = _kernels()
    K, N = 1024, 256
    pack = torch.zeros(N // 16, K // 128, 1088, device="cuda", dtype=torch.uint8)
    xq = torch.zeros(8, K, device="cuda", dtype=torch.int8)
    a_s = torch.zeros(1, device="cuda")
    out = torch.zeros(1, N, device="cuda", dtype=BF16)
    call = lambda **kw: fvk.alpamayo15_w4a8_gemm_grouped([xq.data_ptr()], [a_s.data_ptr()], pack.data_ptr(), [out.data_ptr()],
                                                         kw.get("M", 1), [kw.get("N", N)], kw.get("K", K), kw.get("ld", N),
                                                         kw.get("prefetch", 2), kw.get("warps", 4), _stream(), kw.get("ksplit", 0))
    call()
    for kw in ({"M": 17}, {"N": 200}, {"K": 1000}, {"ld": N - 2}, {"prefetch": 3}, {"warps": 2}, {"ksplit": 3}, {"ksplit": 8},
               {"ksplit": 2, "prefetch": 4, "K": 512}):   # 512 / 128 / 2 groups per warp is not a multiple of 4
        with pytest.raises(Exception, match="alpamayo15_w4a8_gemm_grouped"):
            call(**kw)
    torch.cuda.synchronize()


def test_vision_tower_cutlass_and_graph_match_official_module():
    """A small Qwen3-VL tower (two blocks, one DeepStack merger): the CUTLASS path, eager and as the captured
    graph, against the torch path of the same weights."""
    fvk = _kernels()
    from flash_rt.hardware.thor import fa4_backend
    fa4 = fa4_backend.fa4_fwd()
    if fa4 is None:
        pytest.skip(f"FA4 unavailable: {fa4_backend.status()}")
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    torch.manual_seed(5)
    cfg = Qwen3VLVisionConfig(depth=2, hidden_size=64, intermediate_size=128, num_heads=1, patch_size=4,
                              spatial_merge_size=2, temporal_patch_size=2, out_hidden_size=48,
                              num_position_embeddings=16, deepstack_visual_indexes=[0])
    V = Qwen3VLVisionModel(cfg).eval()
    with torch.no_grad():
        for p in V.parameters():
            p.normal_(0.0, 0.2)
    V = V.to("cuda", BF16)
    grid = torch.tensor([[1, 4, 4], [1, 4, 4]], device="cuda")
    S = int((grid[:, 0] * grid[:, 1] * grid[:, 2]).sum())
    px = torch.randn(S, 3 * 2 * 4 * 4, device="cuda")
    eng = SimpleNamespace(vlm=SimpleNamespace(model=SimpleNamespace(visual=V)), _fvk=fvk, _fa4=fa4, _gemm=_Gemm(fvk),
                          _vis_tables={}, _vis_graphs={}, spatial_merge=2, hidden=48)
    for name in ("_vision_tables", "_vision_cutlass", "_merger", "_check_vision_modules", "_encode", "_encode_groups"):
        setattr(eng, name, getattr(Alpamayo15Thor, name).__get__(eng))
    eng._check_vision_modules()
    with torch.no_grad(), torch.autocast("cuda", dtype=BF16):
        ref_img, ref_deep = V(px.to(BF16), grid_thw=grid)                        # the official module
        img, deep = Alpamayo15Thor._vision_cutlass(eng, px, grid)
        g_img, g_deep = Alpamayo15Thor._encode(eng, px, grid)
        g_img, g_deep = g_img.clone(), g_deep[0].clone()
        px2 = torch.randn_like(px)
        g2_img, g2_deep = Alpamayo15Thor._encode(eng, px2, grid)
        g2_img, g2_deep = g2_img.clone(), g2_deep[0].clone()
        ref2_img, ref2_deep = Alpamayo15Thor._vision_cutlass(eng, px2, grid)
    torch.cuda.synchronize()

    def close(a, e):
        a, e = a.float(), e.float()
        cos = torch.nn.functional.cosine_similarity(a.flatten(), e.flatten(), dim=0)
        assert float(cos) > 0.999, float(cos)
        assert float((a - e).abs().mean()) < 2e-2 * float(e.abs().mean())
    assert img.shape == ref_img.shape == (S // 4, 48) and len(deep) == len(ref_deep) == 1
    close(img, ref_img)
    close(deep[0], ref_deep[0])
    assert torch.equal(g_img, img) and torch.equal(g_deep, deep[0])              # the graph replays the eager path
    assert torch.equal(g2_img, ref2_img) and torch.equal(g2_deep, ref2_deep[0])  # new pixels through the same graph
    assert len(eng._vis_graphs) == 1
@pytest.mark.parametrize("rows,dim,with_delta", [(3086, 4096, True), (37, 4096, False), (300, 12288, True)])
def test_fused_fp8_row_producers_match_separate_kernels(rows, dim, with_delta):
    """RMSNorm -> FP8 rows and SiLU-mul -> FP8 rows give the codes, scales and residual of the separate kernels."""
    fvk = _kernels()
    torch.manual_seed(31)
    s = _stream()
    x = (torch.randn(rows, dim, device="cuda") * 1.5).to(BF16)
    delta = (torch.randn(rows, dim, device="cuda") * 0.3).to(BF16) if with_delta else None
    w = (torch.rand(dim, device="cuda") + 0.5).to(BF16)
    x_ref, x_fused = x.clone(), x.clone()
    h = torch.empty_like(x)
    fvk.alpamayo15_rms_bf16(x_ref.data_ptr(), delta.data_ptr() if with_delta else 0, w.data_ptr(), h.data_ptr(), rows, dim, 1e-6, s)
    q_ref = torch.empty(rows, dim, device="cuda", dtype=torch.uint8); s_ref = torch.empty(rows, device="cuda")
    fvk.alpamayo15_quantize_fp8_rows_bf16(h.data_ptr(), dim, q_ref.data_ptr(), s_ref.data_ptr(), rows, dim, s)
    q = torch.full((rows, dim), 7, device="cuda", dtype=torch.uint8); sc = torch.zeros(rows, device="cuda")
    fvk.alpamayo15_rms_fp8_rows_bf16(x_fused.data_ptr(), delta.data_ptr() if with_delta else 0, w.data_ptr(), q.data_ptr(),
                                     sc.data_ptr(), rows, dim, 1e-6, s)
    torch.cuda.synchronize()
    assert torch.equal(x_fused, x_ref)                                       # the residual update is element-wise
    # the fused producer accumulates the sum of squares over 4-channel vectors: scale within one FP32 ulp, codes equal
    # or off by one where the scale rounding differs
    assert bool((torch.abs(sc - s_ref) <= s_ref * 2 ** -22).all())
    codes = lambda t: t.view(torch.float8_e4m3fn).float()
    assert float((q != q_ref).float().mean()) < 0.02
    assert bool((torch.abs(codes(q) * sc[:, None] - codes(q_ref) * s_ref[:, None]) <= torch.abs(codes(q_ref) * s_ref[:, None]) * 2 ** -3 + s_ref[:, None] * 0.5).all())
    F = dim
    g = (torch.randn(rows, F, device="cuda") * 2).to(BF16); u = (torch.randn(rows, F, device="cuda") * 2).to(BF16)
    act = torch.empty_like(g)
    fvk.alpamayo15_silu_mul_split_bf16(g.data_ptr(), u.data_ptr(), F, act.data_ptr(), rows, F, s)
    fvk.alpamayo15_quantize_fp8_rows_bf16(act.data_ptr(), F, q_ref.data_ptr(), s_ref.data_ptr(), rows, F, s)
    fvk.alpamayo15_silu_mul_fp8_rows_bf16(g.data_ptr(), u.data_ptr(), F, q.data_ptr(), sc.data_ptr(), rows, F, s)
    torch.cuda.synchronize()
    assert torch.equal(sc, s_ref) and torch.equal(q, q_ref)
    # fp32-residual (expert) variant
    res_ref = torch.randn(rows, dim, device="cuda") * 1.5
    res_fused = res_ref.clone()
    fvk.alpamayo15_res_rms_f32_bf16(res_ref.data_ptr(), delta.data_ptr() if with_delta else 0, w.data_ptr(), h.data_ptr(), rows, dim, 1e-6, s)
    fvk.alpamayo15_quantize_fp8_rows_bf16(h.data_ptr(), dim, q_ref.data_ptr(), s_ref.data_ptr(), rows, dim, s)
    fvk.alpamayo15_res_rms_f32_fp8_rows_bf16(res_fused.data_ptr(), delta.data_ptr() if with_delta else 0, w.data_ptr(), q.data_ptr(),
                                             sc.data_ptr(), rows, dim, 1e-6, s)
    torch.cuda.synchronize()
    assert torch.equal(res_fused, res_ref) and torch.equal(sc, s_ref) and torch.equal(q, q_ref)


def test_marlin_w4a8_layout_roundtrip():
    """The Marlin index maps invert the forward tile permutation; the few-row operand packing keeps every nibble."""
    from flash_rt.models.alpamayo15.marlin_w4a8 import (_permute_weights_a8, _scale_perms, _weight_perm_a8, marlin_w4a8_unpack,
                                                        pack_w4a8_small)
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(41)
    K, N = 1024, 256
    G = K // 128
    w = torch.randint(0, 16, (K, N), dtype=torch.int32)
    z = torch.randint(0, 16, (G, N), dtype=torch.int32)
    s_int = torch.randint(0, 4097, (G, N), dtype=torch.int32)
    # forward Marlin packing (vLLM's layout): permuted nibbles into int32 words, permuted scales / zero points
    pw = _permute_weights_a8(w, K, N, _weight_perm_a8())                                 # [K/16, 16 N]
    qweight = torch.zeros(K // 16, N * 2, dtype=torch.int32)
    for i in range(8):
        qweight |= pw.reshape(K // 16, N * 2, 8)[..., i] << (4 * i)
    scale_perm, scale_perm_single = _scale_perms()
    scales = s_int.reshape(-1, len(scale_perm_single))[:, torch.tensor(scale_perm_single)].reshape(G, N).to(torch.int16).view(torch.float16)
    zp = z.reshape(-1, len(scale_perm))[:, torch.tensor(scale_perm)].reshape(G, N)
    qzeros = torch.zeros(G, N // 8, dtype=torch.int32)
    for i in range(8):
        qzeros |= zp.reshape(G, N // 8, 8)[..., i] << (4 * i)
    w_u, z_u, s_u = marlin_w4a8_unpack(qweight, scales, qzeros, K, N)
    assert torch.equal(w_u.cpu().int(), w) and torch.equal(z_u.cpu().int(), z) and torch.equal(s_u.cpu(), s_int)
    packed = pack_w4a8_small(w_u, K, N)
    assert packed.shape == (N // 16, K // 64, 32, 16)
    lo, hi = packed & 0xF, packed >> 4
    assert torch.equal(torch.sort(torch.cat([lo.reshape(-1), hi.reshape(-1)]).int())[0], torch.sort(w.reshape(-1).to(packed.device))[0])


@pytest.mark.parametrize("M,N,K,act,res", [(2880, 3456, 1152, 0, False), (2880, 1152, 1152, 0, True), (2880, 4304, 1152, 1, False),
                                           (2880, 1152, 4304, 0, True), (300, 256, 64, 2, False)])
def test_fp8_gemm_with_bias_activation_residual(M, N, K, act, res):
    """D = act((A8 @ W8.T) * a_s * w_s + bias) (+ C) on the 2-SM FP8 GEMM against the FP32 reference."""
    fvk = _kernels()
    torch.manual_seed(27)
    x = (torch.randn(M, K, device="cuda") * 2).to(BF16)
    w = (torch.randn(N, K, device="cuda") * 0.05).to(BF16)
    bias = (torch.randn(N, device="cuda") * 0.1).to(BF16)
    C = torch.randn(M, N, device="cuda").to(BF16) if res else None
    q8, a_s = quantize_fp8_rows(x)
    w8, w_s = quantize_fp8_rows(w)
    D = torch.empty(M, N, device="cuda", dtype=BF16)
    _Gemm(fvk).fp8_act(q8, a_s, w8, w_s, D, bias=bias, C=C, act=act)
    torch.cuda.synchronize()
    ref = (q8.view(torch.float8_e4m3fn).float() @ w8.view(torch.float8_e4m3fn).float().t()) * a_s[:, None] * w_s[None, :] + bias.float()
    if act == 1:
        ref = torch.nn.functional.gelu(ref, approximate="tanh")
    elif act == 2:
        ref = torch.nn.functional.gelu(ref)
    if C is not None:
        ref = ref + C.float()
    err = (D.float() - ref).abs()
    tol = ref.abs() * 2 ** -7 + 2 ** -6 * float(ref.abs().mean()) + 2 ** -6
    assert bool((err <= tol).all()), float((err / tol).max())
    if res:                           # column blocks of a wider output
        D2 = torch.full((M, N), float("nan"), device="cuda", dtype=BF16)
        half = N // 2
        _Gemm(fvk).fp8_act(q8, a_s, w8[:half], w_s[:half], D2[:, :half], bias=bias[:half], C=C[:, :half], act=act)
        _Gemm(fvk).fp8_act(q8, a_s, w8[half:], w_s[half:], D2[:, half:], bias=bias[half:], C=C[:, half:], act=act)
        torch.cuda.synchronize()
        assert torch.equal(D2, D)


@pytest.mark.parametrize("rows,dim", [(2880, 1152), (37, 1152), (64, 4608)])
def test_layernorm_fp8_rows_matches_separate_kernels(rows, dim):
    fvk = _kernels()
    torch.manual_seed(28)
    s = _stream()
    x = (torch.randn(rows, dim, device="cuda") * 2 + 0.3).to(BF16)
    w = (torch.rand(dim, device="cuda") + 0.5).to(BF16)
    b = (torch.randn(dim, device="cuda") * 0.1).to(BF16)
    h = torch.empty_like(x)
    fvk.alpamayo15_layernorm_bf16(x.data_ptr(), w.data_ptr(), b.data_ptr(), h.data_ptr(), rows, dim, 1e-6, s)
    q_ref = torch.empty(rows, dim, device="cuda", dtype=torch.uint8); s_ref = torch.empty(rows, device="cuda")
    fvk.alpamayo15_quantize_fp8_rows_bf16(h.data_ptr(), dim, q_ref.data_ptr(), s_ref.data_ptr(), rows, dim, s)
    q = torch.full((rows, dim), 7, device="cuda", dtype=torch.uint8); sc = torch.zeros(rows, device="cuda")
    fvk.alpamayo15_layernorm_fp8_rows_bf16(x.data_ptr(), w.data_ptr(), b.data_ptr(), q.data_ptr(), sc.data_ptr(), rows, dim, 1e-6, s)
    torch.cuda.synchronize()
    assert torch.equal(sc, s_ref) and torch.equal(q, q_ref)


# ---------------------------------------------------------------------------------------------------------------------
# request-level kernels (gathers, rotary tables, the sampler, the action expert's projections, the trajectory)
# ---------------------------------------------------------------------------------------------------------------------
def test_request_gathers_scatters_and_casts_match_torch():
    fvk = _kernels()
    torch.manual_seed(5)
    s = _stream()
    table = torch.randn(300, 128, device="cuda").to(BF16)
    idx = torch.randint(0, 300, (17,), device="cuda")
    out = torch.empty(17, 128, device="cuda", dtype=BF16)
    fvk.alpamayo15_gather_rows_bf16(table.data_ptr(), 128, idx.data_ptr(), out.data_ptr(), 128, 17, 128, s)
    base = torch.tensor([3], device="cuda")
    out2 = torch.empty(17, 128, device="cuda", dtype=BF16)
    fvk.alpamayo15_gather_rows_bytes(table.data_ptr(), 256, (idx - 3).clamp_min(0).data_ptr(), base.data_ptr(), out2.data_ptr(), 256, 17, 256, s)
    dst = torch.randn(300, 128, device="cuda").to(BF16); dst_ref = dst.clone()
    src = torch.randn(17, 128, device="cuda").to(BF16)
    uniq = torch.unique(idx)
    fvk.alpamayo15_scatter_rows_bf16(dst.data_ptr(), 128, uniq.data_ptr(), src[:uniq.numel()].data_ptr(), 128, uniq.numel(), 128, 0, s)
    dst_ref.index_copy_(0, uniq, src[:uniq.numel()])
    add = dst.clone(); add_ref = dst_ref.clone()
    fvk.alpamayo15_scatter_rows_bf16(add.data_ptr(), 128, uniq.data_ptr(), src[:uniq.numel()].data_ptr(), 128, uniq.numel(), 128, 1, s)
    add_ref[uniq] = add_ref[uniq] + src[:uniq.numel()]
    f32 = torch.randn(1000, 64, device="cuda"); c16 = torch.empty(1000, 64, device="cuda", dtype=BF16)
    fvk.alpamayo15_cast_f32_bf16(f32.data_ptr(), c16.data_ptr(), f32.numel(), s)
    h16 = torch.randn(1000, 64, device="cuda").half(); c16b = torch.empty_like(c16)
    fvk.alpamayo15_cast_f16_bf16(h16.data_ptr(), c16b.data_ptr(), h16.numel(), s)
    torch.cuda.synchronize()
    assert torch.equal(out, table[idx]) and torch.equal(out2, table[(idx - 3).clamp_min(0) + 3])
    assert torch.equal(dst, dst_ref) and torch.equal(add, add_ref)
    assert torch.equal(c16, f32.to(BF16)) and torch.equal(c16b, h16.to(BF16))


def test_mrope_tables_match_qwen3vl_rotary():
    """The table kernel equals the Qwen3-VL text rotary module's arithmetic (fp32 products of inv_freq and the
    positions, interleaved sections, cos / sin, the BF16 cast) for 3-D positions, and its fp32 form for the expert.
    The reference takes the exact fp32 product: with NVIDIA's Jetson builds of torch, which default to TF32 matmuls,
    the module's (3, 1, D/2, 1) @ (3, 1, 1, P) product rounds positions above 2048 to even values, an artifact of
    that default and not of the module. cos / sin may differ from torch's by one fp32 ulp (libdevice versions)."""
    fvk = _kernels()
    torch.manual_seed(6)
    s = _stream()
    D, P = 128, 97
    inv_freq = 1.0 / (5e6 ** (torch.arange(0, D, 2, device="cuda").float() / D))
    pos3 = torch.randint(0, 5000, (3, P), device="cuda", dtype=torch.int32)
    sec = [24, 20, 20]
    freqs = (inv_freq[None, :] * pos3.float()[:, :, None])             # [3, P, D/2], exact fp32
    ft = freqs[0].clone()
    for dim, off in ((1, 1), (2, 2)):
        ft[..., off:sec[dim] * 3:3] = freqs[dim, ..., off:sec[dim] * 3:3]
    emb = torch.cat((ft, ft), dim=-1)
    for f32 in (0, 1):
        dt = torch.float32 if f32 else BF16
        cos = torch.empty(P, D, device="cuda", dtype=dt); sin = torch.empty(P, D, device="cuda", dtype=dt)
        fvk.alpamayo15_mrope_tables(pos3.data_ptr(), inv_freq.contiguous().data_ptr(), sec[1], sec[2], 1.0, cos.data_ptr(), sin.data_ptr(), P, D, f32, s)
        torch.cuda.synchronize()
        for got, ref in ((cos, emb.cos()), (sin, emb.sin())):
            if f32:
                assert (got - ref).abs().max() <= 2 ** -23, (got - ref).abs().max()           # one ulp at |x| <= 1
            else:
                mism = got != ref.to(BF16)
                assert (got.float() - ref).abs().max() <= 2 ** -7, (got.float() - ref).abs().max()   # one bf16 ulp
                assert mism.float().mean() < 0.02, mism.float().mean()


def _hf_top_p_filter(scores: torch.Tensor, top_p: float, mask_lo: int, mask_n: int, temperature: float):
    scores = scores.float().clone()
    scores[:, mask_lo:mask_lo + mask_n] = float("-inf")
    scores = scores / temperature
    sorted_logits, sorted_indices = torch.sort(scores, descending=False)
    cum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    remove = cum <= (1 - top_p)
    remove[..., -1:] = 0
    remove = remove.scatter(1, sorted_indices, remove)
    return scores.masked_fill(remove, float("-inf")).softmax(dim=-1)


def _fd_top_p_filter(scores: torch.Tensor, top_p: float, mask_lo: int, mask_n: int, temperature: float):
    scores = scores.float().clone()
    scores[:, mask_lo:mask_lo + mask_n] = float("-inf")
    probs = torch.softmax(scores / temperature, dim=-1)
    sp, si = probs.sort(dim=-1, descending=True)
    outside = (sp.cumsum(dim=-1) - sp) > top_p
    sp[outside] = 0.0
    probs = torch.zeros_like(probs).scatter_(-1, si, sp)
    return probs / probs.sum(-1, keepdim=True)


@pytest.mark.parametrize("rule", [0, 1])
def test_sampler_matches_the_decoding_rules(rule):
    """Greedy = masked argmax without a draw; with temperature the kernel returns torch.multinomial's token on the
    filtered probabilities of the official rule (0, transformers' top-p) or FlashDrive's (1) from the same generator
    state, and advances the offset as torch does; the bookkeeping advances the counters."""
    fvk = _kernels()
    from flash_rt.models.alpamayo15.pipeline_thor import torch_philox_blocks, torch_philox_increment, torch_rng_state
    s = _stream()
    gen = torch.cuda.default_generators[torch.cuda.current_device()]
    blocks = torch_philox_blocks()
    V, lo, n_mask = 151936, 100, 50
    scratch = torch.empty(8, V, device="cuda"); ws = torch.zeros(int(fvk.alpamayo15_sample_workspace_bytes(8)), dtype=torch.uint8, device="cuda")
    sp, wp, wn = scratch.data_ptr(), ws.data_ptr(), ws.numel()
    filt = _hf_top_p_filter if rule == 0 else _fd_top_p_filter
    temperature, top_p = 0.6, 0.98
    torch.manual_seed(7 + rule)
    logits = (torch.randn(1, V, device="cuda") * 3).to(BF16)
    logits[0, 7] = 30.0; logits[0, 8] = 30.0                      # a tie at the top (above every random logit): greedy takes the lowest index
    rng = torch.tensor([1234, 8, 0], dtype=torch.int64, device="cuda")
    tok = torch.zeros(8, dtype=torch.long, device="cuda")
    fvk.alpamayo15_sample_tokens_bf16(logits.data_ptr(), 0, 1, V, lo, n_mask, 0.0, 1.0, rule, rng.data_ptr(), blocks, sp, wp, wn, tok.data_ptr(), 0, 0, 0, 0, 0, 0, s)
    torch.cuda.synchronize()
    masked = logits.float().clone(); masked[0, lo:lo + n_mask] = float("-inf")
    assert int(tok[0]) == int(masked.argmax()) == 7 and rng.tolist() == [1234, 8, 0]       # greedy draws nothing
    # torch.multinomial from the same generator state: 1, 2 and 8 rows, fresh logits and offsets every time
    mism = total = 0
    for n in (1, 2, 8):
        for trial in range(12):
            torch.manual_seed(100 * rule + 10 * n + trial)
            logits = (torch.randn(n, V, device="cuda") * 3).to(BF16)
            probs = filt(logits, top_p, lo, n_mask, temperature)
            state = torch_rng_state(gen)
            ref = torch.multinomial(probs, 1).view(-1)
            inc = int(gen.get_offset()) - int(state[1])
            assert inc == torch_philox_increment(n * V, blocks)
            rng.copy_(state)
            fvk.alpamayo15_sample_tokens_bf16(logits.data_ptr(), V, n, V, lo, n_mask, temperature, top_p, rule, rng.data_ptr(), blocks, sp, wp, wn,
                                              tok.data_ptr(), 0, 0, 0, 0, 0, 0, s)
            torch.cuda.synchronize()
            assert int(rng[1]) == int(state[1]) + inc and int(rng[2]) == 0
            mism += int((tok[:n] != ref).sum()); total += n
    assert mism <= total // 100, (mism, total)      # equal up to float rounding of the probabilities (near-equal ratios)
    # the bookkeeping of a decode step (finished rows padded, counters advanced)
    n = 4
    unfinished = torch.tensor([1, 0, 1, 0], device="cuda"); slot = torch.tensor([10], device="cuda"); rpos = torch.tensor([3, 3, 3, 3], device="cuda")
    ln = torch.tensor([5, 5, 5, 5], device="cuda", dtype=torch.int32); gl = torch.tensor([1, 1, 1, 1], device="cuda", dtype=torch.int32)
    fvk.alpamayo15_sample_tokens_bf16(logits.data_ptr(), 0, n, V, lo, n_mask, temperature, top_p, rule, rng.data_ptr(), blocks, sp, wp, wn, tok.data_ptr(),
                                      unfinished.data_ptr(), 77, slot.data_ptr(), rpos.data_ptr(), ln.data_ptr(), gl.data_ptr(), s)
    torch.cuda.synchronize()
    assert int(tok[1]) == 77 and int(tok[3]) == 77 and int(tok[0]) != 77
    assert int(slot) == 11 and rpos.tolist() == [4, 4, 4, 4] and ln.tolist() == [6] * 4 and gl.tolist() == [2] * 4


def test_dflash_propose_and_decide_match_flashdrive_rules():
    fvk = _kernels()
    s = _stream()
    B, stop, traj, period = 8, 50, 60, 70
    proposed = torch.tensor([1, stop, 5, 6, stop, 9, 11], device="cuda")
    first = torch.tensor([3], device="cuda")
    block = torch.zeros(B, dtype=torch.long, device="cuda")
    fvk.alpamayo15_dflash_propose(proposed.data_ptr(), first.data_ptr(), stop, traj, block.data_ptr(), B, s)
    torch.cuda.synchronize()
    follows = torch.nn.functional.pad(proposed[:-1] == stop, (1, 0))
    assert block.tolist() == [3] + torch.where(follows, torch.tensor(traj, device="cuda"), proposed).tolist()
    stats = torch.zeros(3, dtype=torch.long, device="cuda")
    for posterior, expect in (([1, stop, 5, 6, 9, 9, 9, period], None), ([1, stop, 5, 7, 9, 9, 9, 9], None), ([stop, 2, 3, 4, 5, 6, 7, 8], None)):
        post = torch.tensor(posterior, device="cuda")
        fvk.alpamayo15_dflash_decide(block.data_ptr(), post.data_ptr(), stop, period, B, stats.data_ptr(), s)
        torch.cuda.synchronize()
        matches = block[1:] == post[:-1]
        acc = int(matches.cumprod(0).sum())
        bonus = int(post[acc])
        emission = torch.cat([block, post[acc:acc + 1]]).scatter(0, torch.tensor([acc + 1], device="cuda"), post[acc:acc + 1])
        hits = (emission == stop) & (torch.arange(B + 1, device="cuda") <= acc + 1)
        first_stop = int(hits.int().argmax()) if bool(hits.any()) else -1
        assert stats.tolist() == [acc, first_stop, int(bonus == period)]


def test_action_expert_projection_kernels_match_official_modules():
    """Fourier features, the SiLU MLP on the tcgen05 GEMM with RMSNorms, the fp32 LayerNorm, the output projection
    with the Euler update and the unicycle trajectory, each against the official modules."""
    fvk = _kernels()
    pytest.importorskip("alpamayo1_5")
    from alpamayo1_5.models.action_in_proj import PerWaypointActionInProjV2
    from alpamayo1_5.action_space import UnicycleAccelCurvatureActionSpace
    from flash_rt.models.alpamayo15.pipeline_thor import Alpamayo15Thor, _Gemm
    from flash_rt.models.alpamayo15.weights import pack_action_modules
    from types import SimpleNamespace
    torch.manual_seed(8)
    s = _stream()
    A, act_dim, E, n = 64, 2, 2048, 3
    rows = n * A
    ap = PerWaypointActionInProjV2(in_dims=[A, act_dim], out_dim=E, num_enc_layers=2, hidden_size=512, max_freq=100.0, num_fourier_feats=20)
    out_proj = torch.nn.Linear(E, act_dim)
    with torch.no_grad():
        for p_ in list(ap.parameters()) + list(out_proj.parameters()):
            p_.normal_(0.0, 0.05)
    ap = ap.to("cuda", BF16).eval(); out_proj = out_proj.to("cuda", BF16).eval()
    sp = UnicycleAccelCurvatureActionSpace(dt=0.1, n_waypoints=A, accel_mean=0.029, accel_std=0.681, curvature_mean=0.00027, curvature_std=0.0261)
    model = SimpleNamespace(action_in_proj=ap, action_out_proj=out_proj, action_space=sp)
    eng = SimpleNamespace(act_dim=act_dim, n_act=A, e_hidden=E, _fvk=fvk, _gemm=_Gemm(fvk))
    eng._action_in = Alpamayo15Thor._action_in.__get__(eng)
    eng.act = pack_action_modules(model, act_dim, A)
    mk = lambda *shape, dt=BF16: torch.zeros(*shape, device="cuda", dtype=dt)
    eng._eb = {"af": mk(rows, eng.act.k0), "ah0": mk(rows, 512), "ah1": mk(rows, 512), "ae": mk(rows, E)}
    x = torch.randn(n, A, act_dim, device="cuda")
    t = torch.tensor([0.3], device="cuda")
    res = torch.zeros(rows, E, device="cuda")
    eng._action_in(x.view(rows, act_dim), t, rows, res, s)
    with torch.no_grad(), torch.autocast("cuda", dtype=BF16):
        ref = ap(x, t.view(1, 1, 1).expand(n, 1, 1)).float().view(rows, E)
    torch.cuda.synchronize()
    cos = torch.nn.functional.cosine_similarity(res.flatten(), ref.flatten(), dim=0)
    assert float(cos) > 0.999 and float((res - ref).abs().mean()) < 3e-2 * float(ref.abs().mean()), (float(cos), float((res - ref).abs().mean()))
    # output projection + Euler (fp32 state), and the v output (bf16)
    hn = torch.randn(rows, E, device="cuda").to(BF16)
    x2 = x.clone().view(rows, act_dim); v_out = torch.empty(rows, act_dim, device="cuda", dtype=BF16)
    fvk.alpamayo15_action_out_euler(hn.data_ptr(), eng.act.out_w.data_ptr(), eng.act.out_b.data_ptr(), 0.1, x2.data_ptr(), v_out.data_ptr(), rows, E, act_dim, s)
    with torch.no_grad(), torch.autocast("cuda", dtype=BF16):
        v_ref = out_proj(hn)
    torch.cuda.synchronize()
    assert float((v_out.float() - v_ref.float()).abs().max()) <= 2 ** -6 * float(v_ref.float().abs().max()) + 1e-3
    assert torch.allclose(x2, x.view(rows, act_dim) + 0.1 * v_out.float(), atol=1e-6)
    # BF16 Euler step (the streaming backend's state)
    xb = torch.randn(rows, act_dim, device="cuda").to(BF16); xb_ref = xb.clone()
    fvk.alpamayo15_euler_bf16(xb.data_ptr(), v_out.data_ptr(), 0.125, rows * act_dim, s)
    torch.cuda.synchronize()
    assert torch.equal(xb, xb_ref + torch.tensor(0.125, device="cuda", dtype=BF16) * v_out)
    # trajectory integration against the official action space with the same t0 state
    action = torch.randn(n, A, act_dim, device="cuda")
    v0 = torch.tensor([3.0, 7.5, 0.2], device="cuda"); z = torch.tensor([0.1, -0.2, 0.0], device="cuda")
    xyz = torch.zeros(n, A, 3, device="cuda"); rot = torch.zeros(n, A, 3, 3, device="cuda")
    fvk.alpamayo15_action_to_traj(action.data_ptr(), v0.data_ptr(), z.data_ptr(), n, A, 0.1, 0.029, 0.681, 0.00027, 0.0261, xyz.data_ptr(), rot.data_ptr(), s)
    hist = torch.zeros(n, 16, 3, device="cuda"); hist[:, :, 2] = z[:, None]
    hrot = torch.eye(3, device="cuda").expand(n, 16, 3, 3).contiguous()
    with torch.no_grad():
        xyz_ref, rot_ref = sp.action_to_traj(action, hist, hrot, t0_states={"v": v0})
    torch.cuda.synchronize()
    assert torch.allclose(xyz, xyz_ref, atol=2e-3, rtol=1e-4), float((xyz - xyz_ref).abs().max())
    assert torch.allclose(rot, rot_ref, atol=1e-4)


@pytest.mark.parametrize("M,N,K", [(300, 384, 512), (3086, 1024, 4096), (802, 4096, 1024)])
def test_wide_fp8_and_int8_tiles_equal_the_reference_tiles(M, N, K):
    """The 2-SM 256 x 256 FP8 and INT8 tiles tile N differently but reduce K in the same order: identical outputs to
    the 256 x 128 tiles, per-row / per-column scales included."""
    fvk = _kernels()
    s = _stream()
    torch.manual_seed(3)
    A8 = (torch.randn(M, K, device="cuda") * 2).to(torch.float8_e4m3fn).view(torch.uint8)
    W8 = (torch.randn(N, K, device="cuda") * 2).to(torch.float8_e4m3fn).view(torch.uint8)
    a_s = torch.rand(M, device="cuda") + 0.5; w_s = torch.rand(N, device="cuda") * 1e-2
    outs = {}
    for tile in (2, 3):
        ws = torch.empty(max(int(fvk.alpamayo15_fp8_gemm_sm100_workspace(M, N, K, tile)), 16), dtype=torch.uint8, device="cuda")
        D = torch.empty(M, N, device="cuda", dtype=BF16)
        fvk.alpamayo15_fp8_gemm_sm100(A8.data_ptr(), W8.data_ptr(), a_s.data_ptr(), w_s.data_ptr(), D.data_ptr(), M, N, K, ws.data_ptr(),
                                      ws.numel(), s, tile, N)
        outs[tile] = D
    torch.cuda.synchronize()
    ref = (A8.view(torch.float8_e4m3fn).float() @ W8.view(torch.float8_e4m3fn).float().t()) * a_s[:, None] * w_s[None, :]
    assert torch.equal(outs[3], outs[2])
    assert float((outs[2].float() - ref).abs().max()) <= 2 ** -7 * float(ref.abs().max())
    xq = torch.randint(-127, 128, (M, K), device="cuda", dtype=torch.int8)
    wq = torch.randint(-127, 128, (N, K), device="cuda", dtype=torch.int8)
    i_outs = {}
    for tile in (2, 3):
        ws = torch.empty(max(int(fvk.alpamayo15_int8_gemm_sm100_workspace(M, N, K, tile)), 16), dtype=torch.uint8, device="cuda")
        D = torch.empty(M, N, device="cuda", dtype=torch.float16)
        fvk.alpamayo15_int8_gemm_sm100(xq.data_ptr(), wq.data_ptr(), a_s.data_ptr(), w_s.data_ptr(), D.data_ptr(), M, N, K, tile,
                                       ws.data_ptr(), ws.numel(), s)
        i_outs[tile] = D
    torch.cuda.synchronize()
    assert torch.equal(i_outs[3], i_outs[2])


def test_cublaslt_gemms_from_cpp():
    """The C++ cuBLASLt paths: BF16 equals torch.mm up to accumulation order (column blocks of a wider output via ldd,
    one row over an odd N as for the LM head); FP8 with device-side per-tensor scales equals the scaled product."""
    fvk = _kernels()
    s = _stream()
    fvk.alpamayo15_cublaslt_init()
    torch.manual_seed(4)
    M, N, K = 300, 1024, 2048
    A = torch.randn(M, K, device="cuda").to(BF16); W = (torch.randn(N, K, device="cuda") * 0.05).to(BF16)
    D = torch.zeros(M, 2 * N, device="cuda", dtype=BF16)
    fvk.alpamayo15_cublaslt_gemm_bf16(A.data_ptr(), W.data_ptr(), 0, 0, D[:, N:].data_ptr(), M, N, K, 0, 2 * N, 0, s)
    torch.cuda.synchronize()
    ref = (A.float() @ W.float().t())
    assert float((D[:, N:].float() - ref).abs().max()) <= 2 ** -7 * float(ref.abs().max()) and not D[:, :N].any()
    V = 1001                                            # one row over an odd N (the LM head of one token)
    Wv = (torch.randn(V, K, device="cuda") * 0.05).to(BF16); y = torch.zeros(1, V, device="cuda", dtype=BF16)
    fvk.alpamayo15_cublaslt_gemm_bf16(A[:1].data_ptr(), Wv.data_ptr(), 0, 0, y.data_ptr(), 1, V, K, 0, V, 0, s)
    torch.cuda.synchronize()
    refv = A[:1].float() @ Wv.float().t()
    assert float((y.float() - refv).abs().max()) <= 2 ** -7 * float(refv.abs().max())
    A8 = A.to(torch.float8_e4m3fn); W8 = (W * 8).to(torch.float8_e4m3fn)
    sa = torch.tensor([0.5], device="cuda"); sw = torch.tensor([0.125], device="cuda")
    D8 = torch.empty(M, N, device="cuda", dtype=BF16)
    fvk.alpamayo15_cublaslt_gemm_fp8(A8.view(torch.uint8).data_ptr(), W8.view(torch.uint8).data_ptr(), sa.data_ptr(), sw.data_ptr(),
                                     D8.data_ptr(), M, N, K, N, s)
    torch.cuda.synchronize()
    ref8 = (A8.float() @ W8.float().t()) * 0.5 * 0.125
    assert float((D8.float() - ref8).abs().max()) <= 2 ** -7 * float(ref8.abs().max())
