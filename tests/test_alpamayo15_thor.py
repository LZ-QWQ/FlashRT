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
    _TrajTokenMask,
    FP8_E4M3_MAX,
    check_generation_config,
    expert_attention_spans,
    fp8_fragment_layout,
    merge_attention,
    quantize_fp8_rows,
    quantize_int8_rows,
    require_alpamayo15_kernels,
    resolve_preset,
    update_stop_state,
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
    assert found in (set(), set(ALPAMAYO15_KERNEL_SYMBOLS))
    if os.environ.get("FLASHRT_TEST_ALPAMAYO15_REQUIRED") == "1":
        assert found == set(ALPAMAYO15_KERNEL_SYMBOLS)


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


def test_traj_token_mask_range():
    scores = torch.zeros(2, 20)
    out = _TrajTokenMask(5, 4)(None, scores)
    assert torch.isinf(out[:, 5:9]).all() and (out[:, 5:9] < 0).all()
    assert torch.equal(out[:, :5], torch.zeros(2, 5)) and torch.equal(out[:, 9:], torch.zeros(2, 11))


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


@pytest.mark.parametrize("with_delta", [False, True])
def test_vlm_rms_matches_qwen3_rmsnorm(with_delta: bool):
    fvk = _kernels()
    torch.manual_seed(5)
    rows, dim, eps = 9, 4096, 1e-6
    res = torch.randn(rows, dim, device="cuda").to(BF16)
    delta = torch.randn(rows, dim, device="cuda").to(BF16)
    w = (torch.rand(dim, device="cuda") + 0.5).to(BF16)
    ref_res = (res + delta) if with_delta else res.clone()
    ref = _qwen3_head_norm(ref_res, w, eps)
    out = torch.empty_like(res)
    fvk.alpamayo15_rms_bf16(res.data_ptr(), delta.data_ptr() if with_delta else 0, w.data_ptr(), out.data_ptr(),
                            rows, dim, eps, _stream())
    torch.cuda.synchronize()
    assert torch.equal(res, ref_res)
    _assert_bf16_close(out, ref)


def test_expert_fp32_residual_rms():
    fvk = _kernels()
    torch.manual_seed(6)
    rows, dim, eps = 64, 2048, 1e-6
    res = torch.randn(rows, dim, device="cuda")
    delta = torch.randn(rows, dim, device="cuda").to(BF16)
    w = (torch.rand(dim, device="cuda") + 0.5).to(BF16)
    ref_res = res + delta.float()                                     # fp32 + bf16 -> fp32
    hs = ref_res * torch.rsqrt(ref_res.pow(2).mean(-1, keepdim=True) + eps)
    ref = (w.float() * hs).to(BF16)                                   # autocast cast at the next linear
    out = torch.empty(rows, dim, device="cuda", dtype=BF16)
    fvk.alpamayo15_res_rms_f32_bf16(res.data_ptr(), delta.data_ptr(), w.data_ptr(), out.data_ptr(),
                                    rows, dim, eps, _stream())
    torch.cuda.synchronize()
    assert torch.equal(res, ref_res)
    _assert_bf16_close(out, ref)


@pytest.mark.parametrize("n_samples", [1, 3])
def test_expert_qk_prep_matches_autocast_sequence(n_samples):
    fvk = _kernels()
    torch.manual_seed(8)
    A, Hq, Hk, D, eps, S = 64, 16, 8, 128, 1e-6, 80
    rows = n_samples * A
    qkv = (torch.randn(rows, (Hq + 2 * Hk) * D, device="cuda") * 2).to(BF16)
    qw = (torch.rand(D, device="cuda") + 0.5).to(BF16)
    kw = (torch.rand(D, device="cuda") + 0.5).to(BF16)
    ang = torch.randn(rows, D // 2, device="cuda") * 3
    emb = torch.cat((ang, ang), -1)
    cos, sin = emb.cos().contiguous(), emb.sin().contiguous()
    q = qkv[:, :Hq * D].view(rows, Hq, D)
    k = qkv[:, Hq * D:(Hq + Hk) * D].view(rows, Hk, D)
    v = qkv[:, (Hq + Hk) * D:].view(rows, Hk, D)

    def ref(x, w):  # BF16 head norm, then FP32 RoPE, cast to BF16 at SDPA
        n = _qwen3_head_norm(x, w, eps)
        return (n.float() * cos[:, None] + _rot(n).float() * sin[:, None]).to(BF16)

    q_out = torch.empty(rows, Hq, D, device="cuda", dtype=BF16)
    kc = torch.zeros(n_samples, S, Hk, D, device="cuda", dtype=BF16)
    vc = torch.zeros_like(kc)
    slots = torch.tensor([3 + 5 * b for b in range(n_samples)], device="cuda", dtype=torch.long)
    fvk.alpamayo15_expert_qk_prep_bf16(qkv.data_ptr(), qw.data_ptr(), kw.data_ptr(), cos.data_ptr(), sin.data_ptr(),
                                       q_out.data_ptr(), kc.data_ptr(), vc.data_ptr(), slots.data_ptr(), rows,
                                       Hq, Hk, Hk * D, A, S, eps, _stream())
    torch.cuda.synchronize()
    _assert_bf16_close(q_out, ref(q, qw))
    k_ref, written = ref(k, kw), torch.zeros(n_samples, S, dtype=torch.bool, device="cuda")
    for b, st in enumerate(slots.tolist()):  # sample b's action rows start at its slot
        _assert_bf16_close(kc[b, st:st + A], k_ref[b * A:(b + 1) * A])
        assert torch.equal(vc[b, st:st + A], v[b * A:(b + 1) * A])
        written[b, st:st + A] = True
    assert not kc[~written].any() and not vc[~written].any()


@pytest.mark.parametrize("rows,slot,batch_stride", [(11, None, 0), (1, 5, 0), (3, 5, 16)])
def test_vlm_qk_prep_matches_bf16_rope_and_cache_write(rows, slot, batch_stride):
    fvk = _kernels()
    torch.manual_seed(9)
    Hq, Hk, D, eps, S = 32, 8, 128, 1e-6, 16
    qkv = (torch.randn(rows, (Hq + 2 * Hk) * D, device="cuda") * 2).to(BF16)
    qw = (torch.rand(D, device="cuda") + 0.5).to(BF16)
    kw = (torch.rand(D, device="cuda") + 0.5).to(BF16)
    ang = torch.randn(rows, D // 2, device="cuda") * 3
    emb = torch.cat((ang, ang), -1)
    cos, sin = emb.cos().to(BF16).contiguous(), emb.sin().to(BF16).contiguous()
    q = qkv[:, :Hq * D].view(rows, Hq, D)
    k = qkv[:, Hq * D:(Hq + Hk) * D].view(rows, Hk, D)
    v = qkv[:, (Hq + Hk) * D:].view(rows, Hk, D)

    def ref(x, w):  # official BF16 decode RoPE: every product and the sum rounded to BF16
        n = _qwen3_head_norm(x, w, eps)
        return n * cos[:, None] + _rot(n) * sin[:, None]

    nb = rows if batch_stride else 1
    q_out = torch.empty(rows, Hq, D, device="cuda", dtype=BF16)
    kc = torch.zeros(nb, S, Hk, D, device="cuda", dtype=BF16)
    vc = torch.zeros_like(kc)
    d_slot = torch.tensor([slot or 0], device="cuda", dtype=torch.long)
    rpb = 1 if batch_stride else rows
    fvk.alpamayo15_vlm_qk_prep_bf16(qkv.data_ptr(), qw.data_ptr(), kw.data_ptr(), cos.data_ptr(), sin.data_ptr(),
                                    q_out.data_ptr(), kc.data_ptr(), vc.data_ptr(),
                                    d_slot.data_ptr() if slot is not None else 0, rows, Hq, Hk, Hk * D,
                                    rpb, batch_stride, eps, _stream())
    torch.cuda.synchronize()
    start = slot or 0
    written = torch.zeros(nb, S, dtype=torch.bool, device="cuda")
    if batch_stride:  # row b -> sample b, cache row `slot`
        k_got, v_got = kc[:, start], vc[:, start]
        written[:, start] = True
    else:             # rows land contiguously from `slot`
        k_got, v_got = kc[0, start:start + rows], vc[0, start:start + rows]
        written[0, start:start + rows] = True
    _assert_bf16_close(q_out, ref(q, qw))
    _assert_bf16_close(k_got, ref(k, kw))
    assert torch.equal(v_got, v)
    assert not kc[~written].any() and not vc[~written].any()


@pytest.mark.parametrize("fmt", ["fp8", "int8"])
@pytest.mark.parametrize("M,N,K", [(1, 4096, 4096), (3, 40, 64), (8, 6144, 4096), (9, 1000, 12288), (16, 155, 4096),
                                   (1, 65537, 64), (12, 65553, 32)])   # last two: wide (2 tiles/warp) path, odd tile count
def test_w8a16_gemm_matches_dequantized_reference(M, N, K, fmt):
    fvk = _kernels()
    torch.manual_seed(M * 7 + N)
    w = (torch.randn(N, K, device="cuda") * 0.02).to(BF16)
    x = torch.randn(M, K, device="cuda").to(BF16)
    if fmt == "int8":
        codes, scale = quantize_int8_rows(w)
        assert torch.equal(codes.view(torch.int8).abs().amax(dim=1), torch.full((N,), 127, device="cuda", dtype=torch.int8))
        deq = codes.view(torch.int8).float() * scale[:, None]
    else:
        codes, scale = quantize_fp8_rows(w)
        deq = codes.view(torch.float8_e4m3fn).float() * scale[:, None]
    frag, scale_p = fp8_fragment_layout(codes, scale)
    y = torch.full((M, N), float("nan"), device="cuda", dtype=BF16)
    fvk.alpamayo15_w8a16_gemm_bf16(frag.data_ptr(), scale_p.data_ptr(), x.data_ptr(), y.data_ptr(), M, N, K, _stream(),
                                   fmt=1 if fmt == "int8" else 0)
    torch.cuda.synchronize()
    ref = x.float() @ deq.t()
    err = (y.float() - ref).abs()
    # FP32 accumulation in a different order, then one BF16 rounding of the output.
    assert bool((err <= ref.abs() * 2.0 ** -8 + 1e-3 * ref.abs().max()).all()), float(err.max())


def test_w8a16_gemm_rejects_unsupported_shapes():
    fvk = _kernels()
    frag = torch.zeros(16 * 64, device="cuda", dtype=torch.uint8)
    scale = torch.zeros(16, device="cuda")
    x = torch.zeros(17, 64, device="cuda", dtype=BF16)
    y = torch.zeros(17, 16, device="cuda", dtype=BF16)
    for M, K, off in ((17, 64, 0), (1, 48, 0), (1, 64, 8)):   # too many rows, K % 32, misaligned weights
        with pytest.raises(Exception, match="alpamayo15_w8a16_gemm_bf16"):
            fvk.alpamayo15_w8a16_gemm_bf16(frag.data_ptr() + off, scale.data_ptr(), x.data_ptr(), y.data_ptr(),
                                           M, 16, K, _stream())


@pytest.mark.parametrize("M,N,K,act,tile,res", [(3072, 3456, 1152, 0, 0, False), (3072, 1152, 1152, 0, 0, True),
                                                  (3072, 4304, 1152, 1, 0, False), (768, 4608, 4608, 2, 0, False),
                                                  (3072, 1152, 1536, 0, 0, True), (64, 16512, 2048, 0, 1, False),
                                                  (64, 2048, 8256, 0, 1, False), (37, 1032, 72, 1, 0, False),
                                                  (3086, 6144, 4096, 0, 2, False), (300, 4096, 12288, 0, 2, False),
                                                  (2880, 4304, 1152, 1, 2, False), (2880, 1152, 4304, 0, 2, True),
                                                  (720, 4608, 4608, 2, 2, False)])
def test_bf16_cutlass_gemm_matches_torch(M, N, K, act, tile, res):
    """D = act(A @ W.T + bias) (+ C) on tcgen05 against the torch reference (FP32 math, one BF16 rounding)."""
    fvk = _kernels()
    torch.manual_seed(23)
    A = (torch.randn(M, K, device="cuda") * 0.5).to(BF16)
    W = (torch.randn(N, K, device="cuda") * 0.05).to(BF16)
    bias = (torch.randn(N, device="cuda") * 0.1).to(BF16) if tile == 0 else None
    C = torch.randn(M, N, device="cuda").to(BF16) if res else None
    D = torch.empty(M, N, device="cuda", dtype=BF16)
    _Gemm(fvk)(A, W, D, bias=bias, C=C, act=act, tile=tile)
    torch.cuda.synchronize()
    ref = A.float() @ W.float().t()
    if bias is not None:
        ref = ref + bias.float()
    if act == 1:
        ref = torch.nn.functional.gelu(ref, approximate="tanh")
    elif act == 2:
        ref = torch.nn.functional.gelu(ref)
    if C is not None:
        ref = ref + C.float()
    err = (D.float() - ref).abs()
    tol = ref.abs() * 2 ** -7 + 2 ** -7 * (K ** 0.5) * 0.05 * 0.5 * 0.1 + 2 ** -6
    assert bool((err <= tol).all()), float((err / tol).max())
    assert float((D.float() - ref).abs().mean()) < 2e-3 * max(1.0, float(ref.abs().mean()))
    if tile == 2:                         # the 2-SM form reduces K in the same order as the 128-row tile
        D0 = _Gemm(fvk)(A, W, torch.empty_like(D), bias=bias, C=C, act=act, tile=0)
        torch.cuda.synchronize()
        assert torch.equal(D0, D)


@pytest.mark.parametrize("M,N,K,tile", [(64, 3072, 2048, 1), (64, 2048, 8256, 1), (64, 16512, 2048, 1), (9, 256, 64, 1),
                                         (3086, 6144, 4096, 2), (300, 4096, 12288, 2), (300, 4096, 4096, 0)])
def test_fp8_gemm_and_row_quantization_match_reference(M, N, K, tile):
    """Per-row FP8 activation quantization equals the engine's quantize_fp8_rows; the W8A8 GEMM reproduces
    bf16((q_a @ q_w.T) * a_s * w_s) (FP32 accumulation)."""
    fvk = _kernels()
    torch.manual_seed(26)
    x = (torch.randn(M, K + 8, device="cuda") * 2).to(BF16)[:, :K]
    w = (torch.randn(N, K, device="cuda") * 0.05).to(BF16)
    q8 = torch.empty(M, K, device="cuda", dtype=torch.uint8)
    a_s = torch.empty(M, device="cuda", dtype=torch.float32)
    fvk.alpamayo15_quantize_fp8_rows_bf16(x.data_ptr(), x.stride(0), q8.data_ptr(), a_s.data_ptr(), M, K,
                                          torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    q_ref, s_ref = quantize_fp8_rows(x.contiguous())
    assert torch.equal(a_s, s_ref)
    assert torch.equal(q8, q_ref)
    w8, w_s = quantize_fp8_rows(w)
    D = torch.empty(M, N, device="cuda", dtype=BF16)
    _Gemm(fvk).fp8(q8, a_s, w8, w_s, D, tile=tile)
    torch.cuda.synchronize()
    ref = (q8.view(torch.float8_e4m3fn).float() @ w8.view(torch.float8_e4m3fn).float().t()) * a_s[:, None] * w_s[None, :]
    err = (D.float() - ref).abs()
    assert bool((err <= ref.abs() * 2 ** -7 + 2 ** -6 * float(ref.abs().mean())).all()), float(err.max())
    if tile != 1:                     # column blocks of a wider output (leading dimension N): same values, same order
        D2 = torch.full((M, N), float("nan"), device="cuda", dtype=BF16)
        half = N // 2
        _Gemm(fvk).fp8(q8, a_s, w8[:half], w_s[:half], D2[:, :half], tile=tile)
        _Gemm(fvk).fp8(q8, a_s, w8[half:], w_s[half:], D2[:, half:], tile=tile)
        torch.cuda.synchronize()
        assert torch.equal(D2, D)
    deq = x.float() @ w.float().t()
    assert float((D.float() - deq).abs().mean()) < 0.06 * float(deq.abs().mean())   # e4m3 on both operands


@pytest.mark.parametrize("rows,dim", [(3072, 1152), (1029, 1152), (768, 4608), (40, 2048), (5, 8)])
def test_layernorm_matches_torch(rows, dim):
    fvk = _kernels()
    torch.manual_seed(24)
    x = (torch.randn(rows, dim, device="cuda") * 2 + 0.3).to(BF16)
    w = (torch.rand(dim, device="cuda") + 0.5).to(BF16)
    b = (torch.randn(dim, device="cuda") * 0.1).to(BF16)
    out = torch.empty_like(x)
    fvk.alpamayo15_layernorm_bf16(x.data_ptr(), w.data_ptr(), b.data_ptr(), out.data_ptr(), rows, dim, 1e-6,
                                  torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    ref = torch.nn.functional.layer_norm(x.float(), (dim,), w.float(), b.float(), 1e-6)
    assert bool(((out.float() - ref).abs() <= ref.abs() * 2 ** -7 + 2 ** -6).all())
    ref_bf = torch.nn.functional.layer_norm(x, (dim,), w, b, 1e-6)         # the autocast BF16 module path
    assert float((out != ref_bf).float().mean()) < 0.05


def test_silu_mul_split_matches_torch():
    fvk = _kernels()
    torch.manual_seed(25)
    rows, F = 64, 8256
    gu = (torch.randn(rows, 2 * F, device="cuda") * 2).to(BF16)
    out = torch.empty(rows, F, device="cuda", dtype=BF16)
    s = torch.cuda.current_stream().cuda_stream
    fvk.alpamayo15_silu_mul_split_bf16(gu.data_ptr(), gu[:, F:].data_ptr(), 2 * F, out.data_ptr(), rows, F, s)
    torch.cuda.synchronize()
    with torch.autocast("cuda", dtype=BF16):
        ref = torch.nn.functional.silu(gu[:, :F]) * gu[:, F:]
    assert torch.equal(out, ref)
    g, u = gu[:, :F].contiguous(), gu[:, F:].contiguous()         # separate gate / up buffers (leading dimension F)
    out2 = torch.empty_like(out)
    fvk.alpamayo15_silu_mul_split_bf16(g.data_ptr(), u.data_ptr(), F, out2.data_ptr(), rows, F, s)
    torch.cuda.synchronize()
    assert torch.equal(out2, ref)


@pytest.mark.parametrize("kw,msg", [({"decode_weights": "int4"}, "decode_weights"),
                                    ({"expert_weights": "int8"}, "expert_weights"),
                                    ({"prefill_weights": "int8"}, "prefill_weights"),
                                    ({"vision_gemm": "cublas"}, "vision_gemm"),
                                    ({"expert_weights": "fp8", "expert_gemm": "torch"}, "expert_gemm='cutlass'"),
                                    ({"decode_weights": "int8", "max_samples": 17}, "max_samples <= 16"),
                                    ({"decode_weights": "w4a8"}, "paro_checkpoint"),
                                    ({"vision_weights": "fp8", "vision_gemm": "torch"}, "vision_gemm='cutlass'"),
                                    ({"vision_weights": "int8"}, "vision_weights"),
                                    ({"paro_checkpoint": "x"}, "paro_checkpoint"),
                                    ({"preset": "turbo"}, "preset"),
                                    ({"preset": "balanced"}, "paro_checkpoint"),
                                    ({"preset": "balanced", "paro_checkpoint": "x", "expert_weights": "int8"}, "expert_weights"),
                                    ({"lm_head_weights": "int8"}, "lm_head_weights"),
                                    ({"preset": "fast"}, "paro_checkpoint"),
                                    ({"lm_head_weights": "int4", "max_samples": 32}, "max_samples <= 16")])
def test_engine_option_validation(kw, msg):
    with pytest.raises(ValueError, match=msg):
        Alpamayo15Thor(SimpleNamespace(), **kw)


def test_presets_resolve_with_explicit_overrides():
    assert resolve_preset("bf16") == {"decode_weights": "bf16", "expert_weights": "bf16", "prefill_weights": "bf16",
                                      "vision_weights": "bf16", "lm_head_weights": "decode"}
    assert resolve_preset("balanced") == {"decode_weights": "w4a8", "expert_weights": "fp8", "prefill_weights": "fp8",
                                          "vision_weights": "bf16", "lm_head_weights": "decode"}
    assert resolve_preset("fast") == {"decode_weights": "w4a8", "expert_weights": "fp8", "prefill_weights": "fp8",
                                      "vision_weights": "bf16", "lm_head_weights": "int4"}
    assert resolve_preset("balanced", decode_weights="int8", vision_weights=None)["decode_weights"] == "int8"
    assert resolve_preset("bf16", prefill_weights="fp8")["prefill_weights"] == "fp8"
    assert resolve_preset("fast", lm_head_weights="decode")["lm_head_weights"] == "decode"
    with pytest.raises(ValueError, match="preset"):
        resolve_preset("fp8")
    with pytest.raises(ValueError, match="not preset options"):
        resolve_preset("bf16", pdl=True)


@pytest.mark.parametrize("M,Ns,K,ksplit,prefetch,warps", [(1, (4096, 1024, 1024), 4096, 1, 2, 4), (1, (12288, 12288), 4096, 2, 4, 4),
                                                          (1, (4096,), 12288, 0, 2, 4), (8, (1024, 1024), 4096, 2, 2, 8),
                                                          (16, (256, 512, 1024), 2048, 1, 4, 4), (3, (272,), 1024, 0, 2, 8),
                                                          (1, (4096,), 12288, 4, 2, 4), (9, (1024, 512), 4096, 4, 4, 8)])
def test_w4a8_grouped_gemm_is_exact(M, Ns, K, ksplit, prefetch, warps):
    """The grouped few-row W4A8 GEMM computes Marlin's integers exactly for every problem of a group (float64
    reference), writing the column blocks of one output."""
    fvk = _kernels()
    from flash_rt.models.alpamayo15.marlin_w4a8 import pack_w4a8_grouped, pack_w4a8_small
    torch.manual_seed(K + sum(Ns))
    G = K // 128
    Ma = (M + 7) // 8 * 8
    ws = [torch.randint(0, 16, (K, N), device="cuda", dtype=torch.uint8) for N in Ns]
    zs = [torch.randint(0, 16, (G, N), device="cuda", dtype=torch.uint8) for N in Ns]
    sgs = [torch.randint(0, 4097, (G, N), device="cuda", dtype=torch.int32) for N in Ns]
    xqs = [torch.randint(-128, 128, (Ma, K), device="cuda", dtype=torch.int8) for _ in Ns]
    a_ss = [(torch.rand(M, device="cuda") * 2e-2 + 1e-3) * 7e-6 for _ in Ns]
    out = torch.full((M, sum(Ns)), float("nan"), device="cuda", dtype=BF16)
    cols = [sum(Ns[:i]) for i in range(len(Ns))]
    pack = pack_w4a8_grouped(ws, zs, sgs, K)
    assert pack.shape == (sum(Ns) // 16, G, 1088)
    # the fused layout carries every tile's zero points, int16 scales and nibbles
    assert torch.equal(pack[:, :, :16].permute(0, 2, 1).reshape(-1, G).t(), torch.cat(zs, dim=1))
    assert torch.equal(pack[:, :, 16:48].contiguous().view(torch.int16).permute(0, 2, 1).reshape(-1, G).t().int(), torch.cat(sgs, dim=1))
    assert torch.equal(pack[:, :, 64:].reshape(-1, K // 64, 32, 16), torch.cat([pack_w4a8_small(w, K, w.shape[1]) for w in ws]))
    fvk.alpamayo15_w4a8_gemm_grouped([t.data_ptr() for t in xqs], [t.data_ptr() for t in a_ss], pack.data_ptr(),
                                     [out[:, c:].data_ptr() for c in cols], M, list(Ns), K, out.stride(0), prefetch, warps, _stream(), ksplit)
    torch.cuda.synchronize()
    for w, z, sg, xq, a_s, c, N in zip(ws, zs, sgs, xqs, a_ss, cols, Ns):
        wz = w.to(torch.int64) - z.to(torch.int64).repeat_interleave(128, 0)
        part = torch.einsum("mgk,gkn->mgn", xq[:M].to(torch.float64).view(M, G, 128), wz.to(torch.float64).view(G, 128, N))
        total = (part * sg.to(torch.float64).unsqueeze(0)).sum(1)
        assert torch.equal(total, total.round())
        ref = (total.to(torch.int64).to(torch.float32) * a_s.unsqueeze(1)).to(BF16)
        assert torch.equal(out[:, c:c + N], ref)


@pytest.mark.parametrize("rows,K", [(1, 4096), (5, 4096), (16, 1024), (3, 12)])
def test_int8_row_quantization_matches_torch(rows, K):
    fvk = _kernels()
    torch.manual_seed(rows + K)
    x = (torch.randn(rows, K + 8, device="cuda") * 3).to(BF16)[:, :K]          # leading dimension K + 8
    if rows > 1:
        x[1].zero_()                                                          # an all-zero row: scale 1, codes 0
    q = torch.empty(rows, K, device="cuda", dtype=torch.int8)
    scale = torch.empty(rows, device="cuda")
    fvk.alpamayo15_quantize_int8_rows_bf16(x.data_ptr(), x.stride(0), q.data_ptr(), scale.data_ptr(), rows, K, _stream())
    torch.cuda.synchronize()
    amax = x.float().abs().amax(dim=1)
    ref_scale = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
    assert torch.equal(scale, ref_scale)
    ref_q = torch.round(x.float() / ref_scale[:, None]).clamp(-127, 127).to(torch.int8)
    assert torch.equal(q, ref_q)


def test_w4_row_quantization_and_int4_head_match_dequantized_reference():
    """quantize_w4_rows: codes in range, dequantization error within half a step; the INT4 head on the grouped GEMM
    with the per-channel factor equals the integer reference of its own quantization."""
    fvk = _kernels()
    from flash_rt.models.alpamayo15.marlin_w4a8 import W4LmHead, quantize_w4_rows
    torch.manual_seed(3)
    N, K = 2048, 1024
    G = K // 128
    w = (torch.randn(N, K, device="cuda") * 0.03).to(BF16)
    pack, z, sg, cs = quantize_w4_rows(w, chunk_rows=768)                      # chunks that are not a divisor of N
    assert pack.shape == (N // 16, G, 1088) and z.shape == (G, N) and sg.shape == (G, N) and cs.shape == (N,)
    assert int(sg.min()) >= 0 and int(sg.max()) == 4096
    pack_whole, z_whole, sg_whole, cs_whole = quantize_w4_rows(w)
    assert torch.equal(pack, pack_whole) and torch.equal(z, z_whole) and torch.equal(sg, sg_whole) and torch.equal(cs, cs_whole)
    scale_eff = sg.float() * cs[None, :]                                          # [G, N]
    head = W4LmHead(fvk, w, 8)
    assert head.N == N and head.K == K
    x = (torch.randn(5, K, device="cuda") * 2).to(BF16)
    out = torch.full((5, N), float("nan"), device="cuda", dtype=BF16)
    head(x, out, _stream())
    torch.cuda.synchronize()
    # reference: the INT8 rows times the dequantized INT4 weight (the pack inverted tile by tile), in float64
    amax = x.float().abs().amax(dim=1)
    a_scale = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
    xq = torch.round(x.float() / a_scale[:, None]).clamp(-127, 127)
    deq = _dequant_w4(pack, z, sg, cs, N, K)                                   # [N, K] float64
    ref = (xq.double() @ deq.t()) * a_scale.double()[:, None]
    err = (out.double() - ref).abs()
    assert bool((err <= ref.abs() * 2.0 ** -7 + 1e-6).all()), float(err.max())
    # the dequantized weight is within half an effective step of the BF16 weight
    step = scale_eff.t().repeat_interleave(128, dim=1).double()                 # [N, K]
    assert bool(((deq - w.double()).abs() <= step * 0.51 + 1e-7).all())     # half a step, plus the 12-bit scale rounding
    # a vocabulary that is not a multiple of 16: padded tiles, only the valid channels written
    Nv = 2051                                                                 # odd vocabulary: odd leading dimension too
    w_v = (torch.randn(Nv, K, device="cuda") * 0.03).to(BF16)
    head_v = W4LmHead(fvk, w_v, 8)
    assert head_v.N == Nv and head_v.N_pad == 2064 and head_v.w_pack.shape[0] == 2064 // 16
    wide = torch.full((5, Nv + 7), 7.0, device="cuda", dtype=BF16)
    head_v(x, wide[:, :Nv], _stream())
    torch.cuda.synchronize()
    assert torch.equal(wide[:, Nv:], torch.full((5, 7), 7.0, device="cuda", dtype=BF16))
    deq_v = _dequant_w4(head_v.w_pack, head_v.zp, head_v.sg, head_v.col_scale, 2064, K)[:Nv]
    ref_v = (xq.double() @ deq_v.t()) * a_scale.double()[:, None]
    assert bool(((wide[:, :Nv].double() - ref_v).abs() <= ref_v.abs() * 2.0 ** -7 + 1e-6).all())


def _dequant_w4(pack, z, sg, cs, N, K):
    """Invert the fused pack (nibbles after the 64-byte meta block of each tile and group, in the pack_w4a8_small
    layout: n = 16 T + 8 rh + g, k = 64 C + 32 ks + 16 half + 4 t + j) and dequantize."""
    G = K // 128
    pack = pack.view(N // 16, G, 1088)[:, :, 64:].reshape(N // 16, K // 64, 32, 16)
    v = torch.stack([pack & 0xF, pack >> 4], dim=3)                            # [T, C, lane, ks, 16 bytes]
    v = v.view(N // 16, K // 64, 8, 4, 2, 2, 2, 4)                             # [T, C, g, t, ks, half, rh, j]
    v = v.permute(0, 6, 2, 1, 4, 5, 3, 7).reshape(N, K)                        # [T, rh, g | C, ks, half, t, j]
    G = K // 128
    zz = z.t().repeat_interleave(128, dim=1)                                    # [N, K]
    ss = (sg.float() * cs[None, :]).t().repeat_interleave(128, dim=1)
    return (v.double() - zz.double()) * ss.double()


def _rotation_set(K: int, rows: int):
    """A ParoQuant-shaped rotation set (random angles, random in-group pairings, channel scales) with its workspaces."""
    pairs = torch.empty(8, K, dtype=torch.int16)
    for r in range(8):
        for g in range(K // 128):
            perm = torch.randperm(128)
            pr = torch.empty(128, dtype=torch.int16)
            pr[0::2] = perm[:64].to(torch.int16)
            pr[1::2] = perm[64:].to(torch.int16)
            pairs[r, g * 128:(g + 1) * 128] = pr
    alloc = (rows + 7) // 8 * 8
    return {"theta": (torch.rand(8, K // 2) * 6.28).to("cuda", torch.float16).contiguous(), "pairs": pairs.cuda().contiguous(),
            "scales": (torch.rand(K) * 0.5 + 0.75).to("cuda", torch.float16).contiguous(), "igs": torch.full((1,), 7e-6, device="cuda"),
            "xr": torch.empty(rows, K, device="cuda", dtype=torch.float16), "part": torch.empty(rows, K // 128, device="cuda"),
            "xq": torch.zeros(alloc, K, device="cuda", dtype=torch.int8), "a_scale": torch.zeros(alloc, device="cuda")}


@pytest.mark.parametrize("M,Ns,K,cfg", [(1, (4096, 1024, 1024), 4096, (1, 2, 4)), (3, (12288, 12288), 4096, (2, 4, 4)),
                                        (4, (4096,), 12288, (1, 2, 4)), (2, (1024, 512), 4096, (4, 2, 8))])
def test_w4a8_grouped_gemm_with_fused_quantization_matches_two_pass(M, Ns, K, cfg):
    """Rotation + grouped GEMM with the quantization in its prologue gives the two-pass (rotate, quantize, GEMM)
    result bit for bit; the two-pass path's INT8 activations and scales stay as the separate kernels wrote them."""
    fvk = _kernels()
    from flash_rt.models.alpamayo15.marlin_w4a8 import pack_w4a8_grouped
    torch.manual_seed(K + M)
    G = K // 128
    ks, pf, w = cfg
    sets = [_rotation_set(K, M) for _ in Ns]
    x = (torch.randn(M, K + 8, device="cuda") * 2).to(BF16)[:, :K]                  # leading dimension K + 8
    ws = [torch.randint(0, 16, (K, N), device="cuda", dtype=torch.uint8) for N in Ns]
    zs = [torch.randint(0, 16, (G, N), device="cuda", dtype=torch.uint8) for N in Ns]
    sgs = [torch.randint(0, 4097, (G, N), device="cuda", dtype=torch.int32) for N in Ns]
    pack = pack_w4a8_grouped(ws, zs, sgs, K)
    ptrs = lambda key: [s_[key].data_ptr() for s_ in sets]
    s = _stream()
    out_ref = torch.full((M, sum(Ns)), float("nan"), device="cuda", dtype=BF16)
    out = torch.full((M, sum(Ns)), float("nan"), device="cuda", dtype=BF16)
    cols = [sum(Ns[:i]) for i in range(len(Ns))]
    fvk.flashdrive_w4a8_act_prep(x.data_ptr(), x.stride(0), M, K, ptrs("theta"), ptrs("pairs"), ptrs("scales"), ptrs("igs"), ptrs("xr"),
                                 ptrs("part"), ptrs("xq"), ptrs("a_scale"), s)
    fvk.alpamayo15_w4a8_gemm_grouped(ptrs("xq"), ptrs("a_scale"), pack.data_ptr(), [out_ref[:, c:].data_ptr() for c in cols], M, list(Ns), K,
                                     out_ref.stride(0), pf, w, s, ks)
    torch.cuda.synchronize()
    xq_ref = [s_["xq"].clone() for s_ in sets]
    for s_ in sets:                                                                 # stale rotation outputs: must be recomputed
        s_["xr"].fill_(float("nan")); s_["part"].fill_(float("nan")); s_["xq"].fill_(7); s_["a_scale"].fill_(-1.0)
    fvk.flashdrive_w4a8_rotate(x.data_ptr(), x.stride(0), M, K, ptrs("theta"), ptrs("pairs"), ptrs("scales"), ptrs("xr"), ptrs("part"), s)
    region = torch.empty(1 << 20, device="cuda", dtype=torch.uint8)                       # an L2 prefetch region (any content)
    fvk.alpamayo15_w4a8_gemm_grouped_q(ptrs("xr"), ptrs("part"), ptrs("igs"), pack.data_ptr(), [out[:, c:].data_ptr() for c in cols], M,
                                       list(Ns), K, out.stride(0), pf, w, s, ks, [], [], region.data_ptr(), region.numel() - 128)
    torch.cuda.synchronize()
    assert torch.equal(out, out_ref) and not bool(torch.isnan(out).any())
    for s_, q_ref in zip(sets, xq_ref):
        assert bool((s_["xq"] == 7).all()) and bool((s_["a_scale"] == -1.0).all())   # the fused form leaves them alone
        assert q_ref[:M].abs().max() > 0


def test_w4a8_grouped_gemm_fused_quantization_rejects_unsupported_arguments():
    fvk = _kernels()
    K = 1024
    sets = [_rotation_set(K, 8) for _ in range(2)]
    ptrs = lambda key: [s_[key].data_ptr() for s_ in sets]
    pack = torch.zeros((272 + 1024) // 16, K // 128, 1088, device="cuda", dtype=torch.uint8)
    out = torch.zeros(8, 272 + 1024, device="cuda", dtype=BF16)
    call = lambda M, Ns, **kw: fvk.alpamayo15_w4a8_gemm_grouped_q(ptrs("xr")[:len(Ns)], ptrs("part")[:len(Ns)], ptrs("igs")[:len(Ns)],
                                                                 pack.data_ptr(), [out.data_ptr(), out[:, 272:].data_ptr()][:len(Ns)], M, list(Ns),
                                                                 K, out.stride(0), kw.get("prefetch", 2), kw.get("warps", 4), _stream(),
                                                                 kw.get("ksplit", 1))
    with pytest.raises(Exception, match="alpamayo15_w4a8_gemm_grouped_q"):
        call(5, (1024,))                               # more rows than the fused form serves
    with pytest.raises(Exception, match="alpamayo15_w4a8_gemm_grouped_q"):
        call(2, (272, 1024))                           # 17 tiles: the second problem does not start a CTA of 4 tiles
    with pytest.raises(Exception, match="alpamayo15_w4a8_gemm_grouped_q"):   # a prefetch region must be 128-byte aligned
        fvk.alpamayo15_w4a8_gemm_grouped_q(ptrs("xr")[:1], ptrs("part")[:1], ptrs("igs")[:1], pack.data_ptr(), [out.data_ptr()], 1, [1024],
                                           K, out.stride(0), 2, 4, _stream(), 1, [], [], pack.data_ptr() + 64, 4096)
    call(2, (272, 1024), warps=4, ksplit=4)            # one tile per CTA: fine
    torch.cuda.synchronize()


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


def test_vision_tower_cutlass_and_graph_match_torch_tower():
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
                          vision_attention="fa4", vision_gemm="cutlass", _vis_tables={}, _vis_graphs={})
    for name in ("_vision_tables", "_vision", "_vision_cutlass", "_merger", "_check_vision_modules", "_encode"):
        setattr(eng, name, getattr(Alpamayo15Thor, name).__get__(eng))
    eng._check_vision_modules()
    with torch.no_grad(), torch.autocast("cuda", dtype=BF16):
        ref_img, ref_deep = Alpamayo15Thor._vision(eng, px, grid)
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
