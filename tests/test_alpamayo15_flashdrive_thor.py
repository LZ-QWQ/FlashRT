"""Alpamayo 1.5 FlashDrive backend on Thor: contracts shared with FlashDrive's own rollout."""

from __future__ import annotations

import gc
import os
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers")

from flash_rt.models.alpamayo15_flashdrive.pipeline_thor import (  # noqa: E402
    FlashDriveThor,
    contiguous_runs,
    euler_cache_schedule,
    rope_rows,
    rotate_keys_fp32,
    sync_kvlm_all,
)

BF16 = torch.bfloat16


def _fd():
    return pytest.importorskip("flashdrive")


@pytest.mark.parametrize("steps,cache", [(8, [3, 4, 5, 6]), (10, []), (6, [2, 4])])
def test_cache_schedule_matches_euler_with_cache(steps, cache):
    _fd()
    from flashdrive.diffusion import euler_with_cache

    calls = []

    def step_fn(x, t):
        calls.append(round(float(t.flatten()[0]) * steps))
        return torch.ones_like(x)

    diffusion = SimpleNamespace(x_dims=(4, 2))
    euler_with_cache(diffusion, noise=torch.zeros(1, 4, 2), batch_size=1, step_fn=step_fn,
                     cache_steps=cache, device="cpu", inference_step=steps)
    fresh = [i for i, cached in enumerate(euler_cache_schedule(steps, cache)) if not cached]
    assert calls == fresh


def test_cache_schedule_rejects_invalid_steps():
    with pytest.raises(ValueError, match="step 0"):
        euler_cache_schedule(8, [0, 3])
    with pytest.raises(ValueError, match="outside"):
        euler_cache_schedule(8, [8])


def test_expert_key_rotation_matches_flashdrive_formula():
    torch.manual_seed(0)
    S, H, D = 37, 3, 128
    k = torch.randn(S, H, D).to(BF16)
    ang = torch.randn(S, D // 2)
    emb = torch.cat((ang, ang), dim=-1)
    cos, sin = emb.cos(), emb.sin()                                       # FP32 tables, as the expert's rotary
    got = rotate_keys_fp32(k, cos[:, None, :], sin[:, None, :])
    _fd()
    from flashdrive._backbone import apply_rotary_pos_emb_single
    # FlashDrive's own function on its [B, H, S, D] layout, (B, S, D) tables
    ref = apply_rotary_pos_emb_single(k.transpose(0, 1)[None], cos[None], sin[None])[0].transpose(0, 1)
    assert got.dtype == torch.float32
    assert torch.equal(got, ref)
    assert torch.equal(got.to(BF16), ref.to(BF16))


def _alpamayo15_layout(n_sys=40, n_views=4, n_frames=4, frame_len=6, gap=3, tail=9):
    """Token layout of an Alpamayo 1.5 prompt: system text, per view camera/frame labels between
    [VS..VE] frame blocks, then trajectory/text. Returns (frame_ranges, traj_text_range, P)."""
    pos, frame_ranges = n_sys, []
    for _ in range(n_views):
        view = []
        for _ in range(n_frames):
            pos += gap
            view.append((pos, pos + frame_len))
            pos += frame_len
        frame_ranges.append(view)
    return frame_ranges, (pos, pos + tail), pos + tail


def test_streaming_mask_is_causal_over_cache_positions():
    _fd()
    from flashdrive.streaming import create_streaming_attention_mask_sdpa

    frame_ranges, traj, P = _alpamayo15_layout()
    kv_len = P + 25
    mask = create_streaming_attention_mask_sdpa(frame_ranges=frame_ranges, traj_text_range=traj, kv_length=kv_len,
                                                valid_length=P, device="cpu", dtype=torch.float32,
                                                extended_vision_ranges=True)[0, 0]
    qpos = torch.cat([torch.arange(*v[-1]) for v in frame_ranges] + [torch.arange(*traj)])
    causal = torch.arange(kv_len)[None, :] <= qpos[:, None]
    assert torch.equal(mask == 0, causal)


def test_contiguous_runs_of_streaming_rows():
    assert contiguous_runs([3, 4, 5, 9, 10, 20]) == [(3, 6), (9, 11), (20, 21)]
    with pytest.raises(ValueError, match="increasing"):
        contiguous_runs([3, 5, 4])


def test_backend_requires_flashdrive_model():
    _fd()
    from flash_rt.models.alpamayo15_flashdrive.runtime_thor import FlashDriveThorRuntime
    with pytest.raises(TypeError, match="FlashDrive model"):
        FlashDriveThorRuntime(torch.nn.Linear(2, 2), expert="flashdrive")
    with pytest.raises(ValueError, match="unknown expert backend"):
        FlashDriveThorRuntime(torch.nn.Linear(2, 2), expert="sdpa")


def test_action_rejects_unsupported_requests():
    eng = object.__new__(FlashDriveThor)
    eng.m = SimpleNamespace(diffusion=SimpleNamespace(num_inference_steps=10))
    cp = torch.arange(5, 9)
    with pytest.raises(ValueError, match="euler_with_cache"):
        eng._action(total_samples=1, position_ids=None, attention_mask=None, cache_position=cp,
                    diffusion_kwargs={"int_method": "euler"})
    with pytest.raises(ValueError, match="one trajectory sample"):
        eng._action(total_samples=6, position_ids=None, attention_mask=None, cache_position=cp,
                    diffusion_kwargs={"int_method": "euler_with_cache", "cache_steps": [3]})


def test_resolver_entry_and_load_model_redirect():
    from flash_rt.hardware import _PIPELINE_MAP
    assert _PIPELINE_MAP[("alpamayo15_flashdrive", "torch", "thor")] == (
        "flash_rt.frontends.torch.alpamayo15_flashdrive_thor", "Alpamayo15FlashDriveTorchFrontendThor")
    import flash_rt
    with pytest.raises(NotImplementedError, match="Alpamayo15FlashDriveTorchFrontendThor"):
        flash_rt.load_model("unused", config="alpamayo15_flashdrive")


def test_rollout_defaults_match_flashdrive():
    _fd()
    import inspect

    from flashdrive._base import FlashDriveBaseMixin
    from flash_rt.frontends.torch.alpamayo15_flashdrive_thor import FlashDriveCheckpoints, FlashDriveRollout

    r = FlashDriveRollout()
    sig = inspect.signature(FlashDriveBaseMixin.sample_trajectories_streaming).parameters
    for name in ("max_new_tokens", "temperature", "top_p"):          # FlashDrive's own defaults
        assert getattr(r, name) == sig[name].default, name
    # scripts/infer.py: DIFFUSION_STEPS = 8, CACHE_STEPS = [3, 4, 5, 6], one trajectory sample per window
    assert (r.inference_step, list(r.cache_steps), r.num_traj_samples) == (8, [3, 4, 5, 6], 1)
    assert r.diffusion_kwargs() == {"inference_step": 8, "cache_steps": [3, 4, 5, 6], "int_method": "euler_with_cache"}
    c = FlashDriveCheckpoints()
    assert (c.paro, c.dflash) == (c.base + "-PARO", c.base + "-DFlash")


def test_reset_clears_flashdrive_stream_state():
    from flash_rt.models.alpamayo15_flashdrive.pipeline_thor import reset_flashdrive_stream

    calls = []
    model = SimpleNamespace(
        _past_key_values=object(), _cached_streaming_attention_mask=torch.zeros(1),
        _compiled_step_registry={"encode": object(), "dflash_traj_forward": object()},
        vlm=SimpleNamespace(model=SimpleNamespace(
            visual=SimpleNamespace(reset_shape_caches=lambda: calls.append("visual")),
            language_model=SimpleNamespace(reset_shape_caches=lambda: calls.append("lm")))))
    reset_flashdrive_stream(model)
    assert model._past_key_values is None and model._cached_streaming_attention_mask is None
    assert model._compiled_step_registry == {} and sorted(calls) == ["lm", "visual"]


# ---------------------------------------------------------------------------
# kernels (GPU)
# ---------------------------------------------------------------------------
from flash_rt.models.alpamayo15_flashdrive.runtime_thor import FLASHDRIVE_KERNEL_SYMBOLS  # noqa: E402


def _kernels():
    if not torch.cuda.is_available():
        pytest.skip("FlashDrive backend kernels require CUDA")
    try:
        import flash_rt.flash_rt_kernels as fvk
    except Exception as exc:  # pragma: no cover - build dependent
        pytest.skip(f"flash_rt_kernels is unavailable: {exc}")
    if not all(hasattr(fvk, n) for n in FLASHDRIVE_KERNEL_SYMBOLS):
        pytest.skip("built without -DFLASHRT_ENABLE_ALPAMAYO15=ON")
    return fvk


def test_flashdrive_symbol_inventory_is_atomic():
    try:
        import flash_rt.flash_rt_kernels as fvk
    except Exception as exc:  # pragma: no cover - build dependent
        pytest.skip(f"flash_rt_kernels not importable: {exc}")
    found = {n for n in dir(fvk) if n.startswith("flashdrive_")}
    if hasattr(fvk, "alpamayo15_rms_bf16"):       # same build gate as the Alpamayo 1.5 kernels
        assert found == set(FLASHDRIVE_KERNEL_SYMBOLS)
    else:
        assert found == set()


def _rot(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _head_norm(x, w, eps):  # Qwen3VLTextRMSNorm on BF16 activations
    xf = x.float()
    return w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(BF16)


def _bf16_close(a, e, max_frac=0.01):
    """Equal up to one BF16 ulp on a small fraction of elements (FP32 reduction order of the norm)."""
    a, e = a.float(), e.float()
    ulp = torch.where(e == 0, torch.full_like(e, 2.0 ** -133), e.abs() * 2.0 ** -7)
    d = (a - e).abs()
    assert bool((d <= ulp * 1.01).all()), f"max diff {d.max().item()}"
    assert float((d > 0).float().mean()) <= max_frac


def _fd_cache(L_S, Hk, D):  # FlashDrive StaticLayer layout [1, Hk, S, D]
    return torch.zeros(1, Hk, L_S, D, device="cuda", dtype=BF16)


@pytest.mark.parametrize("mode", ["row_map", "slot"])
def test_qk_prep_writes_unrotated_cache_and_rotated_copy(mode):
    fvk = _kernels()
    torch.manual_seed(11)
    rows, Hq, Hk, D, S, eps = 13, 32, 8, 128, 64, 1e-6
    q = (torch.randn(rows, Hq * D, device="cuda") * 2).to(BF16)
    kv = (torch.randn(rows, 2 * Hk * D + 64, device="cuda") * 2).to(BF16)   # k | v | pad: strided leading dim
    k, v = kv[:, :Hk * D], kv[:, Hk * D:2 * Hk * D]
    qw, kw = ((torch.rand(D, device="cuda") + 0.5).to(BF16) for _ in range(2))
    ang = torch.randn(rows, D // 2, device="cuda") * 3
    emb = torch.cat((ang, ang), -1)
    cos, sin = emb.cos().to(BF16).contiguous(), emb.sin().to(BF16).contiguous()
    if mode == "row_map":
        rmap = torch.tensor([2, 5, 6, 7, 20, 21, 22, 23, 24, 40, 41, 50, 63], device="cuda", dtype=torch.long)
        slot, cache_rows = None, rmap
    else:
        rmap, slot = None, torch.tensor([31], device="cuda", dtype=torch.long)
        cache_rows = torch.arange(31, 31 + rows, device="cuda")
    k_raw, v_c = _fd_cache(S, Hk, D), _fd_cache(S, Hk, D)
    k_rope = torch.zeros(S, Hk, D, device="cuda", dtype=BF16)                  # attention copy [S, Hk, D]
    q_out = torch.empty(rows, Hq, D, device="cuda", dtype=BF16)
    fvk.flashdrive_qk_prep_bf16(q.data_ptr(), k.data_ptr(), v.data_ptr(), Hq * D, kv.shape[1], kv.shape[1],
                                qw.data_ptr(), kw.data_ptr(), cos.data_ptr(), sin.data_ptr(), 0, q_out.data_ptr(),
                                k_raw.data_ptr(), D, S * D, k_rope.data_ptr(), Hk * D, D, v_c.data_ptr(), D, S * D,
                                rmap.data_ptr() if rmap is not None else 0, slot.data_ptr() if slot is not None else 0,
                                rows, Hq, Hk, eps, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    from flashdrive._backbone import apply_rotary_pos_emb_single  # FlashDrive's own rotation (BF16 tables)
    n_q = _head_norm(q.view(rows, Hq, D), qw, eps)
    n_k = _head_norm(k.view(rows, Hk, D), kw, eps)
    rope = lambda n: apply_rotary_pos_emb_single(n.transpose(0, 1)[None], cos[None], sin[None])[0].transpose(0, 1)
    _bf16_close(q_out, rope(n_q))
    got_raw = k_raw[0].transpose(0, 1)[cache_rows]                              # [rows, Hk, D]
    _bf16_close(got_raw, n_k)
    assert torch.equal(k_rope[cache_rows], rope(got_raw))                       # rotation of what was cached
    assert torch.equal(v_c[0].transpose(0, 1)[cache_rows], v.view(rows, Hk, D))
    untouched = torch.ones(S, dtype=torch.bool, device="cuda"); untouched[cache_rows] = False
    assert not k_raw[0].transpose(0, 1)[untouched].any() and not k_rope[untouched].any()


@pytest.mark.parametrize("fp32_rule", [0, 1])
def test_rope_rows_matches_flashdrive_rotation(fp32_rule):
    fvk = _kernels()
    torch.manual_seed(12)
    S, Hk, D = 96, 8, 128
    k_raw = (torch.randn(1, Hk, S, D, device="cuda") * 2).to(BF16)
    ang = torch.randn(S, D // 2, device="cuda") * 3
    emb = torch.cat((ang, ang), -1)
    tab = torch.float32 if fp32_rule else BF16
    cos, sin = emb.cos().to(tab).contiguous(), emb.sin().to(tab).contiguous()
    from flashdrive._backbone import apply_rotary_pos_emb_single
    ref = apply_rotary_pos_emb_single(k_raw, cos[None], sin[None])[0].transpose(0, 1).to(BF16)   # [S, Hk, D]
    out = torch.zeros(S, Hk, D, device="cuda", dtype=BF16)
    rows = torch.tensor([0, 7, 8, 33, 95], device="cuda", dtype=torch.long)
    fvk.flashdrive_rope_rows(k_raw.data_ptr(), D, S * D, out.data_ptr(), Hk * D, D, cos.data_ptr(), sin.data_ptr(),
                             fp32_rule, rows.data_ptr(), 0, 0, 0, rows.numel(), Hk, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    assert torch.equal(out[rows], ref[rows])
    mask = torch.ones(S, dtype=torch.bool, device="cuda"); mask[rows] = False
    assert not out[mask].any()
    # contiguous range from a device slot, bounded by a device row count
    out.zero_()
    slot = torch.tensor([10], device="cuda", dtype=torch.long)
    n = torch.tensor([20], device="cuda", dtype=torch.int32)
    fvk.flashdrive_rope_rows(k_raw.data_ptr(), D, S * D, out.data_ptr(), Hk * D, D, cos.data_ptr(), sin.data_ptr(),
                             fp32_rule, 0, 5, slot.data_ptr(), n.data_ptr(), 64, Hk, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    assert torch.equal(out[15:35], ref[15:35])
    assert not out[:15].any() and not out[35:].any()


def test_qk_prep_fp32_rule_matches_expert_autocast_rotation():
    fvk = _kernels()
    torch.manual_seed(13)
    rows, Hq, Hk, D, S, eps = 64, 16, 8, 128, 96, 1e-6
    qkv = (torch.randn(rows, (Hq + 2 * Hk) * D, device="cuda") * 2).to(BF16)   # FlashDrive's fused expert projection
    ld = qkv.shape[1]
    qw, kw = ((torch.rand(D, device="cuda") + 0.5).to(BF16) for _ in range(2))
    ang = torch.randn(rows, D // 2, device="cuda") * 3
    emb = torch.cat((ang, ang), -1)
    cos, sin = emb.cos().contiguous(), emb.sin().contiguous()                   # FP32 expert tables
    slot = torch.tensor([17], device="cuda", dtype=torch.long)
    k_raw, v_c = _fd_cache(S, Hk, D), _fd_cache(S, Hk, D)
    k_rope = torch.zeros(S, Hk, D, device="cuda", dtype=BF16)
    q_out = torch.empty(rows, Hq, D, device="cuda", dtype=BF16)
    fvk.flashdrive_qk_prep_bf16(qkv.data_ptr(), qkv[:, Hq * D:].data_ptr(), qkv[:, (Hq + Hk) * D:].data_ptr(), ld, ld, ld,
                                qw.data_ptr(), kw.data_ptr(), cos.data_ptr(), sin.data_ptr(), 1, q_out.data_ptr(),
                                k_raw.data_ptr(), D, S * D, k_rope.data_ptr(), Hk * D, D, v_c.data_ptr(), D, S * D,
                                0, slot.data_ptr(), rows, Hq, Hk, eps, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    from flashdrive._backbone import apply_rotary_pos_emb_single
    rope = lambda n: apply_rotary_pos_emb_single(n.transpose(0, 1)[None], cos[None], sin[None])[0].transpose(0, 1).to(BF16)
    n_q = _head_norm(qkv[:, :Hq * D].view(rows, Hq, D), qw, eps)
    _bf16_close(q_out, rope(n_q))
    got_raw = k_raw[0].transpose(0, 1)[17:17 + rows]
    _bf16_close(got_raw, _head_norm(qkv[:, Hq * D:(Hq + Hk) * D].view(rows, Hk, D), kw, eps))
    assert torch.equal(k_rope[17:17 + rows], rope(got_raw))
    assert torch.equal(v_c[0].transpose(0, 1)[17:17 + rows], qkv[:, (Hq + Hk) * D:].view(rows, Hk, D))


@pytest.mark.parametrize("rows", [1, 8, 802])
def test_fp16_input_consumers_equal_cast_then_bf16_kernels(rows):
    """The fused W4A8 path feeds Marlin's FP16 outputs straight into qk prep, the residual RMSNorm and SiLU-mul;
    each must equal FlashDrive's .to(torch.bfloat16) cast followed by the BF16 kernel, bit for bit."""
    fvk = _kernels()
    torch.manual_seed(14)
    Hq, Hk, D, S, H, F_, eps = 32, 8, 128, 1024, 4096, 12288, 1e-6
    st = torch.cuda.current_stream().cuda_stream
    bits = lambda t: t.view(torch.int16)
    f16 = lambda *shape: (torch.randn(*shape, device="cuda") * 40).to(torch.float16)  # finer than BF16 precision
    q, k, v = f16(rows, Hq * D), f16(rows, Hk * D), f16(rows, Hk * D)
    qw, kw = ((torch.rand(D, device="cuda") + 0.5).to(BF16) for _ in range(2))
    ang = torch.randn(rows, D // 2, device="cuda") * 3
    emb = torch.cat((ang, ang), -1)
    cos, sin = emb.cos().to(BF16).contiguous(), emb.sin().to(BF16).contiguous()
    slot = torch.tensor([3], device="cuda", dtype=torch.long)
    outs = []
    for f16_in, (qi, ki, vi) in ((1, (q, k, v)), (0, (q.to(BF16), k.to(BF16), v.to(BF16)))):
        o = {"q": torch.empty(rows, Hq, D, device="cuda", dtype=BF16), "kr": _fd_cache(S, Hk, D),
             "k": torch.zeros(S, Hk, D, device="cuda", dtype=BF16), "v": _fd_cache(S, Hk, D)}
        fvk.flashdrive_qk_prep_bf16(qi.data_ptr(), ki.data_ptr(), vi.data_ptr(), Hq * D, Hk * D, Hk * D,
                                    qw.data_ptr(), kw.data_ptr(), cos.data_ptr(), sin.data_ptr(), 0, o["q"].data_ptr(),
                                    o["kr"].data_ptr(), D, S * D, o["k"].data_ptr(), Hk * D, D, o["v"].data_ptr(), D,
                                    S * D, 0, slot.data_ptr(), rows, Hq, Hk, eps, st, f16_in=f16_in)
        outs.append(o)
    for name in outs[0]:
        assert torch.equal(bits(outs[0][name]), bits(outs[1][name])), name

    x0 = (torch.randn(rows, H, device="cuda") * 3).to(BF16)
    d16 = f16(rows, H)
    w = (torch.rand(H, device="cuda") + 0.5).to(BF16)
    xa, xb = x0.clone(), x0.clone()
    ha, hb = torch.empty_like(x0), torch.empty_like(x0)
    fvk.flashdrive_rms_f16delta_bf16(xa.data_ptr(), d16.data_ptr(), w.data_ptr(), ha.data_ptr(), rows, H, eps, st)
    d_bf = d16.to(BF16)
    fvk.alpamayo15_rms_bf16(xb.data_ptr(), d_bf.data_ptr(), w.data_ptr(), hb.data_ptr(), rows, H, eps, st)
    torch.cuda.synchronize()
    assert torch.equal(bits(xa), bits(xb)) and torch.equal(bits(ha), bits(hb))

    g, u = f16(rows, F_) / 8, f16(rows, F_) / 8
    g[0, :4] = torch.tensor([0.0, -0.0, 65504.0, -65504.0], dtype=torch.float16)
    aa, ab = (torch.empty(rows, F_, device="cuda", dtype=BF16) for _ in range(2))
    fvk.flashdrive_silu_mul_f16in_bf16(g.data_ptr(), u.data_ptr(), aa.data_ptr(), rows * F_, st)
    gb, ub = g.to(BF16), u.to(BF16)
    fvk.silu_mul_qwen36_bf16(gb.data_ptr(), ub.data_ptr(), ab.data_ptr(), rows * F_, st)
    torch.cuda.synchronize()
    assert torch.equal(bits(aa), bits(ab))
    ref = (torch.nn.functional.silu(gb) * ub)                                  # FlashDrive's act_fn(gate) * up
    assert torch.equal(bits(aa[torch.isfinite(ref)]), bits(ref[torch.isfinite(ref)]))


@pytest.mark.parametrize("rows,K", [(1, 4096), (5, 4096), (802, 4096), (8, 12288)])
def test_w4a8_act_prep_bit_exact_with_paroquant_and_vllm(rows, K):
    fvk = _kernels()
    pytest.importorskip("paroquant")
    pytest.importorskip("vllm")
    import paroquant.kernels.cuda  # noqa: F401  (registers torch.ops.rotation.rotate)
    from vllm.model_executor.layers.quantization.utils.int8_utils import per_token_quant_int8

    torch.manual_seed(14)
    x = (torch.randn(rows, K, device="cuda") * 3).to(BF16)
    x[0, :7] = torch.tensor([0.0, 1e-6, -2e-5, 65000.0, -3.0e4, 1e-8, 2.5], device="cuda").to(BF16)   # edge values
    sets = []
    for _ in range(3):
        theta = (torch.randn(8, K // 2, device="cuda") * 2).to(torch.float16)
        perm = torch.stack([torch.cat([torch.randperm(128) for _ in range(K // 128)]) for _ in range(8)])
        pairs = perm.to(torch.int16).cuda()                    # 64 disjoint in-group pairs per layer and group
        scales = (torch.rand(1, K, device="cuda") + 0.5).to(torch.float16)
        igs = torch.tensor(6.5e-6, device="cuda", dtype=torch.float32)
        sets.append((theta, pairs, scales, igs))
    xq = [torch.empty(rows, K, device="cuda", dtype=torch.int8) for _ in sets]
    a_s = [torch.empty(rows, device="cuda", dtype=torch.float32) for _ in sets]
    xr = [torch.empty(rows, K, device="cuda", dtype=torch.float16) for _ in sets]
    part = [torch.empty(rows, K // 128, device="cuda", dtype=torch.float32) for _ in sets]
    p = lambda ts: [t.data_ptr() for t in ts]
    fvk.flashdrive_w4a8_act_prep(x.data_ptr(), K, rows, K, p(s[0] for s in sets), p(s[1] for s in sets),
                                 p(s[2] for s in sets), p(s[3] for s in sets), p(xr), p(part), p(xq), p(a_s),
                                 torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    for (theta, pairs, scales, igs), q, a, r in zip(sets, xq, a_s, xr):
        x_rot = torch.ops.rotation.rotate(x.to(torch.float16), pairs, theta, scales, 128)
        q_ref, s_ref = per_token_quant_int8(x_rot)
        assert torch.equal(q, q_ref)
        assert torch.equal(a, (s_ref * igs).view(-1))
        assert torch.equal(r.view(torch.int16), x_rot.view(torch.int16))   # FP16 workspace, bitwise (inf/NaN rows)


@pytest.mark.parametrize("mode,rows,K,tile,cluster", [
    (0, 8, 4096, 0, 0), (1, 8, 4096, 0, 0), (1, 1, 4096, 16, 2), (2, 8, 12288, 0, 0), (1, 16, 4096, 8, 4),
    (1, 9, 3584, 0, 0), (2, 9, 18944, 0, 0),            # verification block: one tile spread over a cluster
    (0, 40, 3584, 16, 2), (1, 40, 3584, 8, 1), (1, 37, 3584, 16, 7), (2, 40, 18944, 16, 0), (2, 33, 18944, 8, 0)])
def test_act_prep_tile_matches_separate_kernels(mode, rows, K, tile, cluster):
    """The fused tile producer + preparation equals the separate kernels bitwise: producer (plain / rms_f16delta /
    silu_mul_f16in) followed by the two-pass ParoQuant preparation; every tile size and cluster shape."""
    fvk = _kernels()
    torch.manual_seed(28)
    s_ = torch.cuda.current_stream().cuda_stream
    n_sets = 3 if mode == 1 else 1
    sets = []
    for _ in range(n_sets):
        theta = (torch.randn(8, K // 2, device="cuda") * 2).to(torch.float16)
        perm = torch.stack([torch.cat([torch.randperm(128) for _ in range(K // 128)]) for _ in range(8)])
        sets.append((theta, perm.to(torch.int16).cuda(), (torch.rand(1, K, device="cuda") + 0.5).to(torch.float16),
                     torch.tensor(6.5e-6, device="cuda", dtype=torch.float32)))
    p = lambda ts: [t.data_ptr() for t in ts]
    eps = 1e-6
    Ra = (rows + 31) // 32 * 32
    w = (torch.rand(K, device="cuda") + 0.5).to(BF16)
    res = torch.randn(rows, K, device="cuda").to(BF16)
    delta = (torch.randn(rows, K, device="cuda") * 0.3).to(torch.float16)
    gate = (torch.randn(rows, K, device="cuda") * 2).to(torch.float16)
    up = torch.randn(rows, K, device="cuda").to(torch.float16)
    x0 = torch.randn(rows, K, device="cuda").to(BF16)
    # reference: separate kernels
    res_ref = res.clone()
    if mode == 0:
        h = x0
    elif mode == 1:
        h = torch.empty_like(res_ref)
        fvk.flashdrive_rms_f16delta_bf16(res_ref.data_ptr(), delta.data_ptr(), w.data_ptr(), h.data_ptr(), rows, K, eps, s_)
    else:
        h = torch.empty(rows, K, device="cuda", dtype=BF16)
        fvk.flashdrive_silu_mul_f16in_bf16(gate.data_ptr(), up.data_ptr(), h.data_ptr(), rows * K, s_)
    xq = [torch.zeros(Ra, K, device="cuda", dtype=torch.int8) for _ in sets]
    a_s = [torch.zeros(Ra, device="cuda", dtype=torch.float32) for _ in sets]
    xr = [torch.empty(rows, K, device="cuda", dtype=torch.float16) for _ in sets]
    part = [torch.empty(rows, K // 128, device="cuda", dtype=torch.float32) for _ in sets]
    fvk.flashdrive_w4a8_act_prep(h.data_ptr(), K, rows, K, p(t[0] for t in sets), p(t[1] for t in sets), p(t[2] for t in sets),
                                 p(t[3] for t in sets), p(xr), p(part), p(xq), p(a_s), s_)
    # fused tile kernel
    res_f = res.clone()
    xq2 = [torch.zeros(Ra, K, device="cuda", dtype=torch.int8) for _ in sets]
    a_s2 = [torch.zeros(Ra, device="cuda", dtype=torch.float32) for _ in sets]
    src = {0: x0, 1: delta, 2: gate}[mode]
    fvk.flashdrive_w4a8_act_prep_tile(mode, src.data_ptr(), up.data_ptr() if mode == 2 else 0, K, res_f.data_ptr() if mode == 1 else 0,
                                      K, w.data_ptr() if mode == 1 else 0, eps, rows, K, p(t[0] for t in sets), p(t[1] for t in sets),
                                      p(t[2] for t in sets), p(t[3] for t in sets), p(xq2), p(a_s2), tile, cluster, s_)
    torch.cuda.synchronize()
    for q, q2, a, a2 in zip(xq, xq2, a_s, a_s2):
        assert torch.equal(q, q2)
        assert torch.equal(a.view(torch.int32), a2.view(torch.int32))
    if mode == 1:
        assert torch.equal(res_f, res_ref)
        # without a delta: plain RMSNorm of the residual (rms_bf16 contract)
        h2 = torch.empty_like(res)
        fvk.alpamayo15_rms_bf16(res.clone().data_ptr(), 0, w.data_ptr(), h2.data_ptr(), rows, K, eps, s_)
        fvk.flashdrive_w4a8_act_prep(h2.data_ptr(), K, rows, K, p(t[0] for t in sets), p(t[1] for t in sets), p(t[2] for t in sets),
                                     p(t[3] for t in sets), p(xr), p(part), p(xq), p(a_s), s_)
        r3 = res.clone()
        fvk.flashdrive_w4a8_act_prep_tile(1, 0, 0, K, r3.data_ptr(), K, w.data_ptr(), eps, rows, K, p(t[0] for t in sets),
                                          p(t[1] for t in sets), p(t[2] for t in sets), p(t[3] for t in sets), p(xq2), p(a_s2),
                                          tile, cluster, s_)
        torch.cuda.synchronize()
        assert torch.equal(r3, res)
        for q, q2 in zip(xq, xq2):
            assert torch.equal(q, q2)


@pytest.mark.parametrize("K", [3584, 18944])
def test_bank_ordered_pair_tables_keep_the_rotation(K):
    """paro_pair_bank_order permutes the pairs of every (layer, group) only; the preparation on the reordered tables is
    bitwise that on the original ones, and each quarter-warp's eight first channels hit eight distinct bank groups
    whenever a residue class has at most eight members."""
    from flash_rt.models.alpamayo15_flashdrive.weights import paro_pair_bank_order
    fvk = _kernels()
    torch.manual_seed(31)
    theta = (torch.randn(8, K // 2, device="cuda") * 2).to(torch.float16)
    perm = torch.stack([torch.cat([torch.randperm(128) for _ in range(K // 128)]) for _ in range(8)])
    pairs = perm.to(torch.int16).cuda()
    th2, pr2 = paro_pair_bank_order(theta, pairs, K)
    assert th2.shape == theta.shape and pr2.shape == pairs.shape and pr2.dtype == pairs.dtype
    a = torch.cat([pairs.view(8, -1, 64, 2).long(), theta.view(8, -1, 64, 1).view(torch.int16).long()], -1)
    b = torch.cat([pr2.view(8, -1, 64, 2).long(), th2.view(8, -1, 64, 1).view(torch.int16).long()], -1)
    key = lambda x: (x[..., 0] * 256 + x[..., 1]) * 65536 + (x[..., 2] & 0xFFFF)
    assert torch.equal(key(a).sort(-1).values, key(b).sort(-1).values)        # same (i, j, theta) multiset per layer/group
    i = pr2.view(8, -1, 64, 2)[..., 0].long()
    res = ((i + (i >> 3)) & 7).view(8, -1, 8, 8)                              # [layer, group, bucket, lane]
    distinct = (torch.nn.functional.one_hot(res, 8).sum(-2) <= 1).all(-1)    # bucket conflict-free
    assert distinct.float().mean() > 0.6                                      # overflow pairs only where a class exceeds 8
    rows = 40
    x = torch.randn(rows, K, device="cuda").to(BF16)
    scales = (torch.rand(1, K, device="cuda") + 0.5).to(torch.float16)
    igs = torch.tensor(6.5e-6, device="cuda", dtype=torch.float32)
    outs = []
    for th, pr in ((theta, pairs), (th2, pr2)):
        xq = torch.zeros(64, K, device="cuda", dtype=torch.int8)
        a_s = torch.zeros(64, device="cuda", dtype=torch.float32)
        fvk.flashdrive_w4a8_act_prep_tile(0, x.data_ptr(), 0, K, 0, K, 0, 0.0, rows, K, [th.data_ptr()], [pr.data_ptr()],
                                          [scales.data_ptr()], [igs.data_ptr()], [xq.data_ptr()], [a_s.data_ptr()], 0, 0,
                                          torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        outs.append((xq, a_s))
    assert torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])


def test_shift_cache_matches_flashdrive_shift_and_rotation():
    """One launch moves every layer's frame blocks f+1 -> f (FlashDrive's gather + scatter) and rotates the
    moved keys into the VLM view (BF16 rule at the destination rows)."""
    fvk = _kernels()
    torch.manual_seed(16)
    L, S, Hk, D = 3, 700, 8, 128
    frames = [[(10, 40), (40, 70), (70, 100), (100, 130)], [(140, 165), (165, 190), (190, 215), (215, 240)],
              [(250, 300), (300, 350), (350, 400), (400, 450)], [(460, 470), (470, 480), (480, 490), (490, 500)]]
    src = torch.cat([torch.arange(s0, s1) for v in frames for (_, _), (s0, s1) in zip(v[:-1], v[1:])]).cuda()
    dst = torch.cat([torch.arange(d0, d1) for v in frames for (d0, d1), (_, _) in zip(v[:-1], v[1:])]).cuda()
    k = [torch.randn(1, Hk, S, D, device="cuda").to(BF16) for _ in range(L)]
    v = [torch.randn(1, Hk, S, D, device="cuda").to(BF16) for _ in range(L)]
    cos = torch.randn(S, D, device="cuda").to(BF16)
    sin = torch.randn(S, D, device="cuda").to(BF16)
    kvlm = torch.randn(L, S, Hk, D, device="cuda").to(BF16)
    k_ref = [t.clone() for t in k]
    v_ref = [t.clone() for t in v]
    kvlm_ref = kvlm.clone()
    for t in k_ref + v_ref:
        t.index_copy_(2, dst, t.index_select(2, src))
    for li in range(L):
        kr = k_ref[li][0].permute(1, 0, 2)                      # [S, Hk, D]
        rows = kr[dst].float()
        c, s_ = cos[dst].float().unsqueeze(1), sin[dst].float().unsqueeze(1)
        half = D // 2
        rot = torch.cat((-rows[..., half:], rows[..., :half]), dim=-1)
        kvlm_ref[li, dst] = ((rows * c).to(BF16).float() + (rot * s_).to(BF16).float()).to(BF16)
    ranges = torch.tensor(frames, dtype=torch.int32, device="cuda")
    kp = torch.tensor([t.data_ptr() for t in k], dtype=torch.int64, device="cuda")
    vp = torch.tensor([t.data_ptr() for t in v], dtype=torch.int64, device="cuda")
    fvk.flashdrive_shift_cache(kp.data_ptr(), vp.data_ptr(), D, S * D, kvlm.data_ptr(), cos.data_ptr(), sin.data_ptr(),
                               ranges.data_ptr(), len(frames), 4, L, S, Hk, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    for a, b in zip(k + v, k_ref + v_ref):
        assert torch.equal(a, b)
    assert torch.equal(kvlm, kvlm_ref)


def test_fa4_call_patterns_match_flashdrive_masks():
    """The backend's three FA4 calls against SDPA with FlashDrive's masks: streaming rows causal by cache position
    (shared-prefix varlen), verification rows causal over [0, cur + 8), the unmasked trajectory-start row. Each
    error must be far below the change an off-by-one key length produces."""
    _kernels()
    from flash_rt.hardware.thor import fa4_backend
    fa4 = fa4_backend.fa4_fwd()
    if fa4 is None:
        pytest.skip(f"FA4 unavailable: {fa4_backend.status()}")
    torch.manual_seed(17)
    S, Hq, Hk, D = 96, 32, 8, 128
    eng = SimpleNamespace(rt=SimpleNamespace(fa4=fa4, n_heads=Hq, head_dim=D, scale=D ** -0.5),
                          _v_view=lambda st, li: st["v"][li].permute(0, 2, 1, 3))
    eng._attend_runs = lambda *args: FlashDriveThor._attend_runs(eng, *args)
    kvlm = torch.randn(1, S, Hk, D, device="cuda").to(BF16)                      # rotated keys [S, Hk, D]
    v = torch.randn(1, Hk, S, D, device="cuda").to(BF16)                         # FlashDrive's value cache
    st = {"kvlm": kvlm, "v": [v], "S": S, "keep": {}}

    def sdpa(q, pos, shift=0):          # q [n, Hq, D]; row i attends keys [0, pos[i] + shift]
        k_ = kvlm[0].float().repeat_interleave(Hq // Hk, 1).transpose(0, 1)     # [Hq, S, D]
        v_ = v[0].float().repeat_interleave(Hq // Hk, 0)                       # [Hq, S, D]
        sc = torch.einsum("nhd,hsd->hns", q.float(), k_) * D ** -0.5
        keep = torch.arange(S, device="cuda")[None, :] <= (torch.tensor(pos, device="cuda") + shift)[:, None]
        sc = sc.masked_fill(~keep[None], float("-inf"))
        return torch.einsum("hns,hsd->nhd", sc.softmax(-1), v_).reshape(len(pos), Hq * D)

    def check(out, q, pos):
        ref = sdpa(q, pos)
        err = (out.float() - ref).abs().mean()
        off = min((sdpa(q, pos, -1) - ref).abs().mean(), (sdpa(q, pos, 1) - ref).abs().mean())
        assert err < 0.1 * off, (float(err), float(off))

    runs = [(10, 20), (40, 52), (70, 86)]                                        # a streaming window's rows
    pos = [r for a, b in runs for r in range(a, b)]
    lens = [b - a for a, b in runs]
    st["keep"]["prefill_meta"] = {
        "cu_q": torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32, device="cuda"),
        "cu_k": torch.zeros(len(runs) + 1, dtype=torch.int32, device="cuda"),
        "used": torch.tensor([b for _, b in runs], dtype=torch.int32, device="cuda"), "max_q": max(lens)}
    st["kvlm"] = kvlm[0][None]                                                   # [layers, S, Hk, D]
    q = torch.randn(len(pos), Hq, D, device="cuda").to(BF16)
    out = torch.empty(len(pos), Hq * D, device="cuda", dtype=BF16)
    FlashDriveThor._attend_prefill(eng, st, 0, q.view(len(pos), -1), out)
    check(out, q, pos)

    cur = 50                                                                     # verification block
    st["keep"]["dec"] = {"used": torch.tensor([cur + 8], dtype=torch.int32, device="cuda")}
    q = torch.randn(8, Hq, D, device="cuda").to(BF16)
    out = torch.empty(8, Hq * D, device="cuda", dtype=BF16)
    FlashDriveThor._attend_verify(eng, st, 0, q.view(8, -1), out)
    check(out, q, list(range(cur, cur + 8)))

    q = torch.randn(1, Hq, D, device="cuda").to(BF16)                           # trajectory-start row: no mask
    out = torch.empty(1, Hq * D, device="cuda", dtype=BF16)
    FlashDriveThor._attend_all_rows(eng, st, 0, q.view(1, -1), out)
    ref = sdpa(q, [S - 1])
    assert float((out.float() - ref).abs().mean()) < 0.1 * float((sdpa(q, [S - 1], -1) - ref).abs().mean())


def _marlin_w4a8_buffers(K, N, seed):
    """Random ParoQuant-style Marlin W4A8 buffers packed by vLLM's own layout functions."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_permute_scales, marlin_zero_points
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import get_weight_perm, marlin_weights
    g = torch.Generator().manual_seed(seed)
    G = K // 128
    w = torch.randint(0, 16, (K, N), generator=g)
    z = torch.randint(0, 16, (G, N), generator=g)
    s = torch.randint(1, 4097, (G, N), generator=g).to(torch.int16)
    s[0, 0] = 4096
    qweight = marlin_weights(w, K, N, 4, get_weight_perm(4, is_a_8bit=True), is_a_8bit=True).cuda()
    scales = marlin_permute_scales(s.view(torch.float16), K, N, 128, is_a_8bit=True).cuda()
    qzeros = marlin_zero_points(z, G, N, 4, is_a_8bit=True).cuda()
    return w, z, s, qweight, scales, qzeros


@pytest.mark.parametrize("M,N,K", [(300, 384, 512), (802, 1024, 1024)])
def test_w4a8_int8_gemm_is_exact_and_matches_marlin(M, N, K):
    fvk = _kernels()
    pytest.importorskip("vllm")
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_empty_g_idx, marlin_make_workspace_new)
    from vllm.scalar_type import scalar_types
    from flash_rt.models.alpamayo15_flashdrive.weights import marlin_w4a8_to_int8
    w, z, s, qweight, scales, qzeros = _marlin_w4a8_buffers(K, N, seed=15)
    w8, sg = marlin_w4a8_to_int8(qweight, scales, qzeros, K, N)
    wd = w - z.repeat_interleave(128, 0)                                       # [K, N]
    assert torch.equal(w8.cpu(), wd.t().to(torch.int8)) and torch.equal(sg.cpu(), s.to(torch.int32))

    torch.manual_seed(16)
    xq = torch.randint(-127, 128, (M, K), dtype=torch.int8, device="cuda")
    # exact integer total (float64 holds it exactly); per-token scales put each row's outputs up to ~1e4 (FP16 range)
    total = xq.double() @ (wd.double() * s.double().repeat_interleave(128, 0)).cuda()
    a_s = (1e4 * (torch.rand(M, 1, device="cuda", dtype=torch.float64) * 0.9 + 0.1)
           / total.abs().amax(dim=1, keepdim=True)).float().contiguous()
    nbytes, n_ones = fvk.flashdrive_w4a8_gemm_sm100_sizes(M, N, K)
    ws = torch.empty(max(int(nbytes), 1), dtype=torch.uint8, device="cuda")
    ones = torch.ones(int(n_ones), dtype=torch.int32, device="cuda")
    out = torch.empty(M, N, dtype=torch.float16, device="cuda")
    fvk.flashdrive_w4a8_gemm_sm100(xq.data_ptr(), w8.data_ptr(), sg.data_ptr(), ones.data_ptr(), a_s.data_ptr(),
                                   out.data_ptr(), M, N, K, ws.data_ptr(), ws.numel(),
                                   torch.cuda.current_stream().cuda_stream)
    ref = (total.float() * a_s).half()          # one conversion of the total to float, the per-token scale, FP16
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))

    dev = torch.device("cuda")
    marlin = ops.marlin_gemm(xq, None, qweight, None, scales, a_s, None, qzeros, marlin_make_empty_g_idx(dev),
                             marlin_make_empty_g_idx(dev), marlin_make_workspace_new(dev), scalar_types.uint4,
                             size_m=M, size_n=N, size_k=K, use_atomic_add=False, use_fp32_reduce=True,
                             is_zp_float=False)
    # Marlin sums FP32 partials of K slices: one FP16 ulp plus that sum's rounding (cancellation) apart at most
    r = marlin.float()
    tol = (r.abs() * 2.0 ** -10).clamp_min(2.0 ** -24) + r.abs().amax(dim=1, keepdim=True) * 2.0 ** -20
    assert bool(((out.float() - r).abs() <= tol).all())
    assert float((out == marlin).float().mean()) > 0.999


# ---------------------------------------------------------------------------
# backend vs FlashDrive on a short synthetic stream (opt-in: real checkpoints)
# ---------------------------------------------------------------------------
_CKPT = os.environ.get("FLASHRT_FLASHDRIVE_CKPT")   # dir holding Alpamayo-1.5-10B{,-PARO,-DFlash}


@pytest.fixture
def _release_model():
    """Each case loads the full checkpoint; the backend and the model reference each other, so free them before
    the next case loads another copy."""
    yield
    gc.collect()
    torch.cuda.empty_cache()


_needs_ckpt = pytest.mark.skipif(not _CKPT, reason="set FLASHRT_FLASHDRIVE_CKPT to the z-lab checkpoint directory")


@pytest.mark.parametrize("M,N,K", [(1, 1024, 4096), (8, 4096, 4096), (7, 1024, 12288), (16, 2048, 4096), (13, 256, 1024)])
def test_w4a8_small_gemm_is_exact(M, N, K):
    """The few-row W4A8 GEMM computes Marlin's integers exactly (float64 reference) and agrees with the tcgen05
    kernel (checked against Marlin itself in the previous test) on the same activations."""
    fvk = _kernels()
    from flash_rt.models.alpamayo15.marlin_w4a8 import pack_w4a8_small
    torch.manual_seed(17)
    G = K // 128
    w = torch.randint(0, 16, (K, N), device="cuda", dtype=torch.uint8)
    z = torch.randint(0, 16, (G, N), device="cuda", dtype=torch.uint8)
    sg = torch.randint(0, 4097, (G, N), device="cuda", dtype=torch.int32)
    Ma = (M + 7) // 8 * 8
    xq = torch.randint(-128, 128, (Ma, K), device="cuda", dtype=torch.int8)
    a_s = (torch.rand(M, device="cuda") * 2e-2 + 1e-3) * 7e-6
    out = torch.empty(M, N, device="cuda", dtype=torch.float16)
    fvk.flashdrive_w4a8_gemm_small(xq.data_ptr(), a_s.data_ptr(), pack_w4a8_small(w, K, N).data_ptr(), z.data_ptr(),
                                   sg.data_ptr(), out.data_ptr(), M, N, K, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    wz = (w.to(torch.int64) - z.to(torch.int64).repeat_interleave(128, 0))            # [K, N]
    part = torch.einsum("mgk,gkn->mgn", xq[:M].to(torch.float64).view(M, G, 128), wz.to(torch.float64).view(G, 128, N))
    total = (part * sg.to(torch.float64).unsqueeze(0)).sum(1)                           # exact in float64 (< 2^53)
    assert torch.equal(total, total.round())
    ref = (total.to(torch.int64).to(torch.float32) * a_s.unsqueeze(1)).to(torch.float16)
    assert torch.equal(out, ref)
    out_t = torch.empty(M, N, device="cuda", dtype=torch.float16)          # streaming warp-per-tile variant
    fvk.flashdrive_w4a8_gemm_tile(xq.data_ptr(), a_s.data_ptr(), pack_w4a8_small(w, K, N).data_ptr(), z.data_ptr(),
                                  sg.data_ptr(), out_t.data_ptr(), M, N, K, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    assert torch.equal(out_t, ref)
    if M >= 1 and N % 8 == 0:
        w8 = wz.to(torch.int8).t().contiguous()
        nbytes, n_ones = fvk.flashdrive_w4a8_gemm_sm100_sizes(M, N, K)
        ws = torch.empty(max(int(nbytes), 1), dtype=torch.uint8, device="cuda")
        ones = torch.ones(int(n_ones) + 1024, dtype=torch.int32, device="cuda")
        out2 = torch.empty(M, N, device="cuda", dtype=torch.float16)
        fvk.flashdrive_w4a8_gemm_sm100(xq[:M].contiguous().data_ptr(), w8.data_ptr(), sg.data_ptr(), ones.data_ptr(),
                                       a_s.data_ptr(), out2.data_ptr(), M, N, K, ws.data_ptr(), ws.numel(),
                                       torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert torch.equal(out2, out)
        out3 = torch.empty(M, N, device="cuda", dtype=torch.float16)      # 256-deep K tile: the same integers
        fvk.flashdrive_w4a8_gemm_sm100(xq[:M].contiguous().data_ptr(), w8.data_ptr(), sg.data_ptr(), ones.data_ptr(),
                                       a_s.data_ptr(), out3.data_ptr(), M, N, K, ws.data_ptr(), ws.numel(),
                                       torch.cuda.current_stream().cuda_stream, 256)
        torch.cuda.synchronize()
        assert torch.equal(out3, out)


def test_pack_w4a8_small_layout():
    """Lane fragments: lane 4g+t, byte 4(2 half + rh) + j = w[16T + 8rh + g, 64C + 32ks + 16half + 4t + j], ks in the nibble."""
    from flash_rt.models.alpamayo15.marlin_w4a8 import pack_w4a8_small
    K, N = 256, 32
    w = torch.randint(0, 16, (K, N), dtype=torch.uint8)
    p = pack_w4a8_small(w, K, N)
    assert p.shape == (N // 16, K // 64, 32, 16)
    for T, C, lane, byte in [(0, 0, 0, 0), (1, 3, 31, 15), (1, 2, 9, 6), (0, 1, 22, 11)]:
        g, t = lane // 4, lane % 4
        a, j = byte // 4, byte % 4
        half, rh = a // 2, a % 2
        n = 16 * T + 8 * rh + g
        for ks in (0, 1):
            k = 64 * C + 32 * ks + 16 * half + 4 * t + j
            nib = (int(p[T, C, lane, byte]) >> (4 * ks)) & 0xF
            assert nib == int(w[k, n])


@pytest.mark.parametrize("M,N,K,tile", [(802, 3584, 3584, 0), (802, 512, 3584, 0), (300, 18944, 3584, 1), (802, 3584, 18944, 0)])
def test_int8_channel_gemm_and_requantization_match_reference(M, N, K, tile):
    """The per-channel INT8 prefill GEMM reproduces half(float(xq @ codes.T) * a_s[m] * c[n]) (int32 accumulation);
    w4a8_to_int8_channel keeps every channel's absmax at 127 and the integer weights within half a step."""
    fvk = _kernels()
    from flash_rt.models.alpamayo15_flashdrive.weights import w4a8_to_int8_channel
    torch.manual_seed(37)
    w8 = torch.randint(-15, 16, (N, K), device="cuda", dtype=torch.int8)
    s = torch.randint(1, 4097, (K // 128, N), device="cuda", dtype=torch.int32)
    codes, c = w4a8_to_int8_channel(w8, s)
    v = w8.int() * s.t().repeat_interleave(128, 1)
    assert torch.equal(codes.int().abs().amax(dim=1), torch.full((N,), 127, device="cuda", dtype=torch.int32))
    assert bool(((codes.float() * c[:, None] - v.float()).abs() <= 0.5 * c[:, None] + 1e-3).all())
    xq = torch.randint(-127, 128, ((M + 31) // 32 * 32, K), device="cuda", dtype=torch.int8)
    a_s = (torch.rand(M, device="cuda") * 2e-5 + 2e-6).to(torch.float32)      # outputs within the FP16 range
    D = torch.empty(M, N, device="cuda", dtype=torch.float16)
    nbytes = fvk.alpamayo15_int8_gemm_sm100_workspace(M, N, K, tile)
    ws = torch.empty(max(int(nbytes), 16), dtype=torch.uint8, device="cuda")
    fvk.alpamayo15_int8_gemm_sm100(xq.data_ptr(), codes.data_ptr(), a_s.data_ptr(), c.data_ptr(), D.data_ptr(), M, N, K, tile,
                                   ws.data_ptr(), ws.numel(), torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    acc = (xq[:M].float() @ codes.float().t())                 # exact: |acc| < 2^24 for these magnitudes? use float64
    acc = (xq[:M].double() @ codes.double().t())
    ref = (acc * a_s.double()[:, None] * c.double()[None, :]).to(torch.float16)
    err = (D.float() - ref.float()).abs()
    assert bool((err <= ref.float().abs() * 2 ** -9 + 1e-6).all()), float(err.max())   # two FP32 products, one FP16 rounding


def test_w4a8_prefill_gemm_forms_are_identical():
    """K tile 128 / 256 and the 1-SM / 2-SM forms of the blockwise W4A8 GEMM produce identical FP16 outputs (integer
    mainloop, one FP32 scale and FP16 rounding per output)."""
    fvk = _kernels()
    torch.manual_seed(41)
    M, N, K = 300, 512, 3584
    xq = torch.randint(-127, 128, ((M + 31) // 32 * 32, K), device="cuda", dtype=torch.int8)
    w8 = torch.randint(-15, 16, (N, K), device="cuda", dtype=torch.int8)
    sg = torch.randint(1, 4097, (K // 128, N), device="cuda", dtype=torch.int32)
    a_s = (torch.rand(M, device="cuda") * 1e-6).float()
    nbytes, n_ones = fvk.flashdrive_w4a8_gemm_sm100_sizes(M, N, K)
    ws = torch.empty(max(int(nbytes), 1), dtype=torch.uint8, device="cuda")
    ones = torch.ones(int(n_ones) + 1024, dtype=torch.int32, device="cuda")
    outs = []
    for ktile, pair in ((128, 0), (256, 0), (128, 1), (256, 1)):
        out = torch.full((M, N), float("nan"), device="cuda", dtype=torch.float16)
        fvk.flashdrive_w4a8_gemm_sm100(xq.data_ptr(), w8.data_ptr(), sg.data_ptr(), ones.data_ptr(), a_s.data_ptr(), out.data_ptr(),
                                       M, N, K, ws.data_ptr(), ws.numel(), torch.cuda.current_stream().cuda_stream, ktile, pair)
        torch.cuda.synchronize()
        outs.append(out)
    ref = ((xq[:M].double() @ (w8.int() * sg.t().repeat_interleave(128, 1)).double().t()) * a_s.double()[:, None]).to(torch.float16)
    assert torch.equal(outs[0], ref) or bool(((outs[0].float() - ref.float()).abs() <= ref.float().abs() * 2 ** -10).all())
    for o in outs[1:]:
        assert torch.equal(o, outs[0])


def test_new_kernels_are_bitwise_repeatable():
    """Every new kernel of the backend returns identical bits on a second call with identical inputs (fixed
    reduction orders, no races)."""
    fvk = _kernels()
    from flash_rt.models.alpamayo15.marlin_w4a8 import pack_w4a8_small
    from flash_rt.models.alpamayo15.pipeline_thor import _Gemm
    from flash_rt.models.alpamayo15_flashdrive.weights import bf16_fragment_layout
    torch.manual_seed(27)
    s_ = torch.cuda.current_stream().cuda_stream
    runs = []
    # W4A8 few-row kernels (both variants) on q / down shapes
    for N, K in ((4096, 4096), (4096, 12288), (1024, 4096)):
        G = K // 128
        w = torch.randint(0, 16, (K, N), device="cuda", dtype=torch.uint8)
        z = torch.randint(0, 16, (G, N), device="cuda", dtype=torch.uint8)
        sg = torch.randint(0, 4097, (G, N), device="cuda", dtype=torch.int32)
        wp = pack_w4a8_small(w, K, N)
        xq = torch.randint(-128, 128, (8, K), device="cuda", dtype=torch.int8)
        a_s = torch.rand(8, device="cuda") * 1e-4
        for fn in (fvk.flashdrive_w4a8_gemm_small, fvk.flashdrive_w4a8_gemm_tile):
            outs = []
            for _ in range(2):
                o = torch.full((8, N), 7.0, device="cuda", dtype=torch.float16)
                fn(xq.data_ptr(), a_s.data_ptr(), wp.data_ptr(), z.data_ptr(), sg.data_ptr(), o.data_ptr(), 8, N, K, s_)
                torch.cuda.synchronize(); outs.append(o)
            runs.append(("w4a8", N, K, torch.equal(outs[0], outs[1])))
    # BF16 few-row kernels
    for N, K in ((4096, 4096), (12288, 4096), (155698, 4096)):
        w = (torch.randn(N, K, device="cuda") * 0.05).to(BF16)
        fr = bf16_fragment_layout(w)
        x = (torch.randn(8, K, device="cuda")).to(BF16)
        for name, fn in (("frag", lambda y: fvk.flashdrive_bf16_gemm_frag(x.data_ptr(), K, fr.data_ptr(), y.data_ptr(), N, 0, 8, N, K, s_)),
                         ("small", lambda y: fvk.flashdrive_bf16_gemm_small(x.data_ptr(), K, w.data_ptr(), y.data_ptr(), N, 0, 8, N, K, s_))):
            outs = []
            for _ in range(2):
                y = torch.full((8, N), 7.0, device="cuda", dtype=BF16)
                fn(y); torch.cuda.synchronize(); outs.append(y)
            runs.append((name, N, K, torch.equal(outs[0], outs[1])))
    # CUTLASS BF16 GEMM with residual, LayerNorm
    A = torch.randn(3072, 1152, device="cuda").to(BF16); W = (torch.randn(1152, 1152, device="cuda") * 0.05).to(BF16)
    bias = torch.randn(1152, device="cuda").to(BF16); C = torch.randn(3072, 1152, device="cuda").to(BF16)
    g = _Gemm(fvk)
    outs = []
    for _ in range(2):
        D = torch.empty(3072, 1152, device="cuda", dtype=BF16); g(A, W, D, bias=bias, C=C); torch.cuda.synchronize(); outs.append(D)
    runs.append(("cutlass_res", 1152, 1152, torch.equal(outs[0], outs[1])))
    outs = []
    for _ in range(2):
        o = torch.empty_like(A)
        fvk.alpamayo15_layernorm_bf16(A.data_ptr(), bias.data_ptr(), bias.data_ptr(), o.data_ptr(), 3072, 1152, 1e-6, s_)
        torch.cuda.synchronize(); outs.append(o)
    runs.append(("layernorm", 3072, 1152, torch.equal(outs[0], outs[1])))
    bad = [r for r in runs if not r[3]]
    assert not bad, bad


@pytest.mark.parametrize("rows,K", [(802, 4096), (8, 4096), (1, 12288), (130, 12288)])
def test_dense_rotation_matches_exact_rotation_and_paro_codes(rows, K):
    """The dense rotation equals the exact (float64) 8-layer rotation of the scaled FP16 input within FP16 GEMM
    error, and its INT8 codes agree with the layer-by-layer kernel on all but near-rounding elements."""
    fvk = _kernels()
    from flash_rt.models.alpamayo15_flashdrive.weights import paro_rotation_matrices
    torch.manual_seed(21)
    x = (torch.randn(rows, K, device="cuda") * 3).to(BF16)
    sets = []
    for _ in range(2):
        theta = (torch.randn(8, K // 2, device="cuda") * 2).to(torch.float16)
        perm = torch.stack([torch.cat([torch.randperm(128) for _ in range(K // 128)]) for _ in range(8)])
        pairs = perm.to(torch.int16).cuda()
        scales = (torch.rand(1, K, device="cuda") + 0.5).to(torch.float16)
        igs = torch.tensor(6.5e-6, device="cuda", dtype=torch.float32)
        sets.append((theta, pairs, scales, igs, paro_rotation_matrices(theta, pairs, K)))
    p = lambda ts: [t.data_ptr() for t in ts]
    s_ = torch.cuda.current_stream().cuda_stream
    Ma = (rows + 7) // 8 * 8
    xr = [torch.empty(rows, K, device="cuda", dtype=torch.float16) for _ in sets]
    part = [torch.empty(rows, K // 128, device="cuda", dtype=torch.float32) for _ in sets]
    xq = [torch.zeros(Ma, K, device="cuda", dtype=torch.int8) for _ in sets]
    a_s = [torch.empty(rows, device="cuda", dtype=torch.float32) for _ in sets]
    fvk.flashdrive_w4a8_rotate_dense(x.data_ptr(), K, rows, K, p(s[4] for s in sets), p(s[2] for s in sets), p(xr), p(part), s_)
    fvk.flashdrive_w4a8_quant(rows, K, p(xr), p(part), p(s[3] for s in sets), p(xq), p(a_s), s_)
    xr2 = [torch.empty(rows, K, device="cuda", dtype=torch.float16) for _ in sets]
    part2 = [torch.empty(rows, K // 128, device="cuda", dtype=torch.float32) for _ in sets]
    xq2 = [torch.zeros(Ma, K, device="cuda", dtype=torch.int8) for _ in sets]
    a_s2 = [torch.empty(rows, device="cuda", dtype=torch.float32) for _ in sets]
    fvk.flashdrive_w4a8_act_prep(x.data_ptr(), K, rows, K, p(s[0] for s in sets), p(s[1] for s in sets),
                                 p(s[2] for s in sets), p(s[3] for s in sets), p(xr2), p(part2), p(xq2), p(a_s2), s_)
    torch.cuda.synchronize()
    G = K // 128
    for (theta, pairs, scales, igs, R), r, pt, q, a, q2, r2, a2 in zip(sets, xr, part, xq, a_s, xq2, xr2, a_s2):
        xs = (x.to(torch.float16) * scales).double().view(rows, G, 128)
        exact = torch.einsum("mgk,gck->mgc", xs, R.double()).reshape(rows, K)    # R itself is FP16-rounded
        err = (r.double() - exact).abs()
        assert bool((err <= exact.abs() * 2 ** -9 + 2 ** -9 * xs.abs().amax(-1).view(rows, G, 1).expand(-1, -1, 128).reshape(rows, K)).all())
        assert torch.equal(pt, r.float().abs().view(rows, G, 128).amax(-1))
        same = float((q[:rows] == q2[:rows]).float().mean())
        assert same > 0.98, same
        assert bool(((q[:rows].int() - q2[:rows].int()).abs() <= 2).all())
        assert torch.allclose(a, a2, rtol=2e-2, atol=0)
        rel = ((r.float() - r2.float()).abs() / r2.float().abs().clamp_min(1e-2)).mean()
        assert float(rel) < 4e-3, float(rel)


@pytest.mark.parametrize("M,N,K,f32", [(8, 4096, 4096, 0), (7, 155697, 4096, 0), (16, 1024, 20480, 1), (1, 12288, 4096, 0),
                                        (13, 4096, 12288, 1), (3, 24, 512, 0)])
def test_bf16_small_gemm_matches_float64(M, N, K, f32):
    """The few-row BF16 GEMM: FP32 accumulation of exact BF16 products (float64 reference within FP32 rounding of
    the dot products), partial last tile guarded, deterministic across runs."""
    fvk = _kernels()
    torch.manual_seed(18)
    Ma = (M + 7) // 8 * 8
    x = (torch.randn(Ma, K + 64, device="cuda") * 0.5).to(BF16)[:, :K]          # strided
    w = (torch.randn(N, K, device="cuda") * 0.05).to(BF16)
    y = torch.empty(M, N + 8, device="cuda", dtype=torch.float32 if f32 else BF16)[:, :N]
    fvk.flashdrive_bf16_gemm_small(x.data_ptr(), x.stride(0), w.data_ptr(), y.data_ptr(), y.stride(0), f32, M, N, K,
                                   torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    ref = x[:M].double() @ w.double().t()
    out = y.double()
    tol = ref.abs() * (2.0 ** -16 + (0 if f32 else 2.0 ** -8)) + 4e-4 * (K / 4096) ** 0.5   # FP32 sums in other orders
    assert bool(((out - ref).abs() <= tol).all()), float((out - ref).abs().max())
    y2 = torch.empty_like(y.contiguous())
    fvk.flashdrive_bf16_gemm_small(x.data_ptr(), x.stride(0), w.data_ptr(), y2.data_ptr(), N, f32, M, N, K,
                                   torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    assert torch.equal(y2, y.contiguous())
    from flash_rt.models.alpamayo15_flashdrive.weights import bf16_fragment_layout
    frag = bf16_fragment_layout(w)
    y3 = torch.empty_like(y2)
    fvk.flashdrive_bf16_gemm_frag(x.data_ptr(), x.stride(0), frag.data_ptr(), y3.data_ptr(), N, f32, M, N, K,
                                  torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    assert bool(((y3.double() - ref).abs() <= tol).all())
    T = (N + 15) // 16
    assert frag.numel() == T * 16 * K
    v = frag.view(T, K // 16, 32, 4, 2)
    for Ti, kc, lane, a in [(0, 0, 0, 0), (T - 1, K // 16 - 1, 31, 3), (0, 1, 13, 2)]:
        g, c = lane // 4, lane % 4
        half, rh = a // 2, a % 2
        n, k = 16 * Ti + 8 * rh + g, 16 * kc + 8 * half + 2 * c
        exp = w[n, k:k + 2] if n < N else torch.zeros(2, device="cuda", dtype=BF16)
        assert torch.equal(v[Ti, kc, lane, a], exp)


def test_draft_qkv_prep_matches_flashdrive_draft_attention():
    """Head RMSNorm + BF16-rule RoPE of the draft's q (positions ctx..ctx+nq-1) and k (0..nk-1) and the copied v
    equal FlashDrive's Qwen3DFlashAttention operand preparation under autocast(bfloat16)."""
    fvk = _kernels()
    from flashdrive.dflash import apply_rotary_pos_emb_tail
    torch.manual_seed(19)
    nq, ctx, Hq, Hk, D = 8, 8, 32, 8, 128
    nk = nq + ctx
    q = torch.randn(nq, Hq * D, device="cuda").to(BF16)
    k = torch.randn(nk, Hk * D, device="cuda").to(BF16)
    v = torch.randn(nk, Hk * D, device="cuda").to(BF16)
    qw = (torch.rand(D, device="cuda") + 0.5).to(BF16)
    kw = (torch.rand(D, device="cuda") + 0.5).to(BF16)
    inv = 1.0 / (1e6 ** (torch.arange(0, D, 2, device="cuda").float() / D))
    freqs = torch.arange(nk, device="cuda").float().unsqueeze(1) * inv.unsqueeze(0)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos().to(BF16), emb.sin().to(BF16)          # Qwen3RotaryEmbedding returns tables in the model dtype
    eps = 1e-6

    def rms(t, w):
        tf = t.float()
        n = (tf * torch.rsqrt(tf.pow(2).mean(-1, keepdim=True) + eps)).to(BF16)
        return (w.to(BF16) * n)                                 # Qwen3RMSNorm: weight * hidden.to(input_dtype)
    qn = rms(q.view(1, nq, Hq, D), qw).transpose(1, 2)
    kn = rms(k.view(1, nk, Hk, D), kw).transpose(1, 2)
    with torch.autocast("cuda", dtype=BF16):
        q_ref, k_ref = apply_rotary_pos_emb_tail(qn, kn, cos.unsqueeze(0), sin.unsqueeze(0))
    q_out = torch.empty(nq, Hq, D, device="cuda", dtype=BF16)
    k_out = torch.empty(nk, Hk, D, device="cuda", dtype=BF16)
    v_out = torch.empty(nk, Hk, D, device="cuda", dtype=BF16)
    fvk.flashdrive_draft_qkv_prep_bf16(q.data_ptr(), k.data_ptr(), v.data_ptr(), Hq * D, Hk * D, Hk * D, qw.data_ptr(),
                                       kw.data_ptr(), cos.data_ptr(), sin.data_ptr(), q_out.data_ptr(), k_out.data_ptr(),
                                       v_out.data_ptr(), nq, ctx, Hq, Hk, eps, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    assert torch.equal(q_out, q_ref[0].transpose(0, 1))
    assert torch.equal(k_out, k_ref[0].transpose(0, 1))
    assert torch.equal(v_out, v.view(nk, Hk, D))


@pytest.mark.parametrize("backends", [
    dict(expert="flashrt", vision="flashrt", prefill="flashrt", decode="flashrt", w4a8="fused", pdl=False),
    dict(expert="flashrt", vision="flashrt", prefill="flashrt", decode="flashrt", w4a8="fused", pdl=True),
], ids=["all_flashrt_pdl_off", "all_flashrt_pdl_on"])
@_needs_ckpt
@pytest.mark.usefixtures("_release_model")
def test_backend_primitives_are_repeatable(backends):
    """Every FlashRT primitive, called twice on identical cache / RNG state, returns identical bits and writes identical
    cache rows (run-to-run reproducibility of the stream)."""
    _kernels()
    from alpamayo1_5 import helper
    from flashdrive.streaming import convert_to_streaming_window
    from flash_rt.frontends.torch.alpamayo15_flashdrive_thor import attach_flashdrive_backend, load_flashdrive

    torch.set_float32_matmul_precision("high")
    root = _CKPT
    model = load_flashdrive(f"{root}/Alpamayo-1.5-10B", f"{root}/Alpamayo-1.5-10B-PARO", f"{root}/Alpamayo-1.5-10B-DFlash")
    names = ["_action", "_encode", "_dflash_prefill", "_shift_streaming_kv_cache", "_dflash_block_step", "_dflash_traj_forward"]
    ref = {n: getattr(model, n) for n in names}
    eng = attach_flashdrive_backend(model, **backends)
    frt = {n: getattr(model, n) for n in names if getattr(model, n) is not ref[n]}
    gen = torch.cuda.default_generators[torch.cuda.current_device()]
    mismatches = []

    def snap():
        pkv = model._past_key_values
        return [(l.keys.clone(), l.values.clone()) for l in pkv.layers] if pkv is not None and pkv.layers[0].is_initialized else None

    def restore(s_):
        for l, (k, v) in zip(model._past_key_values.layers, s_):
            l.keys.copy_(k); l.values.copy_(v)

    def tensors(o):
        if torch.is_tensor(o):
            return [o.detach().clone()]
        return [t for x in (o or []) for t in tensors(x)] if isinstance(o, (list, tuple)) else []

    def wrap(n):
        def f(*a, **k):
            pre, rng = snap(), gen.get_state()
            st_ = eng.rt.active_stream
            kvlm_pre = st_["kvlm"].clone() if st_ is not None and "kvlm" in st_ else None
            out1 = frt[n](*a, **k)
            t1, post1 = tensors(out1), snap()
            kvlm1 = eng.rt.active_stream["kvlm"].clone() if kvlm_pre is not None else None
            if pre is not None:
                restore(pre)
            if kvlm_pre is not None:
                eng.rt.active_stream["kvlm"].copy_(kvlm_pre)
            gen.set_state(rng)
            out2 = frt[n](*a, **k)
            t2, post2 = tensors(out2), snap()
            same_out = all(torch.equal(x, y) for x, y in zip(t1, t2)) and len(t1) == len(t2)
            same_cache = post1 is None or all(torch.equal(a1, a2) and torch.equal(b1, b2) for (a1, b1), (a2, b2) in zip(post1, post2))
            same_kvlm = kvlm1 is None or torch.equal(kvlm1, eng.rt.active_stream["kvlm"])
            if not (same_out and same_cache and same_kvlm):
                mismatches.append((n, same_out, same_cache, same_kvlm))
            return out2
        return f
    for n in frt:
        setattr(model, n, wrap(n))
    proc = helper.get_processor(model.tokenizer)
    vs, ve = (proc.tokenizer.convert_tokens_to_ids(t) for t in ("<|vision_start|>", "<|vision_end|>"))
    g = torch.Generator().manual_seed(0)
    hist = (torch.zeros(1, 1, 16, 3), torch.eye(3).expand(1, 1, 16, 3, 3).contiguous())
    frames = torch.randint(0, 256, (4, 7, 3, 1080, 1920), dtype=torch.uint8, generator=g)
    torch.manual_seed(0)
    for k in range(4):
        msgs = helper.create_message(frames[:, k:k + 4].flatten(0, 1), camera_indices=torch.tensor([0, 1, 2, 6]))
        t = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False, continue_final_message=True,
                                     return_dict=True, return_tensors="pt")
        w = {"tokenized_data": t, "ego_history_xyz": hist[0], "ego_history_rot": hist[1]}
        if k:
            w = convert_to_streaming_window(w, vs, ve)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model.sample_trajectories_streaming(helper.to_device(w, "cuda"), num_traj_samples=1, diffusion_kwargs={
                "inference_step": 8, "cache_steps": [3, 4, 5, 6], "int_method": "euler_with_cache"})
    assert not mismatches, mismatches
    del eng, model
    gc.collect(); torch.cuda.empty_cache()


@pytest.mark.parametrize("backends", [
    dict(expert="flashrt", vision="flashrt", prefill="flashrt", decode="flashrt", w4a8="fused"),
    dict(expert="flashdrive", vision="flashdrive", prefill="flashrt", decode="flashrt", w4a8="flashdrive"),
], ids=["all_flashrt", "mixed"])
@_needs_ckpt
@pytest.mark.usefixtures("_release_model")
def test_backend_primitives_match_flashdrive_on_identical_state(backends):
    _kernels()
    from alpamayo1_5 import helper
    from flashdrive.streaming import convert_to_streaming_window
    from flash_rt.frontends.torch.alpamayo15_flashdrive_thor import attach_flashdrive_backend, load_flashdrive

    torch.set_float32_matmul_precision("high")
    root = _CKPT
    model = load_flashdrive(f"{root}/Alpamayo-1.5-10B", f"{root}/Alpamayo-1.5-10B-PARO", f"{root}/Alpamayo-1.5-10B-DFlash")
    names = ["_action", "_encode", "_dflash_prefill", "_shift_streaming_kv_cache", "_dflash_block_step",
             "_dflash_traj_forward"]
    ref = {n: getattr(model, n) for n in names}
    eng = attach_flashdrive_backend(model, **backends)
    frt = {n: getattr(model, n) for n in names if getattr(model, n) is not ref[n]}   # primitives the backend replaced
    gen = torch.cuda.default_generators[torch.cuda.current_device()]
    seen = {n: [] for n in frt}

    def snap():
        pkv = model._past_key_values
        return [(l.keys.clone(), l.values.clone()) for l in pkv.layers] if pkv is not None and pkv.layers[0].is_initialized else None

    def restore(s_):
        for l, (k, v) in zip(model._past_key_values.layers, s_):
            l.keys.copy_(k); l.values.copy_(v)

    def kvlm_is_rotated_cache():
        st = eng.rt.stream()
        fresh = torch.empty_like(st["kvlm"][0])
        for li in range(eng.rt.n_layers):
            rope_rows(eng.rt, st, li, fresh, st["cos"], st["sin"], 0, st["S"])
            if not torch.equal(fresh, st["kvlm"][li]):
                return False
        return True

    def tensors(o):
        if torch.is_tensor(o):
            return [o.detach().clone()]
        return [t for x in (o or []) for t in tensors(x)] if isinstance(o, (list, tuple)) else []

    cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0))

    def wrap(n):
        def f(*a, **k):
            # Every comparison is reduced to scalars inside the call: three full cache snapshots per call would exhaust
            # Thor's unified memory over the windows of two streams.
            pre, rng = snap(), gen.get_state()
            out = ref[n](*a, **k)                       # FlashDrive (kvlm is not touched: still the rotation of pre)
            r_out, post, rng_ref = tensors(out), snap(), gen.get_state()
            if pre is not None:
                restore(pre)
            gen.set_state(rng)
            f_out = tensors(frt[n](*a, **k))
            invariant = kvlm_is_rotated_cache()          # every FlashRT write keeps the rotated copy exact
            post_f = snap()
            rec = {"rng_eq": torch.equal(gen.get_state(), rng_ref), "invariant": invariant, "out_cos": 1.0, "cache_ok": True,
                   "proposal_eq": None}
            if n == "_dflash_block_step":
                rec["proposal_eq"] = torch.equal(r_out[0], f_out[0])
            if rec["proposal_eq"] is not False:      # a round whose greedy proposals differ has no comparable context
                for x, y in zip(r_out, f_out):
                    if x.dtype in (torch.long, torch.int32):
                        continue
                    rec["out_cos"] = min(rec["out_cos"], cos(x, y))
            if post is not None and rec["proposal_eq"] is not False:   # different block tokens write different rows
                for li, ((kr, vr), (kf, vf)) in enumerate(zip(post, post_f)):
                    if n == "_shift_streaming_kv_cache":
                        rec["cache_ok"] &= torch.equal(kr, kf) and torch.equal(vr, vf)
                        continue
                    if pre is None:
                        continue
                    rows = ((kr != pre[li][0]) | (kf != pre[li][0])).any(dim=(0, 1, 3))    # rows either side wrote
                    if rows.any():
                        rec["cache_ok"] &= cos(kr[:, :, rows], kf[:, :, rows]) > 0.99 and cos(vr[:, :, rows], vf[:, :, rows]) > 0.99
            seen[n].append(rec)
            restore(post); gen.set_state(rng_ref)
            del pre, post_f
            sync_kvlm_all(eng.rt, eng.rt.stream())       # continue from FlashDrive's state
            return out
        return f
    for n in frt:
        setattr(model, n, wrap(n))
    proc = helper.get_processor(model.tokenizer)
    vs, ve = (proc.tokenizer.convert_tokens_to_ids(t) for t in ("<|vision_start|>", "<|vision_end|>"))
    g = torch.Generator().manual_seed(0)
    hist = (torch.zeros(1, 1, 16, 3), torch.eye(3).expand(1, 1, 16, 3, 3).contiguous())

    def stream(n_windows):
        frames = torch.randint(0, 256, (4, n_windows + 3, 3, 1080, 1920), dtype=torch.uint8, generator=g)
        for k in range(n_windows):
            msgs = helper.create_message(frames[:, k:k + 4].flatten(0, 1), camera_indices=torch.tensor([0, 1, 2, 6]))
            t = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False, continue_final_message=True,
                                         return_dict=True, return_tensors="pt")
            w = {"tokenized_data": t, "ego_history_xyz": hist[0], "ego_history_rot": hist[1]}
            if k:
                w = convert_to_streaming_window(w, vs, ve)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model.sample_trajectories_streaming(helper.to_device(w, "cuda"), num_traj_samples=1, diffusion_kwargs={
                    "inference_step": 8, "cache_steps": [3, 4, 5, 6], "int_method": "euler_with_cache"})
    torch.manual_seed(0)
    stream(4)
    eng.reset()                          # an unrelated second stream: FlashDrive's state and the stream graphs rebuilt
    stream(2)
    for n in frt:
        assert seen[n], f"{n} was not exercised"
        for rec in seen[n]:
            assert rec["rng_eq"], f"{n}: random-number consumption differs from FlashDrive"
            assert rec["invariant"], f"{n}: rotated cache copy is stale"
            assert rec["out_cos"] > 0.995, (n, rec["out_cos"])
            assert rec["cache_ok"], n
    if "_dflash_block_step" in seen:
        proposals = [rec["proposal_eq"] for rec in seen["_dflash_block_step"]]
        assert sum(proposals) >= 0.8 * len(proposals)      # greedy draft proposals from the same context
    del eng, model
    gc.collect(); torch.cuda.empty_cache()


@_needs_ckpt
@pytest.mark.usefixtures("_release_model")
def test_backend_stream_lifecycle_buffers_graphs_and_reset():
    """The backend's resources over two streams of the same windows: no stream state before the first window; the
    per-stream graphs captured once (the same graph objects and buffers after the third window as after the
    second); each window's inputs reaching the graphs (different frames, different outputs); ``reset`` ends the
    stream (no state, FlashDrive's cache dropped) and the next stream, started from the same seed, captures its own
    graphs and reproduces the first stream bit for bit; the runtime's expert buffers persist across streams."""
    _kernels()
    from alpamayo1_5 import helper
    from flashdrive.streaming import convert_to_streaming_window
    from flash_rt.frontends.torch.alpamayo15_flashdrive_thor import attach_flashdrive_backend, load_flashdrive

    torch.set_float32_matmul_precision("high")
    root = _CKPT
    model = load_flashdrive(f"{root}/Alpamayo-1.5-10B", f"{root}/Alpamayo-1.5-10B-PARO", f"{root}/Alpamayo-1.5-10B-DFlash")
    eng = attach_flashdrive_backend(model)
    rt = eng.rt
    assert rt.active_stream is None
    expert_ptrs = {k: v.data_ptr() for k, v in rt.expert_buffers().items()}
    proc = helper.get_processor(model.tokenizer)
    vs, ve = (proc.tokenizer.convert_tokens_to_ids(t) for t in ("<|vision_start|>", "<|vision_end|>"))
    frames = torch.randint(0, 256, (4, 8, 3, 1080, 1920), dtype=torch.uint8, generator=torch.Generator().manual_seed(0))
    hist = (torch.zeros(1, 1, 16, 3), torch.eye(3).expand(1, 1, 16, 3, 3).contiguous())

    def state_of(st):
        """The stream's graphs and the addresses of its buffers (cache views, decode state, work buffers)."""
        bufs = {("st", k): v.data_ptr() for k, v in st.items() if torch.is_tensor(v)}
        for key, d in st["keep"].items():
            if isinstance(d, dict):
                bufs.update({(str(key), k): v.data_ptr() for k, v in d.items() if torch.is_tensor(v)})
        return {k: id(g) for k, g in st["graphs"].items()}, bufs

    def run_stream():
        torch.manual_seed(0)
        outs, states = [], []
        for k in range(5):
            msgs = helper.create_message(frames[:, k:k + 4].flatten(0, 1), camera_indices=torch.tensor([0, 1, 2, 6]))
            t = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False, continue_final_message=True,
                                         return_dict=True, return_tensors="pt")
            w = {"tokenized_data": t, "ego_history_xyz": hist[0], "ego_history_rot": hist[1]}
            if k:
                w = convert_to_streaming_window(w, vs, ve)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                xyz, _, extra = model.sample_trajectories_streaming(
                    helper.to_device(w, "cuda"), num_traj_samples=1, return_extra=True,
                    diffusion_kwargs={"inference_step": 8, "cache_steps": [3, 4, 5, 6], "int_method": "euler_with_cache"})
            outs.append(None if xyz is None else (xyz.clone(), str(extra["cot"].reshape(-1)[0])))
            states.append(state_of(rt.active_stream))
        return outs, states

    outs1, states1 = run_stream()
    st1 = rt.active_stream
    assert outs1[0] is None and all(o is not None for o in outs1[1:])            # the first window only prefills
    (g3, b3), (g4, b4) = states1[3], states1[4]                                  # captured once, buffers kept:
    assert all(g4[k] == v for k, v in g3.items()) and all(b4[k] == v for k, v in b3.items())
    assert not torch.equal(outs1[1][0], outs1[2][0])                             # new frames reach the graphs
    eng.reset()
    assert rt.active_stream is None and model._past_key_values is None
    outs2, states2 = run_stream()
    assert rt.active_stream is not st1
    assert not set(states2[4][0].values()) & set(states1[4][0].values())        # the new stream's own graphs
    for o1, o2 in zip(outs1[1:], outs2[1:]):
        assert torch.equal(o1[0], o2[0]) and o1[1] == o2[1]
    assert {k: v.data_ptr() for k, v in rt.expert_buffers().items()} == expert_ptrs
    del eng, model, st1
    gc.collect(); torch.cuda.empty_cache()
