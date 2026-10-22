"""RDNA routing, build identity and mandatory library fallback contracts."""
from types import SimpleNamespace

import pytest

from flash_rt.amd.hardware.rdna.capabilities import RDNA3, RDNA35, RDNA4


@pytest.mark.parametrize("target", [RDNA3, RDNA35, RDNA4])
def test_target_detection_and_lazy_frontend(monkeypatch, target):
    import torch
    from flash_rt.hardware import detect_arch, resolve_pipeline_class

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "test")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(
        gcnArchName=target.gpu_arch + ":xnack-"))
    assert detect_arch() == target.hardware
    cls = resolve_pipeline_class("pi05", "torch", target.hardware)
    assert cls._target == target
    assert cls._pipeline_class.__module__.endswith(f"pi05_{target.backend}.pipeline")
    with pytest.raises(ValueError, match="requires hardware"):
        cls("unused", hardware="amd_cdna4")


@pytest.mark.parametrize("target", [RDNA3, RDNA35, RDNA4])
def test_extension_identity_fails_closed(target):
    info = dict(backend=target.backend, wave_size=32,
                gpu_arch=target.gpu_arch, gfx11_wmma=target.gfx11_wmma)
    kernels = SimpleNamespace(build_info=lambda: info,
                              device_arch=lambda: target.gpu_arch + ":xnack-")
    target.validate(kernels, target.gpu_arch + ":xnack-")
    for field, wrong in (("backend", "cdna4"), ("wave_size", 64),
                         ("gpu_arch", target.gpu_arch + "0"),
                         ("gfx11_wmma", not target.gfx11_wmma)):
        original = info[field]
        info[field] = wrong
        with pytest.raises(RuntimeError, match="requires matching"):
            target.validate(kernels, target.gpu_arch)
        info[field] = original
    with pytest.raises(RuntimeError, match="requires matching"):
        target.validate(kernels, "gfx950")


@pytest.mark.parametrize("hardware", ["amd_rdna3", "amd_rdna4"])
@pytest.mark.parametrize("options", [dict(config="pi0"), dict(framework="jax"),
                                    dict(use_fp4=True), dict(use_fp16=True)])
def test_unsupported_request_fails_before_extension_or_checkpoint(hardware, options):
    import flash_rt
    kwargs = dict(config="pi05", framework="torch", hardware=hardware)
    kwargs.update(options)
    with pytest.raises(ValueError, match="supports"):
        flash_rt.load_model("nonexistent", **kwargs)


@pytest.mark.parametrize("hardware", ["thor", "rtx_sm120", "rtx_sm89", "rtx_sm87"])
def test_cuda_route_does_not_import_rdna_or_change_precision(monkeypatch, hardware):
    import builtins
    import flash_rt
    from flash_rt import _extensions
    from flash_rt import hardware as dispatch

    captured = {}

    class Frontend:
        def __init__(self, checkpoint, num_views=2, hardware=None, use_fp8=True):
            captured.update(hardware=hardware, use_fp8=use_fp8)

    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert not name.startswith("flash_rt.amd"), name
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(_extensions, "require", lambda **kwargs: None)
    monkeypatch.setattr(dispatch, "resolve_pipeline_class", lambda *args: Frontend)
    flash_rt.load_model("unused", hardware=hardware)
    assert captured == dict(hardware=hardware, use_fp8=True)


def test_gfx12_capabilities_override_fast_path_environment(monkeypatch):
    from flash_rt.amd.hardware.rdna35 import Rdna35AttentionBackend, Rdna35GemmBackend

    monkeypatch.setenv("FLASHRT_RDNA35_HIP_ENCODER_ATTN", "1")
    monkeypatch.setenv("FLASHRT_RDNA35_HIP_SMALLM", "1")
    monkeypatch.setenv("FLASHRT_RDNA35_GEMM_AUTOTUNE", "0")
    # gfx12 intentionally has neither WMMA binding. No attribute probing or
    # environment override may turn these kernels back on.
    kernels = SimpleNamespace(RdnaGemmRunner=lambda: object())
    attention = Rdna35AttentionBackend(kernels, supports_encoder=RDNA4.gfx11_wmma)
    gemm = Rdna35GemmBackend(kernels, supports_smallm=RDNA4.gfx11_wmma)
    assert not attention.hip_encoder_gqa
    assert not gemm.hip_smallm
    gemm.prepare_smallm_weight(None)
    assert not gemm._smallm_weights


def _device_target():
    import torch

    if not torch.version.hip or not torch.cuda.is_available():
        pytest.skip("requires a supported ROCm device")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    targets = {t.gpu_arch: t for t in (RDNA3, RDNA35, RDNA4)}
    if arch not in targets:
        pytest.skip("requires a supported RDNA device")
    return targets[arch]


def test_explicit_public_route_initializes_rocm_in_fresh_process():
    import subprocess
    import sys

    target = _device_target()
    # The child must not import torch first: explicit hardware selection used
    # to load the toolkit DSO before the ROCm wheel initialized its libraries.
    code = f"""
import flash_rt
try:
    flash_rt.load_model('/nonexistent/flashrt-rdna-checkpoint',
                        hardware={target.hardware!r}, action_dim=7)
except FileNotFoundError as error:
    assert 'safetensors' in str(error)
else:
    raise AssertionError('missing checkpoint accepted')
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, timeout=60)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("compact", [False, True])
def test_library_fallbacks_preserve_encoder_mask_and_action_projection(monkeypatch, compact):
    import torch
    import torch.nn.functional as F
    from flash_rt.amd.hardware.rdna35 import Rdna35AttentionBackend, Rdna35GemmBackend

    target = _device_target()
    from flash_rt.amd import flash_rt_amd_kernels as kernels
    target.validate(kernels, target.gpu_arch)
    monkeypatch.setenv("FLASHRT_RDNA35_GEMM_AUTOTUNE", "0")
    monkeypatch.setenv("FLASHRT_RDNA35_HIP_ENCODER_ATTN", "1")
    monkeypatch.setenv("FLASHRT_RDNA35_HIP_SMALLM", "1")
    attention = Rdna35AttentionBackend(kernels, supports_encoder=False)
    gemm = Rdna35GemmBackend(kernels, supports_smallm=False)
    generator = torch.Generator(device="cuda").manual_seed(51)
    rows, valid = (559, 559) if compact else (712, 559)
    q = torch.randn(rows, 8, 256, device="cuda", dtype=torch.bfloat16, generator=generator)
    k = torch.randn(rows, 1, 256, device="cuda", dtype=torch.bfloat16, generator=generator)
    v = torch.randn_like(k)
    actual = attention.gqa(q, k, v, valid_prefix=valid, prefix_capacity=rows)
    reference = F.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0).float(),
        k[:valid].transpose(0, 1).unsqueeze(0).float(),
        v[:valid].transpose(0, 1).unsqueeze(0).float(), enable_gqa=True,
    ).squeeze(0).transpose(0, 1).reshape_as(actual)
    torch.testing.assert_close(actual.float(), reference, atol=.01, rtol=.03)

    x = torch.randn(10, 1024, device="cuda", dtype=torch.bfloat16, generator=generator)
    weight = torch.randn(1024, 32, device="cuda", dtype=torch.bfloat16, generator=generator)
    bias = torch.randn(32, device="cuda", dtype=torch.bfloat16, generator=generator)
    out = torch.empty(10, 32, device="cuda", dtype=torch.bfloat16)
    gemm.prepare_smallm_weight(weight)
    assert not gemm._smallm_weights
    assert not gemm.linear_residual(out, x, weight, bias)
    gemm.linear(out, x, weight, bias)
    reference = (x.float() @ weight.float() + bias.float()).bfloat16()
    torch.testing.assert_close(out, reference, atol=.5, rtol=.03)
