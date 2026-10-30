"""N1.7 attention geometries on gfx1100/gfx1151/gfx1201."""

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module", autouse=True)
def rdna_device():
    if not torch.version.hip or not torch.cuda.is_available():
        pytest.skip("requires ROCm and an RDNA GPU")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch not in ("gfx1100", "gfx1151", "gfx1201"):
        pytest.skip("requires gfx1100, gfx1151 or gfx1201")


@pytest.mark.parametrize("batch,sq,sk,hq,hk,dim,causal", [
    (2, 256, 256, 16, 16, 64, False),
    (1, 461, 461, 16, 8, 128, True),
    (1, 461, 461, 32, 32, 64, False),
    (1, 41, 41, 32, 32, 48, False),
    (1, 41, 13, 32, 32, 48, False),
    (1, 41, 448, 32, 32, 48, False),
    (1, 1, 1, 16, 8, 128, True),
    (1, 17, 65, 4, 2, 48, False),
])
def test_attention_reference_and_graph(batch, sq, sk, hq, hk, dim, causal):
    from flash_rt.amd.hardware.rdna.attention import attention_bf16
    torch.manual_seed(17)
    q = torch.randn(batch, sq, hq, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, sk, hk, dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    k_heads = k.float().transpose(1, 2).repeat_interleave(hq // hk, dim=1)
    v_heads = v.float().transpose(1, 2).repeat_interleave(hq // hk, dim=1)
    scores = q.float().transpose(1, 2) @ k_heads.transpose(-1, -2) * dim ** -0.5
    if causal:
        mask = torch.ones(sq, sk, device="cuda", dtype=torch.bool).triu(1)
        scores.masked_fill_(mask, -torch.inf)
    reference = (scores.softmax(-1) @ v_heads).transpose(1, 2)
    actual = attention_bf16(q, k, v, causal=causal)
    torch.testing.assert_close(actual.float(), reference, atol=0.02, rtol=0.02)
    assert torch.nn.functional.cosine_similarity(actual.float().flatten(), reference.flatten(), dim=0) > 0.9999
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        attention_bf16(q, k, v, causal=causal)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = attention_bf16(q, k, v, causal=causal)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, actual)


@pytest.mark.parametrize('width,mode', [(128, 0), (2048, 0), (1536, 1), (1536, 2), (4096, 2)])
def test_fused_normalization(width, mode):
    from flash_rt.amd.hardware.rdna.normalization import normalize
    torch.manual_seed(21)
    x = torch.randn(41, width, device='cuda', dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(width, device='cuda', dtype=torch.bfloat16)
    bias = torch.randn_like(weight)
    for use_residual in (False, True):
        summed = (x + residual) if use_residual else x
        if mode == 0:
            ref = (summed.float() * (summed.float().square().mean(-1, keepdim=True) + 1e-5).rsqrt()).bfloat16() * weight
        else:
            ref = torch.nn.functional.layer_norm(summed, (width,), eps=1e-5)
            if mode == 2:
                ref = ref * (1 + weight) + bias
        actual, total = normalize(x, weight, bias, residual=residual if use_residual else None, mode=mode)
        if use_residual:
            assert torch.equal(total, summed)
        torch.testing.assert_close(actual.float(), ref.float(), atol=0.032, rtol=0.016)
        assert torch.nn.functional.cosine_similarity(actual.double().flatten(), ref.double().flatten(), dim=0) > 0.99999


def test_strided_rope():
    from flash_rt.amd.hardware.rdna.normalization import rope
    x = torch.randn(67, 3, 16, 64, device='cuda', dtype=torch.bfloat16)[:, 0]
    cos, sin = [torch.randn(67, 64, device='cuda') for _ in range(2)]
    a, b = x.float().chunk(2, dim=-1)
    ref = (x.float() * cos[:, None] + torch.cat((-b, a), -1) * sin[:, None]).bfloat16()
    assert torch.equal(rope(x, cos, sin), ref)
