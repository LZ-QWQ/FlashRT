"""Weight preparation for the FlashRT FlashDrive backend: FlashDrive's ParoQuant (Marlin W4A8) linears unpacked into
the tcgen05 prefill GEMM's operands, the per-channel INT8 requantization tier, the pairwise-rotation tables in the
fused activation kernel's bank order (and as dense matrices), and the fragment-ordered BF16 / FP8 / INT8 copies of
the small linears (LM head, DFlash draft) for the few-row GEMMs. Runs once when the backend is installed."""
from __future__ import annotations

import torch

from flash_rt.models.alpamayo15.marlin_w4a8 import marlin_w4a8_unpack
from flash_rt.models.alpamayo15.weights import fp8_fragment_layout, quantize_fp8_rows, quantize_int8_rows


def marlin_w4a8_to_int8(qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor, K: int, N: int,
                        maps: dict | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """``(w8, s)``: ``w8 [N, K]`` int8 = ``w - z`` and ``s [K/128, N]`` int32 (the tcgen05 prefill GEMM operands)."""
    w, z, s = marlin_w4a8_unpack(qweight, scales, qzeros, K, N, maps)
    G = K // 128
    w8 = (w.to(torch.int16) - z.to(torch.int16).repeat_interleave(128, 0)).to(torch.int8).t().contiguous()
    return w8, s


def w4a8_to_int8_channel(w8: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel INT8 requantization of Marlin's group-scaled integers: ``w8 [N, K]`` int8 (w - z) and ``s [K/128, N]``
    int32 group scales give the exact integer weight ``w8 * s[group]``; its per-channel symmetric INT8 codes
    (``round(v / c)``, ``c = absmax / 127``) and the fp32 channel scale ``c`` (1 for an all-zero channel) are returned
    for the per-channel INT8 prefill GEMM (``D = half(acc * a_scale[m] * c[n])``). A precision tier: the codes carry the
    group-scale ratios of a channel with at most half a step of error."""
    N, K = w8.shape
    v = w8.to(torch.int32) * s.t().repeat_interleave(128, 1)           # [N, K] exact integers (|w8| <= 15, s <= 4096)
    amax = v.abs().amax(dim=1).to(torch.float32)
    c = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
    codes = torch.round(v.to(torch.float32) / c[:, None]).clamp_(-127.0, 127.0).to(torch.int8).contiguous()
    return codes, c.contiguous()


def paro_pair_bank_order(theta: torch.Tensor, pairs: torch.Tensor, K: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Reorder the 64 pairs of every (layer, channel group) of a ParoQuant rotation so that the eight pairs a
    quarter-warp of the tile kernel handles together touch eight distinct shared-memory bank groups with their first
    channel (bank group = 16-byte chunk index ``c + c // 8`` mod 8 of the 8-row tile layout). The pairs of a layer are
    disjoint, so their order does not change the rotation; returns (theta, pairs) in the new order."""
    G = K // 128
    pr = pairs.view(8, G, 64, 2).to(torch.int64)
    th = theta.view(8, G, 64)
    i = pr[..., 0]
    res = (i + (i >> 3)) & 7
    ar = torch.arange(64, device=pairs.device)
    order = torch.argsort(res * 64 + ar, dim=-1)                       # pairs sorted by bank residue (stable)
    res_s = torch.gather(res, -1, order)
    counts = torch.nn.functional.one_hot(res, 8).sum(-2)                # [8, G, 8]
    starts = counts.cumsum(-1) - counts
    rank = ar - torch.gather(starts, -1, res_s)                         # rank within the residue class
    main = rank < 8
    pos = torch.where(main, rank * 8 + res_s, torch.zeros_like(rank))   # bucket = rank, lane-in-bucket = residue
    taken = torch.zeros_like(rank).scatter_add(-1, pos, main.long()) > 0
    free_order = torch.argsort(taken.long() * 64 + ar, dim=-1)           # free positions first, ascending
    ovf = (~main).long().cumsum(-1) - 1
    pos = torch.where(main, pos, torch.gather(free_order, -1, ovf.clamp(min=0)))
    new_pr = torch.empty_like(pr).scatter_(-2, pos.unsqueeze(-1).expand(-1, -1, -1, 2),
                                           torch.gather(pr, -2, order.unsqueeze(-1).expand(-1, -1, -1, 2)))
    new_th = torch.empty_like(th).scatter_(-1, pos, torch.gather(th, -1, order))
    return new_th.reshape(8, K // 2).contiguous(), new_pr.reshape(8, K).to(pairs.dtype).contiguous()


def paro_rotation_matrices(theta: torch.Tensor, pairs: torch.Tensor, K: int) -> torch.Tensor:
    """The 128 x 128 matrix of every channel group of a ParoQuant rotation: ``R[g][c][k]`` such that
    ``y[g*128 + c] = sum_k R[g][c][k] x[g*128 + k]`` equals the 8 Givens layers applied in order (FP32 products of
    the layer cosines/sines, returned as FP16 for the tensor-core kernel)."""
    G = K // 128
    R = torch.eye(128, device=theta.device, dtype=torch.float32).repeat(G, 1, 1)         # [G, 128, 128]
    th = theta.float().view(8, G, 64)
    pr = pairs.view(8, G, 64, 2).long()
    gidx = torch.arange(G, device=theta.device).unsqueeze(1)
    for r in range(8):
        c, s_ = torch.cos(th[r]).unsqueeze(-1), torch.sin(th[r]).unsqueeze(-1)       # [G, 64, 1]
        i, j = pr[r, :, :, 0], pr[r, :, :, 1]                                        # [G, 64]
        Ri, Rj = R[gidx, i], R[gidx, j]                                              # rows i, j: [G, 64, 128]
        R[gidx, i] = c * Ri + s_ * Rj
        R[gidx, j] = c * Rj - s_ * Ri
    return R.to(torch.float16).contiguous()


def bf16_fragment_layout(w: torch.Tensor) -> torch.Tensor:
    """Reorder a [N, K] bf16 weight into the ``flashdrive_bf16_gemm_frag`` operand: ``[ceil(N/16)][K/16][32 lanes][16 B]``,
    lane ``4g + c`` holding the four ``mma.m16n8k16`` A registers of one 16-wide k step (rows g, g+8; k 2c and 8+2c).
    Rows are zero-padded to a multiple of 16."""
    N, K = w.shape
    if K % 16:
        raise ValueError(f"K={K} must be a multiple of 16")
    T = (N + 15) // 16
    if T * 16 != N:
        w = torch.cat([w, w.new_zeros(T * 16 - N, K)])
    # n = 16T + 8rh + g ; k = 16kc + 8half + 2c + e  ->  [T, kc, g, c, half, rh, e]  (register a = 2half + rh)
    frag = w.view(T, 2, 8, K // 16, 2, 4, 2).permute(0, 3, 2, 5, 4, 1, 6).contiguous()
    return frag.view(-1)


class Bf16Fragment:
    """Fragment-ordered BF16 copy of a [N, K] linear weight for the streaming few-row GEMM."""

    def __init__(self, w: torch.Tensor) -> None:
        with torch.no_grad():
            self.frag = bf16_fragment_layout(w.detach())
        self.shape = tuple(w.shape)


class Fp8Fragment:
    """Per-row one-byte copy of a [N, K] linear weight (``fmt`` "fp8": E4M3 codes, "int8": symmetric INT8 codes; both
    with per-row FP32 scales) in the ``alpamayo15_w8a16_gemm_bf16`` fragment order."""

    def __init__(self, w: torch.Tensor, fmt: str = "fp8") -> None:
        with torch.no_grad():
            codes, scale = (quantize_int8_rows if fmt == "int8" else quantize_fp8_rows)(w.detach())
            self.frag, self.scale = fp8_fragment_layout(codes, scale)
        self.shape = tuple(w.shape)
        self.fmt = 1 if fmt == "int8" else 0
