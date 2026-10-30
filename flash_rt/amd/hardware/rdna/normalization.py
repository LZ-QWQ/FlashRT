"""Wave32 BF16 normalization and rotate-half primitives.

Intermediate BF16 casts preserve the materialized transformer's rounding,
including RMSNorm's weight multiplication and AdaLayerNorm's affine steps.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _norm(X, W, B, R, O, SUM, WIDTH: tl.constexpr, EPS: tl.constexpr,
          MODE: tl.constexpr, RESIDUAL: tl.constexpr, BLOCK: tl.constexpr):
    col = tl.arange(0, BLOCK)
    row = tl.program_id(0)
    x = tl.load(X + row * WIDTH + col, col < WIDTH, 0).to(tl.float32)
    if RESIDUAL:
        r = tl.load(R + row * WIDTH + col, col < WIDTH, 0).to(tl.float32)
        x = (x + r).to(tl.bfloat16).to(tl.float32)
        tl.store(SUM + row * WIDTH + col, x, col < WIDTH)
    if MODE == 0:  # Weighted RMSNorm: normalized BF16, then BF16 multiply.
        variance = tl.sum(x * x, 0) / WIDTH
        y = (x * tl.rsqrt(variance + EPS)).to(tl.bfloat16).to(tl.float32)
        y = y * tl.load(W + col, col < WIDTH, 0).to(tl.float32)
    else:
        mean = tl.sum(x, 0) / WIDTH
        centered = tl.where(col < WIDTH, x - mean, 0.0)
        variance = tl.sum(centered * centered, 0) / WIDTH
        y = (centered * tl.rsqrt(variance + EPS)).to(tl.bfloat16).to(tl.float32)
        if MODE == 2:  # AdaLayerNorm; scale and shift are shared across rows.
            scale = tl.load(W + col, col < WIDTH, 0).to(tl.float32)
            shift = tl.load(B + col, col < WIDTH, 0).to(tl.float32)
            gamma = (1.0 + scale).to(tl.bfloat16).to(tl.float32)
            y = (y * gamma).to(tl.bfloat16).to(tl.float32) + shift
    tl.store(O + row * WIDTH + col, y, col < WIDTH)


def normalize(x, weight=None, shift=None, *, residual=None, mode=0, eps=1e-5):
    if x.dtype != torch.bfloat16 or x.device.type != "cuda":
        raise TypeError("RDNA normalization requires a BF16 device tensor")
    if x.ndim < 2 or x.numel() == 0 or mode not in (0, 1, 2) or eps <= 0:
        raise ValueError("unsupported RDNA normalization shape, mode or epsilon")
    width = x.shape[-1]
    parameters = (weight,) if mode == 0 else ((weight, shift) if mode == 2 else ())
    for value in parameters:
        if (value is None or value.shape != (width,) or value.dtype != x.dtype
                or value.device != x.device or not value.is_contiguous()):
            raise ValueError("normalization parameters must be contiguous BF16 vectors on the input device")
    if residual is not None:
        if residual.shape != x.shape or residual.dtype != x.dtype or residual.device != x.device:
            raise ValueError("normalization residual must match the input shape, dtype and device")
        residual = residual.contiguous()
    x = x.contiguous()
    out = torch.empty_like(x)
    summed = torch.empty_like(x) if residual is not None else out
    _norm[(x.numel() // width,)](
        x, weight, shift, residual, out, summed, width, eps, mode,
        residual is not None, triton.next_power_of_2(width),
        num_warps=4, enable_fp_fusion=False)
    return out, summed


@triton.jit
def _rope(X, C, S, O, SIZE: tl.constexpr, HEADS: tl.constexpr, DIM: tl.constexpr,
          XS: tl.constexpr, XH: tl.constexpr, CS: tl.constexpr, SS: tl.constexpr,
          BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    channel = index % DIM
    head = index // DIM % HEADS
    row = index // (DIM * HEADS)
    offset = row * XS + head * XH
    value = tl.load(X + offset + channel, index < SIZE, 0).to(tl.float32)
    other = tl.load(X + offset + (channel + DIM // 2) % DIM, index < SIZE, 0).to(tl.float32)
    other = tl.where(channel < DIM // 2, -other, other)
    cos = tl.load(C + row * CS + channel, index < SIZE, 0).to(tl.float32)
    sin = tl.load(S + row * SS + channel, index < SIZE, 0).to(tl.float32)
    tl.store(O + index, value * cos + other * sin, index < SIZE)


def rope(x, cos, sin):
    if x.dtype != torch.bfloat16 or x.device.type != "cuda":
        raise TypeError("RDNA RoPE requires a BF16 device tensor")
    if (x.ndim != 3 or x.numel() == 0 or x.shape[-1] % 2 or x.stride(-1) != 1
            or any(t.shape != (x.shape[0], x.shape[-1]) or t.device != x.device
                   or t.dtype not in (torch.bfloat16, torch.float32) or t.stride(-1) != 1
                   for t in (cos, sin))):
        raise ValueError("RDNA RoPE requires SHD input and matching SD cosine/sine tables")
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    _rope[(triton.cdiv(x.numel(), 256),)](
        x, cos, sin, out, x.numel(), x.shape[1], x.shape[2],
        x.stride(0), x.stride(1), cos.stride(0), sin.stride(0), 256,
        num_warps=4, enable_fp_fusion=False)
    return out
