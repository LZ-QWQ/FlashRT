"""BF16 transformer primitives shared by the RDNA targets."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..rdna35.gemm import Rdna35GemmBackend


class RdnaTransformerOps:
    """Library GEMMs and explicit BF16 rounding at transformer boundaries.

    The Torch implementation is also the numerical/performance baseline for
    new fused kernels. All operators run on the caller's current stream.
    """

    def __init__(self, kernels, *, library_gemm=True, long_attention=None,
                 smallm_shapes=(), gemm_autotune=None):
        self.fused = True
        self.long_attention = long_attention
        self.smallm_shapes = frozenset(smallm_shapes)
        self.gemm = (Rdna35GemmBackend(kernels, autotune=gemm_autotune)
                     if library_gemm else None)

    def prepare_linear(self, weight, rows):
        if (self.gemm is not None and 1 <= rows <= 48
                and tuple(weight.shape) in self.smallm_shapes):
            self.gemm.prepare_smallm_weight(weight)

    def linear(self, x, weight, bias=None):
        shape = (*x.shape[:-1], weight.shape[1])
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        if self.gemm is None:
            return F.linear(flat, weight.t(), bias).view(shape)
        out = torch.empty((flat.shape[0], weight.shape[1]),
                          device=x.device, dtype=x.dtype)
        self.gemm.linear(out, flat, weight, bias)
        return out.view(shape)

    @staticmethod
    def layer_norm(x, weight=None, bias=None, eps=1e-5):
        return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)

    def rms_norm(self, x, weight, eps=1e-6):
        if self.fused:
            from .normalization import normalize
            return normalize(x, weight, mode=0, eps=eps)[0]
        # Qwen3 rounds the normalized value before its BF16 learned weight multiply.
        y = x.float()
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)
        return y.to(x.dtype) * weight

    def ada_layer_norm(self, x, scale, shift, eps=1e-5):
        if self.fused:
            from .normalization import normalize
            return normalize(x, scale, shift, mode=2, eps=eps)[0]
        # Preserve the reference's BF16 round after norm, 1+scale, and multiply.
        y = F.layer_norm(x, (x.shape[-1],), eps=eps)
        return y * (1 + scale) + shift

    def rope(self, x, cos, sin):
        if self.fused:
            from .normalization import rope
            return rope(x, cos, sin)
        # Qwen3-VL rotate-half uses FP32 arithmetic, then returns the input dtype.
        first, second = x.float().chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        return (x.float() * cos[:, None].float()
                + rotated * sin[:, None].float()).to(x.dtype)

    def attention(self, q, k, v, *, causal=False):
        if self.long_attention is not None and q.shape[1] >= 128:
            return self.long_attention(q, k, v, causal=causal)
        return self.sdpa(q, k, v, causal=causal)

    @staticmethod
    def sdpa(q, k, v, *, causal=False):
        """BSHD multi-head attention, including native GQA."""
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            is_causal=causal, enable_gqa=q.shape[2] != k.shape[2])
        return y.transpose(1, 2).contiguous()

    def residual_layer_norm(self, update, residual, scale=None, shift=None, eps=1e-5):
        if self.fused:
            from .normalization import normalize
            return normalize(update, scale, shift, residual=residual,
                             mode=1 if scale is None else 2, eps=eps)
        summed = update + residual
        normalized = (self.layer_norm(summed, eps=eps) if scale is None
                      else self.ada_layer_norm(summed, scale, shift, eps))
        return normalized, summed
