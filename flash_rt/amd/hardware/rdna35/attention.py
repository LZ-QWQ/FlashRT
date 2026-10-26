"""Compatibility alias for :mod:`flash_rt.amd.hardware.rdna.attention`."""

from ..rdna.attention import RdnaAttentionBackend
from ..rdna.attention import RdnaAttentionBackend as Rdna35AttentionBackend

__all__ = ["RdnaAttentionBackend", "Rdna35AttentionBackend"]
