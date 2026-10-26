"""Compatibility alias for :mod:`flash_rt.amd.hardware.rdna.gemm`."""

from ..rdna.gemm import RdnaGemmBackend
from ..rdna.gemm import RdnaGemmBackend as Rdna35GemmBackend

__all__ = ["RdnaGemmBackend", "Rdna35GemmBackend"]
