"""Exact build/device pairs for the BF16 wave32 source set."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RdnaTarget:
    hardware: str
    gpu_arch: str
    backend: str
    gfx11_wmma: bool

    def validate(self, kernels, device_arch: str) -> None:
        info = dict(kernels.build_info())
        build = str(info.get("gpu_arch", "")).split(":", 1)[0]
        device = device_arch.split(":", 1)[0]
        extension_device = str(kernels.device_arch()).split(":", 1)[0]
        if (info.get("backend") != self.backend
                or info.get("wave_size") != 32
                or info.get("gfx11_wmma") is not self.gfx11_wmma
                or (build, device, extension_device) != (self.gpu_arch,) * 3):
            raise RuntimeError(
                f"{self.hardware} requires matching {self.gpu_arch} RDNA kernels "
                f"and device; build={info!r}, device={device_arch!r}, "
                f"extension_device={extension_device!r}. "
                f"Rebuild with bash scripts/amd/build_amd.sh {self.gpu_arch}")


RDNA3 = RdnaTarget("amd_rdna3", "gfx1100", "rdna3", True)
RDNA4 = RdnaTarget("amd_rdna4", "gfx1201", "rdna4", False)
