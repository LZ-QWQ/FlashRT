"""RDNA3/4 initialization for the existing BF16 Pi0.5 lifecycle."""
import torch

from flash_rt.amd.frontends.torch.pi05_rdna35 import Pi05TorchFrontendAmdRdna35


class Pi05TorchFrontendRdna(Pi05TorchFrontendAmdRdna35):
    """Reuse checkpoint loading, prompt/cache handling and inference unchanged."""

    def _validate_backend(self, hardware, use_fp8):
        target = self._target
        if hardware not in (None, target.hardware):
            raise ValueError(
                f"{type(self).__name__} requires hardware={target.hardware!r}, "
                f"got {hardware!r}")
        if use_fp8:
            raise ValueError(f"{target.hardware} currently supports BF16 only")
        if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
            raise RuntimeError(f"{target.hardware} requires a ROCm build of PyTorch")
        device_arch = str(getattr(
            torch.cuda.get_device_properties(torch.cuda.current_device()),
            "gcnArchName", "unknown"))
        if device_arch.split(":", 1)[0] != target.gpu_arch:
            raise RuntimeError(
                f"{target.hardware} requires {target.gpu_arch}; "
                f"running device architecture is {device_arch!r}")

        from flash_rt.amd import flash_rt_amd_kernels as fvk

        target.validate(fvk, device_arch)
        return fvk
