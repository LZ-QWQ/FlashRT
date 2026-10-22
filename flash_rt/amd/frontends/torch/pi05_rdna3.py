"""Pi0.5 BF16 frontend for AMD RDNA3."""
from flash_rt.amd.frontends.torch.pi05_rdna import (
    Pi05TorchFrontendRdna, _build_time_embeddings,
)
from flash_rt.amd.hardware.rdna.capabilities import RDNA3
from flash_rt.amd.models.pi05_rdna3.pipeline import Pi05PipelineRdna3


class Pi05TorchFrontendAmdRdna3(Pi05TorchFrontendRdna):
    _target = RDNA3
    _pipeline_class = Pi05PipelineRdna3
