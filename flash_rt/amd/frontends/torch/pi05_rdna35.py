"""Pi0.5 BF16 frontend for AMD RDNA35."""
from flash_rt.amd.frontends.torch.pi05_rdna import (
    Pi05TorchFrontendRdna, _build_time_embeddings,
)
from flash_rt.amd.hardware.rdna.capabilities import RDNA35
from flash_rt.amd.models.pi05_rdna35.pipeline import Pi05PipelineRdna35


class Pi05TorchFrontendAmdRdna35(Pi05TorchFrontendRdna):
    _target = RDNA35
    _pipeline_class = Pi05PipelineRdna35
