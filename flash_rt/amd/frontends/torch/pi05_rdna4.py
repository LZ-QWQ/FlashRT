"""Pi0.5 BF16 frontend for AMD RDNA4."""
from flash_rt.amd.frontends.torch.pi05_rdna import (
    Pi05TorchFrontendRdna, _build_time_embeddings,
)
from flash_rt.amd.hardware.rdna.capabilities import RDNA4
from flash_rt.amd.models.pi05_rdna4.pipeline import Pi05PipelineRdna4


class Pi05TorchFrontendAmdRdna4(Pi05TorchFrontendRdna):
    _target = RDNA4
    _pipeline_class = Pi05PipelineRdna4
