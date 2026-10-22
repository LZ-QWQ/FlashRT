"""Pi0.5 BF16 compute binding for RDNA4."""
from flash_rt.amd.models.pi05_rdna35.pipeline import Pi05PipelineRdna35


class Pi05PipelineRdna4(Pi05PipelineRdna35):
    """Reuse the shared Pi0.5 tensor pipeline with target-selected providers."""
