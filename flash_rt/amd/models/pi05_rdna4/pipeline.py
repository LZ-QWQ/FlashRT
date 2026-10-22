"""Pi0.5 BF16 compute binding for RDNA4."""
from flash_rt.amd.models.pi05_rdna.pipeline import (
    Pi05PipelineRdna, VIS_L, VIS_D, VIS_H, VIS_NH, VIS_HD,
    ENC_L, ENC_D, ENC_H, ENC_NH, ENC_HD,
    DEC_L, DEC_D, DEC_H, DEC_NH, DEC_HD, PATCHES_PER_VIEW, ACTION_DIM,
)


class Pi05PipelineRdna4(Pi05PipelineRdna):
    """Use the shared tensor pipeline and target-selected RDNA providers."""
