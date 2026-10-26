"""gfx1100 attention policy over the shared wave32 implementation."""

from ..rdna.attention import RdnaAttentionBackend


class Rdna3AttentionBackend(RdnaAttentionBackend):
    def __init__(self, kernels):
        super().__init__(kernels)
        # W7900D: SDPA wins at 655 dense rows (190 vs 314 us), while
        # native wins with padding (314 vs 1262 us at 712/655 rows).
        self.hip_dense_encoder_gqa = False
