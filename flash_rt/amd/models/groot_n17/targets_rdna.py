"""GR00T N1.7 BF16 operator policies, selected once for each RDNA target."""


def make_transformer_ops(kernels, capabilities):
    from flash_rt.amd.hardware.rdna.attention import attention_bf16
    from flash_rt.amd.hardware.rdna.transformer import RdnaTransformerOps

    # Matched BF16 microbenchmarks and full-model validation select the
    # long-sequence attention backend and profitable small-M weight layouts.
    policies = {
        "gfx1100": (attention_bf16, ((1536, 1536), (1536, 1024), (1024, 1024), (1024, 1536)), None),
        # Timed selection among hipBLASLt algorithms changes BF16 rounding on
        # gfx1151/gfx1201 enough to exceed the physical-action regression gate.
        # Keep the library's first heuristic; small-M WMMA is unchanged.
        "gfx1151": (attention_bf16, (), False),
        "gfx1201": (None, ((1536, 1024), (1024, 1024), (1024, 1536)), False),
    }
    attention, shapes, autotune = policies[capabilities.gpu_arch]
    return RdnaTransformerOps(kernels, long_attention=attention, smallm_shapes=shapes,
                             gemm_autotune=autotune)
