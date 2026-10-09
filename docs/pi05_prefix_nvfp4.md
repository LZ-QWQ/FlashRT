# Pi0.5 on NVFP4 (RTX 5090, sm_120a): an optional tier

Build with `-DGPU_ARCH=120 -DFLASHRT_ENABLE_PI05_SKINNY=ON
-DFLASHRT_ENABLE_PI05_NVFP4=ON` to use this optional tier. Both options
default to OFF.

The tier runs the Pi0.5 vision tower, the Gemma-2B FFNs and the action
decoder on NVFP4 (e2m1 values, UE4M3 scales per 16 elements, a per-tensor
global scale). Every quantization happens inside the kernel that produces the
operand — a norm, a GEMM epilogue or a decoder consumer — so the graph has no
standalone quantize or dequantize pass around the 4-bit GEMMs.

It is a tier, not the default: against the FP8 skinny profile it trades about
1e-4 of action cosine (vs the FP16 engine) for ~1.46x lower E2E latency.
Qualify it per task before deploying.

## Use

```python
model = flash_rt.load_model(ckpt, config="pi05", framework="torch",
                            hardware="rtx_sm120", num_views=2, use_fp4=True)
```

`use_fp4=True` on `rtx_sm120` selects both halves of the tier. They can also be
chosen separately on the frontend:

```python
rt = Pi05TorchFrontendRtx(ckpt, num_views=2,
                          prefix_precision="nvfp4",    # default "fp8"
                          decoder_precision="nvfp4")   # default "fp8"
```

or with `FLASHRT_PI05_PREFIX_PRECISION` / `FLASHRT_PI05_DECODER_PRECISION`.
The tier needs the FP8 frontend (`use_fp8=True`): its calibration supplies the
static global scales below. `reload_weights` re-quantizes in place.

## What runs where

| stage | precision | kernels per layer |
|---|---|---|
| SigLIP attention (27 layers) | NVFP4 | LayerNorm → e2m1; QKV GEMM with the bias in its epilogue; attention read straight from the QKV rows (`fused_qkv_attention`, head_dim 72 padded to 80, no head-split pass); output → e2m1; O GEMM with bias + residual in its epilogue |
| SigLIP FFN | NVFP4 | LayerNorm → e2m1; up GEMM with bias + GELU → e2m1 epilogue (hidden padded 4304 → 4352); down GEMM with bias + residual |
| Gemma attention (18 layers) | FP8 | 4-bit Q/K at head_dim 256 costs ~0.2 action cosine on real frames, so Q/K/V/O stay FP8; FA2 runs unsplit on SM120 |
| Gemma FFN | NVFP4 | residual + RMSNorm → e2m1; gate\|up GEMM over row-interleaved weights with GeGLU → e2m1 in the epilogue; even-column compaction; down GEMM (stream-K) |
| decoder (10 steps x 18 layers) | NVFP4 | skinny W4A4 family: weights prefetched into registers before the programmatic-dependent-launch wait, K-split partials, fused consumers (residual + AdaRMSNorm → e2m1, gate GELU → e2m1, Q/K RoPE) |

Static global scales: the inputs of the down projections are GELU / GeGLU
outputs whose range reaches ~1e4 in late Gemma layers, beyond the 2688 that a
global scale of 1 can represent. Their global scale is the FP8 calibration's
`amax / 448`, applied as the producing epilogue's norm constant and folded into
the down GEMM's alpha. Normalized inputs run with a global scale of 1. Weights
use `amax / (6 * 448)`; the prefix weights also search the block scale for the
least squared error (on the decoder that search measured slightly worse, so its
block scales round up).

`FLASHRT_PI05_ENC_O_NVFP4=1` (opt-in) moves the Gemma attention output
projection to NVFP4 as well: ~1% faster, action cosine min 0.9992 → 0.9990 on
the frames below.

## Numbers

RTX 5090, 2 views, `pi05_libero_pytorch`, synchronized observation-to-final-
action E2E with `tools/bench_pi05_e2e.py` (state in the prompt every frame,
200 iterations, 8 LIBERO frames, one container and fixture for all rows):

| `--profile` | prefix | decoder | p50 | p95 |
|---|---|---|---:|---:|
| `default` | FP8 | FP8 | 20.89 ms | 21.11 ms |
| `skinny` | FP8 | FP8 skinny | 15.13 ms | 15.41 ms |
| `nvfp4` | NVFP4 | FP8 skinny | 11.23 ms | 11.53 ms |
| `nvfp4-full` | NVFP4 | NVFP4 skinny | 10.37 ms | 10.63 ms |

The captured graph of the `nvfp4-full` tier replays in 8.24 ms (vision 1.75,
encoder 3.3, decoder 3.25 ms under CUPTI); the rest of the E2E is the host-side
tokenization, image upload and action download.

Agreement on 24 real LIBERO frames, same prompt and pinned noise, against the
FP16 frontend:

| tier | action cosine mean | min |
|---|---:|---:|
| FP8 prefix + FP8 skinny decoder | 0.99998 | 0.99997 |
| NVFP4 prefix + NVFP4 decoder | 0.99986 | 0.99915 |

Replays are bit-identical. `tests/test_pi05_prefix_nvfp4.py` gates the tier
against FP8, the NVFP4 decoder, the batched pipeline and a weight reload.

## Limits

- sm_120a only.
- The Gemma attention (FP8 projections, FA2 at head_dim 256, ~31 us per layer)
  is now the largest prefix cost after the GEMMs.
- The numbers above are the engineering gate; closed-loop success rates decide
  per task whether a deployment uses the tier.
