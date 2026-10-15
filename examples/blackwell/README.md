# FlashRT on RTX 5090 (Blackwell, SM120)

## Prerequisites

- RTX 5090 or other Blackwell GPU
- CUDA Toolkit 13.0+
- Python 3.10+, PyTorch 2.x

## Step 1: Install Dependencies

For the full install path (Docker / native, CMake build) see
[`docs/INSTALL.md`](../../docs/INSTALL.md). Minimum:

```bash
pip install torch pybind11 pyyaml numpy safetensors
# NOTE: pip install flash-attn is NOT required — FlashRT vendors
# Flash-Attention 2 source and builds it into flash_rt_fa2.so during
# cmake. See README §Build for details.
```

## Step 2: Download CUTLASS and Build

```bash
cd FlashRT

# CUTLASS (header-only)
git clone --depth 1 --branch v4.4.2 \
  https://github.com/NVIDIA/cutlass.git third_party/cutlass

# Build flash_rt_kernels module
mkdir build && cd build
cmake ..              # auto-detects SM120
make -j$(nproc)
make install          # installs .so → flash_rt/
cd ..
```

Verify:

```bash
python -c "import flash_rt; print(flash_rt.__version__)"
```

## Step 3: Download Checkpoint

```bash
# Pi0.5 LIBERO checkpoint (requires OpenPI access)
python -c "from openpi.models import download; download('pi05_libero_pytorch')"
```

## Step 4: Run Evaluation

```bash
python examples/blackwell/eval_libero.py \
  --checkpoint /path/to/pi05_libero_pytorch \
  --task_suite libero_spatial
```

## VLA latency (RTX 5090)

Pi0.5 below reports observation-to-final-action wall-clock p50 for the
fastest verified tiers. The GROOT tables report pure CUDA Graph replay p50
(`cuda.Event` around `graph.replay()`). Replay excludes graph-external work
such as image normalization, H2D upload, D2H action download, post-process
un-normalization, and Python wrapper overhead.

### Pi0.5

| Tier (2 views, `pi05_libero`) | Observation-to-action p50 | Frequency |
|---|---:|---:|
| **JAX/Orbax NVFP4 prefix + decoder** | **8.78 ms** | **114 Hz** |
| Torch NVFP4 prefix + decoder (`use_fp4=True`) | **9.31 ms** | **107 Hz** |

These timings use a fixed text prompt without state tokens, and exclude
loading, calibration and graph capture. They were measured in separate runs.
The JAX result is the fastest verified tier: RTX 5090, cuBLASLt 13.1, 8 real
LIBERO observations, 30 warmups and 200 synchronized measured calls. Numerical
checks and the exact boundary are in the
[JAX compatibility follow-up](../../docs/pi05_prefix_nvfp4.md#compatibility-follow-up-2026-10-16).
Select it with the [Orbax frontend precision options](../../docs/pi05_prefix_nvfp4.md#jax--orbax-frontend-compatibility).
The Torch tier's method and quality gates are in
[Pi0.5 NVFP4 numbers](../../docs/pi05_prefix_nvfp4.md#numbers).

### GROOT N1.6

Use a trained embodiment such as `gr1`; the base checkpoint's default
placeholder embodiment emits untrained actions.

T=50, padded production horizon:

| Views | Replay p50 | Frequency |
|-------|------------|-----------|
| 1 view | **11.90 ms** | 84 Hz |
| 2 views | **13.08 ms** | 76 Hz |
| 3 views | **13.92 ms** | 72 Hz |

T=16, LIBERO-style short horizon:

| Views | Replay p50 | Frequency |
|-------|------------|-----------|
| 1 view | **11.31 ms** | 88 Hz |
| 2 views | **12.53 ms** | 80 Hz |
| 3 views | **13.36 ms** | 75 Hz |

GROOT N1.6 E2E precision is cosine **0.9992** vs the Isaac-GR00T
`Gr00tN1d6` reference on `gr1`, with matched noise and matched
post-vlln backbone features.

### Pi0-FAST

50-token end-to-end, Orbax/JAX frontend:

| Mode | Quickstart P50 | Throughput |
|------|---------------:|-----------:|
| **Default** (`decode_cuda_graph=False`) | **147.4 ms** | **~340 tok/s** |
| **Max-perf** (`decode_cuda_graph=True`) | **122.9 ms** | **~410 tok/s** |

Detailed Pi0-FAST per-token breakdown: default is about 12 ms prefill
+ 2.87 ms per decode token; max-perf is about 11 ms prefill + 2.39 ms
per decode token.

## Troubleshooting

**`ModuleNotFoundError: flash_rt_fa2`**: build step skipped or `cp *.so ../flash_rt/` not run after `make`. See [`docs/INSTALL.md`](../../docs/INSTALL.md) §6.

**`CMake Error: pybind11 not found`**: Run `pip install pybind11`.

**`ENABLE_NVFP4: DISABLED`**: Expected if GPU is not SM120. NVFP4 is optional (FP8 is the primary path).
