# Pi0.5 BF16 on AMD RDNA

The PyTorch Pi0.5 frontend supports these exact ROCm targets:

| GPU | ISA | `hardware` | Encoder attention | Action projection |
| --- | --- | --- | --- | --- |
| Radeon Pro W7900 / W7900D | `gfx1100` | `amd_rdna3` | gfx11 WMMA / SDPA | gfx11 WMMA / hipBLASLt |
| Ryzen AI Max+ 395 / Radeon 8060S | `gfx1151` | `amd_rdna35` | gfx11 WMMA / SDPA | gfx11 WMMA / hipBLASLt |
| Radeon AI PRO R9700 | `gfx1201` | `amd_rdna4` | SDPA | hipBLASLt |

All three share the Pi0.5 tensor pipeline, checkpoint conversion and wave32
BF16 providers. Each has its own frontend and compute binding. The gfx12
build excludes the two gfx11 WMMA implementations and their Python bindings;
its library fallbacks preserve the encoder mask and action residual contracts.
Environment switches cannot enable those missing kernels. Other gfx targets
are not implied by a generation name.

## Container

From the repository root, build one target per image:

```bash
docker build -f docker/Dockerfile.rocm10.1.0 \
  --build-arg GPU_ARCH=gfx1100 -t flashrt:rocm10.1-gfx1100 .
# Use GPU_ARCH=gfx1201 for R9700, or gfx1151 for Strix Halo.
```

The recipe starts from the pinned ROCm 10.1.0 Ubuntu 24.04 **dev** image,
installs Python 3.13.13, and installs PyTorch 2.14.0 and Triton 3.8.0 from
AMD's `stable.repo.amd.com/rocm/whl-next` index, including only the three
RDNA device packages listed above. The native extension builds
against `/opt/rocm`. It does not install CUDA extras or include checkpoints.
The `environment` stage can also be built alone for development.
`ROCM_IMAGE` can select a registry mirror of the same base digest;
`UV_PYTHON_INSTALL_MIRROR` optionally selects a Python release download mirror.

Select an idle GPU's render node (confirm the PCI mapping in
`/dev/dri/by-path`). Run with the host device groups, especially when using
an unprivileged UID:

```bash
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri/renderD128 \
  --group-add "$(stat -c %g /dev/kfd)" \
  --group-add "$(stat -c %g /dev/dri/renderD128)" \
  --security-opt seccomp=unconfined --shm-size=8g \
  -v /path/to/shared/models:/models:ro \
  -e FLASH_RT_PALIGEMMA_TOKENIZER=/models/paligemma_tokenizer.model \
  flashrt:rocm10.1-gfx1100
```

The host needs a working amdgpu/KFD driver for the device; a container does
not install the host driver. Verify `torch.cuda.is_available()` and
`torch.cuda.get_device_properties(0).gcnArchName` inside the container.
The ROCm PyTorch API intentionally retains the `torch.cuda` namespace.

For an existing ROCm environment, install the runtime dependencies and use
an editable checkout, then build the standalone AMD target:

```bash
python -m pip install --no-deps -e .
ROCM_PATH=/opt/rocm bash scripts/amd/build_amd.sh gfx1100
python -c 'import torch; from flash_rt.amd import flash_rt_amd_kernels as k; print(k.build_info())'
```

Initialize ROCm PyTorch before importing the native extension directly;
ROCm wheels initialize their bundled runtime libraries. `load_model()`
handles this order. Rebuild when changing ISA or Python ABI: a checkout
contains only one `flash_rt_amd_kernels` module. The frontend rejects a
mismatched build/device pair before loading the checkpoint.

## Load and predict

Provide an OpenPI-compatible `model.safetensors` checkpoint, matching
normalization statistics and the PaliGemma SentencePiece tokenizer (see
[the tokenizer instructions](../flash_rt/utils/paligemma_tokenizer.py)).
The robot output schema is explicit; LIBERO uses `action_dim=7`.

```python
import numpy as np
import flash_rt

model = flash_rt.load_model(
    "/models/pi05_libero", config="pi05", framework="torch",
    hardware="auto", use_fp8=False, action_dim=7,
)
actions = model.predict(
    images=[base_rgb, wrist_rgb],  # two uint8 arrays, (224, 224, 3)
    prompt="pick up the object",
    state=normalized_state,       # normalize with the checkpoint statistics
)
assert actions.shape == (model.pipeline.chunk_size, 7)
assert np.isfinite(actions).all()
```

`num_views=3` adds the right wrist camera. `config.json.action_horizon`
determines the chunk length (default 10); `num_steps` controls the denoising
schedule. Set `action_dim`, or supply `output_action_dim` in `config.json`.
Constant robot channels are retained, not inferred as padding.

This initial tier supports **Pi0.5 / torch / BF16 only**. The historical
`use_fp8=True` default warns and selects BF16; pass `use_fp8=False` explicitly.
FP4, FP16, FA4, other model/framework combinations and named precision tiers
are rejected. Leave `precision="auto"`. This does not add CDNA, CUDA, JAX,
NPU or MPS functionality.

`predict()` uses eager inference. The existing `cache_frames` schedule and
low-level fixed-noise full/decoder Graph APIs remain available. The legacy
`FLASHRT_RDNA35_*` provider switches also apply to RDNA3/4; the names are
retained for compatibility. For correctness qualification, set
`FLASHRT_RDNA35_GEMM_AUTOTUNE=0` to disable the existing lazy GEMM tuning.
No performance claim is made by this initial port.

## Qualification

The public API smoke test uses synthetic images and your real checkpoint:

```bash
python examples/pi05_rdna_quickstart.py --checkpoint /models/pi05_libero --action-dim 7
```

Run on each target with its matching extension:

```bash
export FLASHRT_RDNA35_GEMM_AUTOTUNE=0
python -m pytest -q -ra \
  tests/test_amd_rdna_targets.py tests/test_amd_rdna35_routing.py \
  tests/test_amd_rdna35_ops.py tests/test_amd_arch_gate.py \
  tests/test_amd_routing.py tests/test_pi05_torch_pipeline.py

export FLASH_RT_PI05_RDNA_CKPT=/models/pi05_libero
export FLASH_RT_PI05_ACTION_DIM=7
python -m pytest -q -ra tests/test_amd_rdna35_model.py
```

The historical test filenames now cover all three RDNA targets. gfx12 skips
only native gfx11 WMMA tests and runs the explicit library fallback tests.
Missing/mismatched extensions fail on a supported visible GPU. The old
`FLASH_RT_PI05_RDNA35_CKPT` variable is also accepted.

Model tests cover finite actions, fixed-noise determinism, eager/Graph parity,
full/decoder-only cache refresh and native/library numerical parity. Optional
`FLASH_RT_PI05_REFERENCE` points to an independently generated OpenPI NPZ
fixture with checkpoint hash and producer revision; without that fixture,
these checks establish implementation consistency, not independent policy
accuracy or robot task success.
