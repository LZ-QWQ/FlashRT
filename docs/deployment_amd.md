# FlashRT on AMD GPUs

This is the model-independent guide to the AMD backend: supported
hardware, how to build the extension, how the tree is laid out, how
hardware routing works, and how to run the tests.

Per-model deployment guides:

| Model | Guide | Status |
|---|---|---|
| Pi0.5 | [deployment_amd_pi05.md](deployment_amd_pi05.md) | CDNA4 FP8/BF16; RDNA 3/3.5/4 BF16 |
| GROOT N1.7 | [deployment_amd_groot_n17.md](deployment_amd_groot_n17.md) | FP8 backbone + bf16 DiT |

## Supported hardware

The native HIP extension has architecture-selected source sets for gfx942
(CDNA3 / MI300), gfx950 (CDNA4 / MI350), gfx1100 (RDNA 3 / W7900D),
gfx1151 (RDNA 3.5 / Radeon 8060S), and gfx1201 (RDNA 4 / AI PRO R9700).
RDNA implementations remain in the existing `csrc/amd` layout and
use an `_rdna` filename suffix; they do not compile or reuse the CDNA
wave64/MFMA kernels.

The extension is built for exactly one architecture. CDNA3 uses E4M3 FNUZ
(maximum finite value 240), CDNA4 uses OCP E4M3 (maximum finite value 448),
and the RDNA backend is BF16-only. Using a mismatched source set can produce
wrong results rather than a slow fallback, so the restriction is enforced at
build time and frontend initialization:

- **At build time** — `scripts/amd/build_amd.sh` accepts the exact gfx942,
  gfx950, gfx1151, gfx1100, and gfx1201 source sets and rejects other
  `GPU_ARCH` values. Set
  `FLASHRT_AMD_ALLOW_ARCH=1` to override when bringing up a port to a
  future architecture.
- **At CDNA frontend init** — the frontend reads the extension's
  `device_arch()` (the running device's `gcnArchName`, e.g.
  `gfx950:sramecc+:xnack-`) and `build_info()["gpu_arch"]` (the
  compile-time target) and raises `RuntimeError` unless both exactly match.
  This fires before the checkpoint is touched, so forcing
  an explicit AMD hardware target on the other generation fails immediately instead
  of computing garbage.
- **At RDNA frontend init** — the Pi0.5 frontend requires an extension whose
  `build_info()` reports the selected RDNA backend, wave32, and the exact
  gfx target, with a matching visible device. RDNA3/4 also check the compiled
  WMMA capability. A CDNA extension cannot be
  loaded through the RDNA route, or vice versa.

## Build

The AMD sources live in a standalone tree with their own build entry
point; the root (CUDA) CMake project is not involved and does not need
CUDA or CUTLASS present.

```bash
bash scripts/amd/build_amd.sh gfx950
# or, on MI300X:
bash scripts/amd/build_amd.sh gfx942
# or, on RDNA 3.5:
PYTHON=.venv/bin/python bash scripts/amd/build_amd.sh gfx1151
# or, on W7900D / AI PRO R9700 (one target per checkout):
bash scripts/amd/build_amd.sh gfx1100
# bash scripts/amd/build_amd.sh gfx1201
```

Output: `flash_rt/amd/flash_rt_amd_kernels*.so`. CMake selects the source
set for the target generation.

Dependencies:

- **hipBLASLt** — ships with ROCm and is required by both AMD source sets.
  RDNA 3.5 uses an instance-local BF16 algorithm cache populated by runtime
  measurements before HIP Graph capture.
- **No vendored third-party code.** There is no CUTLASS/CK checkout to
  manage; everything else is HIP C++ plus compiler MFMA intrinsics on CDNA4
  or WMMA builtins on RDNA 3.5.
- **aiter** (strongly recommended for CDNA4) — AMD's assembly flash-attention
  library. When importable it serves the attention sites; otherwise the
  backend falls back to torch SDPA. Attention is the largest kernel
  bucket on this backend, so the fallback is expensive: measured on the
  Pi0.5 quickstart, one variable changed, same process and node —

  | Attention path | Median |
  |---|---|
  | aiter (default when importable) | 16.3 ms |
  | torch SDPA (aiter absent, or `FVK_AMD_ATTN=sdpa`) | 22.2 ms |

  If a deployment measures roughly 6 ms above the published numbers,
  check that aiter is importable in the serving environment first.

## Pi0.5 on RDNA 3 and RDNA 4

Initial support is limited to `config="pi05"`, `framework="torch"`, BF16,
and these exact build/device pairs:

| Validated GPU | ISA / hardware name | Encoder attention / action projection |
|---|---|---|
| Radeon Pro W7900D | `gfx1100` / `amd_rdna3` | Existing gfx11 WMMA / small-M WMMA |
| Radeon AI PRO R9700 | `gfx1201` / `amd_rdna4` | PyTorch SDPA / hipBLASLt BF16 |

The gfx12 build excludes the two gfx11-only WMMA translation units and their
bindings. Its providers always use the library paths for those operations,
even if `FLASHRT_RDNA35_HIP_ENCODER_ATTN=1` or `FLASHRT_RDNA35_HIP_SMALLM=1`.
Other wave32 BF16 operators and decoder attention reuse the RDNA source set.
The frontend validates the selected ISA, backend, wave size and WMMA
capability before loading weights. FP4, FP16, FA4 and other model/framework
combinations are not supported. The historical `use_fp8=True` API default
is explicitly disabled with a warning; pass `use_fp8=False` for BF16.

The new targets reuse the existing RDNA Pi0.5 frontend lifecycle and the
model-owned tensor pipeline. Halo keeps its existing initialization checks
and defaults. The CUDA and CDNA runtime routes are unchanged. Existing
`FLASHRT_RDNA35_*` controls retain their names; default `predict()` is eager
with `cache_frames=1` (fresh vision/encoder work on every frame).

### Environment and validation

Both discrete GPUs were validated with ROCm 10.1.0, Python 3.13.13,
`torch==2.14.0+rocm10.1.0` and
`triton==3.8.0+git669b31ac.rocm10.1.0` from
`https://stable.repo.amd.com/rocm/whl-next`. The PyTorch installation included
the `device-gfx1100` and `device-gfx1201` extras. The HIP module links against
ROCm's hipBLASLt; NumPy, safetensors, SentencePiece and pybind11 are also
required. These are the tested versions, not a claim of a minimum version.
Import PyTorch before directly importing the native extension; the new public
hardware routes initialize the ROCm wheel libraries in that order.

Checkpoint: [Fisher-Wang/pi05-libero-pytorch](https://huggingface.co/Fisher-Wang/pi05-libero-pytorch),
revision `ab107fbe6661e8abfec0f13dc27bfd3fafe24c69`,
`model.safetensors` SHA256
`0f8c489e37b01c72251c45f2e73595894f3933fc6297f4f1cf95fc8737db4c74`.
This FP32 checkpoint is converted to BF16 by the existing loader. LIBERO
requires `action_dim=7`; its action horizon is 10. Configure the local
PaliGemma SentencePiece tokenizer as described in the
[Pi0.5 guide](deployment_amd_pi05.md).

```bash
export FLASH_RT_PI05_RDNA_CKPT=/path/to/pi05-libero-pytorch
export FLASH_RT_PI05_ACTION_DIM=7
export FLASH_RT_PALIGEMMA_TOKENIZER=/path/to/paligemma_tokenizer.model
export OMP_NUM_THREADS=8
python -m pytest \
  tests/test_amd_rdna_targets.py tests/test_amd_rdna35_routing.py \
  tests/test_amd_rdna35_ops.py tests/test_amd_rdna35_model.py \
  tests/test_amd_arch_gate.py tests/test_amd_routing.py \
  tests/test_pi05_torch_pipeline.py tests/test_amd_no_global_patching.py \
  tests/test_amd_cdna3_capabilities.py -q -ra -p no:cacheprovider
```

The `rdna35_ops/model` filenames are retained for existing users and commands;
they now exercise the shared operators/model on all three RDNA targets.
`FLASH_RT_PI05_RDNA35_CKPT` remains accepted as a legacy checkpoint variable.
RDNA3/4-specific routing and capability contracts live in
`test_amd_rdna_targets.py`; Halo routing and CDNA tests stay separate.

The above suite passed 184 tests on W7900D and 161 on R9700 (17 / 40 skips).
Both ran ten real-checkpoint cases: finite actions, fixed-noise determinism,
eager/full/decoder graph parity, prompt/state cache invalidation, temporal
cache scheduling and native/library numerical parity. Graph parity is exact;
the existing native/library acceptance bound is cosine >= 0.999. R9700 skips
23 gfx11-only WMMA cases. Remaining skips cover unavailable hardware,
optional Hugging Face dependencies and the independent OpenPI reference
fixture. Halo also rebuilt successfully on its existing ROCm 10.0 environment
(178 passed, 23 skipped; no checkpoint configured). CUDA routing is covered by
mocked API regression tests; CUDA/CDNA device execution was not revalidated.
This is synthetic-input correctness evidence, not a LIBERO task-success result.

### Reproduce inference and timing

With the environment variables above, this runs the public BF16 API and
separately measures the internal full-pipeline graph. Build/import, weight
loading, graph capture and lazy GEMM autotuning are outside the timed loops.
No `FLASHRT_RDNA35_*` overrides were used.

```python
import os
import time
import numpy as np
import torch
import flash_rt

torch.manual_seed(17)
model = flash_rt.load_model(
    os.environ["FLASH_RT_PI05_RDNA_CKPT"], config="pi05", framework="torch",
    hardware="auto", use_fp8=False, num_views=2, num_steps=10,
    action_dim=7, cache_frames=1)
rng = np.random.default_rng(17)
images = [rng.integers(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(2)]
state = np.zeros(8, dtype=np.float32)

def predict():
    return model.predict(images=images, prompt="pick up the object", state=state)

actions = predict()
assert actions.shape == (10, 7) and np.isfinite(actions).all()
print("actions:", actions.shape, actions[0])

def measure(run):
    for _ in range(20):
        run()
    samples = []
    for _ in range(50):
        torch.cuda.synchronize()
        started = time.perf_counter()
        run()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000)
    return np.percentile(samples, [50, 95])

print("predict median/p95 ms:", measure(predict))
frontend = model.pipeline
frontend._fill_images({"images": images})
noise = torch.from_numpy(rng.standard_normal((10, 32)).astype(np.float32)).to(
    device="cuda", dtype=torch.bfloat16)
eager = frontend.forward_with_fixed_noise(frontend._image_buf, noise).clone()
graph = frontend.forward_with_fixed_noise(
    frontend._image_buf, noise, use_graph=True).clone()
torch.testing.assert_close(graph, eager, atol=0, rtol=0)
print("full graph replay median/p95 ms:", measure(frontend.pipeline._graph.replay))
```

Measurements on the validated checkpoint, batch 1, two 224x224 RGB views,
ten denoising steps, action shape `(10, 7)`, zero eight-dimensional state,
`"pick up the object"` prompt (143 tokens; encoder length 655), seed 17,
20 warmups and 50 measured iterations:

| GPU | Full `predict()` median / p95 | Full graph replay median / p95 |
|---|---|---|
| W7900D | 88.91 / 89.73 ms | 86.27 / 86.48 ms |
| AI PRO R9700 | 57.88 / 59.64 ms | 56.01 / 56.17 ms |

Both metrics are synchronized host wall time. `predict()` includes image
preparation, input transfers and CPU action postprocessing/output. Graph
replay excludes those steps and reuses fixed GPU inputs; it is not the
public API latency. Graph output matched eager output exactly on both GPUs.
These are initial support measurements on shared servers, without performance
tuning or an upstream speedup claim. W7900D had one 323.60 ms `predict()`
outlier; its mean was 93.72 ms (R9700 mean: 58.15 ms).

## Layout

```
csrc/amd/                    standalone HIP tree (own CMake entry point)
  bindings.cpp               pybind module flash_rt_amd_kernels
  bindings_rdna.cpp          gfx1151-only pybind surface
  gemm/                      hipBLASLt + CDNA MFMA / RDNA WMMA kernels
  arch.h                     FP8 format, datatype, and capability selection
  attention/                 CDNA and `_rdna` encoder/decoder attention
  kernels/                   norm / activation / quantize / fusion families;
                              `_rdna` files are selected only for gfx1151
flash_rt/amd/
  core/hip_buffer.py         ctypes device memory over libamdhip64
  core/hip_graph.py          ctypes HIP graph capture / instantiate / replay
  hardware/cdna3|cdna4/      generation-specific attention backends
  hardware/rdna35/           BF16 attention and GEMM providers
  models/<model>/pipeline.py pointer-only forward passes
  frontends/torch/<model>.py weight load, calibration, capture, infer
  models/pi05_rdna35/        BF16 RDNA Pi0.5 pipeline
  frontends/torch/pi05_rdna35.py
```

The pybind entry points keep the same `uintptr_t` pointer + stream ABI as the
CUDA module. RDNA-only entries carry an `_rdna` suffix so they cannot be
silently routed on CDNA. CDNA model frontends warm up and replay a captured
HIP graph on their production path. The RDNA Pi0.5 frontend defaults to eager
execution and `cache_frames=1`; explicit values greater than one enable
decoder-only frames that reuse the last encoded K/V prefix.

## Hardware routing

Auto-detection uses the device's `gcnArchName`: gfx942 maps to `amd_cdna3`,
gfx950 maps to `amd_cdna4`, gfx1151 maps to `amd_rdna35`, gfx1100 maps
to `amd_rdna3`, and gfx1201 maps to `amd_rdna4`. An unregistered
ROCm architecture raises rather than falling through to an NVIDIA table entry.

```python
import flash_rt

model = flash_rt.load_model(checkpoint_dir, config="pi05",
                            framework="torch")            # auto-detected
model = flash_rt.load_model(checkpoint_dir, config="pi05",
                            framework="torch",
                            hardware="amd_cdna3")         # explicit MI300X
model = flash_rt.load_model(checkpoint_dir, config="pi05",
                            framework="torch",
                            hardware="amd_cdna4")         # explicit
model = flash_rt.load_model(checkpoint_dir, config="pi05",
                            framework="torch",
                            hardware="amd_rdna35",
                            use_fp8=False)                 # gfx1151 BF16
```

Expected failures:

| Situation | Error |
|---|---|
| Extension not built | `ImportError` naming `flash_rt_amd_kernels` and the build command |
| CDNA build/device mismatch | `RuntimeError` naming the device and build architectures |
| RDNA build/backend/wave size/device mismatch | `RuntimeError` naming the required RDNA source set and gfx target |
| Model/framework not ported to AMD | `RuntimeError` from pipeline resolution |
| Thor-only options (`use_fp4_decoder`, `use_fa4`) | `ValueError` naming the supported hardware |

## Build environment knob

Model and architecture-specific runtime knobs are documented in the
per-model guides. The only shared build escape hatch is:

| Env | Default | Meaning |
|---|---|---|
| `FVK_AMD_ATTN` | `aiter` | attention backend: `aiter` or `sdpa` (torch fallback) |
| `FLASHRT_FP8_NT_AUTOTUNE` | `auto` | timed hipBLASLt algorithm selection at setup; `off` uses heuristic top-1 |
| `FLASHRT_FP8_ALGO_POOL` | `16` | candidate pool depth for the timed selection. Deeper pools (64/128) sometimes find faster algorithms but widen run-to-run pick variance, and a single mis-timed trial can ship a slow algorithm |
| `FLASHRT_AMD_ALLOW_ARCH` | `0` | build-script escape hatch for an unregistered AMD ISA port |

`FVK_AMD_ATTN`, `FLASHRT_FP8_NT_AUTOTUNE`, and `FLASHRT_FP8_ALGO_POOL`
belong to the CDNA runtime. The RDNA Pi0.5 backend has its own
`FLASHRT_RDNA35_*` controls and never reads those CDNA FP8/aiter settings.

## Tests

```bash
python -m pytest tests/test_amd_routing.py tests/test_amd_extension.py \
                 tests/test_amd_hip_graph.py tests/test_amd_kernel_parity.py \
                 tests/test_amd_pi05_model.py \
                 tests/test_amd_rdna_targets.py \
                 tests/test_amd_rdna35_routing.py \
                 tests/test_amd_rdna35_ops.py \
                 tests/test_amd_rdna35_model.py -v
```

| File | Covers | Skips when |
|---|---|---|
| `test_amd_routing.py` | pipeline-map entries, `detect_arch()` ROCm branches (including unregistered-target rejection), `load_model` failure modes | mostly runs anywhere; extension-dependent cases skip without the `.so` |
| `test_amd_extension.py` | `build_info()` / `device_arch()` coherence, required-symbol inventory of every bound kernel | extension not importable |
| `test_amd_hip_graph.py` | buffer round-trips with pattern data, capture → instantiate → replay, byte-identical repeat replays | no ROCm device or extension |
| `test_amd_kernel_parity.py` | numerical parity of the kernel surface against torch references on real-distribution inputs, including FP8 byte-exactness and the seqused fixed-shape attention path | no ROCm device or extension |
| `test_amd_pi05_model.py` | end-to-end model load, graph capture, pinned-noise determinism, exact/fixed prompt modes, FP8 and BF16 | no checkpoint (see the per-model guide for the environment variables) |
| `test_amd_rdna35_routing.py` | gfx1151 routing, lazy imports, BF16/profile contracts, and CDNA/Triton isolation | torch unavailable; hardware-only cases require gfx1151 |
| `test_amd_rdna_targets.py` | RDNA3/4 routing, exact build/device/capability matching, gfx12 library fallbacks, CUDA route isolation and legacy Halo initialization | numerical cases require gfx1100 or gfx1201 |
| `test_amd_rdna35_ops.py` | RDNA native-kernel parity, validation, graph replay, and library fallbacks | no supported RDNA device; gfx11 WMMA-only cases skip on gfx1201; a missing/mismatched extension fails |
| `test_amd_rdna35_model.py` | RDNA Pi0.5 finite output, determinism, full-graph parity, and optimized/fallback parity | no supported RDNA device or checkpoint |
| `test_amd_groot_routing.py` | GROOT N1.7 pipeline-map entry, precision-tier contracts, attention-backend site/layer validation | extension-dependent cases skip without the `.so` |
| `test_amd_groot_model.py` | GROOT N1.7 end-to-end: kernel backbone, finite actions, pinned-noise determinism, optional reference cosine | no checkpoint |

Everything skips cleanly with a stated reason on machines without ROCm,
so the suite is safe to run in a CUDA-only CI.

## Measuring

- Report **medians** after warmup. Report a minimum only as a lower
  bound, never as the headline.
- On CDNA4, timed hipBLASLt algorithm selection moves cross-run medians by roughly
  ±0.2–0.35 ms. Compare arms **inside one process** where possible.
- When judging output cosine against a saved reference, **pin the
  denoise noise** to the exact array the reference was generated with.
  A fresh random draw shifts cosine by about 1e-3, which is the same
  magnitude as a real numerical regression and will mask it.
- Some ROCm profiler reports fold dispatch gaps into the reported
  kernel time. Use them for ranking buckets; take absolute per-call
  numbers from isolated in-graph chains.

## HIP versus CUDA notes

- CDNA4 uses wave64 reductions from `common_hip.h`; RDNA 3.5 uses wave32
  reductions from `common_hip_rdna.h`. HIP has no CUDA-style `*_sync`
  shuffle variants.
- The three-argument graph instantiate is `hipGraphInstantiateWithFlags`.
- Memcpy-kind and capture-mode enum values match CUDA numerically
  (validated on hardware by the runtime-seam smoke test).
- FP8 storage comes from `csrc/amd/arch.h`: E4M3 FNUZ with
  `HIP_R_8F_E4M3_FNUZ` on CDNA3, and OCP E4M3 with `HIP_R_8F_E4M3` on
  CDNA4. Weights and activation scales must be regenerated for the target.
- hipBLASLt matmul is column-major; the GEMM runner uses the
  operand-swap form (`D_col = B_col @ A_col`), the same trick the CUDA
  FP8 paths use.
- gfx950 exposes `V_MFMA_F32_16X16X32_FP8_FP8` and `..._BF16` with
  per-lane contiguous 8-byte fragments; the hand-written GEMMs repack
  weights at setup into per-lane consumption order so each workgroup
  streams its weight tile linearly.
- On gfx942 the packed FNUZ FP8 and packed BF16 Pi0.5 decoder paths are
  available. MXFP4 remains unavailable; unsupported forms route through
  hipBLASLt. Encoder attention uses AITER.
- Every HIP runtime call in the Python layer is return-code checked; a
  failed launch, copy or synchronise raises instead of letting a stale
  buffer be read back as a result.

Packed BF16 routing uses `build_info()`'s `packed_bf16_shapes` (M, N, K)
and `packed_bf16_layout` together. CDNA3 declares the Pi0.5 decoder forms
(10, 2560, 1024), (10, 1024, 2048), and (10, 8192, 1024) with
`mfma_k16_lane4` packing. CDNA4 declares the GROOT square projections
(M, 1536, 1536), for 1 <= M <= 48, with `mfma_k32_lane8` packing.
Both frontends require a matching shape and layout before packing; GROOT
therefore retains hipBLASLt on CDNA3. The broad
`supports_packed_bf16_mfma` flag alone does not authorize a model route.
