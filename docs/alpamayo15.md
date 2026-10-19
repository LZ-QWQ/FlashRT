# Alpamayo 1.5 on Jetson AGX Thor

FlashRT runs NVIDIA [Alpamayo 1.5](https://github.com/NVlabs/alpamayo1.5)
(`nvidia/Alpamayo-1.5-10B`) on Jetson AGX Thor (`sm_110`) in two forms:

- **Full requests** (`flashrt_full`): the official request procedure
  (`sample_trajectories_from_data_with_vlm_rollout`) on FlashRT kernels: the
  same multi-camera prompt, chain-of-causation reasoning with the official
  sampler, the same stop rule, all 10 flow-matching steps of the diffusion
  expert, the official noise draw and trajectory integration. Reference:
  the official model.
- **Streaming** (`fd_flashrt`): [FlashDrive](https://github.com/z-lab/flashdrive)'s
  streaming procedure (frame K/V reuse across windows, DFlash speculative
  reasoning, W4A8 language model, a fine-tuned expert with cached velocities)
  on FlashRT kernels. Reference: FlashDrive's own implementation.

The two procedures produce different outputs and are **not comparable with
each other**; every table below compares each form with its own reference on
the same inputs, windows and seeds.

| Arm | Weights | Procedure | Execution |
|---|---|---|---|
| `official` | `nvidia/Alpamayo-1.5-10B` | full 16-frame request, 10 expert steps | official model, SDPA |
| `flashrt_full` | the same, plus `z-lab/Alpamayo-1.5-10B-PARO` for the `balanced` / `fast` presets | the same full request | FlashRT engine (`Alpamayo15Thor`) |
| `flashdrive` | `z-lab/Alpamayo-1.5-10B`, `-PARO`, `-DFlash` | streaming: K/V reuse, DFlash, W4A8, velocity caching | FlashDrive, `torch.compile` |
| `fd_flashrt` | the same z-lab weights | the same streaming procedure | FlashDrive orchestration on FlashRT backends (`FlashDriveThor`) |

Contents: [platform](#platform-and-pinned-dependencies), [installation](#installation),
[checkpoints and data](#checkpoints-and-input-data), [usage](#usage),
[benchmark protocol](#benchmark-protocol), [results](#results-on-thor),
[precision tiers](#precision-tiers), [execution contracts](#execution-contracts),
[tests](#tests), [limitations](#limitations).

## Platform and pinned dependencies

Validated on Jetson AGX Thor (SM110, 20 SMs) with NVIDIA PyTorch 26.05:
Python 3.12.3, torch 2.12.0a0+5aff3928d8.nv26.05, CUDA 13.2, transformers
4.57.1, physical-ai-av 0.2.0, PyAV 19.0.1, CUTLASS v4.4.2 (vendored).

| Source | Full-request environment | Streaming environment |
|---|---|---|
| NVlabs/alpamayo1.5 | 36aeb4c | a5bd40c |
| NVlabs/alpamayo | not required | 4cda35d |
| z-lab/flashdrive | not required | 2626517 |
| vLLM | not required | v0.16.0, source-built SM110 W4A8 Marlin |
| ParoQuant | not required | 0.1.15 |
| FlashAttention 4 (`flash_attn/cute`, CuTe DSL forward) | vendored forward-only subset `csrc/attention/flash_attn_4_src` (snapshot 2025-07-04, BSD-3-Clause, `VENDOR.md`); runtime `nvidia-cutlass-dsl==4.5.1`, `quack-kernels==0.4.1` (`thor-fa4` extra) | the same |

Keep the platform's torch, torchvision and Triton: the upstream pins of these
projects target other builds. The model dependencies are optional and are not
imported by the FlashRT package itself. FA4 is the attention kernel of both
forms (prefill, decode, expert and vision attention); it is a CuTe DSL kernel
compiled at first use by `nvidia-cutlass-dsl`, loaded only through
`flash_rt.hardware.thor.fa4_backend`, and the engines refuse to start without
it.

## Installation

Run inside an ARM64 NVIDIA GPU environment with the CUDA toolkit (for Docker
on Jetson, configure the NVIDIA runtime). Start in the FlashRT checkout; the
directories below are relative and user-chosen.

Environments and sources:

```bash
python -m venv --system-site-packages .venv-official
python -m venv --system-site-packages .venv-flashdrive
mkdir -p dependencies
git clone https://github.com/NVlabs/alpamayo1.5 dependencies/alpamayo-official
git -C dependencies/alpamayo-official checkout 36aeb4c
git clone https://github.com/NVlabs/alpamayo1.5 dependencies/alpamayo-streaming
git -C dependencies/alpamayo-streaming checkout a5bd40c
git clone https://github.com/NVlabs/alpamayo dependencies/alpamayo-r1
git -C dependencies/alpamayo-r1 checkout 4cda35d
git clone https://github.com/z-lab/flashdrive dependencies/flashdrive
git -C dependencies/flashdrive checkout 2626517
git clone --branch v0.16.0 --depth 1 https://github.com/vllm-project/vllm dependencies/vllm
git clone --branch v4.4.2 --depth 1 https://github.com/NVIDIA/cutlass third_party/cutlass
```

Python packages (both environments), then the model packages per environment:

```bash
for env_dir in .venv-official .venv-flashdrive; do
  "$env_dir/bin/python" -m pip install accelerate==1.15.0 av==19.0.1 einops==0.8.2 \
    hydra-colorlog==1.2.0 hydra-core==1.3.7 pandas==3.0.2 physical-ai-av==0.2.0 \
    pillow==12.2.0 transformers==4.57.1 huggingface_hub==0.36.2 tokenizers==0.22.2 \
    safetensors==0.7.0 matplotlib seaborn pytest pybind11 cmake ninja uv_build setuptools wheel
  "$env_dir/bin/python" -m pip install --no-deps -e .
done
.venv-official/bin/python -m pip install --no-deps dependencies/alpamayo-official
.venv-flashdrive/bin/python -m pip install --no-deps dependencies/alpamayo-streaming \
  dependencies/alpamayo-r1 dependencies/flashdrive paroquant==0.1.15
.venv-flashdrive/bin/python -m pip install cbor2 gguf rich
VLLM_TARGET_DEVICE=empty .venv-flashdrive/bin/python -m pip install \
  --no-deps --no-build-isolation dependencies/vllm
source .venv-flashdrive/bin/activate
PYTHON=python bash scripts/build_alpamayo15_marlin.sh dependencies/vllm build-marlin
```

The Marlin recipe builds only vLLM's W4A8 operator schema and the SM110
kernels it needs, in a fresh build directory; other vLLM CUDA operations stay
unavailable. Do not reuse an extension built for another GPU or torch ABI.

FA4 runtime (CuTe DSL) and the FlashRT kernels. `FLASHRT_ENABLE_ALPAMAYO15`
is Thor-only and off by default; the Cosmos groups are build dependencies of
the GEMM helpers, not a change of model. On CUDA 13.2 compile FA4 with
`CUTE_DSL_ARCH=sm_101a` and `FLASH_ATTENTION_ARCH=sm_100a` (`sm_110a` is
rejected by that libNVVM); set the same two variables in the shell that runs
the benchmarks.

```bash
python - <<'PY'
from importlib.metadata import version
from pathlib import Path
Path('dependencies/platform-constraints.txt').write_text(
    ''.join(f'{name}=={version(name)}\n' for name in ('torch', 'torchvision', 'triton')))
PY
python -m pip install -c dependencies/platform-constraints.txt apache-tvm-ffi==0.1.14.post1 torch-c-dlpack-ext==0.1.5
python -m pip install -c dependencies/platform-constraints.txt -e '.[thor-fa4]'
export CUTE_DSL_ARCH=sm_101a FLASH_ATTENTION_ARCH=sm_100a
cmake -S . -B build-alpamayo -DGPU_ARCH=110 -DFLASHRT_ENABLE_ALPAMAYO15=ON \
  -DFLASHRT_ENABLE_COSMOS3_EDGE=ON -DFLASHRT_ENABLE_COSMOS3_REASONER=ON
cmake --build build-alpamayo --target flash_rt_kernels flash_rt_fp4 -j4
python -c 'import flash_rt.flash_rt_kernels, flash_rt.flash_rt_fp4'
python -c 'import vllm._C, torch; assert hasattr(torch.ops._C, "marlin_gemm")'
python -m pip freeze > dependencies/flashdrive-environment.lock.txt
.venv-official/bin/python -m pip install -c dependencies/platform-constraints.txt \
  apache-tvm-ffi==0.1.14.post1 torch-c-dlpack-ext==0.1.5 -e '.[thor-fa4]'
```

`flash_rt_kernels` holds every kernel of both forms; `flash_rt_fp4` adds the
NVFP4 GEMM and quantizers of the full-request `fast` preset (the `bf16` and
`balanced` presets and the streaming backend do not need it).

## Checkpoints and input data

Authenticate with `hf auth login` for gated resources; keep credentials out
of source and command lines. The official configuration references auxiliary
processor repositories: obtain access to them or prepopulate a verified
offline cache, and do not substitute another tokenizer or image processor.

| Checkpoint | Revision | Used by |
|---|---|---|
| nvidia/Alpamayo-1.5-10B | 7aba8293c09993f2e125c6819df05d7fa3e873ea | `official`, `flashrt_full` |
| z-lab/Alpamayo-1.5-10B-PARO | d029319a | `flashrt_full` (`balanced`, `fast`), `flashdrive`, `fd_flashrt` |
| z-lab/Alpamayo-1.5-10B | 03637f1f3acac01d8bf5a094147d6352570e6c46 | `flashdrive`, `fd_flashrt` |
| z-lab/Alpamayo-1.5-10B-DFlash | 2496c456 | `flashdrive`, `fd_flashrt` |
| nvidia/PhysicalAI-Autonomous-Vehicles | 33f9bf447ed3bcb7d545ce13f4226f824214fafb | input clips |

`z-lab/Alpamayo-1.5-10B` carries the official VLM tensors unchanged and a
fine-tuned action expert; `-PARO` is a full model file with the ParoQuant
W4A8 language model (its other tensors equal the base up to an FP16 round
trip of values below about 1e-5); `-DFlash` holds the draft model. The
weights are NVIDIA's (non-commercial license); FlashDrive's code is MIT.

Resolve the pinned snapshots into a dedicated cache before benchmarking, and
do not update them between comparator runs:

```bash
export HF_HOME="$PWD/.hf-cache"
python - <<'CACHE_PY'
from pathlib import Path
from huggingface_hub import snapshot_download
pins = {
    "nvidia/Alpamayo-1.5-10B": "7aba8293c09993f2e125c6819df05d7fa3e873ea",
    "z-lab/Alpamayo-1.5-10B": "03637f1f3acac01d8bf5a094147d6352570e6c46",
    "z-lab/Alpamayo-1.5-10B-PARO": "d029319a",
    "z-lab/Alpamayo-1.5-10B-DFlash": "2496c456",
}
for repo, revision in pins.items():
    snapshot = Path(snapshot_download(repo_id=repo, revision=revision))
    refs = snapshot.parent.parent / "refs"
    refs.mkdir(exist_ok=True)
    (refs / "main").write_text(snapshot.name)
CACHE_PY
```

Cache the auxiliary files before setting `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`; leave offline mode unset while preparing dataset
windows. FlashDrive's loader selects its optimized path from a `z-lab/` repo
ID, so run it from the pinned cache; the FlashRT frontends also accept local
directories.

Input windows come from the official loader (it streams the needed features;
a 120-window pack can take several GB of video chunk transport):

```bash
.venv-official/bin/python benchmarks/alpamayo15_thor.py --prepare \
  --clip 87147a1b-3eef-4c25-94d2-ec7718a49a7a --start-us 1700000 \
  --stride-us 100000 --windows 120 --output data/example.pt
```

A window is 4 cameras x 4 frames plus the ego history; consecutive windows
are 0.1 s apart. The streaming procedure requires that spacing.

## Usage

### Full-request engine

```python
from flash_rt.frontends.torch.alpamayo15_thor import Alpamayo15TorchFrontendThor

fe = Alpamayo15TorchFrontendThor("nvidia/Alpamayo-1.5-10B",
                                 engine_kwargs={"paro_checkpoint": "<Alpamayo-1.5-10B-PARO dir>"})
out = fe.infer(image_frames,      # (n_cameras, n_frames, 3, H, W) uint8
               camera_indices,    # (n_cameras,)
               ego_history_xyz,   # (1, 1, 16, 3) ego frame at t0
               ego_history_rot,   # (1, 1, 16, 3, 3)
               seed=42)
out["pred_xyz"], out["pred_rot"], out["cot"][0]
```

The frontend loads the official model, wires the engine over it and captures
the CUDA graphs. Several trajectory samples: construct with `max_samples=16`
and call `fe.infer(..., num_traj_samples=16)` (`pred_xyz` is then
`(1, 1, 16, 64, 3)`); the prompt is encoded once and shared by the samples.

Presets (`engine_kwargs={"preset": ...}`):

| Preset | What runs | Needs |
|---|---|---|
| `"balanced"` (default) | W4A8 ParoQuant decode projections, FP8 expert, FP8 prefill projections; BF16 vision | `paro_checkpoint` (the engine refuses to start without it) |
| `"bf16"` | the official weights everywhere | nothing beyond the official checkpoint |
| `"fast"` | `balanced` with NVFP4 (W4A4) prefill projections and an INT4 LM head | `paro_checkpoint`, the `flash_rt_fp4` module |

Single tiers override the preset's choice: `decode_weights` (`"bf16"`,
`"fp8"`, `"int8"`, `"w4a8"`), `expert_weights` (`"bf16"`, `"fp8"`),
`prefill_weights` (`"bf16"`, `"fp8"`, `"nvfp4"`), `vision_weights` (`"bf16"`,
`"fp8"`), `lm_head_weights` (`"decode"`, `"int4"`); for example
`engine_kwargs={"preset": "bf16", "decode_weights": "int8"}`. Other options:
`pdl` (programmatic dependent launch, off). Every tier other than `bf16` changes
numerics; the [results](#results-on-thor) give the measured effect of each
preset and the [precision tiers](#precision-tiers) section describes them.

### Streaming backend

```python
from flash_rt.frontends.torch.alpamayo15_flashdrive_thor import (
    Alpamayo15FlashDriveTorchFrontendThor, FlashDriveCheckpoints)

fe = Alpamayo15FlashDriveTorchFrontendThor(
    FlashDriveCheckpoints(base="z-lab/Alpamayo-1.5-10B",          # repo IDs from the
                          paro="z-lab/Alpamayo-1.5-10B-PARO",     # pinned cache, or
                          dflash="z-lab/Alpamayo-1.5-10B-DFlash"))# local directories
for window in windows:                 # consecutive windows of one clip, 0.1 s apart
    out = fe.step(window["image_frames"], window["camera_indices"],
                  window["ego_history_xyz"], window["ego_history_rot"])
    # out is None for the first window of a stream (prefill only)
fe.reset()                             # before an unrelated stream
```

Calls run under `torch.autocast("cuda", dtype=torch.bfloat16)`, as
FlashDrive's `scripts/infer.py` does. `FlashDriveRollout` holds FlashDrive's
controls (token budget 128, temperature 0.6, top-p 0.98, 8 Euler steps,
cached steps 3-6, one sample). `reset()` ends a stream completely. By default
every streaming primitive runs on FlashRT (`backends=` selects per group, see
[execution contracts](#execution-contracts)); FlashDrive's `torch_compile`
setting applies only to groups left on FlashDrive. Precision tiers of the
backend, off by default: `decode_weights="int8"` (LM head and draft linears),
`prefill_weights="int8"` (streaming-window projections), `expert_weights="fp8"`;
pass them in `backends=`, for example
`backends={"decode_weights": "int8", "prefill_weights": "int8", "expert_weights": "fp8"}`.

## Benchmark protocol

One runner drives the four arms on identical input packs:

```bash
export CUTE_DSL_ARCH=sm_101a FLASH_ATTENTION_ARCH=sm_100a
.venv-official/bin/python benchmarks/alpamayo15_thor.py --arm official \
  --inputs data/example.pt --seed 0 --output results/official.json
.venv-official/bin/python benchmarks/alpamayo15_thor.py --arm flashrt_full \
  --inputs data/example.pt --seed 0 --output results/flashrt-balanced.json          # default preset
.venv-official/bin/python benchmarks/alpamayo15_thor.py --arm flashrt_full --preset bf16 \
  --inputs data/example.pt --seed 0 --output results/flashrt-bf16.json
.venv-official/bin/python benchmarks/alpamayo15_thor.py --arm flashrt_full --preset fast \
  --inputs data/example.pt --seed 0 --output results/flashrt-fast.json
.venv-flashdrive/bin/python benchmarks/alpamayo15_thor.py --arm flashdrive \
  --inputs data/example.pt --seed 0 --output results/flashdrive.json
.venv-flashdrive/bin/python benchmarks/alpamayo15_thor.py --arm fd_flashrt \
  --inputs data/example.pt --seed 0 --output results/fd-flashrt.json
.venv-flashdrive/bin/python benchmarks/alpamayo15_thor.py --arm fd_flashrt \
  --engine-kw decode_weights=int8,prefill_weights=int8,expert_weights=fp8 \
  --inputs data/example.pt --seed 0 --output results/fd-flashrt-tiers.json
```

`--paro` (default `z-lab/Alpamayo-1.5-10B-PARO`, a repo ID in the pinned
cache or a local directory) supplies the ParoQuant weights to the `balanced`
and `fast` presets and to the streaming arms; `--engine-kw k=v,...` passes
single tiers to the engine or backend. Use locked GPU, memory and CPU clocks,
no competing workload and the site's GPU scheduler lock; load only packs you
created or trust (torch serialization).

Protocol of every table below:

- Inputs: nine clips of `nvidia/PhysicalAI-Autonomous-Vehicles` (the example
  clip `87147a1b` with 120 windows and eight clips with 60), windows from
  t0 = 1.7 s, 0.1 s apart, built by the official loader; 4 cameras x 4
  frames, 128 reasoning tokens, one trajectory sample. Windows 0-2 of each
  clip are warm-up and excluded; 573 steady windows per seed.
- Seeds 0, 1 and 2 (1719 windows per arm). Full requests re-seed every window
  with `seed * 1000 + k` as the official example does; streaming arms seed
  once per clip. Reasoning is sampled (temperature 0.6), so all comparisons
  are distribution-level; paired differences are per (seed, clip, window).
- Latency is the model time per window: tokenized inputs on the host to the
  trajectory on the host, synchronized, in a warm process (the runner's
  `request_ms` adds preprocessing and tokenization on top; graph capture,
  compilation and loading are reported separately as `load_s` and in the
  excluded early windows). Peak memory is `torch.cuda.max_memory_allocated`.
- Error is the ADE / FDE (meters, xy) of the single sampled trajectory
  against the logged future. It compares implementations of the same
  procedure; it is not an accuracy evaluation of the model and not a
  closed-loop driving metric.
- Brackets are 95% bootstrap intervals over the nine clips (the interval
  width, about +-0.05 m ADE, comes from the clips, not from the number of
  windows).
- The official standalone example uses a 256-token budget; all arms here use
  128. Do not mix budgets.

The runner reports the same statistics per window; the tables below were
produced with the same packs, windows and seeds.

## Results on Thor

Jetson AGX Thor, MAXN with locked clocks (GPC 1575 MHz, EMC 4266 MHz),
NGC PyTorch 26.05, the protocol above.

### Full requests

Nine clips x three seeds (1719 windows per arm):

| Path | mean ms | median ms | p90 ms | vs official (mean / median) | peak GB | ADE m | paired ADE vs official | paired ADE vs `bf16` |
|---|---|---|---|---|---|---|---|---|
| Official Alpamayo 1.5 (BF16, SDPA) | 3322 | 3168 | 3755 | - | 21.6 | 2.845 | - | |
| FlashRT `preset="bf16"` | 2097 | 1952 | 2495 | 1.58x / 1.62x | 22.3 | 2.852 | +0.008 [-0.009, +0.032] | - |
| FlashRT `preset="balanced"` (default) | 899 | 847 | 1021 | 3.70x / 3.74x | 35.0 | 2.830 | -0.015 [-0.045, +0.016] | -0.022 [-0.064, +0.009] |
| FlashRT `preset="fast"` | 812 | 761 | 925 | 4.09x / 4.16x | 32.0 | 2.844 | -0.001 [-0.082, +0.089] | -0.008 [-0.098, +0.079] |

`fast` against `balanced`: -87 ms [-95, -80], ADE +0.014 [-0.048, +0.074].
FDE moves with ADE (`fast` vs official +0.012 [-0.313, +0.308]). Every
interval includes zero: on this evaluation the presets are indistinguishable
from the official model and from each other. The `bf16` engine generates the
official model's token stream on 2 of 1719 windows: BF16 logits carry many
near-ties, and a near-tie step resolves differently once any kernel changes
its reduction order (as between the official model's SDPA and
FlashAttention-2 backends), even though the sampler draws exactly the
official random numbers; `balanced` generates the `bf16` engine's tokens on
1447 of 1719, `fast` on 1122.

Stage times of the FlashRT engine on the example clip (117 steady windows,
seed 0; CUDA-event means per request):

| Preset | mean ms | median ms | p90 ms | vision | LM prefill | decode ms / token | expert | peak GB |
|---|---|---|---|---|---|---|---|---|
| `bf16` | 1886 | 1882 | 2156 | 121 | 537 | 67.4 | 213 | 22.3 |
| `balanced` | 833 | 809 | 892 | 121 | 246 | 22.1 | 126 | 35.0 |
| `fast` | 737 | 745 | 827 | 121 | 184 | 20.8 | 126 | 32.0 |

Reasoning length averages 16 tokens on this clip (19 over the nine clips), so
decode is the largest stage of the `balanced` and `fast` presets. The sampler
kernel costs 0.12 ms per step. Each GEMM runs on the faster of FlashRT's
tcgen05 instantiations and cuBLASLt (called from C++) at its shape, measured in
place on Thor: the FP8 prefill projections on a 2-SM 256 x 256 x 128 tile with
per-row scales (360-375 TOPS; cuBLASLt reaches 287-357 with per-tensor scales
and 196 with per-row scales), the `bf16` preset's gate / up / down prefill
projections, its decode projections and every preset's one-row first-token
logits on cuBLASLt, its q|k|v and o on the BF16 256 x 256 tile.

### Streaming

Nine clips x three seeds (1719 windows per arm), FlashDrive's procedure
and settings (`scripts/infer.py`):

| Path | mean ms | median ms | p90 ms | vs FlashDrive (mean / median) | peak GB | ADE m | paired ADE vs FlashDrive |
|---|---|---|---|---|---|---|---|
| FlashDrive (unmodified, `torch.compile` max-autotune) | 958 | 856 | 1057 | - | 13.3 | 2.763 | - |
| FlashDrive + FlashRT backend (defaults) | 462 | 400 | 513 | 2.07x / 2.14x | 26.5 | 2.742 | +0.035 [-0.008, +0.067] |
| + all three tiers (`decode_weights="int8"`, `prefill_weights="int8"`, `expert_weights="fp8"`) | 342 | 296 | 389 | 2.80x / 2.89x | 27.6 | 2.744 | +0.024 [-0.038, +0.087] |

The FlashDrive rows pair on 1602 windows (its runs lack the example clip's
seed 0); the backend rows cover 1719. The three tiers together against the
default backend: -120 ms [-146, -103], ADE +0.002 [-0.071, +0.061]; the
backend samples FlashDrive's token stream on 743 of 1602 windows. FlashDrive
averages 4.7 DFlash rounds per window and 5.1 reasoning tokens per round, the
backend 4.9 and 5.2. Cold costs: FlashDrive's first window of a stream takes
4.3 s and its second 148 s (`torch.compile`, warm compile cache, one clip per
process); the backend ran each seed's nine clips in one process with
`reset()` between them, first window 0.7 s and second 1.1 s (CUDA graph
capture; 17 s and 19 s for the first stream of a process, which also builds
the FA4 kernels). The backend's peak memory includes the INT8 weight copy of
its tcgen05 W4A8 GEMM.

Stage times of the FlashRT backend on the example clip (117 steady windows,
seed 0; CUDA-event means per window, the DFlash column per round):

| Configuration | mean ms | median ms | p90 ms | vision | streaming prefill | DFlash round | expert | rounds / window |
|---|---|---|---|---|---|---|---|---|
| defaults | 458 | 402 | 511 | 27 | 144 | 35.8 | 87 | 4.6 |
| all three tiers | 342 | 306 | 389 | 27 | 96 | 29.4 | 53 | 4.5 |

Window latency follows the number of DFlash rounds, which varies with the
sampled reasoning, so compare stage columns between rows. A round is 8
verification rows through the language model plus the draft, the sampler
(0.18 ms for the 8 rows) and the acceptance decisions, all in one graph.

On identical cache and random state (12 windows, each FlashRT primitive
against FlashDrive eager; FlashDrive's compiled primitives against eager for
scale), the minimum cosine similarity of outputs / written cache rows:

| Primitive | FlashRT vs eager | FlashDrive compiled vs eager |
|---|---|---|
| vision | 0.9991 / - | 0.9988 / - |
| streaming prefill | 0.9993 / 0.9974 | 0.9993 / 0.9973 |
| DFlash round | 0.9987 / 0.9970 | 0.9090 / 0.8164 |
| trajectory start | - / 0.9986 | - / 0.9986 |
| expert | 1.0000 / 0.9999 | 1.0000 / 0.9994 |

Random-number consumption matched FlashDrive on every call, and the rotated
cache copy equalled the rotation of FlashDrive's cache after every FlashRT
primitive.

## Precision tiers

### Full-request engine

The `bf16` preset keeps the official BF16 weights everywhere. The other tiers
trade numerics for weight bytes; each changes the sampled tokens and is
validated by the nine-clip evaluation above.

- `decode_weights="w4a8"` (`balanced`, `fast`): the reasoning-decode
  projections run on the ParoQuant W4A8 language model of
  `z-lab/Alpamayo-1.5-10B-PARO` (learned pairwise rotations, INT4 weights with
  128-wide groups and zero points, per-token INT8 activations). One grouped
  few-row GEMM launch serves the linears that share an input (q|k|v, o,
  gate|up, down), on a weight layout that streams each tile's zero points and
  integer group scales with its nibbles; the activation quantization runs in
  the GEMM prologue and the next group's rotation tables are prefetched into
  the L2 from inside the GEMM. The LM head runs as INT8 (`balanced`) or on an
  own INT4 linear with 128-wide groups and per-channel factors (`fast`,
  `lm_head_weights="int4"`). `"fp8"` and `"int8"` are per-row copies of the
  official weights on a W8A16 kernel. Decode drops from 66.6 to 22.2 ms per
  token (21 with the INT4 head).
- `expert_weights="fp8"`: W8A8 expert projections (per-row FP8 activations
  and weights, scales in the GEMM epilogue), 212 -> 126 ms per request.
- `prefill_weights="fp8"`: W8A8 prefill projections on a 2-SM 256 x 256 FP8
  tcgen05 tile, the RMSNorm and SiLU-mul producers writing FP8 rows with their
  per-row scales directly; 537 -> 246 ms. `"nvfp4"` (`fast`): W4A4 on a 2-SM 256 x 256 NVFP4 tile (e2m1
  weights and activations, UE4M3 block-16 scales; per-tensor weight scales,
  per-row activation scales written by the fused producers, per-row alphas in
  the GEMM epilogue); 246 -> 184 ms. It needs the `flash_rt_fp4` module.
- Activation scales are per row and computed by the producer kernels that
  already hold the row (RMSNorm, SiLU-mul, LayerNorm, the attention-output
  pass), so they cost one in-register reduction in kernels that run at DRAM
  bandwidth; the weights' scales are static. Calibrated per-tensor static
  activation scales would not speed up these producers, and the per-row FP8
  GEMM on the 256 x 256 tile is faster than cuBLASLt's per-tensor FP8 GEMM at
  every prefill shape, so the tiers keep the per-row form (no calibration set).
- `vision_weights="fp8"` is available but slower than the BF16 tower on
  Thor (the per-row quantization of the attention and GELU outputs costs more
  than the GEMMs save) and not part of any preset.
- The prompt's last-token logits stay on the BF16 LM head in every preset
  (sampling the first token from a quantized head measured ADE +0.055
  [+0.001, +0.161]). The vision blocks' fc1 on cuBLASLt measured about 3 ms
  faster per request but its last-bit differences resample windows (ADE
  +0.047 [-0.007, +0.153]); the tower stays on the FlashRT GEMMs.

### Streaming backend

Off by default; numerics changes validated together on nine clips x three
seeds above: `decode_weights="int8"` (per-row symmetric INT8 copies of the LM
head and draft linears on the W8A16 kernel; `"fp8"` exists, INT8 keeps more
weight precision for the same bytes), `prefill_weights="int8"` (the
streaming-window projections as per-channel INT8 requantizations of Marlin's
group-scaled integers on a plain INT8 2-SM 256 x 256 tcgen05 GEMM, 276-370 TOPS on Thor
against 128-140 for the blockwise W4A8 GEMM), `expert_weights="fp8"` (W8A8
expert, as in the full-request engine). `act_rot="dense"` runs the ParoQuant
rotation as one 128 x 128 FP16 matrix per channel group (one FP16 rounding
instead of eight); `pdl=True` launches every kernel with programmatic
dependent launch (measured neutral on Thor).

## Execution contracts

Hot path: in both forms every GPU operation of a request after host
tokenization is launched from FlashRT's C++ code. GEMMs run on tcgen05 (CUTLASS
instantiations with bias, activation, residual and quantization folded into
the epilogues), on FlashRT's few-row kernels or, where it is faster at the
shape (`bf16` preset prefill gate / up / down and decode projections, the
one-row LM head), on cuBLASLt called from C++; attention runs on FA4, and the rest on the
model-local kernels of `csrc/kernels/alpamayo15_*.cu`: RMSNorm / LayerNorm /
SiLU-mul producers, embedding and image-row gathers and scatters, the M-RoPE
tables, the sampler with the decode bookkeeping, the action expert's input and
output projections with the Euler step, the unicycle trajectory integration
and, for streaming, the DFlash proposal and acceptance decisions. The only
torch operation left on the GPU is the official noise draw of the diffusion
expert (`torch.randn`), kept as the official draw point; memcpys move the
sampled tokens to the host for the stop rule. Each frame grid's vision tower,
each decode step (full request) or DFlash round (streaming), each streaming
window's prefill and the whole expert are CUDA graphs. The host prepares a
request (tokenization, trajectory-history tokens, M-RoPE positions and
FlashDrive's stream bookkeeping) before the graphs run. The
[kernel sources per phase](#kernel-sources-per-phase) below are the profiler's
account of this.

Two phases are launched kernel by kernel from C++ instead of being captured:
the full-request LM prefill and the first window of a stream. Their length
follows the prompt, and capturing them would only remove launch gaps that
are not there. A `balanced` prefill runs 516 kernels in 247 ms, about 0.5 ms
each, while a launch costs a few microseconds, so the GPU never waits for the
host: the profiler finds 0.4 ms of idle time between those kernels in total.
A graph could save at most that 0.4 ms per request, about 0.05%. The first
window of a stream runs once per stream and is not on the steady-state path.

Code layout: `flash_rt/models/alpamayo15/pipeline_thor.py` is the full-request
engine (buffers, graphs, the request path), `weights.py` its weight
preparation (fused projections, the tiers' copies, the action modules'
operands), `marlin_w4a8.py` the W4A8 layouts; `flash_rt/models/alpamayo15_flashdrive/`
holds the streaming backend and its `weights.py`; the frontends under
`flash_rt/frontends/torch/` do the official / FlashDrive preprocessing and
model loading (`load_flashdrive`); kernels live in `csrc/kernels/alpamayo15_*`,
`csrc/gemm/alpamayo15_*_sm100.cu`, `csrc/quantize/alpamayo15_nvfp4_rows.cu` and
the `alpamayo15_nvfp4_*` sources of `flash_rt_fp4`, all behind
`FLASHRT_ENABLE_ALPAMAYO15`.

### Full-request engine

| Stage | Official (Hugging Face eager) | FlashRT Thor |
|---|---|---|
| Vision tower (27 layers, 16 frames) | per-frame attention calls, FP32 LayerNorm/RoPE round trips, CPU-built position tables | one CUDA graph per frame grid: patch embedding (+ position embedding), qkv / proj / fc1 / fc2 and the mergers on 2-SM 256 x 256 tcgen05 BF16 GEMMs with bias, GELU and residual fused into the epilogues, LayerNorm kernel, fused in-place FP32 RoPE, batched FA4 attention on the interleaved q/k/v, per-grid cached tables |
| LM prefill (36 layers) | eager modules, `DynamicCache` | embedding and image rows gathered by kernels, the M-RoPE tables from host-computed positions, fused q/k/v GEMM, fused head RMSNorm + M-RoPE writing K/V straight into a static cache, FA4 causal attention, fused residual + RMSNorm, vectorized SiLU-mul; the tier's producers quantize their rows in place; the first token sampled by the sampler kernel |
| Reasoning decode | `generate()` | one CUDA graph per token: embedding and table gathers, grouped few-row GEMMs, FA4 decode attention over the cache (one split), the sampler kernel (trajectory-token mask, temperature, top-p, torch's multinomial draw) and the step's counters |
| Diffusion expert (10 Euler steps) | 10 eager expert calls (or bounded per-length graphs) | all 10 steps in one CUDA graph: the action input projection (Fourier features, SiLU MLP with the activation in the GEMM epilogue, LayerNorm), tcgen05 GEMMs (64-row tiles, fused q/k/v and gate\|up weights, split SiLU-mul), the output projection with the Euler step; one graph serves every reasoning length |
| Trajectory | `action_to_traj` (unicycle integration, tensor ops) | one kernel from the expert's state and the host-computed t0 velocity |
| `num_traj_samples = N` | prompt (images included) repeated N times through vision and prefill | vision and prefill once; the prompt K/V is shared by all samples |

The static cache holds `[prompt | reasoning | action]` per sample. Each
sample's 64 action-token K/V rows are written right after its visible prefix,
so a single sample attends in one non-causal pass with a device-side length.
With several samples, attention is one pass over the shared prompt for all
samples plus a per-sample variable-length suffix pass, merged by their
log-sum-exp.

Numerics of the `bf16` preset: the model-local kernels reproduce the rounding
sequence of the official module graph under `torch.autocast(bfloat16)` (VLM:
BF16 activations, Qwen3 RMSNorm rounding the normalized value and then the
weight product, RoPE rounding each product and the sum; expert: FP32 residual
stream and RoPE, BF16 operands at attention; vision: FP32 RoPE with one BF16
rounding). `tests/test_alpamayo15_thor.py` checks each kernel against these op
sequences (vision RoPE bit-exact, the others within one BF16 ulp from FP32
reduction order). GEMM and attention kernels keep FP32 accumulation but
differ from the eager kernels in reduction order, so near-tie sampling steps
can choose a different token, exactly as switching the official model between
its SDPA and FlashAttention-2 backends does. The tcgen05 GEMM epilogues round
bias + activation (+ residual) once where the module graph rounds after each
op. With several samples, the shared-prompt and per-sample attention passes
are merged from BF16 partial outputs (one extra rounding) and the reasoning
LM head runs as a vocab-major GEMM, so per-sample numerics are not
bit-identical to a single-sample run.

The sampler kernel reproduces `torch.multinomial(probs, 1)` on the CUDA
generator: torch draws an exponential variate per vocabulary entry
(Philox4x32-10 words assigned per element by its distribution kernel,
curand's uniform, the fast `__logf`) and takes `argmax(probs / q)`; the kernel
computes the same words for the tokens kept by top-p, the same ratios, and
advances the generator's offset by torch's increment. The engine starts each
request from the generator's current seed and offset and moves the offset
forward after decode, so the official noise draw that follows reads the same
random stream as in the official procedure. Given equal probabilities the
token is identical; the probabilities differ at the float-rounding level, so
only near-equal ratios can pick differently.

Rotary tables: the official model computes the Qwen3-VL text rotary tables
with a `(3, 1, D/2, 1) @ (3, 1, 1, P)` matmul that NVIDIA's Jetson torch build
runs in TF32 by default (`torch.backends.cuda.matmul.allow_tf32 = True`),
rounding positions above 2048 to even values and above 4096 to multiples of
4. The engine computes the module's arithmetic exactly (fp32 products). This
is a difference between the official model's Thor default and its own
definition: on the example clip the official model with TF32 disabled samples
the same tokens on 23 of 24 windows and moves every trajectory slightly (ADE
by about 0.005; one window resamples); the engine's tables equal the exact
form.

### Streaming backend

What FlashDrive changes against the official procedure, kept as is:

| | Official Alpamayo 1.5 | FlashDrive |
|---|---|---|
| Prompt per window | 4 cameras x 4 frames, full prefill | first window full; later windows prefill only each camera's newest frame plus trajectory/text, older frames' K/V shifted in a pre-RoPE cache |
| Reasoning | autoregressive sampling | DFlash: a 2-layer draft proposes 7 tokens per round, the target verifies them in one forward and samples; stops at `<cot_end>` or when a sampled bonus token is `.`; 128-token budget |
| Language model | BF16 | W4A8 (ParoQuant, Marlin) |
| Action expert | official weights, 10 Euler steps | fine-tuned weights, 8 steps, velocity reused on steps 3-6, BF16 state |

`FlashDriveThor(model, expert=..., vision=..., prefill=..., decode=..., w4a8=...)`
installs FlashRT implementations of FlashDrive's primitives on a loaded
FlashDrive model (every group by default; a group set to `"flashdrive"`
keeps FlashDrive's execution, eager or compiled):

| Group | FlashDrive primitives | FlashRT implementation |
|---|---|---|
| `expert` | `_action` | the 8-step Euler loop with cached velocities in one CUDA graph: FP32 residual stream, fused head RMSNorm + FP32 RoPE, FA4 attention with a device-side key length, FlashDrive's fused q/k/v and gate/up weights, tcgen05 GEMMs with a fused SiLU-mul; the action input projection (Fourier features, SiLU MLP, LayerNorm) and the output projection with the Euler step as kernels |
| `vision` | `_encode` | the full-request engine's vision tower, one CUDA graph per frame grid |
| `prefill` | `_first_prefill`, `_dflash_prefill`, `_shift_streaming_kv_cache` | the first window of a stream and every streaming window as 36 decoder layers on FlashRT kernels (fused head norm + RoPE writing the cache, FA4, fused residual + RMSNorm, SiLU-mul) with FlashDrive's W4A8 projections; the first window's bookkeeping (padded position table, frame ranges, shift indices) computed on the host from the token ids; the window's rows attend by one FA4 varlen call; one CUDA graph per stream for streaming windows; one kernel moves every layer's frame blocks and rotates the moved keys |
| `decode` (needs `prefill="flashrt"`) | `_dflash_block_step`, `_dflash_traj_forward` | draft proposal (two-layer draft on FlashRT kernels, greedy by the sampler kernel), the proposal and acceptance rules as kernels, 8-row verification, posterior sampling by the sampler kernel (FlashDrive's top-p rule, torch's multinomial draw from the generator's state) in one CUDA graph per round; the trajectory-start forward as a one-row graph; the LM head on a few-row BF16 GEMM |
| `w4a8="fused"` (default) | the W4A8 linears inside the above | one activation pass per shared input (FP16 cast, ParoQuant pairwise rotation, per-token INT8 quantization; over the streaming window as one kernel per row tile together with the producer), the FP16 GEMM outputs read directly by the next kernel |
| the W4A8 GEMMs of `w4a8="fused"` | vLLM's Marlin kernel | FlashRT GEMMs computing Marlin's integer arithmetic: an INT8 tcgen05 GEMM on the unpacked `w - z` weights for row tiles (first-window and streaming-window prefill) and a few-row `mma.sync` kernel on the packed nibbles, zero points and integer group scales for verification and trajectory-start rows; the unpacked layout is checked against Marlin once per shape when the backend is built |

Cache contract: FlashDrive's `StaticCache` stays the store of record (keys
un-rotated and values, `[1, Hk, S, 128]` per layer, one position table per
stream). FlashDrive rotates the whole cache on every forward; the backend
keeps a rotated copy equal to FlashDrive's eager rotation for every row the
FlashRT primitives read, updated wherever a row is written (first-window
prefill, streaming windows, the frame-block shift, verification rows including
rejected draft rows, the trajectory-start row and the expert's action rows).
The expert rotates the cache with its own FP32 tables once per window, as
FlashDrive's expert does. FlashDrive's masks become FA4 lengths: a streaming
window's rows attend causally by cache position, verification rows causally
over `[0, cur + 8)`, the trajectory-start row every cache row, the expert rows
`[0, action_start + 64)`.

Numerics: the fused W4A8 activation pass is bit-exact with ParoQuant's
rotation kernel and vLLM's per-token INT8 quantization. The tcgen05 W4A8 GEMM
computes Marlin's integer arithmetic exactly (per 128-channel group the INT8 x
`(w - z)` sums times the integer group scale, accumulated in int32, converted
to float once and scaled per token); Marlin converts partial sums over K
slices to float and adds them in FP32, so the two agree except where Marlin's
FP32 sum rounds to the neighbouring FP16 value (0.001-0.01% of outputs on the
checkpoint's layers, by one FP16 ulp where observed). The weights are unpacked
through vLLM's layout definitions and checked against Marlin when the backend
is built. The other primitives reproduce FlashDrive's eager operation
sequence (rounding points, RNG consumption) with different kernels, so results
differ from FlashDrive eager at the reduction-order level, no more than
FlashDrive's own eager and compiled executions differ from each other (the
cosine table above).

### Kernel sources per phase

The profiler's account of the hot path on the example clip (windows 8-10,
`torch.profiler`, every GPU kernel inside each phase annotation attributed to
its source; kernel time per phase instance). Full request, `balanced` preset:

| Phase | kernel ms | FlashRT kernels | FlashRT CUTLASS tcgen05 GEMMs | FA4 | other |
|---|---|---|---|---|---|
| vision tower (16 frames) | 121.7 | 86 launches, 25.9 ms | 117 launches, 77.8 ms | 27 launches, 17.5 ms | 1 memcpy (the pixel copy into the graph input), 0.55 ms |
| LM prefill | 246.7 | 191 launches, 72.9 ms | 288 launches, 149.8 ms | 36 launches, 18.5 ms | cuBLASLt from C++: the first-token logits, 1 launch, 5.6 ms |
| decode step (one token) | 21.8 | 438 launches, 19.8 ms | - | 36 launches, 2.0 ms | none |
| diffusion expert (10 steps) | 126.1 | 1860 launches, 12.2 ms | 1470 launches, 93.9 ms | 360 launches, 20.0 ms | none |

`fast` has the same sources (prefill 181.6 ms, decode step 20.7 ms). In `bf16`
the prefill (539.7 ms) runs 72 cuBLASLt GEMMs for gate / up (203.8 ms) and 144
for the down projection's column blocks (112.8 ms) next to 72 tcgen05 GEMMs
for q|k|v and o (119.9 ms), and each decode step (67.6 ms) runs its
projections on cuBLASLt (65.0 ms). The trajectory integration is one kernel
after the expert graph. No ATen kernel runs in any full-request phase; the
official noise draw runs between the decode and expert phases.

Streaming backend, defaults:

| Phase | kernel ms | FlashRT kernels | FlashRT CUTLASS tcgen05 GEMMs | FA4 | other |
|---|---|---|---|---|---|
| first window of a stream | 621.7 | 374 launches, 146.5 ms | 477 launches, 407.2 ms | 99 launches, 61.6 ms | FlashDrive's cache allocation and bookkeeping: 202 ATen launches, 3.2 ms (zero fills of the new cache layers, position-table and index-table construction), 157 memcpys, 3.4 ms |
| vision (4 frames) | 26.8 | 86 launches, 3.9 ms | 117 launches, 19.1 ms | 27 launches, 3.8 ms | 1 memcpy, 0.07 ms |
| streaming prefill | 144.7 | 191 launches, 49.2 ms | 252 launches, 87.6 ms | 36 launches, 6.8 ms | 5 ATen copies of the window's inputs, 0.01 ms; memcpys 1.1 ms |
| DFlash round | 36.0 | 720 launches, 33.9 ms | - | 38 launches, 2.0 ms | 7 ATen launches, 0.02 ms (copies of FlashDrive's block tokens, context and cache position into the graph's inputs, one 1-element add); 43 memcpys, 0.05 ms |
| trajectory start | 23.5 | 688 launches, 21.4 ms | - | 36 launches, 2.0 ms | 43 memcpys, 0.04 ms |
| frame-block shift | 3.5 | 1 launch, 3.5 ms | - | - | 4 ATen launches, 0.02 ms (FlashDrive's bookkeeping) |
| expert (8 steps) | 87.4 | 682 launches, 5.7 ms | 588 launches, 73.6 ms | 144 launches, 8.1 ms | the noise draw and 3 fills, 0.005 ms |

The ATen entries of the streaming backend are the copies of the inputs that
FlashDrive's orchestration hands to each primitive and its own cache
initialization in the first window; no computation of a primitive runs
outside FlashRT kernels.

## Tests

```bash
python -m pytest -q -rs tests/test_alpamayo15_thor.py tests/test_alpamayo15_flashdrive_thor.py
python benchmarks/alpamayo15_thor.py --help
bash -n scripts/build_alpamayo15_marlin.sh
```

Set `FLASHRT_TEST_ALPAMAYO15_REQUIRED=1` to make missing full-request kernels
fail instead of skip. For the streaming checkpoint integration tests set
`FLASHRT_FLASHDRIVE_CKPT` to a local directory containing `Alpamayo-1.5-10B`,
`Alpamayo-1.5-10B-PARO` and `Alpamayo-1.5-10B-DFlash`. Inspect skip reasons;
a skipped integration test is not a passing hardware test.

## Limitations

- Full requests: one scene per request (`num_traj_sets=1`),
  `num_traj_samples` up to the engine's `max_samples` (16 with the quantized
  decode tiers); the navigation-guided CFG entry point is not accelerated.
  The engine rewires the official model's q/k/v and gate/up parameters into
  views of fused tensors; do not move the model or replace those parameters
  after constructing the frontend. The K/V cache is sized for
  `max_prompt + max_new_tokens + 64` rows per sample.
- Streaming: one trajectory sample per window; Alpamayo 1.5 checkpoints
  only; streams are reset explicitly and the first window of a stream returns
  no trajectory; the backend keeps an INT8 copy of the language model's
  linear weights (about 7 GB) next to Marlin's INT4 buffers.
- The offline trajectory error on nine clips compares implementations of a
  procedure; it is not evidence of closed-loop driving quality.
- Video frames are decoded by the official loader on the host CPU. With the
  same PyAV/FFmpeg versions the decoded YUV planes are identical on Thor
  (aarch64) and x86, but the YUV -> RGB conversion differs by up to 2 levels in
  sampled pixels, so pixel inputs and therefore sampled outputs can differ
  slightly between platforms.
