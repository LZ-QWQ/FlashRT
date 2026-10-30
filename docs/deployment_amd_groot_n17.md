# GROOT N1.7 on AMD Instinct (MI300X / CDNA3 and MI350X / CDNA4)

Model-specific guide. For hardware support, the build, routing, shared
environment knobs and the test suite, see
[deployment_amd.md](deployment_amd.md) first.

For W7900D, Strix Halo and R9700, see [GR00T N1.7 BF16 on AMD RDNA](#gr00t-n17-bf16-on-amd-rdna). The Instinct sections below describe the CDNA route.

## Performance

GROOT uses OCP E4M3 and the packed small-M path on gfx950. On gfx942 it uses
E4M3 FNUZ and hipBLASLt for GEMMs that have no CDNA3 packed implementation.

GROOT N1.7 (3.1 B: Qwen3-VL ViT + truncated Cosmos-Reason2 LLM + VL
adapter + 32-layer DiT action head), 2 camera views, 4-step flow
matching, action horizon 40. Median **full-frame** end-to-end latency —
a fresh observation through the backbone graph plus the action chain,
i.e. what a serving loop pays per frame:

### MI350X (gfx950)

| Configuration | Median latency | vs eager |
|---|---|---|
| PyTorch eager (official policy) | 67.9 ms | 1.00× |
| torch.compile (max-autotune) | 77.9 ms | 0.87× |
| **FlashRT AMD, FP8 backbone + bf16 DiT** | **16.0 ms** | **4.24×** |

`torch.compile` is slower than eager for this model on this stack; the
judged baseline is therefore eager.

Scope note: the frame cost splits into roughly 7 ms of backbone graph
replay and 9 ms of action chain (the 4 denoise steps over the 32-layer
DiT). Deployments that reuse backbone features across frames pay only
the action chain.

Accuracy: combined denormalized-action cosine against the reference is
**0.9995** with the initial noise pinned (per-modality: end-effector and
joint targets above 0.999; the 1-D gripper signal is near-constant over
a trajectory so its cosine is naturally lower and is not a useful gate).

### MI300X (gfx942)

Measured on this RunPod with ROCm 7.2.4 and AITER, from DROID trajectory 1,
step 0, using the official two-view preprocessing path (277 sequence tokens
and 1,024 vision tokens), 50 warmup replays and 100 measured calls:

| Boundary | Min | Median | p95 | Max |
|---|---:|---:|---:|---:|
| Backbone + action chain | 19.52 ms | 19.58 ms | 19.77 ms | 34.37 ms |
| Backbone graph | 8.05 ms | 8.10 ms | 8.14 ms | 8.18 ms |
| Four-step action chain | 11.39 ms | 11.41 ms | 11.43 ms | 16.88 ms |

The timed full boundary starts with the tensors emitted by the official GROOT
processor and ends with the normalized action tensor. Video decode and CPU
processor work are outside it. Graph replay is bit-identical. After official
denormalization with the same real state, all 680 action values compare with
cosine **0.999934**, maximum absolute error **0.04519**, and mean absolute
error **0.00593** against the official policy output.

## Usage

```python
import flash_rt

model = flash_rt.load_model(checkpoint_dir, config="groot_n17",
                            framework="torch", num_views=2)
fe = model.pipeline

# aux is the official processor's output bundle for the observation
# (input_ids, visual/text embeds, rope tables, pixel features, grid_thw).
fe.set_prompt(aux=aux, prompt=instruction_text)

state_normed = fe.normalize_state(state_dict)
out = fe.infer(state_normed, aux=aux)          # fresh observation
actions = fe.denormalize_action(out, state_dict=state_dict)
```

Omitting `aux` from `infer` reuses the backbone features computed at
`set_prompt` and runs the action chain only.

## Precision

The tier here is an **FP8 backbone with an unquantized bf16 DiT action
head**. Note this is deliberately *less* quantized than the Thor and RTX
FP8 frontends, which additionally run the DiT FFN and fused
self-attention QKV in FP8 (`_DIT_USE_FP8 = True`); that DiT calibration
path is not ported to the AMD frontend yet, so it remains available headroom
rather than a shipped capability.

- ViT, DeepStack mergers, the truncated LLM and the VL self-attention
  adapter run FP8 GEMMs with per-tensor activation scales.
- The DiT action head, the state/action encoders and the decoder stay
  bf16; they are never quantized.
- Activation scales are calibrated once and cached to disk. A warm
  `set_prompt` loads the cache and runs kernels only; a cold cache runs
  a one-time reference pass purely to extract activation maxima.

This CDNA route has no BF16-only tier: `use_fp8=False` without `use_fp16=True` is
rejected rather than silently ignored, and the non-quantized full-FP16
reference tier is not ported to AMD (`use_fp16=True` raises
`NotImplementedError`).

## Attention

One backend serves all five attention sites — ViT self-attention, the
causal LLM self-attention, the VL adapter, and the DiT self- and
cross-attention. On the measured GROOT geometries aiter beats torch SDPA
at every site (1.19×–1.83×), including head_dim 48 and the GQA causal
site, so aiter is the default and `FVK_AMD_ATTN=sdpa` is the fallback.

The LLM K/V slots hold the model's native 8 KV heads: aiter performs
GQA internally, so the head-expansion step the CUDA path needs is not
run here.

## Fusion and kernel routing

Two AMD-specific paths carry most of the speedup over a direct port.
Both are controlled by environment knobs so either can be A/B'd against
the decomposed form in one process.

**Fused GEMM epilogues** (`FVK_AMD_FUSED_EPILOGUE`, default on) —
hipBLASLt supports fused FP8 bias and bias+GELU epilogues that
the CUDA SM120 path cannot use, so the backbone runs
`fp8_nn_bias`/`fp8_nn_gelu_bias` instead of a descale GEMM followed by
separate bias and activation kernels. The DiT and the frontend's own
graph closures likewise use `bf16_nn_bias`, `bf16_nn_bias_gelu` and
`bf16_nn_bias_res` (residual accumulation in the epilogue), and the
norm→quantize chains collapse into the fused norm kernels. Together this
removes on the order of a thousand kernel launches per frame.

Numerics note: a fused epilogue adds the bias on the FP32 accumulator
before the output rounding, where the decomposed form adds it after.
The difference is last-ULP and is judged by the end-to-end cosine gate,
not by bit equality.

On gfx942 this switch is also decisive: the same real frame measured
**19.72 ms median with fusion** and **29.42 ms without fusion**. Those A/B
runs used 20 warmups and 50 calls; the longer result above is the reported
absolute benchmark.

**Packed-weight MFMA GEMMs** (`FVK_AMD_DIT_GEMM`, default `smallm`) —
the DiT projections are a small-M, weight-bandwidth-bound shape
(M = 41 action+state tokens). A hand-written MFMA kernel with weights
pre-packed at setup into per-lane consumption order beats the autotuned
hipBLASLt kernel by 1.41× on that shape (951 GB/s versus 673 GB/s
effective weight bandwidth) and is routed for the Q/K/V/O projections
only. The FFN shapes measured slower than the library and deliberately
stay on hipBLASLt — routing is by measured shape list, not blanket
substitution. Set `FVK_AMD_DIT_GEMM=hipblaslt` to disable the routing;
the packing costs roughly 430 MB of additional weight storage. This route is
a gfx950 capability for GROOT's BF16 DiT shapes. On gfx942 the capability
layer rejects packed BF16 MFMA and uses hipBLASLt, even when the default knob
remains `smallm`.

## Environment knobs

Shared knobs are in [deployment_amd.md](deployment_amd.md). GROOT-specific:

| Env | Default | Meaning |
|---|---|---|
| `FVK_AMD_FUSED_EPILOGUE` | `1` | fused GEMM epilogues and fused norm→quantize chains; `0` selects the decomposed form (also switches the DiT driver back to the hardware-independent forward) |
| `FVK_AMD_DIT_GEMM` | `smallm` | DiT projection GEMMs: `smallm` requests packed-weight MFMA where supported (gfx950); otherwise hipBLASLt is used |

Both are read at setup and graph-build time, never on the inference
path; changing them after the graphs are built has no effect.

## Tests

```bash
export FLASH_RT_GROOT_N17_AMD_CKPT=<groot_n17_checkpoint_dir>
python -m pytest tests/test_amd_groot_routing.py \
                 tests/test_amd_groot_model.py -v
```

`test_amd_groot_routing.py` covers the pipeline-map entry, the
`load_model` precision-tier contracts and the attention backend's site
and layer-index validation. `test_amd_groot_model.py` covers the
end-to-end path: frontend construction, `set_prompt` through the FP8
kernel backbone, finite actions of the expected shape, bit-identical
results for a pinned initial noise, and — when a reference fixture is
provided — a cosine gate against it. Both skip with a stated reason
without ROCm, the extension, or the checkpoint.

## Reproducing the numbers

1. Build for the installed architecture (see the general guide) with aiter
   importable — without it attention falls back to torch SDPA and every
   number here shifts.
2. Capture an aux fixture from official preprocessing, then run:

   ```bash
   taskset -c 48-63 python benchmarks/groot_n17_amd_latency.py \
     --ckpt <groot_n17_checkpoint_dir> --aux <single_observation.pt> \
     --warmup 50 --iters 100 --json-out groot-amd.json
   ```

   `48-63` is the compact GPU-local CPU set for the measured pod. Determine
   the selected GPU's NUMA node before choosing a set on another host.
3. Judge accuracy on the same call with the initial noise pinned to the
   array the reference was generated with, comparing denormalized
   actions per modality.
4. To attribute a change, flip one knob at a time in a single process:
   `FVK_AMD_FUSED_EPILOGUE=0/1` for the fusion tier and
   `FVK_AMD_DIT_GEMM=hipblaslt/smallm` for the projection routing.

# GR00T N1.7 BF16 on AMD RDNA

The `groot_n17` Torch route supports the unquantized BF16 model on Radeon PRO W7900D (`gfx1100`), Strix Halo / Radeon 8060S (`gfx1151`), and Radeon AI PRO R9700 (`gfx1201`). It runs fresh camera observations through the vision tower, DeepStack, truncated language model, VL adapter and four-step DiT, then decodes physical actions with the official GR00T processor.

The validated configuration is `nvidia/GR00T-N1.7-3B`, DROID relative EEF/joint embodiment, batch one, two cameras with two history frames each, and action horizon 40. FP8, FP16, NVFP4, RTC and robot closed-loop task accuracy are outside this path.

Pass `use_fp8=False` explicitly to `load_model` for this route. FP8/FP4/FP16
requests are rejected before backend loading rather than converted to BF16.

## Container

Build the optional `groot` target from the repository root:

```bash
docker build -f docker/Dockerfile.rocm.rdna --target groot \
  --build-arg GPU_ARCH=gfx1100 --build-arg BUILD_JOBS=8 \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" \
  -t flashrt:groot-rdna-gfx1100 .
```

Use `gfx1151` for Strix Halo or `gfx1201` for R9700. The image uses Ubuntu 24.04's Python 3.12, ROCm 10.1.0, AMD PyTorch 2.14 and matching torchvision/Triton. Processor dependencies use a separate Python environment that reuses the base AMD runtime; the default Docker target and Pi0.5 environment remain unchanged.

Rebuild the image or native RDNA extension when migrating from Python 3.13
or an extension reporting `backend="rdna35"`. Python extension binaries must
match the interpreter, and the current runtime requires `backend="rdna"`
plus the exact `hardware` and `gpu_arch` identity. Do not copy an old extension
into the new environment; see [AMD build instructions](deployment_amd.md#build).

The image provides FlashRT, Python dependencies and the pinned Isaac-GR00T processor implementation. Model weights, tokenizer data, image-processing configuration, observations and reference fixtures are prepared separately and mounted at runtime. Building the image does not require Hugging Face credentials.

## Prepare model resources

Download on a host with Hugging Face access, using `hf` from `huggingface-hub` (install with `python -m pip install "huggingface-hub[cli]==0.36.0"` if needed). Use an existing user-owned model directory outside the FlashRT checkout/build context. Set `MODEL_DIR` to its absolute path to reuse existing downloads; the commands below use `$HOME/models` by default.

Cosmos is gated: first obtain access to [nvidia/Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B) with your Hugging Face account. If authentication is needed, use `hf auth login` interactively on the download host, or provide `HF_TOKEN` through your existing credential mechanism. Do not pass credentials as Docker build arguments or copy them into the image. Use the official endpoint for authenticated downloads.

```bash
export HF_ENDPOINT=https://huggingface.co
export MODEL_DIR="${MODEL_DIR:-$HOME/models}"
mkdir -p "$MODEL_DIR"

hf download nvidia/GR00T-N1.7-3B \
  --revision 2fc962b973bccdd5d8ce4f67cc63b264d6886495 \
  --local-dir "$MODEL_DIR/GR00T-N1.7-3B" \
  --include 'model*.safetensors' model.safetensors.index.json \
    config.json processor_config.json statistics.json embodiment_id.json

hf download nvidia/Cosmos-Reason2-2B \
  config.json preprocessor_config.json video_preprocessor_config.json \
  tokenizer.json tokenizer_config.json vocab.json merges.txt chat_template.json \
  --revision 9ce19a195e423419c349abfc86fd07178b230561 \
  --local-dir "$MODEL_DIR/Cosmos-Reason2-2B"
```

The GR00T checkpoint already contains the backbone weights used by FlashRT. The Cosmos command downloads only the eight listed processor/tokenizer files; no separate Cosmos model weights are needed. Keep both revisions fixed when reproducing the validated configuration.

The resulting layout is:

```text
$MODEL_DIR/
├── GR00T-N1.7-3B/
│   ├── model-00001-of-00002.safetensors
│   ├── model-00002-of-00002.safetensors
│   ├── model.safetensors.index.json
│   ├── config.json
│   ├── processor_config.json
│   ├── statistics.json
│   └── embodiment_id.json
└── Cosmos-Reason2-2B/
    ├── config.json
    ├── preprocessor_config.json
    ├── video_preprocessor_config.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── vocab.json
    ├── merges.txt
    └── chat_template.json
```

## Run offline

Copy these two directories to the inference host if it differs from the download host, and set `MODEL_DIR` there to their parent directory. Prepare an observation directory as described under [Quickstart and latency](#quickstart-and-latency), then mount both paths read-only. With the image and resources already present, inference needs no Hugging Face login or network access:

```bash
export OBSERVATION_DIR=/path/to/observations
export RENDER_DEVICE=/dev/dri/renderD128
docker run --rm -it --device /dev/kfd --device "$RENDER_DEVICE" \
  --group-add "$(stat -c %g /dev/kfd)" \
  --group-add "$(stat -c %g "$RENDER_DEVICE")" --ipc=host \
  --network none -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v "$MODEL_DIR:/models:ro" -v "$OBSERVATION_DIR:/observations:ro" \
  flashrt:groot-rdna-gfx1100
```

Select the appropriate `RENDER_DEVICE` and image tag on a multi-GPU host. The group IDs are read from the host's device nodes. The local paths inside the container are `/models/GR00T-N1.7-3B` for the checkpoint and `/models/Cosmos-Reason2-2B` for the processor/tokenizer data. Without Docker, build the extension as described in [deployment_amd.md](deployment_amd.md), provide the processor environment from the `groot` Docker stage, and use the corresponding host paths with the same offline environment variables.

## Inference from raw observations

```python
import flash_rt
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

checkpoint = "/models/GR00T-N1.7-3B"
processor = Gr00tN1d7Processor.from_pretrained(
    checkpoint, model_name="/models/Cosmos-Reason2-2B")
model = flash_rt.load_model(
    checkpoint, config="groot_n17", framework="torch", num_views=2,
    use_fp8=False, fuse_image_embeds=True)
frontend = model.pipeline
frontend.set_hf_processor(processor)
state, aux = frontend.prepare_observation(observation)
model.set_prompt(aux=aux)  # tokenized instruction and camera grid
normalized_actions = model.infer(state, aux=aux)
actions = frontend.denormalize_action(
    normalized_actions, state_dict=observation["state"])
```

`observation` uses the official batched structure:

- `video`: camera-name → `uint8` array of shape `(1, 2, height, width, 3)`.
- `state`: state-modality-name → array of shape `(1, 1, channels)`.
- `language`: language-modality-name → nested list `[[instruction]]`.

Use the camera, state and language names in the checkpoint's `processor_config.json`. `prepare_observation` runs only the official processor and constructs prompt M-RoPE metadata; it does not execute a model or bind a prompt. The instruction comes from `observation["language"]`. `set_prompt(aux=..., prompt=...)` accepts optional descriptive text, but the token IDs in `aux` determine the model input.

As with the existing GR00T N1.7 frontends, use explicit setup, normalized-state inference and physical-action decoding rather than `VLAModel.predict(images)`. Both `model.infer(...)` and `frontend.predict(...)` return a normalized FP32 tensor of shape `(1, action_horizon, 132)`. `denormalize_action` returns a dictionary of physical action channels; the validated DROID embodiment has 17 active channels in total.

For each new raw observation, call `prepare_observation`, pass its fresh state and `aux` to `infer`, then decode using that observation's physical state. Omitting `aux` explicitly reuses the most recent observation, initially the one passed to `set_prompt`; always supply fresh aux when the cameras change. Reuse setup while token IDs, grid, masks and RoPE metadata stay unchanged. When the instruction or camera grid changes, call `set_prompt(aux=new_aux)` before inference to invalidate the old graph. Preparation alone does not change the active prompt, and mismatching metadata is rejected by `infer`. Fresh observation tensors, state and noise are copied into persistent buffers; visual features and cross-attention K/V are recomputed for each supplied observation.

RDNA selects one of the existing N1.7 input contracts at construction:

- **Raw patches** with `fuse_image_embeds=True`, as on the Orin fused-input path: `pixel_values` of shape `(patches, 1536)`, `input_ids`, `grid_thw`, `visual_pos_masks`, `rope_cos` and `rope_sin`. `prepare_observation` produces this bundle without an official model forward. FlashRT runs patch projection, all 24 vision layers, the merger, language model and action head.
- **Projected CUDA/CDNA aux** with `fuse_image_embeds=False` (the RDNA default): `pixel_features` of shape `(patches, 1024)` already includes patch projection and positional embeddings; `llm_input_embeds` of shape `(1, tokens, 2048)` contains fused text/image embeddings. Grid, mask and RoPE keys have the same meanings. FlashRT recomputes the first 18 vision layers for DeepStack, then the language model and action head, as on the existing projected-input routes. The caller must refresh both feature tensors for every new observation. Their preparation includes model computation and must be included in any end-to-end timing.

The explicit constructor mode determines which fields are read, even when both forms are present. Extra raw fields never change projected behavior. Switching modes requires constructing another frontend. Orin defaults to `fuse_image_embeds=True`; specify this parameter explicitly when sharing raw-input code across Orin and RDNA. Setup owns copies of inputs and metadata; changing a caller tensor after setup cannot silently mutate captured state. `attention_mask`, when supplied, must describe an unpadded batch of one.

An existing CUDA/CDNA caller can reuse its aux preparation and this inference workflow, selecting RDNA hardware and BF16 at construction:

```python
model = flash_rt.load_model(checkpoint, config="groot_n17", framework="torch",
                            hardware="amd_rdna3", use_fp8=False)
frontend = model.pipeline
# For offline use, inject the processor configured with local tokenizer files.
frontend.set_hf_processor(processor)
model.set_prompt(aux=setup_aux, prompt=instruction)
state = frontend.normalize_state(state_dict)  # state.<modality> keys also accepted
normalized = model.infer(state, aux=observation_aux, initial_noise=noise,
                         num_inference_timesteps=4, action_horizon=40,
                         num_timestep_buckets=1000, use_dit_graph=True)
actions = frontend.denormalize_action(normalized, state_dict=state_dict)
```

The processor is loaded lazily from the checkpoint if not injected. For offline deployment, keep the explicit local processor construction used by the example. The image and prompt preparation helper is shared with Orin. Orin adds optional `prepare_observation` and `set_hf_processor` methods; its existing model and GPU preprocessing implementation, and the CDNA compute path, remain unchanged.

`use_dit_graph` controls the denoising graph with the existing N1.7 meaning. `load_model(use_backbone_graph=...)` controls backbone capture independently on Orin/RDNA. For eager execution, construct with `use_backbone_graph=False` and pass `infer(use_dit_graph=False)`. The existing RDNA `infer(use_graph=False)` alias remains accepted for compatibility. With both enabled and fresh aux, RDNA captures the complete model in one graph. Positive denoising-step/time-bucket counts and a horizon up to the checkpoint maximum are supported; changing them rebuilds the schedule and graph. Omitted values return to the checkpoint/construction defaults. The first inference after setup or a schedule change may include GEMM tuning, Triton compilation and capture. A frontend instance is single-stream and is not reentrant.

For fixed-noise validation, pass `initial_noise` to `infer`. It must have shape `(1, action_horizon, 132)` (the unbatched form is also accepted); only the embodiment's active action channels enter numerical comparisons.

## Continuous camera frames (Orin-compatible API)

After raw-mode setup, `frames=` updates only the image input, reusing the prompt
metadata. The optional preparation helper and public calls below work on both
Orin and RDNA in their supported configurations:

```python
from flash_rt.models.groot_n17.observation import frames_from_observation

camera_order = processor.modality_configs[frontend.embodiment_tag]["video"].modality_keys
frames = frames_from_observation(next_observation, camera_order)
state = frontend.normalize_state({
    "state." + k: v for k, v in next_observation["state"].items()})
normalized_actions = model.infer(state, frames=frames)
actions = frontend.denormalize_action(
    normalized_actions, state_dict=next_observation["state"])
```

`frames` is `(images, H, W, 3)` RGB uint8: a NumPy array, CPU/device tensor, or
ordered sequence of equally sized arrays/tensors. A dictionary is rejected.
For the two-camera DROID checkpoint, include both history times: old exterior,
old wrist, current exterior, current wrist. The helper reads camera order
explicitly; dictionary insertion order must not choose image-token assignment.
No history is accumulated or synthesized inside `infer`.

`frames` requires `fuse_image_embeds=True`. Supplying both `frames` and
`aux["pixel_values"]` is rejected. Optional metadata supplied in `aux` must
match setup. Changes to the processed grid require new `set_prompt` metadata;
a raw resolution change that leaves the processed grid unchanged is accepted.

RDNA runs the checkpoint's official eval image transform and HF image processor
on the host, omitting per-frame text tokenization, M-RoPE construction and
lossless PIL round trips. This preserves the original DROID
`letter_box_transform=False` configuration (1,792 patches); Orin's square-image
GPU resize must not be substituted (it would produce 1,024 patches).
Device tensors are accepted but copied to the host. The resulting patches are
uploaded into the existing model graph buffers. Both transfer directions and
image preparation belong in end-to-end timing. This implementation makes no
claim of GPU image-preprocessing speedup or of matching Orin's measured latency.

## Quickstart and latency

Save a raw observation as an NPZ containing `video.<camera>` and `state.<modality>` arrays with the shapes above, plus an `instruction` string. Then run:

```bash
python examples/amd/groot_n17_rdna.py \
  --checkpoint /models/GR00T-N1.7-3B \
  --processor /models/Cosmos-Reason2-2B \
  --observation /observations/droid.npz --input frames --benchmark 50
```

`observation_p50_ms` includes frame packing, CPU image/state processing, device staging, model execution and physical-action decoding. Setup and warmup are excluded. With `--input aux`, it also includes repeated text tokenization; `--input frames` reuses prompt metadata. The example reports this complete workflow boundary only, including fresh visual computation.

For the default projected contract, use `examples/amd/groot_n17_projected.py`
with `--checkpoint`, `--observation` and `--hardware`. Its NPZ contains
`pixel_features`, `llm_input_embeds`, `grid_thw`, `visual_pos_masks`, `rope_cos`,
`rope_sin` and `state.<modality>` arrays. It uses the shared RTX SM120/CDNA/RDNA
setup, normalization, inference and decoding calls; add `--fp8` for the RTX/CDNA
FP8 routes. The processor/tokenizer referenced by the checkpoint must be
available locally or in the Hugging Face cache for offline use. Producing the
projected features requires model work, so this saved-feature example does not
report end-to-end latency.

## Validation

An independent official-policy fixture contains raw observations, normalized state, unprojected image patches, projected `pixel_features` and `llm_input_embeds`, prompt/RoPE metadata, fixed initial noise, normalized actions and decoded physical actions. It is generated outside FlashRT using the same checkpoint and preprocessing configuration.

```bash
export FLASH_RT_GROOT_N17_RDNA_CKPT=/models/GR00T-N1.7-3B
export FLASH_RT_GROOT_N17_RDNA_REFERENCE=/observations/official-reference.pt
export FLASH_RT_GROOT_N17_PROCESSOR=/models/Cosmos-Reason2-2B
python -m pytest -q tests/test_amd_rdna_groot_ops.py \
  tests/test_amd_rdna_groot_routing.py tests/test_amd_rdna_groot_model.py tests/test_groot_n17_observation.py
```

The tests check operator references, all three routing entries, official preprocessing, active-channel action precision, physical-component errors, graph/eager parity, independent graph controls, frame/state/noise updates, explicit prompt changes, schedule changes, caller-buffer ownership, padding rejection, and the CUDA/CDNA projected-aux call contract. Numerical tolerances are BF16 regression gates, not a guarantee of robot task success.

## Implementation

The RDNA frontend owns weights, preprocessing and graph lifetime; `pipeline_rdna.py` owns model mathematics. Shared RDNA transformer primitives provide BF16 normalization/RoPE fusion, native GQA and library GEMMs. The model's `targets_rdna.py` selects long-sequence attention and small-M projection implementations during setup. On gfx1151 and gfx1201, GR00T fixes hipBLASLt to its first heuristic instead of timing-selecting among algorithms: varying BF16 accumulation order exceeded the physical-action regression gate in repeated runs. The existing small-M WMMA selection is retained. The GEMM provider accepts an optional setup policy; existing Pi0.5 callers keep their original environment-controlled tuning behavior. The public loader forwards the two optional Orin/RDNA controls, and Orin gains processor preparation/injection helpers; existing CUDA/CDNA model computation is unchanged.

Model semantics follow [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T/tree/9c7e746b2cd37a810070a98ef41d290a07e806c2) and Transformers 4.57.3. In particular, BF16 intermediate rounding is retained in rotary buffers, weighted RMSNorm, AdaLayerNorm and embodiment-conditioned projections.
