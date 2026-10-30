"""Official GR00T N1.7 observation preparation, independent of GPU hardware.

These helpers execute processor transforms only, never a model forward.
Camera ordering and preprocessing are defined by the checkpoint's processor.
"""

from __future__ import annotations

import numpy as np
import torch

from flash_rt.models.groot_n17.mrope_table import (
    RopeConfig, apply_interleaved_mrope, build_position_ids_for_segments,
    compute_inv_freq,
)


def _process_observation(processor, observation, embodiment_tag, *, num_views,
                        image_history):
    """Run the official image, instruction and state transforms once.

    Input is the official batched observation dictionary with ``video``,
    ``state`` and ``language`` groups. No model forward is needed to prepare it.
    """
    from gr00t.data.embodiment_tags import EmbodimentTag

    if not isinstance(observation, dict) or any(
            not isinstance(observation.get(key), dict) for key in ("video", "state", "language")):
        raise ValueError("observation requires video, state and language dictionaries")
    if len(observation["video"]) != num_views:
        raise ValueError(f"expected {num_views} camera streams")
    flat = {}
    for group in ("video", "state", "language"):
        for key, value in observation[group].items():
            if group == "language":
                # Policy observations carry B x language-history; inference
                # consumes the latest instruction for each batch item.
                value = np.asarray(value)
                if value.ndim != 2 or value.shape[0] != 1 or value.shape[1] < 1:
                    raise ValueError("language must have shape (1, history)")
                flat[key] = value[:, -1].tolist()
            else:
                value = np.asarray(value)
                if group == "video":
                    if (value.ndim != 5 or value.shape[:2] != (1, image_history)
                            or value.shape[-1] != 3 or value.dtype != np.uint8):
                        raise ValueError("video must be uint8 (1, image_history, height, width, 3)")
                elif value.ndim != 3 or value.shape[:2] != (1, 1):
                    raise ValueError("state must have shape (1, 1, channels)")
                flat[f"{group}.{key}"] = value
    tag = EmbodimentTag.resolve(embodiment_tag)
    batch = processor.process_observation(flat, tag)
    return batch


def _prompt_aux(batch, image_token_id, *, device):
    """Build N1.7 M-RoPE from token segments, independent of the backbone."""
    ids = batch["input_ids"].detach().cpu()
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError("GR00T N1.7 supports batch size one")
    mask = ids == image_token_id
    grids = batch["image_grid_thw"].detach().cpu()
    lengths, kinds, segments = [], [], []
    tokens = mask[0].tolist()
    offset, image = 0, 0
    while offset < len(tokens):
        visual = tokens[offset]
        end = offset + 1
        while end < len(tokens) and tokens[end] == visual:
            end += 1
        lengths.append(end - offset)
        kinds.append("image" if visual else "text")
        if visual:
            if image >= len(grids):
                raise ValueError("more image-token segments than image grids")
            segments.append(tuple(int(v) for v in grids[image]))
            image += 1
        else:
            segments.append(None)
        offset = end
    if image != len(grids):
        raise ValueError("image grids do not match the tokenized observation")
    config = RopeConfig()
    positions = build_position_ids_for_segments(
        segment_lengths=lengths, segment_kinds=kinds, segment_grids=segments,
        cfg=config, device=device)
    # The official BF16 policy casts inv_freq along with the model. Text
    # rotary then promotes that rounded buffer back to FP32 for the angles.
    inv = compute_inv_freq(config, device=device).bfloat16().float()
    freqs = positions[..., None].float() * inv
    freqs = apply_interleaved_mrope(freqs, config.mrope_section)
    angles = torch.cat((freqs, freqs), dim=-1).squeeze(0)
    cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
    return {"input_ids": ids, "visual_pos_masks": mask, "grid_thw": grids,
            "rope_cos": cos.unsqueeze(0), "rope_sin": sin.unsqueeze(0),
            "attention_mask": batch["attention_mask"],
            "pixel_values": batch["pixel_values"]}


def prepare_observation(processor, observation, embodiment_tag, *, num_views,
                        metadata=None):
    """Return normalized state and raw aux using the official eval processor.

    ``metadata`` is an optional previous result; it only avoids rebuilding
    M-RoPE when token IDs and image grids match. Pixels and state stay fresh.
    This function does not bind a prompt or execute model weights.
    """
    processor.eval()
    video = processor.modality_configs[embodiment_tag]["video"]
    batch = _process_observation(
        processor, observation, embodiment_tag, num_views=num_views,
        image_history=len(video.delta_indices))
    if (metadata is None
            or not torch.equal(batch["input_ids"], metadata["input_ids"])
            or not torch.equal(batch["image_grid_thw"], metadata["grid_thw"])):
        image_id = processor.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        metadata = _prompt_aux(batch, image_id, device="cpu")
    aux = {k: v for k, v in metadata.items() if k not in ("pixel_values", "attention_mask")}
    aux.update(pixel_values=batch["pixel_values"], attention_mask=batch["attention_mask"])
    return batch["state"], aux


def frames_from_observation(observation, camera_order):
    """Flatten batch-one video into (history * cameras, H, W, 3) uint8.

    Official order is time first, then cameras in checkpoint modality order.
    The temporal copies inside each vision patch are unrelated to this history.
    """
    videos = observation["video"]
    if set(videos) != set(camera_order) or not camera_order:
        raise ValueError("video keys must match the checkpoint camera order")
    values = [np.asarray(videos[key]) for key in camera_order]
    if any(v.ndim != 5 or v.shape[0] != 1 or v.shape[-1] != 3
           or v.dtype != np.uint8 or v.shape != values[0].shape for v in values):
        raise ValueError("video must be matching uint8 (1, history, H, W, 3) arrays")
    frames = np.stack([v[0] for v in values], axis=1)
    return frames.reshape(-1, *frames.shape[-3:])


def _host_frames(frames):
    """Accept the same tensor/ndarray/sequence forms as Orin's frames= API."""
    def array(value):
        if isinstance(value, torch.Tensor):
            if value.dtype != torch.uint8:
                raise ValueError("frames must be uint8 camera bytes")
            return value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            return value
        raise ValueError("frames must contain tensors or numpy arrays")

    if isinstance(frames, (list, tuple)):
        if not frames:
            raise ValueError("frames must not be empty")
        frames = np.stack([array(frame) for frame in frames])
    elif isinstance(frames, (torch.Tensor, np.ndarray)):
        frames = array(frames)
    else:
        raise ValueError("frames must be a tensor, ndarray or ordered sequence, not a mapping")
    if frames.dtype != np.uint8:
        raise ValueError("frames must be uint8 camera bytes")
    if frames.ndim != 4 or frames.shape[-1] != 3 or any(n == 0 for n in frames.shape):
        raise ValueError("frames must have shape (images, H, W, 3) with nonempty dimensions")
    return np.ascontiguousarray(frames)


def process_frames(processor, frames, expected_grid):
    """Prepare fresh pixels without re-tokenizing or executing model weights.

    Uses the checkpoint's official eval image transforms and HF image processor,
    including non-square resize, crop and smart-resize rules. It omits the
    lossless PIL round trips in process_observation. Processing is on the host;
    device frames are copied to host explicitly. No approximate GPU resize is
    substituted, and no CPU/transfer cost is hidden in the reported boundary.
    """
    frames = _host_frames(frames)
    if frames.shape[0] != expected_grid.shape[0]:
        raise ValueError("frames count must match every camera/history image from set_prompt")
    processor.eval()
    if processor.use_albumentations:
        from gr00t.model.gr00t_n1d7.image_augmentations import apply_with_replay
        images, _ = apply_with_replay(processor.eval_image_transform, frames)
    else:
        images = processor.eval_image_transform(
            torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0))[0]
        images = list(images)
    batch = processor.processor.image_processor(
        images=images, input_data_format="channels_first", return_tensors="pt")
    if not torch.equal(batch["image_grid_thw"].cpu(), expected_grid.cpu()):
        raise ValueError("frames changed image grid; call set_prompt with the new observation")
    return batch["pixel_values"]
