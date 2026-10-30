#!/usr/bin/env python3
"""Run GR00T N1.7 BF16 from a raw DROID observation on AMD RDNA."""

import argparse
import json
import time

import numpy as np
import torch

import flash_rt
from flash_rt.models.groot_n17.observation import frames_from_observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processor", required=True, help="Cosmos-Reason2 tokenizer/image processor directory")
    parser.add_argument("--observation", required=True, help="NPZ with video.*, state.* and instruction arrays")
    parser.add_argument("--hardware", default="auto", help="RDNA target, or rtx_sm87 for Orin")
    parser.add_argument("--input", choices=("frames", "aux"), default="frames")
    parser.add_argument("--embodiment", default="oxe_droid_relative_eef_relative_joint")
    parser.add_argument("--benchmark", type=int, default=0, help="number of measured predictions after warmup")
    args = parser.parse_args()
    if args.benchmark < 0:
        parser.error("--benchmark must be nonnegative")
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

    with np.load(args.observation, allow_pickle=False) as data:
        observation = {group: {key.split(".", 1)[1]: data[key] for key in data.files
                               if key.startswith(group + ".")}
                       for group in ("video", "state")}
        instruction = str(data["instruction"].item())
    processor = Gr00tN1d7Processor.from_pretrained(args.checkpoint, model_name=args.processor)
    model = flash_rt.load_model(args.checkpoint, config="groot_n17", framework="torch",
                               num_views=len(observation["video"]), use_fp8=False,
                               hardware=args.hardware, embodiment_tag=args.embodiment,
                               fuse_image_embeds=True)
    model.pipeline.set_hf_processor(processor)
    language_key = processor.modality_configs[model.pipeline.embodiment_tag]["language"].modality_keys[0]
    observation["language"] = {language_key: [[instruction]]}
    frontend = model.pipeline
    _, aux = frontend.prepare_observation(observation)
    model.set_prompt(aux=aux, prompt=instruction)
    camera_order = processor.modality_configs[frontend.embodiment_tag]["video"].modality_keys

    def run_observation():
        # Submit fresh frames/state each time. Reinitialize if instruction/grid
        # changes; preparation itself does not change the active prompt.
        if args.input == "frames":
            frames = frames_from_observation(observation, camera_order)
            state = frontend.normalize_state({"state." + k: v for k, v in observation["state"].items()})
            normalized = model.infer(state, frames=frames)
        else:
            state, aux = frontend.prepare_observation(observation)
            normalized = model.infer(state, aux=aux)
        return frontend.denormalize_action(normalized, state_dict=observation["state"])

    actions = run_observation()
    report = {"hardware": getattr(frontend, "hardware", args.hardware), "input": args.input,
              "action_shapes": {k: list(v.shape) for k, v in actions.items()},
              "finite": all(bool(torch.isfinite(v).all()) for v in actions.values())}
    if args.benchmark:
        for _ in range(5):
            run_observation()
        wall = []
        for _ in range(args.benchmark):
            started = time.perf_counter()
            run_observation()
            wall.append((time.perf_counter() - started) * 1000)
        report.update(observation_p50_ms=float(np.median(wall)),
                      timing_scope="frame preparation, transfers, state normalization, model and physical decoding; excludes setup/warmup")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
