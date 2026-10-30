#!/usr/bin/env python3
"""Run the projected GR00T N1.7 contract shared by RTX SM120, CDNA and RDNA.

The NPZ contains pixel_features, llm_input_embeds, grid_thw, visual_pos_masks,
rope_cos, rope_sin, and state.<modality> arrays. These features must come from
the same checkpoint and observation. Their production includes model work;
this example does not report end-to-end performance from saved features.
"""

import argparse
import json

import numpy as np
import torch
import flash_rt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--observation", required=True)
    parser.add_argument("--hardware", default="auto")
    parser.add_argument("--fp8", action="store_true", help="select the RTX/CDNA FP8 route; omit on RDNA")
    parser.add_argument("--embodiment", default="oxe_droid_relative_eef_relative_joint")
    args = parser.parse_args()
    model = flash_rt.load_model(args.checkpoint, config="groot_n17", hardware=args.hardware,
                               use_fp8=args.fp8, embodiment_tag=args.embodiment)
    frontend = model.pipeline
    with np.load(args.observation, allow_pickle=False) as data:
        aux = {k: torch.from_numpy(data[k]) for k in (
            "pixel_features", "llm_input_embeds", "grid_thw", "visual_pos_masks", "rope_cos", "rope_sin")}
        state_dict = {k: data[k] for k in data.files if k.startswith("state.")}
    frontend.set_prompt(aux=aux)
    state = frontend.normalize_state(state_dict)
    normalized = frontend.infer(state, aux=aux)
    actions = frontend.denormalize_action(normalized, state_dict=state_dict)
    print(json.dumps({k: v.tolist() for k, v in actions.items()}))


if __name__ == "__main__":
    main()
