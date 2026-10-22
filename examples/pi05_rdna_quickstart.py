"""Synthetic-input Pi0.5 BF16 smoke test on a supported AMD RDNA GPU.

Set FLASH_RT_PALIGEMMA_TOKENIZER to the PaliGemma SentencePiece model.
This checks the public inference path, not robot task accuracy.
"""
import argparse
import json

import numpy as np
import torch

import flash_rt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--action-dim", type=int, default=None,
                        help="Robot output width; LIBERO uses 7")
    parser.add_argument("--hardware", default="auto",
                        choices=("auto", "amd_rdna3", "amd_rdna35", "amd_rdna4"))
    parser.add_argument("--num-views", type=int, choices=(2, 3), default=2)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--prompt", default="pick up the object")
    args = parser.parse_args()

    from flash_rt.hardware import detect_arch
    hardware = detect_arch() if args.hardware == "auto" else args.hardware
    if hardware not in ("amd_rdna3", "amd_rdna35", "amd_rdna4"):
        parser.error(f"requires a supported RDNA GPU, detected {hardware!r}")
    torch.manual_seed(17)
    model = flash_rt.load_model(
        args.checkpoint, config="pi05", framework="torch", hardware=hardware,
        use_fp8=False, action_dim=args.action_dim, num_views=args.num_views,
        num_steps=args.num_steps,
    )
    rng = np.random.default_rng(17)
    images = [rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
              for _ in range(args.num_views)]
    actions = model.predict(images=images, prompt=args.prompt)
    expected = (model.pipeline.chunk_size, model.pipeline.action_dim)
    if actions.shape != expected or not np.isfinite(actions).all():
        raise RuntimeError(f"invalid Pi0.5 output: shape={actions.shape}, expected={expected}")
    from flash_rt.amd import flash_rt_amd_kernels
    print(json.dumps({"hardware": hardware, "torch": torch.__version__,
                      "build": dict(flash_rt_amd_kernels.build_info()),
                      "actions_shape": list(actions.shape), "finite": True,
                      "sample": actions[0].tolist()}, indent=2))


if __name__ == "__main__":
    main()
