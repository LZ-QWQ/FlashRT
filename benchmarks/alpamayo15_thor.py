# SPDX-License-Identifier: Apache-2.0
"""Reproduce full-request and streaming Alpamayo 1.5 arms on real input packs.

Prepare a pack with --prepare using the official dataset loader. Downloads and
model construction are outside timed requests; preprocessing is included in
request_ms. Load only packs you created or trust (torch serialization).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch


def summarize(values):
    return {"n": len(values), **{k: float(v) for k, v in zip(
        ("mean", "p50", "p90", "p99"),
        (np.mean(values), *np.percentile(values, (50, 90, 99))))}} if values else {"n": 0}


def parse_options(text):
    """``k=v,k=v`` -> dict; values ``True`` / ``False`` / ``None`` and numbers are converted, the rest stay strings."""
    out = {}
    for item in filter(None, (text or "").split(",")):
        key, _, value = item.partition("=")
        if not key or not _:
            raise ValueError(f"option {item!r} is not key=value")
        literal = {"True": True, "False": False, "None": None}
        if value in literal:
            out[key] = literal[value]
        else:
            try:
                out[key] = int(value)
            except ValueError:
                try:
                    out[key] = float(value)
                except ValueError:
                    out[key] = value
    return out


def local_dir(repo_or_dir, revision=None):
    """A local directory as is; a Hugging Face repo ID through the (offline-capable) cache."""
    if os.path.isdir(repo_or_dir):
        return repo_or_dir
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=repo_or_dir, revision=revision)


def prepare(args):
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
    import physical_ai_av
    if args.windows < 4 or args.stride_us <= 0:
        raise ValueError("use at least four windows and a positive stride")
    interface = physical_ai_av.PhysicalAIAVDatasetInterface(revision=args.dataset_revision)
    windows = [load_physical_aiavdataset(args.clip, t0_us=args.start_us + i * args.stride_us,
                                       avdi=interface) for i in range(args.windows)]
    pack = {"clip_id": args.clip, "dataset_revision": interface.revision,
            "start_us": args.start_us, "stride_us": args.stride_us, "windows": windows}
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pack, destination)
    print(json.dumps({"clip_id": args.clip, "windows": len(windows),
                      "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}))


def run(args):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (11, 0):
        raise RuntimeError("this benchmark requires Jetson AGX Thor (SM110)")
    from alpamayo1_5 import helper
    torch.set_float32_matmul_precision("high")
    pack = torch.load(args.inputs, map_location="cpu", weights_only=False)
    windows = pack["windows"]
    streaming = args.arm in ("flashdrive", "fd_flashrt")
    if streaming and pack["stride_us"] != 100000:
        raise ValueError("documented streaming protocol requires consecutive 0.1-second windows")
    if len(windows) <= args.warmup_windows:
        raise ValueError("pack has no windows remaining after warmup exclusion")
    start = time.perf_counter()
    if args.arm in ("official", "flashrt_full"):
        if args.arm == "official":
            from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
            model = Alpamayo1_5.from_pretrained(args.checkpoint, dtype=torch.bfloat16,
                    attn_implementation="sdpa").to("cuda").eval()
            processor = helper.get_processor(model.tokenizer)
            if args.official_expert_graph:
                model.enable_diffusion_expert_cuda_graph(max_batch_size=1, max_graphs=4)
        else:
            from flash_rt.frontends.torch.alpamayo15_thor import Alpamayo15TorchFrontendThor
            engine = parse_options(args.engine_kw)
            engine.setdefault("preset", args.preset)
            if engine["preset"] != "bf16" or engine.get("decode_weights") == "w4a8":
                engine.setdefault("paro_checkpoint", local_dir(args.paro))      # the ParoQuant W4A8 language model
            frontend = Alpamayo15TorchFrontendThor(args.checkpoint, max_new_tokens=args.max_new_tokens, engine_kwargs=engine)
            model, processor = frontend.model, frontend.processor
    elif args.arm == "flashdrive":
        import flashdrive
        from flashdrive.streaming import convert_to_streaming_window
        # FlashDrive's loader leaves the LM head row of its added mask token at a random initialization: a fixed seed
        # makes the reference the same model in every process (the caller's generators untouched)
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(0)
            model = flashdrive.from_pretrained(args.base, torch_compile="max-autotune")
        from flashdrive import FlashDriveMixin
        if not isinstance(model, FlashDriveMixin):
            raise ValueError("author loader needs a z-lab repo ID; use its pinned offline HF cache")
        processor = helper.get_processor(model.tokenizer)
    else:
        from flash_rt.frontends.torch.alpamayo15_flashdrive_thor import (
            Alpamayo15FlashDriveTorchFrontendThor, FlashDriveCheckpoints, FlashDriveRollout)
        frontend = Alpamayo15FlashDriveTorchFrontendThor(
            FlashDriveCheckpoints(args.base, args.paro, args.dflash),
            rollout=FlashDriveRollout(max_new_tokens=args.max_new_tokens), backends=parse_options(args.engine_kw) or None)
    load_s = time.perf_counter() - start
    rows = []
    torch.manual_seed(args.seed)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for index, data in enumerate(windows):
            torch.cuda.synchronize()
            begin = time.perf_counter()
            if args.arm == "fd_flashrt":
                result = frontend.step(data["image_frames"], data["camera_indices"],
                                       data["ego_history_xyz"], data["ego_history_rot"])
                xyz = None if result is None else result["pred_xyz"]
            else:
                messages = helper.create_message(frames=data["image_frames"].flatten(0, 1),
                             camera_indices=data["camera_indices"], num_frames_per_camera=4)
                tokens = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                         continue_final_message=True, return_dict=True, return_tensors="pt")
                inputs = {"tokenized_data": tokens, "ego_history_xyz": data["ego_history_xyz"],
                          "ego_history_rot": data["ego_history_rot"]}
                if streaming and index > 0:
                    inputs = convert_to_streaming_window(
                        inputs, model.tokenizer.convert_tokens_to_ids("<|vision_start|>"),
                        model.tokenizer.convert_tokens_to_ids("<|vision_end|>"))
                inputs = helper.to_device(inputs, "cuda")
                if streaming:
                    result = model.sample_trajectories_streaming(
                        inputs, max_new_tokens=args.max_new_tokens, num_traj_samples=1, return_extra=True,
                        diffusion_kwargs={"inference_step": 8, "cache_steps": [3, 4, 5, 6],
                                          "int_method": "euler_with_cache"})
                    xyz = result[0]
                else:
                    torch.cuda.manual_seed_all(args.seed * 1000 + index)
                    if args.arm == "official":
                        result = model.sample_trajectories_from_data_with_vlm_rollout(
                            inputs, top_p=0.98, temperature=0.6, num_traj_samples=1,
                            max_generation_length=args.max_new_tokens, return_extra=True)
                    else:
                        result = frontend.engine.sample_trajectories(inputs, num_traj_samples=1)
                    xyz = result[0]
            xyz = None if xyz is None else xyz.cpu()
            if xyz is not None and not torch.isfinite(xyz).all():
                raise RuntimeError("model returned a non-finite trajectory")
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - begin) * 1000
            row = {"window": index, "request_ms": elapsed, "has_trajectory": xyz is not None}
            if xyz is not None:
                distance = torch.linalg.vector_norm(
                    xyz.reshape(-1, 64, 3)[0, :, :2] - data["ego_future_xyz"].reshape(-1, 64, 3)[0, :, :2], dim=-1)
                row.update(ade_m=float(distance.mean()), fde_m=float(distance[-1]))
            rows.append(row)
            print(json.dumps(row), flush=True)
    steady = [r for r in rows if r["window"] >= args.warmup_windows and r["has_trajectory"]]
    report = {"arm": args.arm, "clip_id": pack["clip_id"], "dataset_revision": pack["dataset_revision"],
              "input_sha256": hashlib.sha256(Path(args.inputs).read_bytes()).hexdigest(),
              "seed": args.seed, "num_traj_samples": 1, "max_new_tokens": args.max_new_tokens,
              "warmup_windows": args.warmup_windows, "load_s": load_s,
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "steady_request_ms": summarize([r["request_ms"] for r in steady]),
              "steady_ade_m": summarize([r["ade_m"] for r in steady]),
              "steady_fde_m": summarize([r["fde_m"] for r in steady]), "windows": rows}
    if not steady:
        raise RuntimeError("no trajectory-producing steady windows")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "windows"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--arm", choices=("official", "flashrt_full", "flashdrive", "fd_flashrt"))
    parser.add_argument("--inputs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--clip", default="87147a1b-3eef-4c25-94d2-ec7718a49a7a")
    parser.add_argument("--start-us", type=int, default=1700000)
    parser.add_argument("--stride-us", type=int, default=100000)
    parser.add_argument("--windows", type=int, default=12)
    parser.add_argument("--dataset-revision", default="33f9bf447ed3bcb7d545ce13f4226f824214fafb")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-windows", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--checkpoint", default="nvidia/Alpamayo-1.5-10B")
    parser.add_argument("--base", default="z-lab/Alpamayo-1.5-10B")
    parser.add_argument("--paro", default="z-lab/Alpamayo-1.5-10B-PARO")
    parser.add_argument("--dflash", default="z-lab/Alpamayo-1.5-10B-DFlash")
    parser.add_argument("--official-expert-graph", action="store_true")
    parser.add_argument("--preset", choices=("bf16", "balanced", "fast"), default="balanced",
                        help="flashrt_full: the engine's precision preset (balanced and fast load the ParoQuant checkpoint of --paro)")
    parser.add_argument("--engine-kw", default="",
                        help="k=v,... options of the FlashRT engine (flashrt_full) or backend (fd_flashrt), e.g. decode_weights=int8")
    args = parser.parse_args()
    if args.prepare:
        prepare(args)
    else:
        if not args.arm or not args.inputs:
            parser.error("--arm and --inputs are required for a benchmark")
        run(args)


if __name__ == "__main__":
    main()
