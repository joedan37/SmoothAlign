#!/usr/bin/env python3
"""Cache frozen GRAM encoder outputs for a fair, large-batch IEMOCAP study."""

import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from iemocap_common import (
    IemocapTripletDataset,
    LABEL_PROMPTS,
    collate_iemocap,
    load_pretrained_model,
    pooled_text_outputs,
    pooled_video_outputs,
    set_seed,
)


@torch.no_grad()
def cache_split(model, args, split, device):
    dataset = IemocapTripletDataset(
        args.dataset_dir,
        split,
        num_video_frames=args.num_video_frames,
        image_size=args.image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_iemocap,
    )
    tensor_keys = ("text_final", "text_middle", "video_final", "video_middle", "audio")
    chunks = {key: [] for key in tensor_keys}
    ids, labels, texts = [], [], []
    started = time.time()
    for step, batch in enumerate(loader, start=1):
        text_final, text_middle = pooled_text_outputs(
            model, batch["texts"], device, need_middle=True, fp16=args.fp16
        )
        video_final, video_middle = pooled_video_outputs(
            model, batch["video"], device, need_middle=True, fp16=args.fp16
        )
        chunks["text_final"].append(text_final.cpu())
        chunks["text_middle"].append(text_middle.cpu())
        chunks["video_final"].append(video_final.cpu())
        chunks["video_middle"].append(video_middle.cpu())
        chunks["audio"].append(batch["audio"].float().cpu())
        ids.extend(batch["ids"])
        labels.extend(batch["labels"])
        texts.extend(batch["texts"])
        if step % args.log_steps == 0 or step == len(loader):
            done = min(step * args.batch_size, len(dataset))
            print(
                f"cache split={split} samples={done}/{len(dataset)} elapsed={time.time() - started:.1f}s",
                flush=True,
            )

    payload = {key: torch.cat(value, dim=0).contiguous() for key, value in chunks.items()}
    payload.update(
        {
            "ids": ids,
            "labels": labels,
            "texts": texts,
            "split": split,
            "num_video_frames": args.num_video_frames,
            "image_size": args.image_size,
        }
    )
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--pretrain-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-video-frames", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--log-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required to cache IEMOCAP encoder features.")
    model = load_pretrained_model(args.pretrain_dir, device, drop_audio_encoder=True)

    anchor_labels = sorted(LABEL_PROMPTS)
    anchor_prompts = [LABEL_PROMPTS[label] for label in anchor_labels]
    anchor_final, anchor_middle = pooled_text_outputs(
        model, anchor_prompts, device, need_middle=True, fp16=args.fp16
    )
    anchor_path = os.path.join(args.cache_dir, "class_text_anchors.pt")
    torch.save(
        {
            "labels": anchor_labels,
            "prompts": anchor_prompts,
            "text_final": anchor_final.cpu(),
            "text_middle": anchor_middle.cpu(),
        },
        anchor_path,
    )
    print("Saved class text anchors.", flush=True)

    # Keep generated metadata portable; user-supplied local paths never enter
    # cache payloads or the manifest.
    manifest = {
        "settings": {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "num_video_frames": args.num_video_frames,
            "image_size": args.image_size,
            "seed": args.seed,
        },
        "splits": {},
    }
    for split in ("train", "test"):
        path = os.path.join(args.cache_dir, f"{split}.pt")
        if os.path.exists(path) and not args.overwrite:
            payload = torch.load(path, map_location="cpu")
            print(f"Using existing {split} cache ({len(payload['ids'])} samples).", flush=True)
        else:
            payload = cache_split(model, args, split, device)
            torch.save(payload, path)
            print(f"Saved {split} cache.", flush=True)
        manifest["splits"][split] = {
            "samples": len(payload["ids"]),
            "tensor_shapes": {
                key: list(payload[key].shape)
                for key in ("text_final", "text_middle", "video_final", "video_middle", "audio")
            },
        }
    with open(os.path.join(args.cache_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print("Feature caching complete.", flush=True)


if __name__ == "__main__":
    main()
