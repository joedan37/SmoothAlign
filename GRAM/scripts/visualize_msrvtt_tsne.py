#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
from easydict import EasyDict as edict
from PIL import Image, ImageDraw, ImageFont
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data import data_registry  # noqa: E402
from model import model_registry  # noqa: E402


DEFAULT_RUNS = {
    "version1": {
        "hps": os.path.join(
            ROOT,
            "output/gram/finetune_msrvtt_from_GRAM_pretrained_5modalities/log/hps.json",
        ),
        "checkpoint": os.path.join(
            ROOT,
            "output/gram/finetune_msrvtt_from_GRAM_pretrained_5modalities/ckpt/"
            "best_ret%tvas--msrvtt_ret_ret_area_forward.pt",
        ),
    },
    "version2": {
        "hps": os.path.join(
            ROOT,
            "output/gram/version2_gaussian_prior_stable_all_volume_"
            "finetune_msrvtt_from_GRAM_pretrained_5modalities/log/hps.json",
        ),
        "checkpoint": os.path.join(
            ROOT,
            "output/gram/version2_gaussian_prior_stable_all_volume_"
            "finetune_msrvtt_from_GRAM_pretrained_5modalities/ckpt/"
            "best_ret%tvas--msrvtt_ret_ret_area_forward.pt",
        ),
    },
}

CATEGORY_KEYWORDS = {
    "music": [
        "music",
        "song",
        "sing",
        "singer",
        "guitar",
        "piano",
        "dance",
        "dancing",
        "band",
        "concert",
    ],
    "sports": [
        "football",
        "soccer",
        "basketball",
        "baseball",
        "tennis",
        "golf",
        "skate",
        "snowboard",
        "ski",
        "surf",
        "race",
        "racing",
        "wrestl",
        "gymnastics",
    ],
    "cooking": [
        "cook",
        "cooking",
        "food",
        "kitchen",
        "recipe",
        "dish",
        "meal",
        "eat",
        "eating",
        "bake",
        "chef",
    ],
    "vehicle": [
        "car",
        "truck",
        "train",
        "bus",
        "motorcycle",
        "bike",
        "bicycle",
        "vehicle",
        "driving",
        "road",
        "boat",
        "plane",
    ],
    "animal": [
        "dog",
        "cat",
        "horse",
        "animal",
        "bird",
        "fish",
        "puppy",
        "cow",
    ],
    "game": [
        "game",
        "video game",
        "minecraft",
        "grand theft auto",
        "gta",
        "player",
    ],
}

COLORS = {
    "music": (0, 158, 115),
    "sports": (230, 159, 0),
    "cooking": (213, 94, 0),
    "vehicle": (0, 114, 178),
    "animal": (204, 121, 167),
    "game": (86, 180, 233),
}

MODALITY_MARKERS = {
    "T": "circle",
    "V": "square",
    "A": "triangle",
}


def load_font(size):
    font_dir = os.environ.get(
        "SMOOTHALIGN_FONT_DIR",
        os.path.join(os.path.dirname(__file__), "..", "assets", "fonts"),
    )
    candidates = [
        os.path.join(font_dir, "DejaVuSans.ttf"),
        os.path.join(font_dir, "DejaVuSans-Bold.ttf"),
        os.path.join(font_dir, "LiberationSans-Regular.ttf"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def classify_caption(caption):
    text = caption.lower()
    hits = []
    for label, words in CATEGORY_KEYWORDS.items():
        if any(word in text for word in words):
            hits.append(label)
    return hits[0] if len(hits) == 1 else None


def select_indices(anno_path, max_per_class, seed):
    annos = json.load(open(anno_path))
    buckets = defaultdict(list)
    seen_video_ids = set()
    for idx, item in enumerate(annos):
        video_id = item.get("video_id")
        if video_id in seen_video_ids:
            continue
        label = classify_caption(item.get("desc", ""))
        if label is None:
            continue
        buckets[label].append(idx)
        seen_video_ids.add(video_id)

    rng = random.Random(seed)
    selected = []
    for label in CATEGORY_KEYWORDS:
        ids = list(buckets[label])
        rng.shuffle(ids)
        selected.extend((idx, label) for idx in ids[:max_per_class])

    selected.sort(key=lambda x: x[0])
    indices = [idx for idx, _ in selected]
    labels_by_index = {idx: label for idx, label in selected}
    counts = {label: sum(1 for _, item_label in selected if item_label == label) for label in CATEGORY_KEYWORDS}
    return indices, labels_by_index, counts


def load_hps(path, batch_size, num_workers):
    hps = json.load(open(path))
    args = edict(hps)
    args.local_rank = 0
    args.run_cfg = edict(args.run_cfg)
    args.model_cfg = edict(args.model_cfg)
    args.data_cfg = edict(args.data_cfg)
    args.data_cfg.train = [edict(x) for x in args.data_cfg.get("train", [])]
    args.data_cfg.val = [edict(x) for x in args.data_cfg.val]
    args.run_cfg.use_ddp = False
    args.run_cfg.fp16 = False
    args.run_cfg.bf16 = False
    args.model_cfg.checkpointing = False
    args.model_cfg.fp16 = False
    args.data_cfg.val[0].batch_size = batch_size
    args.data_cfg.val[0].n_workers = num_workers
    args.data_cfg.val[0].training = False
    return args


def load_model(model_cfg, checkpoint_path, device):
    model = model_registry[model_cfg.model_type](model_cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    checkpoint = model.modify_checkpoint(checkpoint)
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    print(f"Loaded {checkpoint_path}")
    print(f"  missing keys: {len(missing)}")
    print(f"  unexpected keys: {len(unexpected)}")
    model.to(device)
    model.eval()
    return model


def make_subset_loader(args, indices):
    d_cfg = args.data_cfg.val[0]
    dataset = data_registry[d_cfg.type](d_cfg, args)
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=args.data_cfg.val[0].batch_size,
        shuffle=False,
        num_workers=args.data_cfg.val[0].n_workers,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        drop_last=False,
    )
    return loader


def move_batch_to_device(batch, device):
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def extract_features(run_name, run_cfg, indices, labels_by_index, batch_size, num_workers, device, cache_dir):
    cache_path = os.path.join(cache_dir, f"{run_name}_features.pt")
    if os.path.exists(cache_path):
        print(f"Using cached features: {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    args = load_hps(run_cfg["hps"], batch_size=batch_size, num_workers=num_workers)
    loader = make_subset_loader(args, indices)
    model = load_model(args.model_cfg, run_cfg["checkpoint"], device)

    records = []
    offset = 0
    with torch.no_grad():
        for step, batch in enumerate(loader):
            current_indices = indices[offset : offset + len(batch["ids"])]
            offset += len(batch["ids"])
            batch = move_batch_to_device(batch, device)
            output = model(batch, args.data_cfg.val[0].task, compute_loss=False)
            feat_t = output["feat_t"].detach().float().cpu()
            feat_v = output["feat_v"].detach().float().cpu()
            feat_a = output["feat_a"].detach().float().cpu()
            captions = batch["raw_captions"]
            ids = batch["ids"]
            for i, anno_idx in enumerate(current_indices):
                label = labels_by_index[anno_idx]
                records.append(
                    {
                        "anno_idx": anno_idx,
                        "video_id": ids[i],
                        "caption": captions[i],
                        "label": label,
                        "T": feat_t[i],
                        "V": feat_v[i],
                        "A": feat_a[i],
                    }
                )
            print(f"{run_name}: extracted batch {step + 1}/{len(loader)}")

    payload = {"run": run_name, "records": records}
    torch.save(payload, cache_path)
    print(f"Saved features: {cache_path}")
    return payload


def l2_normalize(array):
    norm = np.linalg.norm(array, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return array / norm


def build_tsne_points(features_by_run, seed, perplexity):
    vectors = []
    meta = []
    for run_name, payload in features_by_run.items():
        for record in payload["records"]:
            for modality in ("T", "V", "A"):
                vectors.append(record[modality].numpy())
                meta.append(
                    {
                        "run": run_name,
                        "modality": modality,
                        "label": record["label"],
                        "video_id": record["video_id"],
                        "anno_idx": record["anno_idx"],
                        "caption": record["caption"],
                    }
                )

    vectors = l2_normalize(np.asarray(vectors, dtype=np.float32))
    max_perplexity = max(5, min(perplexity, (len(vectors) - 1) // 3))
    coords = TSNE(
        n_components=2,
        perplexity=max_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(vectors)
    for item, coord in zip(meta, coords):
        item["x"] = float(coord[0])
        item["y"] = float(coord[1])
    return meta


def draw_marker(draw, x, y, color, marker, radius):
    if marker == "circle":
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(30, 30, 30))
    elif marker == "square":
        draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(30, 30, 30))
    elif marker == "triangle":
        pts = [(x, y - radius - 1), (x - radius - 1, y + radius), (x + radius + 1, y + radius)]
        draw.polygon(pts, fill=color, outline=(30, 30, 30))
    else:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def scale_points(points, left, top, width, height):
    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    pad_x = (max_x - min_x) * 0.08 + 1e-6
    pad_y = (max_y - min_y) * 0.08 + 1e-6
    min_x -= pad_x
    max_x += pad_x
    min_y -= pad_y
    max_y += pad_y
    for p in points:
        p["px"] = left + (p["x"] - min_x) / (max_x - min_x) * width
        p["py"] = top + height - (p["y"] - min_y) / (max_y - min_y) * height


def draw_tsne(meta, output_path):
    runs = list(DEFAULT_RUNS.keys())
    width, height = 1900, 980
    margin_x, margin_y = 70, 90
    panel_gap = 60
    legend_h = 150
    panel_w = (width - 2 * margin_x - panel_gap) // 2
    panel_h = height - margin_y - legend_h - 40

    image = Image.new("RGB", (width, height), (248, 249, 252))
    draw = ImageDraw.Draw(image)
    title_font = load_font(30)
    label_font = load_font(22)
    small_font = load_font(18)

    for idx, run in enumerate(runs):
        left = margin_x + idx * (panel_w + panel_gap)
        top = margin_y
        draw.rectangle((left, top, left + panel_w, top + panel_h), fill=(235, 238, 245), outline=(205, 210, 220))
        for gx in range(1, 5):
            x = left + gx * panel_w / 5
            draw.line((x, top, x, top + panel_h), fill=(220, 224, 232))
        for gy in range(1, 5):
            y = top + gy * panel_h / 5
            draw.line((left, y, left + panel_w, y), fill=(220, 224, 232))

        title = f"t-SNE Visualization of {run.upper()} latent space"
        tw = draw.textlength(title, font=title_font)
        draw.text((left + (panel_w - tw) / 2, 35), title, fill=(25, 30, 40), font=title_font)
        points = [dict(p) for p in meta if p["run"] == run]
        scale_points(points, left + 25, top + 25, panel_w - 50, panel_h - 50)
        for p in points:
            draw_marker(
                draw,
                p["px"],
                p["py"],
                COLORS[p["label"]],
                MODALITY_MARKERS[p["modality"]],
                6,
            )

    legend_top = height - legend_h + 15
    draw.text((margin_x, legend_top), "Modality", fill=(30, 30, 30), font=label_font)
    lx = margin_x + 125
    for modality, marker in MODALITY_MARKERS.items():
        draw_marker(draw, lx, legend_top + 15, (70, 70, 70), marker, 7)
        draw.text((lx + 18, legend_top + 3), modality, fill=(30, 30, 30), font=small_font)
        lx += 80

    draw.text((margin_x, legend_top + 55), "Pseudo categories", fill=(30, 30, 30), font=label_font)
    lx = margin_x + 215
    ly = legend_top + 58
    for label in CATEGORY_KEYWORDS:
        color = COLORS[label]
        draw.rectangle((lx, ly + 3, lx + 22, ly + 25), fill=color, outline=(30, 30, 30))
        draw.text((lx + 30, ly), label, fill=(30, 30, 30), font=small_font)
        lx += 145

    note = "Same selected samples and one shared t-SNE fit are used for both panels."
    draw.text((margin_x, height - 35), note, fill=(80, 80, 80), font=small_font)
    image.save(output_path)
    print(f"Saved plot: {output_path}")


def compute_distance_metrics(features_by_run, output_csv):
    rows = []
    for run_name, payload in features_by_run.items():
        by_label = defaultdict(list)
        for record in payload["records"]:
            by_label[record["label"]].append(record)

        for label, records in by_label.items():
            tv = []
            ta = []
            va = []
            for rec in records:
                t = rec["T"].numpy()
                v = rec["V"].numpy()
                a = rec["A"].numpy()
                tv.append(float(np.linalg.norm(t - v)))
                ta.append(float(np.linalg.norm(t - a)))
                va.append(float(np.linalg.norm(v - a)))
            rows.append(
                {
                    "run": run_name,
                    "label": label,
                    "n": len(records),
                    "same_sample_TV_l2": np.mean(tv),
                    "same_sample_TA_l2": np.mean(ta),
                    "same_sample_VA_l2": np.mean(va),
                }
            )

        all_records = payload["records"]
        rows.append(
            {
                "run": run_name,
                "label": "ALL",
                "n": len(all_records),
                "same_sample_TV_l2": np.mean([float(np.linalg.norm(r["T"].numpy() - r["V"].numpy())) for r in all_records]),
                "same_sample_TA_l2": np.mean([float(np.linalg.norm(r["T"].numpy() - r["A"].numpy())) for r in all_records]),
                "same_sample_VA_l2": np.mean([float(np.linalg.norm(r["V"].numpy() - r["A"].numpy())) for r in all_records]),
            }
        )

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run", "label", "n", "same_sample_TV_l2", "same_sample_TA_l2", "same_sample_VA_l2"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved metrics: {output_csv}")
    return rows


def write_points_csv(meta, output_csv):
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run", "modality", "label", "video_id", "anno_idx", "x", "y", "caption"],
        )
        writer.writeheader()
        for row in meta:
            writer.writerow(row)
    print(f"Saved t-SNE coordinates: {output_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "output/gram/tsne_msrvtt_v1_v2"))
    parser.add_argument("--max-per-class", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perplexity", type=int, default=35)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = os.path.join(args.output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA is not available. Feature extraction on CPU will be slow.")

    anno_path = json.load(open(DEFAULT_RUNS["version1"]["hps"]))["data_cfg"]["val"][0]["txt"]
    indices, labels_by_index, counts = select_indices(anno_path, args.max_per_class, args.seed)
    print(f"Selected {len(indices)} MSRVTT samples: {counts}")
    with open(os.path.join(args.output_dir, "selected_samples.json"), "w") as f:
        json.dump(
            {
                "annotation": anno_path,
                "max_per_class": args.max_per_class,
                "counts": counts,
                "indices": indices,
                "labels_by_index": labels_by_index,
            },
            f,
            indent=2,
        )

    features_by_run = {}
    for run_name, run_cfg in DEFAULT_RUNS.items():
        features_by_run[run_name] = extract_features(
            run_name,
            run_cfg,
            indices,
            labels_by_index,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            cache_dir=cache_dir,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    meta = build_tsne_points(features_by_run, seed=args.seed, perplexity=args.perplexity)
    write_points_csv(meta, os.path.join(args.output_dir, "tsne_points.csv"))
    draw_tsne(meta, os.path.join(args.output_dir, "msrvtt_v1_v2_tsne.png"))
    rows = compute_distance_metrics(features_by_run, os.path.join(args.output_dir, "distance_metrics.csv"))

    summary = {
        "selected_counts": counts,
        "distance_metrics": rows,
        "plot": os.path.join(args.output_dir, "msrvtt_v1_v2_tsne.png"),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
