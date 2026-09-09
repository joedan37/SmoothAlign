#!/usr/bin/env python3
"""Create a GRAM Figure-4-style V1/V5 comparison with quantitative checks."""

import argparse
import csv
import json
import math
import os
import random
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from iemocap_common import (
    COLORS,
    LABEL_NAMES,
    LABEL_PROMPTS,
    compute_iemocap_gaussian_prior,
    gaussian_confidence_gate,
    load_adapter_state,
    load_pretrained_model,
    pooled_text_outputs,
    project_features,
    project_prior_features,
    set_seed,
)


def save_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def normalize(tensor):
    return F.normalize(tensor.float(), dim=-1)


@torch.no_grad()
def project_model(args, checkpoint, cache, classes, class_anchors, device):
    model = load_pretrained_model(args.pretrain_dir, device, drop_audio_encoder=True)
    checkpoint_payload = load_adapter_state(model, checkpoint, device)
    if class_anchors is None:
        text_final = cache["text_final"].to(device)
        text_middle = cache["text_middle"].to(device)
    else:
        anchor_label_to_row = {label: idx for idx, label in enumerate(class_anchors["labels"])}
        anchor_rows = torch.tensor([anchor_label_to_row[label] for label in cache["labels"]])
        text_final = class_anchors["text_final"].index_select(0, anchor_rows).to(device)
        text_middle = class_anchors["text_middle"].index_select(0, anchor_rows).to(device)
    video_final = cache["video_final"].to(device)
    audio = cache["audio"].to(device)
    feat_t, feat_v, feat_a = project_features(model, text_final, video_final, audio)

    if class_anchors is None:
        prompts = [LABEL_PROMPTS[label] for label in classes]
        prompt_pooled, _ = pooled_text_outputs(model, prompts, device, need_middle=False, fp16=args.fp16)
    else:
        prompt_rows = torch.tensor([anchor_label_to_row[label] for label in classes])
        prompt_pooled = class_anchors["text_final"].index_select(0, prompt_rows).to(device)
    prompt_features = normalize(model.contra_head_t(prompt_pooled))

    video_middle = cache["video_middle"].to(device)
    prior_feats = project_prior_features(model, text_middle, video_middle, audio)
    result = {
        "T": feat_t.cpu(),
        "V": feat_v.cpu(),
        "A": feat_a.cpu(),
        "prompt": prompt_features.cpu(),
        "prior_T": prior_feats[0].cpu(),
        "prior_V": prior_feats[1].cpu(),
        "prior_A": prior_feats[2].cpu(),
        "checkpoint_meta": {
            key: checkpoint_payload.get(key)
            for key in ("version", "epoch", "global_step")
        },
    }
    del model, text_final, video_final, text_middle, video_middle, audio
    torch.cuda.empty_cache()
    return result


def choose_classes(train_cache, count):
    counts = Counter(train_cache["labels"])
    return [label for label, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:count]], counts


def select_test_indices(test_cache, classes, max_per_class, seed):
    rng = random.Random(seed)
    selected = []
    counts = {}
    for label in classes:
        candidates = [idx for idx, value in enumerate(test_cache["labels"]) if value == label]
        rng.shuffle(candidates)
        chosen = sorted(candidates[:max_per_class])
        counts[label] = len(chosen)
        selected.extend(chosen)
    return selected, counts


def build_plot_vectors(projected, selected, labels, classes):
    vectors, metadata = [], []
    for class_index, label in enumerate(classes):
        vectors.append(projected["prompt"][class_index].numpy())
        metadata.append({"label": label, "modality": "Text label", "sample_index": -1})
    for index in selected:
        for key, modality in (("A", "Audio"), ("V", "Video")):
            vectors.append(projected[key][index].numpy())
            metadata.append({"label": labels[index], "modality": modality, "sample_index": index})
    return np.asarray(vectors, dtype=np.float32), metadata


def embed_tsne(vectors, seed, perplexity):
    vectors = vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)
    n_components = min(50, vectors.shape[1], vectors.shape[0] - 1)
    reduced = PCA(n_components=n_components, random_state=seed).fit_transform(vectors)
    reduced = reduced / np.clip(np.linalg.norm(reduced, axis=1, keepdims=True), 1e-12, None)
    actual_perplexity = min(perplexity, max(5, (len(vectors) - 1) // 3))
    coords = TSNE(
        n_components=2,
        perplexity=actual_perplexity,
        metric="cosine",
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(reduced)
    return coords, actual_perplexity


def tsne_metrics(coords, metadata):
    class_labels = [item["label"] for item in metadata]
    modality_labels = [item["modality"] for item in metadata]
    return {
        "class_silhouette": float(silhouette_score(coords, class_labels)),
        "modality_silhouette": float(silhouette_score(coords, modality_labels)),
    }


def label_ids_for(labels):
    mapping = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    return torch.tensor([mapping[label] for label in labels], dtype=torch.long)


def pair_metrics(left, right, labels):
    left = normalize(left)
    right = normalize(right)
    similarity = left @ right.T
    n = similarity.size(0)
    label_ids = label_ids_for(labels)
    eye = torch.eye(n, dtype=torch.bool)
    same = label_ids[:, None].eq(label_ids[None, :]) & ~eye
    different = ~label_ids[:, None].eq(label_ids[None, :])
    nearest = similarity.masked_fill(eye, -float("inf")).argmax(dim=1)
    neighbor_accuracy = label_ids[nearest].eq(label_ids).float().mean()
    paired = similarity.diag().mean()
    same_mean = similarity[same].mean()
    different_mean = similarity[different].mean()
    return {
        "paired_cosine": float(paired),
        "same_class_nonpaired_cosine": float(same_mean),
        "different_class_cosine": float(different_mean),
        "false_negative_preservation_gap": float(same_mean - different_mean),
        "cross_modal_1nn_class_accuracy": float(neighbor_accuracy),
    }


def representation_metrics(projected, labels):
    metrics = {
        "T_to_V": pair_metrics(projected["T"], projected["V"], labels),
        "T_to_A": pair_metrics(projected["T"], projected["A"], labels),
        "V_to_A": pair_metrics(projected["V"], projected["A"], labels),
    }
    cloud = normalize(projected["T"] + projected["V"] + projected["A"])
    metrics["semantic_cloud_class_silhouette"] = float(
        silhouette_score(cloud.numpy(), labels, metric="cosine")
    )
    modality_centroids = [normalize(projected[key].mean(dim=0, keepdim=True))[0] for key in ("T", "V", "A")]
    centroid_cosines = [
        float(torch.dot(modality_centroids[i], modality_centroids[j]))
        for i in range(3)
        for j in range(i + 1, 3)
    ]
    metrics["mean_modality_centroid_cosine"] = float(np.mean(centroid_cosines))
    return metrics


def prior_metrics(projected, selected, labels, args):
    index = torch.tensor(selected, dtype=torch.long)
    features = [projected[key].index_select(0, index) for key in ("prior_T", "prior_V", "prior_A")]
    prior_raw = compute_iemocap_gaussian_prior(features, tau=args.gaussian_tau, kl_reduction="mean")
    prior = gaussian_confidence_gate(
        prior_raw,
        features,
        temp=args.confidence_temp,
        power=args.confidence_power,
        floor=args.confidence_floor,
        subtract_random_baseline=args.confidence_random_baseline,
    )
    selected_labels = [labels[idx] for idx in selected]
    label_ids = label_ids_for(selected_labels)
    eye = torch.eye(len(selected), dtype=torch.bool)
    same = label_ids[:, None].eq(label_ids[None, :]) & ~eye
    different = ~label_ids[:, None].eq(label_ids[None, :])
    return {
        "raw_same_class": float(prior_raw[same].mean()),
        "raw_different_class": float(prior_raw[different].mean()),
        "raw_gap": float(prior_raw[same].mean() - prior_raw[different].mean()),
        "gated_same_class": float(prior[same].mean()),
        "gated_different_class": float(prior[different].mean()),
        "gated_gap": float(prior[same].mean() - prior[different].mean()),
        "same_class_repulsion_weight": float(1.0 - prior[same].mean()),
        "different_class_repulsion_weight": float(1.0 - prior[different].mean()),
    }


def font(size, bold=False):
    font_dir = os.environ.get(
        "SMOOTHALIGN_FONT_DIR",
        os.path.join(os.path.dirname(__file__), "assets", "fonts"),
    )
    candidates = [
        os.path.join(font_dir, "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
        os.path.join(font_dir, "LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def rgb(hex_color):
    value = hex_color.lstrip("#")
    return tuple(int(value[offset : offset + 2], 16) for offset in (0, 2, 4))


def marker(draw, x, y, color, shape, radius):
    outline = (36, 43, 52)
    if shape == "square":
        draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=color, outline=outline, width=1)
    elif shape == "triangle":
        draw.polygon(
            [(x, y - radius - 1), (x - radius - 1, y + radius), (x + radius + 1, y + radius)],
            fill=color,
            outline=outline,
        )
    else:
        points = []
        for index in range(10):
            angle = -math.pi / 2 + index * math.pi / 5
            current = radius if index % 2 == 0 else radius * 0.45
            points.append((x + current * math.cos(angle), y + current * math.sin(angle)))
        draw.polygon(points, fill=color, outline=outline)


def scale_coords(coords, box):
    left, top, width, height = box
    minimum = coords.min(axis=0)
    maximum = coords.max(axis=0)
    span = np.maximum(maximum - minimum, 1e-6)
    padding = span * 0.08
    minimum -= padding
    maximum += padding
    span = maximum - minimum
    x = left + (coords[:, 0] - minimum[0]) / span[0] * width
    y = top + height - (coords[:, 1] - minimum[1]) / span[1] * height
    return np.column_stack([x, y])


def draw_panel(image, box, coords, metadata, title, metrics, seed):
    draw = ImageDraw.Draw(image)
    left, top, width, height = box
    draw.rounded_rectangle(
        (left, top, left + width, top + height), radius=12, fill=(245, 247, 250), outline=(205, 211, 220), width=2
    )
    header = font(25, bold=True)
    small = font(16)
    draw.text((left + 20, top + 14), title, fill=(28, 35, 45), font=header)
    metric_text = (
        f"class silhouette {metrics['class_silhouette']:.3f}   |   "
        f"modality silhouette {metrics['modality_silhouette']:.3f}   |   seed {seed}"
    )
    draw.text((left + 22, top + 50), metric_text, fill=(87, 96, 110), font=small)
    plot_box = (left + 34, top + 82, width - 68, height - 112)
    px, py, pw, ph = plot_box
    for index in range(1, 5):
        gx = px + index * pw / 5
        gy = py + index * ph / 5
        draw.line((gx, py, gx, py + ph), fill=(222, 226, 233), width=1)
        draw.line((px, gy, px + pw, gy), fill=(222, 226, 233), width=1)
    scaled = scale_coords(coords, plot_box)
    style = {"Audio": ("triangle", 6), "Video": ("square", 6), "Text label": ("star", 15)}
    for modality in ("Audio", "Video", "Text label"):
        shape, radius = style[modality]
        for point, item in zip(scaled, metadata):
            if item["modality"] != modality:
                continue
            marker(draw, point[0], point[1], rgb(COLORS[item["label"]]), shape, radius)


def draw_comparison(path, panels, classes, seed, subtitle, controlled=False):
    width, height = 1900, 900
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    title_font = font(34, bold=True)
    subtitle_font = font(18)
    title = "IEMOCAP false-negative stress test on the held-out test split" if controlled else "IEMOCAP latent-space comparison on the held-out test split"
    title_width = draw.textlength(title, font=title_font)
    draw.text(((width - title_width) / 2, 24), title, fill=(24, 31, 41), font=title_font)
    subtitle_width = draw.textlength(subtitle, font=subtitle_font)
    draw.text(((width - subtitle_width) / 2, 69), subtitle, fill=(91, 99, 112), font=subtitle_font)
    boxes = [(45, 110, 885, 650), (970, 110, 885, 650)]
    v1_coords, v1_metadata, v1_metrics = panels["v1"]
    v5_coords, v5_metadata, v5_metrics = panels["v5"]
    draw_panel(image, boxes[0], v1_coords, v1_metadata, "(a) Original GRAM (V1)", v1_metrics, seed)
    draw_panel(image, boxes[1], v5_coords, v5_metadata, "(b) SmoothGRAM (V5)", v5_metrics, seed)

    legend_font = font(19)
    legend_y = 797
    x = 70
    draw.text((x, legend_y), "Modality", fill=(35, 42, 52), font=legend_font)
    x += 105
    for label, shape in (("Text label", "star"), ("Audio", "triangle"), ("Video", "square")):
        marker(draw, x + 10, legend_y + 12, (110, 118, 130), shape, 10 if shape == "star" else 7)
        draw.text((x + 27, legend_y), label, fill=(45, 52, 62), font=legend_font)
        x += 145
    x += 45
    draw.text((x, legend_y), "Emotion class", fill=(35, 42, 52), font=legend_font)
    x += 155
    for label in classes:
        draw.rectangle((x, legend_y + 4, x + 18, legend_y + 22), fill=rgb(COLORS[label]), outline=(35, 42, 52))
        draw.text((x + 27, legend_y), LABEL_NAMES.get(label, label), fill=(45, 52, 62), font=legend_font)
        x += 150
    note = "Colors encode emotion classes; marker shapes encode modalities. Lower modality silhouette indicates less modality-domain separation."
    draw.text((70, 848), note, fill=(92, 100, 113), font=font(16))
    image.save(path, quality=95)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--test-cache", required=True)
    parser.add_argument("--class-anchor-cache", default=None)
    parser.add_argument("--pretrain-dir", required=True)
    parser.add_argument("--v1-checkpoint", required=True)
    parser.add_argument("--v5-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--class-count", type=int, default=3)
    parser.add_argument("--max-per-class", type=int, default=40)
    parser.add_argument("--perplexity", type=int, default=25)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 21, 42, 84, 168])
    parser.add_argument("--primary-seed", type=int, default=42)
    parser.add_argument("--sample-seed", type=int, default=2025)
    parser.add_argument("--gaussian-tau", type=float, default=0.5)
    parser.add_argument("--confidence-temp", type=float, default=0.07)
    parser.add_argument("--confidence-power", type=float, default=0.5)
    parser.add_argument("--confidence-floor", type=float, default=0.10)
    parser.add_argument("--confidence-random-baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.primary_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for IEMOCAP comparison visualization.")
    train_cache = torch.load(args.train_cache, map_location="cpu")
    test_cache = torch.load(args.test_cache, map_location="cpu")
    class_anchors = torch.load(args.class_anchor_cache, map_location="cpu") if args.class_anchor_cache else None
    classes, train_counts = choose_classes(train_cache, args.class_count)
    selected, selected_counts = select_test_indices(
        test_cache, classes, args.max_per_class, args.sample_seed
    )
    projected = {
        "v1": project_model(args, args.v1_checkpoint, test_cache, classes, class_anchors, device),
        "v5": project_model(args, args.v5_checkpoint, test_cache, classes, class_anchors, device),
    }
    torch.save(projected, os.path.join(args.output_dir, "projected_test_features.pt"))

    labels = test_cache["labels"]
    metrics = {
        "classes": classes,
        "train_class_counts": dict(train_counts),
        "selected_test_counts": selected_counts,
        "selected_ids": [test_cache["ids"][idx] for idx in selected],
        "controlled_repeated_class_anchor_stress_test": class_anchors is not None,
        "high_dimensional": {
            version: representation_metrics(value, labels) for version, value in projected.items()
        },
        "v5_prior_on_selected_test_samples": prior_metrics(projected["v5"], selected, labels, args),
        "tsne_stability": {},
        "checkpoints": {
            version: value["checkpoint_meta"] for version, value in projected.items()
        },
    }
    plot_payload = {}
    vectors_by_version = {}
    metadata_by_version = {}
    for version, value in projected.items():
        vectors_by_version[version], metadata_by_version[version] = build_plot_vectors(
            value, selected, labels, classes
        )

    all_seeds = list(dict.fromkeys(args.seeds + [args.primary_seed]))
    for seed in all_seeds:
        panels = {}
        metrics["tsne_stability"][str(seed)] = {}
        for version in ("v1", "v5"):
            coords, actual_perplexity = embed_tsne(vectors_by_version[version], seed, args.perplexity)
            current_metrics = tsne_metrics(coords, metadata_by_version[version])
            current_metrics["perplexity"] = actual_perplexity
            metrics["tsne_stability"][str(seed)][version] = current_metrics
            panels[version] = (coords, metadata_by_version[version], current_metrics)
            if seed == args.primary_seed:
                plot_payload[version] = panels[version]
        seed_path = os.path.join(args.output_dir, f"iemocap_v1_v5_tsne_seed_{seed}.png")
        draw_comparison(
            seed_path,
            panels,
            classes,
            seed,
            f"Top three training classes; {args.max_per_class} fixed test samples per class; PCA -> cosine t-SNE",
            controlled=class_anchors is not None,
        )

    primary_path = os.path.join(args.output_dir, "iemocap_v1_vs_v5_fig4_comparison.png")
    draw_comparison(
        primary_path,
        plot_payload,
        classes,
        args.primary_seed,
        f"Top three training classes; {args.max_per_class} fixed test samples per class; PCA -> cosine t-SNE",
        controlled=class_anchors is not None,
    )
    save_json(os.path.join(args.output_dir, "metrics.json"), metrics)
    with open(os.path.join(args.output_dir, "tsne_stability.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "version", "class_silhouette", "modality_silhouette", "perplexity"])
        for seed, values in metrics["tsne_stability"].items():
            for version, current in values.items():
                writer.writerow(
                    [seed, version, current["class_silhouette"], current["modality_silhouette"], current["perplexity"]]
                )
    print("Saved primary comparison.", flush=True)
    print(json.dumps(metrics["high_dimensional"], indent=2), flush=True)
    print(json.dumps(metrics["v5_prior_on_selected_test_samples"], indent=2), flush=True)


if __name__ == "__main__":
    main()
