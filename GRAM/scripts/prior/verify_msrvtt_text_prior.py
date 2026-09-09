#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
from pathlib import Path
from statistics import NormalDist

import numpy as np


def log(message):
    print(message, flush=True)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)
        f.write("\n")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_msrvtt_1ka_captions(ret_annotation, caption_annotation, split_size, captions_per_video):
    ret_data = load_json(ret_annotation)
    cap_data = load_json(caption_annotation)
    cap_map = {item["video_id"]: item["desc"] for item in cap_data}

    video_ids = []
    seen = set()
    for item in ret_data:
        video_id = item["video_id"]
        if video_id in seen:
            continue
        seen.add(video_id)
        video_ids.append(video_id)
        if len(video_ids) >= split_size:
            break

    rows = []
    missing = []
    bad_count = []
    for video_id in video_ids:
        captions = cap_map.get(video_id)
        if captions is None:
            missing.append(video_id)
            continue
        if len(captions) < captions_per_video:
            bad_count.append((video_id, len(captions)))
            continue
        rows.append({"video_id": video_id, "captions": captions[:captions_per_video]})

    if missing:
        raise RuntimeError(f"Missing caption annotations for {len(missing)} videos, first={missing[:5]}")
    if bad_count:
        raise RuntimeError(f"Not enough captions for {len(bad_count)} videos, first={bad_count[:5]}")
    if len(rows) != split_size:
        raise RuntimeError(f"Expected {split_size} videos but got {len(rows)}")
    return rows


def encode_texts_with_bert(captions_by_video, encoder_cfg):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    model_path = encoder_cfg["model_path"]
    local_files_only = bool(encoder_cfg.get("local_files_only", True))
    pooling = encoder_cfg.get("pooling", "mean")
    max_length = int(encoder_cfg.get("max_length", 64))
    batch_size = int(encoder_cfg.get("batch_size", 128))
    normalize = bool(encoder_cfg.get("normalize_features", True))

    requested_device = encoder_cfg.get("device", "cuda")
    device = torch.device(requested_device if requested_device == "cuda" and torch.cuda.is_available() else "cpu")

    log(f"Loading frozen BERT tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only)
    log(f"Loading frozen BERT model from {model_path}")
    model = AutoModel.from_pretrained(model_path, local_files_only=local_files_only)
    model.eval()
    if bool(encoder_cfg.get("freeze", True)):
        for param in model.parameters():
            param.requires_grad_(False)
    model.to(device)
    log(f"BERT loaded on {device}; pooling={pooling}; normalize={normalize}")

    flat_texts = []
    for item in captions_by_video:
        flat_texts.extend(item["captions"])

    features = []
    with torch.no_grad():
        for start in range(0, len(flat_texts), batch_size):
            batch_texts = flat_texts[start:start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded)
            if pooling == "cls":
                batch_feat = output.last_hidden_state[:, 0]
            elif pooling == "pooler":
                batch_feat = output.pooler_output
            elif pooling == "mean":
                mask = encoded["attention_mask"].unsqueeze(-1).to(output.last_hidden_state.dtype)
                batch_feat = (output.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            else:
                raise ValueError(f"Unsupported BERT pooling: {pooling}")
            if normalize:
                batch_feat = F.normalize(batch_feat, dim=-1)
            features.append(batch_feat.detach().cpu().float().numpy())
            log(f"Encoded {min(start + batch_size, len(flat_texts))}/{len(flat_texts)} captions")

    features = np.concatenate(features, axis=0)
    n_video = len(captions_by_video)
    n_caption = len(captions_by_video[0]["captions"])
    return features.reshape(n_video, n_caption, features.shape[-1])


def cosine_error(sample_mean, true_mean, eps=1e-12):
    numerator = np.sum(sample_mean * true_mean, axis=-1)
    denominator = np.linalg.norm(sample_mean, axis=-1) * np.linalg.norm(true_mean, axis=-1)
    return 1.0 - numerator / np.maximum(denominator, eps)


def run_lln(features, lln_cfg, seed):
    rng = np.random.default_rng(seed)
    n_video, n_caption, _ = features.shape
    mu = features.mean(axis=1)
    rows = []
    per_k_errors = {}

    k_min = int(lln_cfg.get("k_min", 1))
    k_max = int(lln_cfg.get("k_max", n_caption - 1))
    repeats = int(lln_cfg.get("bootstrap_repeats", 100))

    for k in range(k_min, k_max + 1):
        errors = np.empty((n_video, repeats), dtype=np.float64)
        for video_idx in range(n_video):
            for repeat_idx in range(repeats):
                subset = rng.choice(n_caption, size=k, replace=False)
                sample_mean = features[video_idx, subset].mean(axis=0)
                errors[video_idx, repeat_idx] = cosine_error(sample_mean[None, :], mu[video_idx][None, :])[0]

        video_mean = errors.mean(axis=1)
        video_std = errors.std(axis=1, ddof=0)
        rows.append({
            "k": k,
            "mean_error": float(video_mean.mean()),
            "mean_bootstrap_std": float(video_std.mean()),
            "global_std": float(errors.reshape(-1).std(ddof=0)),
            "ci95_video_mean": float(1.96 * video_mean.std(ddof=1) / math.sqrt(n_video)),
        })
        per_k_errors[k] = errors
        log(f"LLN k={k}: mean_error={rows[-1]['mean_error']:.6f}, mean_bootstrap_std={rows[-1]['mean_bootstrap_std']:.6f}")

    return rows, per_k_errors


def run_residual_pca(features):
    mu = features.mean(axis=1, keepdims=True)
    residuals = (features - mu).reshape(-1, features.shape[-1]).astype(np.float64)
    residuals = residuals - residuals.mean(axis=0, keepdims=True)

    log(f"Running PCA with SVD on residual matrix {residuals.shape}")
    _, singular_values, vt = np.linalg.svd(residuals, full_matrices=False)
    components = vt[:2]
    coords = residuals @ components.T
    eigenvalues = (singular_values ** 2) / max(residuals.shape[0] - 1, 1)
    explained_ratio = eigenvalues[:2] / eigenvalues.sum()
    return residuals, coords, explained_ratio


def qq_statistics(pc1, standardize=True):
    values = np.asarray(pc1, dtype=np.float64)
    if standardize:
        values = (values - values.mean()) / values.std(ddof=0)
    ordered = np.sort(values)
    n = ordered.size
    normal = NormalDist()
    probs = (np.arange(1, n + 1) - 0.5) / n
    theoretical = np.array([normal.inv_cdf(float(p)) for p in probs], dtype=np.float64)
    corr = np.corrcoef(theoretical, ordered)[0, 1]
    slope, intercept = np.polyfit(theoretical, ordered, deg=1)
    return theoretical, ordered, {
        "qq_corr": float(corr),
        "qq_r2": float(corr ** 2),
        "qq_fit_slope": float(slope),
        "qq_fit_intercept": float(intercept),
        "pc1_mean": float(values.mean()),
        "pc1_std": float(values.std(ddof=0)),
        "pc1_skew": float(((values - values.mean()) ** 3).mean() / max(values.std(ddof=0) ** 3, 1e-12)),
        "pc1_excess_kurtosis": float(((values - values.mean()) ** 4).mean() / max(values.std(ddof=0) ** 4, 1e-12) - 3.0),
    }


def save_lln_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_lln_plot(path_base, rows, figure_cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.array([row["k"] for row in rows])
    y = np.array([row["mean_error"] for row in rows])
    std = np.array([row["mean_bootstrap_std"] for row in rows])

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(x, y, color="#1f4e8c", linewidth=2.5, marker="o", markersize=4, label="Mean cosine error")
    plt.fill_between(x, np.maximum(y - std, 0.0), y + std, color="#7db7e8", alpha=0.28, label="+/- 1 std")
    plt.xlabel("Number of observed captions k")
    plt.ylabel("Cosine distance to semantic mean")
    plt.xticks(x)
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    save_figure(plt, path_base, figure_cfg)


def save_qq_plot(path_base, theoretical, ordered, stats, figure_cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo = min(theoretical.min(), ordered.min())
    hi = max(theoretical.max(), ordered.max())
    plt.figure(figsize=(5.4, 5.4))
    plt.scatter(theoretical, ordered, s=4, alpha=0.35, color="#244c8f", edgecolors="none")
    plt.plot([lo, hi], [lo, hi], color="#c7352f", linewidth=2.0, label="y = x")
    plt.xlabel("Theoretical standard normal quantiles")
    plt.ylabel("Empirical PC1 residual quantiles")
    plt.title(f"Q-Q plot (R^2={stats['qq_r2']:.4f})")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    save_figure(plt, path_base, figure_cfg)


def save_kde_plot(path_base, coords, explained_ratio, figure_cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6.0, 5.4))
    try:
        import seaborn as sns
        sns.kdeplot(x=coords[:, 0], y=coords[:, 1], levels=12, fill=True, cmap="Blues", thresh=0.02)
        sns.kdeplot(x=coords[:, 0], y=coords[:, 1], levels=10, color="#1f4e8c", linewidths=0.8)
    except Exception as exc:
        log(f"seaborn.kdeplot failed ({exc}); falling back to histogram contour")
        hist, x_edges, y_edges = np.histogram2d(coords[:, 0], coords[:, 1], bins=100, density=True)
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        plt.contourf(x_centers, y_centers, hist.T, levels=12, cmap="Blues")
        plt.contour(x_centers, y_centers, hist.T, levels=10, colors="#1f4e8c", linewidths=0.7)
    plt.scatter(coords[:, 0], coords[:, 1], s=1, alpha=0.04, color="black")
    plt.xlabel(f"PC1 ({explained_ratio[0] * 100:.2f}% var.)")
    plt.ylabel(f"PC2 ({explained_ratio[1] * 100:.2f}% var.)")
    plt.title("2D KDE of centered caption residuals")
    plt.grid(False)
    plt.tight_layout()
    save_figure(plt, path_base, figure_cfg)


def save_figure(plt, path_base, figure_cfg):
    dpi = int(figure_cfg.get("dpi", 300))
    formats = figure_cfg.get("formats", ["png"])
    for fmt in formats:
        path = Path(f"{path_base}.{fmt}")
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=dpi)
        log(f"Saved figure: {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit-videos", type=int, default=None)
    parser.add_argument("--stage", choices=["all", "extract", "analyze"], default="all")
    args = parser.parse_args()

    cfg = load_json(args.config)
    if args.limit_videos is not None:
        cfg["split_size"] = args.limit_videos

    set_seed(int(cfg.get("seed", 0)))
    output_dir = Path(cfg["output_dir"])
    if args.limit_videos is not None:
        output_dir = output_dir.parent / f"{output_dir.name}_limit{args.limit_videos}"
    features_dir = output_dir / "features"
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
    for directory in [features_dir, metrics_dir, figures_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    save_json(output_dir / "run_config.json", cfg)
    log(f"Output dir: {output_dir}")

    feature_path = features_dir / "bert_text_features.npz"

    features = None
    if args.stage in ("all", "extract"):
        captions_by_video = load_msrvtt_1ka_captions(
            cfg["ret_annotation"],
            cfg["caption_annotation"],
            int(cfg["split_size"]),
            int(cfg["captions_per_video"]),
        )
        save_json(output_dir / "samples_used.json", captions_by_video)
        log(f"Loaded {len(captions_by_video)} videos and {len(captions_by_video) * len(captions_by_video[0]['captions'])} captions")

        features = encode_texts_with_bert(captions_by_video, cfg["encoder"])
        np.savez_compressed(
            feature_path,
            features=features.astype(np.float32),
            video_ids=np.array([item["video_id"] for item in captions_by_video]),
        )
        log(f"Saved BERT text features with shape {features.shape}")

    if args.stage == "extract":
        log("Feature extraction stage completed.")
        return

    if features is None:
        if not feature_path.exists():
            raise RuntimeError(f"Missing feature cache for analyze stage: {feature_path}")
        packed = np.load(feature_path, allow_pickle=True)
        features = packed["features"].astype(np.float32)
        log(f"Loaded cached BERT text features with shape {features.shape}")

    lln_rows, _ = run_lln(features, cfg["lln"], int(cfg.get("seed", 0)) + 17)
    save_lln_csv(metrics_dir / "lln_metrics.csv", lln_rows)
    save_json(metrics_dir / "lln_metrics.json", lln_rows)
    save_lln_plot(figures_dir / "figure5a_lln_cosine_error", lln_rows, cfg.get("figures", {}))

    _, coords, explained_ratio = run_residual_pca(features)
    np.savez_compressed(
        features_dir / "residual_pca_coords.npz",
        coords=coords.astype(np.float32),
        explained_ratio=explained_ratio.astype(np.float64),
    )
    theoretical, ordered, qq_stats = qq_statistics(
        coords[:, 0],
        standardize=bool(cfg.get("gaussian", {}).get("standardize_pc1", True)),
    )
    gaussian_metrics = {
        **qq_stats,
        "pc1_explained_ratio": float(explained_ratio[0]),
        "pc2_explained_ratio": float(explained_ratio[1]),
        "num_residuals": int(coords.shape[0]),
    }
    save_json(metrics_dir / "gaussian_metrics.json", gaussian_metrics)
    save_qq_plot(figures_dir / "figure5b_pc1_qq", theoretical, ordered, qq_stats, cfg.get("figures", {}))
    save_kde_plot(figures_dir / "figure5c_residual_pca_kde", coords, explained_ratio, cfg.get("figures", {}))

    log("Prior theory verification completed.")


if __name__ == "__main__":
    main()
