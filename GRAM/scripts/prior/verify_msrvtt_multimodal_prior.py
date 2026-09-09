#!/usr/bin/env python3
"""Empirical probabilistic-cloud validation on MSR-VTT text, video, and audio.

The experiment deliberately separates two scopes:
1. Text/video/audio rows test repeated observations inside each modality.
2. The joint T-V-A row tests heterogeneous observations after the frozen GRAM
   pretraining model maps them into its shared contrastive space.

No SmoothGRAM fine-tuning checkpoint is used.  The script reports quantitative
diagnostics and never changes thresholds after seeing the results.
"""

import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DISPLAY_NAMES = {
    "text": "Text captions (frozen BERT)",
    "video": "Video frames (frozen EVA-CLIP)",
    "audio": "Audio windows (frozen BEATs)",
    "joint_tva": "Joint T-V-A observations (frozen GRAM space)",
}


def log(message):
    print(message, flush=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def recursive_edict(obj):
    from easydict import EasyDict

    if isinstance(obj, dict):
        return EasyDict({key: recursive_edict(value) for key, value in obj.items()})
    if isinstance(obj, list):
        return [recursive_edict(value) for value in obj]
    return obj


def load_msrvtt_rows(ret_annotation, caption_annotation, split_size, captions_per_video):
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
    for video_id in video_ids:
        captions = cap_map.get(video_id)
        if captions is None or len(captions) < captions_per_video:
            raise RuntimeError(f"Missing captions for {video_id}: got {0 if captions is None else len(captions)}")
        rows.append({"video_id": video_id, "captions": captions[:captions_per_video]})
    if len(rows) != split_size:
        raise RuntimeError(f"Expected {split_size} videos but loaded {len(rows)}")
    return rows


def resolve_media_path(directory, video_id, extensions):
    directory = Path(directory)
    for extension in extensions:
        candidate = directory / f"{video_id}{extension}"
        if candidate.exists():
            return candidate
    matches = sorted(directory.glob(f"{video_id}.*"))
    return matches[0] if matches else None


def load_frozen_gram(pretrain_dir, device):
    import torch
    from model import model_registry

    pretrain_dir = Path(pretrain_dir)
    hps = load_json(pretrain_dir / "log" / "hps.json")
    cfg = recursive_edict(hps["model_cfg"])
    cfg.model_type = "gram"
    cfg.checkpointing = False
    cfg.fp16 = False
    cfg.captioner_mode = False

    cwd = Path.cwd()
    os.chdir(ROOT)
    try:
        model = model_registry[cfg.model_type](cfg)
        checkpoint_path = pretrain_dir / "ckpt" / "model_step_249.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            checkpoint = checkpoint["model"]
        checkpoint = {key.replace("module.", ""): value for key, value in checkpoint.items()}
        checkpoint = model.modify_checkpoint(checkpoint)
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    finally:
        os.chdir(cwd)

    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    log(f"Loaded frozen GRAM pretraining checkpoint: {checkpoint_path}")
    log(f"Checkpoint load: missing={len(missing)}, unexpected={len(unexpected)}")
    return model


def normalize_numpy(features, eps=1e-12):
    denominator = np.linalg.norm(features, axis=-1, keepdims=True)
    return features / np.maximum(denominator, eps)


def reuse_text_features(rows, source_path, observations):
    packed = np.load(source_path, allow_pickle=True)
    features = packed["features"].astype(np.float32)
    source_ids = [str(value) for value in packed["video_ids"].tolist()]
    index = {video_id: row for row, video_id in enumerate(source_ids)}
    requested = [item["video_id"] for item in rows]
    missing = [video_id for video_id in requested if video_id not in index]
    if missing:
        raise RuntimeError(f"Text feature cache misses {len(missing)} requested videos, first={missing[:5]}")
    selected = features[[index[video_id] for video_id in requested], :observations]
    return normalize_numpy(selected.astype(np.float32)), np.asarray(requested)


def encode_projected_text(model, rows, observations, encoder_cfg, device):
    import torch
    import torch.nn.functional as F

    tokenizer = model.multimodal_encoder.tokenizer
    texts = []
    for item in rows:
        texts.extend(item["captions"][:observations])
    batch_size = int(encoder_cfg.get("text_batch_size", 128))
    max_length = int(encoder_cfg.get("text_max_length", 64))
    use_fp16 = bool(encoder_cfg.get("fp16", True)) and device.type == "cuda"
    raw_parts, projected_parts = [], []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            current = texts[start:start + batch_size]
            tokens = tokenizer(
                current,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            with torch.cuda.amp.autocast(enabled=use_fp16):
                output = model.multimodal_encoder.bert(
                    input_ids=tokens.input_ids,
                    attention_mask=tokens.attention_mask,
                ).last_hidden_state
                raw = model.pool_text_for_contra(output)
                projected = F.normalize(model.contra_head_t(raw), dim=-1)
            raw_parts.append(F.normalize(raw.float(), dim=-1).cpu().numpy())
            projected_parts.append(projected.float().cpu().numpy())
            log(f"Encoded shared-space text {min(start + batch_size, len(texts))}/{len(texts)}")

    n_video = len(rows)
    raw = np.concatenate(raw_parts).reshape(n_video, observations, -1).astype(np.float32)
    projected = np.concatenate(projected_parts).reshape(n_video, observations, -1).astype(np.float32)
    return raw, projected


def uniform_video_frames(path, count):
    import cv2

    capture = cv2.VideoCapture(str(path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        capture.release()
        raise RuntimeError(f"Cannot determine frame count: {path}")
    frame_indices = np.rint(np.linspace(0, total - 1, count)).astype(np.int64)
    frames = []
    last_valid = None
    for frame_index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if ok and frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            last_valid = frame
        if last_valid is None:
            capture.release()
            raise RuntimeError(f"Cannot decode initial frame from {path}")
        frames.append(last_valid.copy())
    capture.release()
    return frames


def preprocess_video_frames(frames, resolution):
    import torch
    from torchvision.transforms import CenterCrop, Compose, Normalize, Resize

    tensors = [torch.from_numpy(frame).permute(2, 0, 1).float().div_(255.0) for frame in frames]
    transform = Compose([
        Resize(resolution, antialias=True),
        CenterCrop(resolution),
        Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    ])
    return torch.stack([transform(tensor) for tensor in tensors])


def encode_video_observations(model, rows, cfg, device):
    import torch
    import torch.nn.functional as F

    count = int(cfg["observations"]["video"])
    encoder_cfg = cfg["encoder"]
    resolution = int(encoder_cfg.get("vision_resolution", 224))
    batch_size = int(encoder_cfg.get("vision_batch_size", 8))
    use_fp16 = bool(encoder_cfg.get("fp16", True)) and device.type == "cuda"
    raw_all, projected_all, valid_ids, failures = [], [], [], []

    for row_index, item in enumerate(rows):
        video_id = item["video_id"]
        path = resolve_media_path(cfg["video_dir"], video_id, [".mp4", ".avi", ".webm", ".mkv"])
        if path is None:
            failures.append({"video_id": video_id, "reason": "missing video"})
            continue
        try:
            frames = uniform_video_frames(path, count)
            pixels = preprocess_video_frames(frames, resolution)
            raw_parts, projected_parts = [], []
            with torch.no_grad():
                for start in range(0, count, batch_size):
                    current = pixels[start:start + batch_size].to(device, non_blocking=True).unsqueeze(1)
                    with torch.cuda.amp.autocast(enabled=use_fp16):
                        output = model.forward_vision_encoder(current)
                        raw = model.pool_vision_for_contra(output)
                        projected = F.normalize(model.contra_head_v(raw), dim=-1)
                    raw_parts.append(F.normalize(raw.float(), dim=-1).cpu().numpy())
                    projected_parts.append(projected.float().cpu().numpy())
            raw_all.append(np.concatenate(raw_parts).astype(np.float32))
            projected_all.append(np.concatenate(projected_parts).astype(np.float32))
            valid_ids.append(video_id)
        except Exception as exc:
            failures.append({"video_id": video_id, "reason": repr(exc)})
        if (row_index + 1) % 20 == 0 or row_index + 1 == len(rows):
            log(f"Video observations: processed={row_index + 1}/{len(rows)}, valid={len(valid_ids)}, failed={len(failures)}")

    if not valid_ids:
        raise RuntimeError("No video observations were extracted")
    return np.stack(raw_all), np.stack(projected_all), np.asarray(valid_ids), failures


def uniform_audio_segments(path, count, sample_rate, segment_seconds):
    import librosa

    waveform, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    waveform = np.asarray(waveform, dtype=np.float32)
    window = max(int(round(sample_rate * segment_seconds)), 1)
    if waveform.size < window:
        waveform = np.pad(waveform, (0, window - waveform.size))
    max_start = max(waveform.size - window, 0)
    starts = np.rint(np.linspace(0, max_start, count)).astype(np.int64)
    return np.stack([waveform[start:start + window] for start in starts])


def waveforms_to_fbank(segments, encoder_cfg):
    import torch
    import torchaudio

    sample_rate = int(encoder_cfg.get("audio_sample_rate", 16000))
    mel_bins = int(encoder_cfg.get("audio_mel_bins", 64))
    mean = float(encoder_cfg.get("audio_fbank_mean", 15.41663))
    std = float(encoder_cfg.get("audio_fbank_std", 6.55582))
    banks = []
    for segment in segments:
        waveform = torch.from_numpy(segment).float().unsqueeze(0) * (2 ** 15)
        bank = torchaudio.compliance.kaldi.fbank(
            waveform,
            num_mel_bins=mel_bins,
            sample_frequency=sample_rate,
            frame_length=25,
            frame_shift=10,
            dither=0.0,
        )
        banks.append((bank - mean) / (std * 2.0))
    target = max(bank.shape[0] for bank in banks)
    padded = []
    for bank in banks:
        if bank.shape[0] < target:
            bank = torch.nn.functional.pad(bank, (0, 0, 0, target - bank.shape[0]))
        padded.append(bank[:target])
    return torch.stack(padded)


def encode_audio_observations(model, rows, cfg, device):
    import torch
    import torch.nn.functional as F

    count = int(cfg["observations"]["audio"])
    encoder_cfg = cfg["encoder"]
    batch_size = int(encoder_cfg.get("audio_batch_size", 16))
    sample_rate = int(encoder_cfg.get("audio_sample_rate", 16000))
    segment_seconds = float(encoder_cfg.get("audio_segment_seconds", 2.0))
    use_fp16 = bool(encoder_cfg.get("fp16", True)) and device.type == "cuda"
    raw_all, projected_all, valid_ids, failures = [], [], [], []

    for row_index, item in enumerate(rows):
        video_id = item["video_id"]
        path = resolve_media_path(cfg["audio_dir"], video_id, [".mp3", ".wav", ".m4a", ".mkv"])
        if path is None:
            failures.append({"video_id": video_id, "reason": "missing audio"})
            continue
        try:
            segments = uniform_audio_segments(path, count, sample_rate, segment_seconds)
            banks = waveforms_to_fbank(segments, encoder_cfg)
            raw_parts, projected_parts = [], []
            with torch.no_grad():
                for start in range(0, count, batch_size):
                    current = banks[start:start + batch_size].to(device, non_blocking=True).unsqueeze(1)
                    with torch.cuda.amp.autocast(enabled=use_fp16):
                        output = model.forward_audio_encoder(current)
                        raw = model.pool_audio_for_contra(output)
                        projected = F.normalize(model.contra_head_a(raw), dim=-1)
                    raw_parts.append(F.normalize(raw.float(), dim=-1).cpu().numpy())
                    projected_parts.append(projected.float().cpu().numpy())
            raw_all.append(np.concatenate(raw_parts).astype(np.float32))
            projected_all.append(np.concatenate(projected_parts).astype(np.float32))
            valid_ids.append(video_id)
        except Exception as exc:
            failures.append({"video_id": video_id, "reason": repr(exc)})
        if (row_index + 1) % 20 == 0 or row_index + 1 == len(rows):
            log(f"Audio observations: processed={row_index + 1}/{len(rows)}, valid={len(valid_ids)}, failed={len(failures)}")

    if not valid_ids:
        raise RuntimeError("No audio observations were extracted")
    return np.stack(raw_all), np.stack(projected_all), np.asarray(valid_ids), failures


def cosine_error(sample_mean, reference_mean, eps=1e-12):
    numerator = np.sum(sample_mean * reference_mean, axis=-1)
    denominator = np.linalg.norm(sample_mean, axis=-1) * np.linalg.norm(reference_mean, axis=-1)
    return 1.0 - numerator / np.maximum(denominator, eps)


def run_lln(features, lln_cfg, seed):
    from scipy.stats import spearmanr

    features = np.asarray(features, dtype=np.float32)
    rng = np.random.default_rng(seed)
    n_event, n_observation, _ = features.shape
    reference = features.mean(axis=1)
    k_min = int(lln_cfg.get("k_min", 1))
    k_max = min(int(lln_cfg.get("k_max", n_observation - 1)), n_observation - 1)
    repeats = int(lln_cfg.get("bootstrap_repeats", 100))
    event_batch = max(int(lln_cfg.get("event_batch_size", 4)), 1)
    rows = []

    for k in range(k_min, k_max + 1):
        event_means = np.empty(n_event, dtype=np.float64)
        event_stds = np.empty(n_event, dtype=np.float64)
        all_errors = []
        for start in range(0, n_event, event_batch):
            current = features[start:start + event_batch]
            bsz = current.shape[0]
            order = np.argsort(rng.random((bsz, repeats, n_observation)), axis=2)[..., :k]
            expanded = np.broadcast_to(current[:, None, :, :], (bsz, repeats, n_observation, current.shape[-1]))
            selected = np.take_along_axis(expanded, order[..., None], axis=2)
            sample_mean = selected.mean(axis=2)
            errors = cosine_error(sample_mean, reference[start:start + bsz, None, :])
            event_means[start:start + bsz] = errors.mean(axis=1)
            event_stds[start:start + bsz] = errors.std(axis=1, ddof=0)
            all_errors.append(errors.reshape(-1))
        flattened = np.concatenate(all_errors)
        rows.append({
            "k": k,
            "mean_error": float(event_means.mean()),
            "mean_bootstrap_std": float(event_stds.mean()),
            "global_std": float(flattened.std(ddof=0)),
            "ci95_event_mean": float(1.96 * event_means.std(ddof=1) / math.sqrt(n_event)),
        })
        log(f"LLN k={k}/{k_max}: mean_error={rows[-1]['mean_error']:.6f}")

    x = np.asarray([row["k"] for row in rows], dtype=np.float64)
    y = np.asarray([row["mean_error"] for row in rows], dtype=np.float64)
    rho = float(spearmanr(x, y).statistic)
    reduction = float((y[0] - y[-1]) / max(abs(y[0]), 1e-12))
    monotone_fraction = float(np.mean(np.diff(y) <= 0.0)) if len(y) > 1 else 1.0
    summary = {
        "spearman_rho_k_vs_error": rho,
        "relative_error_reduction": reduction,
        "monotone_step_fraction": monotone_fraction,
        "initial_mean_error": float(y[0]),
        "final_mean_error": float(y[-1]),
        "num_events": int(n_event),
        "num_observations_per_event": int(n_observation),
    }
    return rows, summary


def residual_pca(features, gaussian_cfg, seed):
    from sklearn.decomposition import PCA

    center = features.mean(axis=1, keepdims=True)
    residuals = (features - center).reshape(-1, features.shape[-1]).astype(np.float32)
    residuals -= residuals.mean(axis=0, keepdims=True)
    solver = gaussian_cfg.get("pca_solver", "randomized")
    iterations = int(gaussian_cfg.get("pca_iterations", 7))
    pca = PCA(n_components=2, svd_solver=solver, iterated_power=iterations, random_state=seed)
    coords = pca.fit_transform(residuals)
    return residuals, coords.astype(np.float64), pca.explained_variance_ratio_.astype(np.float64)


def qq_statistics(values, standardize=True):
    values = np.asarray(values, dtype=np.float64)
    if standardize:
        values = (values - values.mean()) / max(values.std(ddof=0), 1e-12)
    ordered = np.sort(values)
    n = ordered.size
    probs = (np.arange(1, n + 1) - 0.5) / n
    normal = NormalDist()
    theoretical = np.array([normal.inv_cdf(float(probability)) for probability in probs])
    corr = np.corrcoef(theoretical, ordered)[0, 1]
    slope, intercept = np.polyfit(theoretical, ordered, deg=1)
    centered = values - values.mean()
    std = max(values.std(ddof=0), 1e-12)
    stats = {
        "qq_corr": float(corr),
        "qq_r2": float(corr ** 2),
        "qq_fit_slope": float(slope),
        "qq_fit_intercept": float(intercept),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "skew": float((centered ** 3).mean() / std ** 3),
        "excess_kurtosis": float((centered ** 4).mean() / std ** 4 - 3.0),
    }
    return theoretical, ordered, stats


def random_projection_diagnostics(residuals, count, seed):
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(residuals.shape[1], count)).astype(np.float32)
    directions /= np.maximum(np.linalg.norm(directions, axis=0, keepdims=True), 1e-12)
    projected = residuals @ directions
    r2_values = []
    for column in range(projected.shape[1]):
        _, _, stats = qq_statistics(projected[:, column], standardize=True)
        r2_values.append(stats["qq_r2"])
    return {
        "count": int(count),
        "qq_r2_mean": float(np.mean(r2_values)),
        "qq_r2_std": float(np.std(r2_values)),
        "qq_r2_min": float(np.min(r2_values)),
        "qq_r2_max": float(np.max(r2_values)),
    }


def evaluate_support(lln_summary, gaussian_metrics, criteria):
    checks = {
        "lln_rank_trend": lln_summary["spearman_rho_k_vs_error"] <= float(criteria["lln_spearman_rho_max"]),
        "lln_error_reduction": lln_summary["relative_error_reduction"] >= float(criteria["lln_relative_error_reduction_min"]),
        "pc1_qq_linearity": gaussian_metrics["pc1"]["qq_r2"] >= float(criteria["pc1_qq_r2_min"]),
        "pc1_skew": abs(gaussian_metrics["pc1"]["skew"]) <= float(criteria["pc1_abs_skew_max"]),
        "pc1_excess_kurtosis": abs(gaussian_metrics["pc1"]["excess_kurtosis"]) <= float(criteria["pc1_abs_excess_kurtosis_max"]),
    }
    return {"checks": checks, "all_prespecified_checks_pass": bool(all(checks.values()))}


def save_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, path_base, figure_cfg):
    formats = figure_cfg.get("formats", ["png"])
    dpi = int(figure_cfg.get("dpi", 300))
    for extension in formats:
        path = Path(f"{path_base}.{extension}")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        log(f"Saved figure: {path}")


def thin_sorted_xy(x, y, maximum):
    if len(x) <= maximum:
        return x, y
    index = np.linspace(0, len(x) - 1, maximum).round().astype(np.int64)
    return x[index], y[index]


def plot_lln_axis(axis, rows, title=None):
    x = np.asarray([row["k"] for row in rows])
    y = np.asarray([row["mean_error"] for row in rows])
    std = np.asarray([row["mean_bootstrap_std"] for row in rows])
    axis.plot(x, y, color="#1f4e8c", linewidth=2.3, marker="o", markersize=3.4)
    axis.fill_between(x, np.maximum(y - std, 0.0), y + std, color="#7db7e8", alpha=0.28)
    axis.set_xlabel("Number of observations $k$")
    axis.set_ylabel("Cosine distance to event mean")
    axis.grid(True, alpha=0.23)
    if title:
        axis.set_title(title)


def plot_qq_axis(axis, theoretical, ordered, stats, figure_cfg, title=None):
    x, y = thin_sorted_xy(theoretical, ordered, int(figure_cfg.get("max_qq_points", 6000)))
    low = min(float(theoretical.min()), float(ordered.min()))
    high = max(float(theoretical.max()), float(ordered.max()))
    axis.scatter(x, y, s=4, alpha=0.34, color="#244c8f", edgecolors="none")
    axis.plot([low, high], [low, high], color="#c7352f", linewidth=1.8)
    axis.set_xlabel("Theoretical normal quantiles")
    axis.set_ylabel("Empirical PC1 quantiles")
    if title:
        axis.set_title(f"{title}\n$R^2$={stats['qq_r2']:.4f}")
    axis.grid(True, alpha=0.23)


def plot_kde_axis(axis, coords, explained_ratio, figure_cfg, seed, title=None):
    import seaborn as sns

    maximum = int(figure_cfg.get("max_kde_points", 20000))
    if len(coords) > maximum:
        rng = np.random.default_rng(seed)
        coords = coords[rng.choice(len(coords), size=maximum, replace=False)]
    try:
        sns.kdeplot(x=coords[:, 0], y=coords[:, 1], levels=12, fill=True, cmap="Blues", thresh=0.02, ax=axis)
        sns.kdeplot(x=coords[:, 0], y=coords[:, 1], levels=10, color="#1f4e8c", linewidths=0.7, ax=axis)
    except Exception as exc:
        log(f"KDE failed ({exc}); using histogram contours")
        hist, x_edges, y_edges = np.histogram2d(coords[:, 0], coords[:, 1], bins=90, density=True)
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        axis.contourf(x_centers, y_centers, hist.T, levels=12, cmap="Blues")
    axis.scatter(coords[:, 0], coords[:, 1], s=1, alpha=0.035, color="black")
    axis.set_xlabel(f"PC1 ({explained_ratio[0] * 100:.2f}% var.)")
    axis.set_ylabel(f"PC2 ({explained_ratio[1] * 100:.2f}% var.)")
    if title:
        axis.set_title(title)


def save_modality_figures(modality, result, output_dir, figure_cfg, seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    name = DISPLAY_NAMES[modality]
    figure_dir = Path(output_dir) / "figures" / modality

    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    plot_lln_axis(axis, result["lln_rows"], name)
    fig.tight_layout()
    save_figure(fig, figure_dir / "lln_mean_convergence", figure_cfg)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(5.4, 5.4))
    plot_qq_axis(axis, result["theoretical"], result["ordered"], result["gaussian_metrics"]["pc1"], figure_cfg, name)
    fig.tight_layout()
    save_figure(fig, figure_dir / "pc1_qq", figure_cfg)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.0, 5.4))
    plot_kde_axis(axis, result["coords"], result["explained_ratio"], figure_cfg, seed, name)
    fig.tight_layout()
    save_figure(fig, figure_dir / "residual_pc1_pc2_kde", figure_cfg)
    plt.close(fig)


def save_composite(results, output_dir, figure_cfg, seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [key for key in ("text", "video", "audio", "joint_tva") if key in results]
    fig, axes = plt.subplots(len(order), 3, figsize=(15.5, 4.1 * len(order)))
    if len(order) == 1:
        axes = np.asarray([axes])
    for row_index, modality in enumerate(order):
        result = results[modality]
        name = DISPLAY_NAMES[modality]
        plot_lln_axis(axes[row_index, 0], result["lln_rows"], name)
        plot_qq_axis(
            axes[row_index, 1], result["theoretical"], result["ordered"],
            result["gaussian_metrics"]["pc1"], figure_cfg, name,
        )
        plot_kde_axis(
            axes[row_index, 2], result["coords"], result["explained_ratio"],
            figure_cfg, seed + row_index, name,
        )
    axes[0, 0].text(-0.18, 1.18, "(a) Mean convergence", transform=axes[0, 0].transAxes, fontsize=14, fontweight="bold")
    axes[0, 1].text(-0.10, 1.18, "(b) PC1 Q-Q", transform=axes[0, 1].transAxes, fontsize=14, fontweight="bold")
    axes[0, 2].text(-0.10, 1.18, "(c) PC1-PC2 residual KDE", transform=axes[0, 2].transAxes, fontsize=14, fontweight="bold")
    fig.suptitle("Probabilistic semantic-cloud diagnostics across modalities", fontsize=18, fontweight="bold", y=1.005)
    fig.tight_layout()
    save_figure(fig, Path(output_dir) / "figures" / "appendix_multimodal_probabilistic_cloud", figure_cfg)
    plt.close(fig)


def align_feature_sets(feature_sets):
    common = set(feature_sets["text_projected"][1].tolist())
    common &= set(feature_sets["video_projected"][1].tolist())
    common &= set(feature_sets["audio_projected"][1].tolist())
    ordered = [video_id for video_id in feature_sets["text_projected"][1].tolist() if video_id in common]
    aligned = {}
    for key in ("text_projected", "video_projected", "audio_projected"):
        features, ids = feature_sets[key]
        lookup = {video_id: index for index, video_id in enumerate(ids.tolist())}
        aligned[key] = features[[lookup[video_id] for video_id in ordered]]
    return aligned, np.asarray(ordered)


def extract_features(cfg, rows, output_dir):
    import torch

    feature_dir = Path(output_dir) / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    text_count = int(cfg["observations"]["text"])
    joint_count = int(cfg["observations"]["joint_per_modality"])

    text_raw, text_ids = reuse_text_features(rows, cfg["reuse_text_features"], text_count)
    np.savez_compressed(feature_dir / "text_raw.npz", features=text_raw, video_ids=text_ids)
    log(f"Saved reused text features: {text_raw.shape}")

    requested = cfg["encoder"].get("device", "cuda")
    device = torch.device(requested if requested == "cuda" and torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("A CUDA device is required for EVA-CLIP and BEATs feature extraction")
    model = load_frozen_gram(cfg["pretrained_gram_dir"], device)

    text_model_raw, text_projected = encode_projected_text(model, rows, joint_count, cfg["encoder"], device)
    np.savez_compressed(
        feature_dir / "text_shared.npz",
        raw_features=text_model_raw,
        projected_features=text_projected,
        video_ids=np.asarray([item["video_id"] for item in rows]),
    )

    video_raw, video_projected, video_ids, video_failures = encode_video_observations(model, rows, cfg, device)
    np.savez_compressed(
        feature_dir / "video_features.npz",
        raw_features=video_raw,
        projected_features=video_projected,
        video_ids=video_ids,
    )
    save_json(Path(output_dir) / "metrics" / "video_failures.json", video_failures)

    audio_raw, audio_projected, audio_ids, audio_failures = encode_audio_observations(model, rows, cfg, device)
    np.savez_compressed(
        feature_dir / "audio_features.npz",
        raw_features=audio_raw,
        projected_features=audio_projected,
        video_ids=audio_ids,
    )
    save_json(Path(output_dir) / "metrics" / "audio_failures.json", audio_failures)

    del model
    torch.cuda.empty_cache()
    log("Multimodal feature extraction completed")


def load_analysis_features(output_dir, cfg):
    feature_dir = Path(output_dir) / "features"
    text = np.load(feature_dir / "text_raw.npz", allow_pickle=True)
    text_shared = np.load(feature_dir / "text_shared.npz", allow_pickle=True)
    video = np.load(feature_dir / "video_features.npz", allow_pickle=True)
    audio = np.load(feature_dir / "audio_features.npz", allow_pickle=True)
    joint_count = int(cfg["observations"]["joint_per_modality"])

    feature_sets = {
        "text_projected": (text_shared["projected_features"][:, :joint_count].astype(np.float32), text_shared["video_ids"]),
        "video_projected": (video["projected_features"][:, :joint_count].astype(np.float32), video["video_ids"]),
        "audio_projected": (audio["projected_features"][:, :joint_count].astype(np.float32), audio["video_ids"]),
    }
    aligned, joint_ids = align_feature_sets(feature_sets)
    joint = np.concatenate([
        aligned["text_projected"], aligned["video_projected"], aligned["audio_projected"]
    ], axis=1)
    joint = normalize_numpy(joint.astype(np.float32))
    log(f"Joint T-V-A feature tensor: {joint.shape}; common events={len(joint_ids)}")

    return {
        "text": (text["features"].astype(np.float32), text["video_ids"]),
        "video": (video["raw_features"].astype(np.float32), video["video_ids"]),
        "audio": (audio["raw_features"].astype(np.float32), audio["video_ids"]),
        "joint_tva": (joint, joint_ids),
    }


def analyze_features(cfg, output_dir):
    feature_sets = load_analysis_features(output_dir, cfg)
    seed = int(cfg.get("seed", 0))
    results = {}
    summary = {
        "experiment_name": cfg["experiment_name"],
        "criteria_declared_before_analysis": cfg["support_criteria"],
        "scope_note": (
            "Text/video/audio rows analyze repeated observations within each modality. "
            "The joint row analyzes balanced text, video, and audio observations in the frozen "
            "GRAM pretraining space. Results are empirical diagnostics, not a formal proof of "
            "multivariate Gaussianity."
        ),
        "modalities": {},
    }

    for modality_index, (modality, (features, video_ids)) in enumerate(feature_sets.items()):
        log(f"Analyzing {modality}: features={features.shape}")
        lln_rows, lln_summary = run_lln(features, cfg["lln"], seed + 101 * (modality_index + 1))
        residuals, coords, explained_ratio = residual_pca(features, cfg["gaussian"], seed + modality_index)
        theoretical, ordered, pc1_stats = qq_statistics(
            coords[:, 0], standardize=bool(cfg["gaussian"].get("standardize_pc1", True))
        )
        _, _, pc2_stats = qq_statistics(
            coords[:, 1], standardize=bool(cfg["gaussian"].get("standardize_pc1", True))
        )
        projection_stats = random_projection_diagnostics(
            residuals,
            int(cfg["gaussian"].get("random_projection_count", 16)),
            seed + 1000 + modality_index,
        )
        gaussian_metrics = {
            "pc1": pc1_stats,
            "pc2": pc2_stats,
            "pc1_explained_ratio": float(explained_ratio[0]),
            "pc2_explained_ratio": float(explained_ratio[1]),
            "random_projection_diagnostics": projection_stats,
            "num_residuals": int(coords.shape[0]),
            "feature_dimension": int(features.shape[-1]),
        }
        support = evaluate_support(lln_summary, gaussian_metrics, cfg["support_criteria"])
        metric_dir = Path(output_dir) / "metrics" / modality
        save_csv(metric_dir / "lln_metrics.csv", lln_rows)
        save_json(metric_dir / "lln_metrics.json", lln_rows)
        save_json(metric_dir / "lln_summary.json", lln_summary)
        save_json(metric_dir / "gaussian_metrics.json", gaussian_metrics)
        save_json(metric_dir / "support_assessment.json", support)
        np.savez_compressed(
            Path(output_dir) / "features" / f"{modality}_residual_pca.npz",
            coords=coords.astype(np.float32),
            explained_ratio=explained_ratio,
            video_ids=video_ids,
        )
        result = {
            "lln_rows": lln_rows,
            "lln_summary": lln_summary,
            "coords": coords,
            "explained_ratio": explained_ratio,
            "theoretical": theoretical,
            "ordered": ordered,
            "gaussian_metrics": gaussian_metrics,
            "support": support,
        }
        results[modality] = result
        summary["modalities"][modality] = {
            "display_name": DISPLAY_NAMES[modality],
            "feature_shape": list(features.shape),
            "lln": lln_summary,
            "gaussian": gaussian_metrics,
            "support": support,
        }
        save_modality_figures(modality, result, output_dir, cfg["figures"], seed + modality_index)

    save_composite(results, output_dir, cfg["figures"], seed)
    summary["all_modalities_pass_all_prespecified_checks"] = bool(
        all(item["support"]["all_prespecified_checks_pass"] for item in summary["modalities"].values())
    )
    save_json(Path(output_dir) / "metrics" / "validation_summary.json", summary)
    log(json.dumps({key: value["support"] for key, value in summary["modalities"].items()}, indent=2))
    log("Multimodal probabilistic-cloud analysis completed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["all", "extract", "analyze"], default="all")
    parser.add_argument("--limit-videos", type=int, default=None)
    args = parser.parse_args()

    cfg = load_json(args.config)
    if args.limit_videos is not None:
        cfg["split_size"] = int(args.limit_videos)
    set_seed(int(cfg.get("seed", 0)))

    output_dir = Path(cfg["output_dir"])
    if args.limit_videos is not None:
        output_dir = output_dir.parent / f"{output_dir.name}_limit{args.limit_videos}"
    for directory in (output_dir / "features", output_dir / "metrics", output_dir / "figures"):
        directory.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "run_config.json", cfg)

    rows = load_msrvtt_rows(
        cfg["ret_annotation"],
        cfg["caption_annotation"],
        int(cfg["split_size"]),
        int(cfg["captions_per_video"]),
    )
    save_json(output_dir / "samples_used.json", rows)
    log(f"Output directory: {output_dir}")
    log(f"Loaded {len(rows)} MSR-VTT events")

    if args.stage in ("all", "extract"):
        extract_features(cfg, rows, output_dir)
    if args.stage in ("all", "analyze"):
        analyze_features(cfg, output_dir)


if __name__ == "__main__":
    main()
