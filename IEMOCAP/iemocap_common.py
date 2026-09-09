import csv
import json
import os
import pickle
import random
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from PIL import Image, ImageDraw, ImageFont
from sklearn.manifold import TSNE
from torch.utils.data import Dataset


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "GRAM"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model import model_registry  # noqa: E402
from utils.volume import volume_computation3  # noqa: E402


LABEL_NAMES = {
    "neu": "Neutral",
    "fru": "Frustrated",
    "ang": "Angry",
    "sad": "Sad",
    "exc": "Excited",
    "hap": "Happy",
}

LABEL_PROMPTS = {
    "neu": "a person speaks in a neutral emotion",
    "fru": "a person speaks with frustration",
    "ang": "a person speaks with anger",
    "sad": "a person speaks with sadness",
    "exc": "a person speaks with excitement",
    "hap": "a person speaks with happiness",
}

COLORS = {
    "neu": "#009E73",
    "fru": "#E69F00",
    "ang": "#D55E00",
    "sad": "#0072B2",
    "exc": "#CC79A7",
    "hap": "#56B4E9",
}


def recursive_edict(obj):
    if isinstance(obj, dict):
        return edict({key: recursive_edict(value) for key, value in obj.items()})
    if isinstance(obj, list):
        return [recursive_edict(value) for value in obj]
    return obj


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pretrained_model(pretrain_dir, device, drop_audio_encoder=True):
    hps_path = os.path.join(pretrain_dir, "log", "hps.json")
    ckpt_path = os.path.join(pretrain_dir, "ckpt", "model_step_249.pt")
    hps = json.load(open(hps_path))
    cfg = recursive_edict(hps["model_cfg"])
    cfg.model_type = "gram"
    cfg.checkpointing = False
    cfg.fp16 = False
    cfg.captioner_mode = False

    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        model = model_registry[cfg.model_type](cfg)
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            checkpoint = checkpoint["model"]
        checkpoint = {key.replace("module.", ""): value for key, value in checkpoint.items()}
        checkpoint = model.modify_checkpoint(checkpoint)
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    finally:
        os.chdir(cwd)

    if drop_audio_encoder and hasattr(model, "audio_encoder"):
        model.audio_encoder = None
    print("Loaded pretrained model.", flush=True)
    print(f"Missing keys: {len(missing)}; unexpected keys: {len(unexpected)}", flush=True)
    model.to(device)
    model.eval()
    return model


def set_trainable_heads_only(model):
    for param in model.parameters():
        param.requires_grad = False
    trainable_modules = [model.contra_head_t, model.contra_head_v, model.contra_head_a]
    for module in trainable_modules:
        for param in module.parameters():
            param.requires_grad = True
    model.contra_temp.requires_grad = True
    return [param for param in model.parameters() if param.requires_grad]


def adapter_state_dict(model):
    return {
        "contra_head_t": model.contra_head_t.state_dict(),
        "contra_head_v": model.contra_head_v.state_dict(),
        "contra_head_a": model.contra_head_a.state_dict(),
        "contra_temp": model.contra_temp.detach().cpu(),
    }


def load_adapter_state(model, checkpoint_path, device):
    payload = torch.load(checkpoint_path, map_location="cpu")
    state = payload["adapter"] if "adapter" in payload else payload
    model.contra_head_t.load_state_dict(state["contra_head_t"])
    model.contra_head_v.load_state_dict(state["contra_head_v"])
    model.contra_head_a.load_state_dict(state["contra_head_a"])
    if "contra_temp" in state:
        model.contra_temp.data.copy_(state["contra_temp"].to(model.contra_temp.device))
    model.to(device)
    model.eval()
    return payload


def read_iemocap_items(dataset_dir, split):
    path = os.path.join(dataset_dir, f"{split}.txt")
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            items.append({"id": parts[0], "label": parts[1], "text": parts[2]})
    return items


def load_pickle_tensor(path, key):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data[key].float()


def prepare_video_tensor(images, num_frames, image_size):
    total_frames = images.size(0)
    if total_frames >= num_frames:
        frame_idx = torch.linspace(0, total_frames - 1, steps=num_frames).long()
        images = images.index_select(0, frame_idx)
    else:
        pad = images[-1:].repeat(num_frames - total_frames, 1, 1, 1)
        images = torch.cat([images, pad], dim=0)

    _, _, height, width = images.shape
    crop = min(height, width)
    top = (height - crop) // 2
    left = (width - crop) // 2
    images = images[:, :, top : top + crop, left : left + crop]
    if crop != image_size:
        images = F.interpolate(images, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return images


class IemocapTripletDataset(Dataset):
    def __init__(self, dataset_dir, split, num_video_frames, image_size, classes=None, max_per_class=None, seed=42):
        self.dataset_dir = dataset_dir
        self.split = split
        self.split_dir = os.path.join(dataset_dir, split)
        self.num_video_frames = num_video_frames
        self.image_size = image_size
        items = read_iemocap_items(dataset_dir, split)
        if classes:
            items = [item for item in items if item["label"] in set(classes)]
        if max_per_class is not None:
            rng = random.Random(seed)
            by_label = defaultdict(list)
            for item in items:
                by_label[item["label"]].append(item)
            trimmed = []
            for label, label_items in by_label.items():
                label_items = list(label_items)
                rng.shuffle(label_items)
                trimmed.extend(label_items[:max_per_class])
            items = sorted(trimmed, key=lambda item: (item["label"], item["id"]))
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        sample_id = item["id"]
        audio_path = os.path.join(self.split_dir, "audio", f"{sample_id}.pkl")
        video_path = os.path.join(self.split_dir, "video", f"{sample_id}.pkl")
        audio = load_pickle_tensor(audio_path, "audio_feature")
        images = load_pickle_tensor(video_path, "images")
        images = prepare_video_tensor(images, self.num_video_frames, self.image_size)
        return {
            "id": sample_id,
            "label": item["label"],
            "text": item["text"],
            "audio": audio,
            "video": images,
        }


def collate_iemocap(batch):
    return {
        "ids": [item["id"] for item in batch],
        "labels": [item["label"] for item in batch],
        "texts": [item["text"] for item in batch],
        "audio": torch.stack([item["audio"] for item in batch], dim=0),
        "video": torch.stack([item["video"] for item in batch], dim=0),
    }


@torch.no_grad()
def pooled_text_outputs(model, texts, device, need_middle=False, fp16=True):
    tokens = model.multimodal_encoder.tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=model.max_caption_len,
        return_tensors="pt",
    ).to(device)
    with torch.cuda.amp.autocast(enabled=fp16):
        output = model.multimodal_encoder.bert(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
            output_hidden_states=need_middle,
        )
    pooled = model.pool_text_for_contra(output.last_hidden_state).float()
    middle = None
    if need_middle:
        layer_index = model.multimodal_encoder.bert.config.num_hidden_layers // 2
        middle = model.pool_text_for_contra(output.hidden_states[layer_index]).float()
    return pooled, middle


@torch.no_grad()
def pooled_video_outputs(model, videos, device, need_middle=False, fp16=True):
    videos = videos.to(device, non_blocking=True)
    with torch.cuda.amp.autocast(enabled=fp16):
        if need_middle:
            output = model.forward_vision_encoder(
                videos,
                return_intermediate_features=True,
                intermediate_layer=None,
            )
            final_output, middle_output = output if isinstance(output, tuple) else (output, output)
        else:
            final_output = model.forward_vision_encoder(videos)
            middle_output = None
    pooled = model.pool_vision_for_contra(final_output).float()
    middle = model.pool_vision_for_contra(middle_output).float() if need_middle else None
    return pooled, middle


def project_features(model, text_pooled, video_pooled, audio_features):
    feat_t = F.normalize(model.contra_head_t(text_pooled), dim=-1)
    feat_v = F.normalize(model.contra_head_v(video_pooled), dim=-1)
    feat_a = F.normalize(model.contra_head_a(audio_features), dim=-1)
    return feat_t.float(), feat_v.float(), feat_a.float()


@torch.no_grad()
def project_prior_features(model, text_middle, video_middle, audio_features):
    feat_t = F.normalize(model.contra_head_t(text_middle), dim=-1)
    feat_v = F.normalize(model.contra_head_v(video_middle), dim=-1)
    feat_a = F.normalize(model.contra_head_a(audio_features), dim=-1)
    return feat_t.float(), feat_v.float(), feat_a.float()


def encode_training_batch(model, batch, device, need_prior=False, fp16=True):
    text_pooled, text_middle = pooled_text_outputs(model, batch["texts"], device, need_middle=need_prior, fp16=fp16)
    video_pooled, video_middle = pooled_video_outputs(model, batch["video"], device, need_middle=need_prior, fp16=fp16)
    audio = batch["audio"].to(device, non_blocking=True).float()
    feat_t, feat_v, feat_a = project_features(model, text_pooled, video_pooled, audio)
    prior_feats = None
    if need_prior:
        prior_feats = project_prior_features(model, text_middle, video_middle, audio)
    return feat_t, feat_v, feat_a, prior_feats


def compute_iemocap_gaussian_prior(modality_features, tau=1.0, eps=1e-6, kl_reduction="mean"):
    """IEMOCAP uses pre-extracted audio features, so normalize KL by feature dimension."""
    valid_features = [feature for feature in modality_features if feature is not None]
    if len(valid_features) < 2:
        raise ValueError("compute_iemocap_gaussian_prior needs at least two modalities.")

    pooled_features = []
    for feature in valid_features:
        if feature.dim() > 2:
            feature = feature.reshape(feature.shape[0], -1, feature.shape[-1]).mean(dim=1)
        pooled_features.append(F.normalize(feature.float(), dim=-1))

    stacked_features = torch.stack(pooled_features, dim=0)
    mu = stacked_features.mean(dim=0)
    var = stacked_features.var(dim=0, unbiased=False).clamp_min(eps)

    mu_i = mu.unsqueeze(1)
    mu_j = mu.unsqueeze(0)
    var_i = var.unsqueeze(1)
    var_j = var.unsqueeze(0)

    kl_terms_ij = (var_i / var_j) + ((mu_j - mu_i).pow(2) / var_j) - 1.0 + torch.log(var_j / var_i)
    kl_terms_ji = (var_j / var_i) + ((mu_i - mu_j).pow(2) / var_i) - 1.0 + torch.log(var_i / var_j)
    if kl_reduction == "sum":
        kl_ij = 0.5 * kl_terms_ij.sum(dim=-1)
        kl_ji = 0.5 * kl_terms_ji.sum(dim=-1)
    elif kl_reduction == "mean":
        kl_ij = 0.5 * kl_terms_ij.mean(dim=-1)
        kl_ji = 0.5 * kl_terms_ji.mean(dim=-1)
    else:
        raise ValueError(f"Unsupported KL reduction: {kl_reduction}")

    symmetric_kl = 0.5 * (kl_ij + kl_ji)
    prior = torch.exp(-symmetric_kl / max(float(tau), eps)).clamp(min=0.0, max=1.0)
    prior.fill_diagonal_(1.0)
    return prior.to(dtype=valid_features[0].dtype)


def prior_matrix_stats(prior_matrix):
    batch_size = prior_matrix.size(0)
    eye = torch.eye(batch_size, device=prior_matrix.device, dtype=torch.bool)
    offdiag = prior_matrix.masked_select(~eye)
    if offdiag.numel() == 0:
        return {
            "offdiag_mean": 0.0,
            "offdiag_max": 0.0,
            "offdiag_min": 0.0,
            "offdiag_nonzero_ratio": 0.0,
        }
    return {
        "offdiag_mean": float(offdiag.mean().detach().cpu()),
        "offdiag_max": float(offdiag.max().detach().cpu()),
        "offdiag_min": float(offdiag.min().detach().cpu()),
        "offdiag_nonzero_ratio": float((offdiag > 1e-6).float().mean().detach().cpu()),
    }


def gaussian_confidence_gate(
    prior_matrix,
    modality_features,
    temp=0.07,
    power=0.5,
    floor=0.25,
    subtract_random_baseline=False,
):
    pooled_features = [F.normalize(feature.float(), dim=-1) for feature in modality_features if feature is not None]
    if len(pooled_features) < 2:
        return prior_matrix
    batch_size = pooled_features[0].size(0)
    pair_confidences = []
    for idx_i in range(len(pooled_features)):
        for idx_j in range(idx_i + 1, len(pooled_features)):
            logits = pooled_features[idx_i] @ pooled_features[idx_j].T
            logits = logits / max(float(temp), 1e-6)
            row_prob = F.softmax(logits, dim=1).diag()
            col_prob = F.softmax(logits, dim=0).diag()
            confidence = torch.sqrt((row_prob * col_prob).clamp_min(0.0))
            if subtract_random_baseline and batch_size > 1:
                random_prob = 1.0 / batch_size
                confidence = ((confidence - random_prob) / (1.0 - random_prob)).clamp(0.0, 1.0)
            pair_confidences.append(confidence)
    sample_confidence = torch.stack(pair_confidences, dim=0).mean(dim=0)
    sample_confidence = sample_confidence.clamp(0.0, 1.0).pow(max(float(power), 1e-6))
    if floor > 0:
        floor = min(max(float(floor), 0.0), 1.0)
        sample_confidence = floor + (1.0 - floor) * sample_confidence
    pair_confidence = torch.sqrt(sample_confidence.unsqueeze(1) * sample_confidence.unsqueeze(0)).to(prior_matrix)
    gated = prior_matrix * pair_confidence
    gated.fill_diagonal_(1.0)
    return gated


def apply_prior_to_logits(logits, prior_matrix):
    targets = torch.arange(logits.size(0), device=logits.device)
    hard_push_mask = (1.0 - prior_matrix).clamp(min=1e-6, max=1.0).to(logits)
    weighted_logits = logits + hard_push_mask.log()
    weighted_logits.scatter_(1, targets.unsqueeze(1), logits.gather(1, targets.unsqueeze(1)))
    return weighted_logits


def gram_volume_loss(
    model,
    feat_t,
    feat_v,
    feat_a,
    prior_feats=None,
    tau=1.0,
    confidence=True,
    prior_kl_reduction="mean",
    confidence_floor=0.25,
    subtract_confidence_random_baseline=False,
    return_stats=False,
):
    volume = volume_computation3(feat_t, feat_v, feat_a)
    temp = model.contra_temp.clamp(min=1e-3, max=1.0)
    logits_t2va = -volume / temp
    logits_va2t = logits_t2va.T
    stats = {}
    if prior_feats is not None:
        prior = compute_iemocap_gaussian_prior(list(prior_feats), tau=tau, kl_reduction=prior_kl_reduction)
        stats.update({f"prior_raw_{key}": value for key, value in prior_matrix_stats(prior).items()})
        if confidence:
            prior = gaussian_confidence_gate(
                prior,
                list(prior_feats),
                floor=confidence_floor,
                subtract_random_baseline=subtract_confidence_random_baseline,
            )
            stats.update({f"prior_gated_{key}": value for key, value in prior_matrix_stats(prior).items()})
        logits_t2va = apply_prior_to_logits(logits_t2va, prior)
        logits_va2t = apply_prior_to_logits(logits_va2t, prior.T)
    targets = torch.arange(feat_t.size(0), device=feat_t.device)
    loss = (
        F.cross_entropy(logits_t2va, targets, label_smoothing=0.1)
        + F.cross_entropy(logits_va2t, targets, label_smoothing=0.1)
    ) / 2
    if return_stats:
        return loss, stats
    return loss


def l2_normalize(array):
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms


def select_visualization_items(dataset_dir, split, classes, max_per_class, seed):
    items = read_iemocap_items(dataset_dir, split)
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for item in items:
        if item["label"] in classes:
            by_label[item["label"]].append(item)
    selected = []
    counts = {}
    for label in classes:
        candidates = list(by_label[label])
        rng.shuffle(candidates)
        chosen = sorted(candidates[:max_per_class], key=lambda item: item["id"])
        counts[label] = len(chosen)
        selected.extend(chosen)
    selected.sort(key=lambda item: (item["label"], item["id"]))
    return selected, counts


@torch.no_grad()
def extract_visualization_features(model, dataset_dir, split, selected, classes, num_video_frames, image_size, device, fp16=True):
    label_text_features = {}
    prompts = [LABEL_PROMPTS[label] for label in classes]
    text_pooled, _ = pooled_text_outputs(model, prompts, device, need_middle=False, fp16=fp16)
    projected = F.normalize(model.contra_head_t(text_pooled), dim=-1).float().cpu()
    for idx, label in enumerate(classes):
        label_text_features[label] = projected[idx]

    records = []
    split_dir = os.path.join(dataset_dir, split)
    for idx, item in enumerate(selected):
        sample_id = item["id"]
        audio = load_pickle_tensor(os.path.join(split_dir, "audio", f"{sample_id}.pkl"), "audio_feature")
        video = load_pickle_tensor(os.path.join(split_dir, "video", f"{sample_id}.pkl"), "images")
        video = prepare_video_tensor(video, num_video_frames, image_size).unsqueeze(0)
        audio = audio.unsqueeze(0).to(device).float()
        video_pooled, _ = pooled_video_outputs(model, video, device, need_middle=False, fp16=fp16)
        feat_v = F.normalize(model.contra_head_v(video_pooled), dim=-1).float().cpu()[0]
        feat_a = F.normalize(model.contra_head_a(audio), dim=-1).float().cpu()[0]
        records.append({"id": sample_id, "label": item["label"], "text": item["text"], "V": feat_v, "A": feat_a})
        if (idx + 1) % 20 == 0:
            print(f"Visual features: {idx + 1}/{len(selected)}", flush=True)
    return {"classes": classes, "label_text_features": label_text_features, "records": records}


def build_tsne_rows(payload, seed, perplexity):
    vectors = []
    rows = []
    for label in payload["classes"]:
        vectors.append(payload["label_text_features"][label].numpy())
        rows.append(
            {
                "sample_id": f"class_text_{label}",
                "label": label,
                "label_name": LABEL_NAMES.get(label, label),
                "modality": "Text label",
                "utterance": LABEL_PROMPTS.get(label, label),
            }
        )
    for record in payload["records"]:
        for modality in ("V", "A"):
            vectors.append(record[modality].numpy())
            rows.append(
                {
                    "sample_id": record["id"],
                    "label": record["label"],
                    "label_name": LABEL_NAMES.get(record["label"], record["label"]),
                    "modality": "Video" if modality == "V" else "Audio",
                    "utterance": record["text"],
                }
            )
    vectors = l2_normalize(np.asarray(vectors, dtype=np.float32))
    actual_perplexity = min(perplexity, max(5, (len(vectors) - 1) // 3))
    coords = TSNE(
        n_components=2,
        perplexity=actual_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(vectors)
    for row, coord in zip(rows, coords):
        row["x"] = float(coord[0])
        row["y"] = float(coord[1])
    return rows, actual_perplexity


def save_points_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "label", "label_name", "modality", "x", "y", "utterance"],
        )
        writer.writeheader()
        writer.writerows(rows)


def load_font(size, bold=False):
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


def hex_to_rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[idx : idx + 2], 16) for idx in (0, 2, 4))


def draw_marker(draw, x, y, color, marker, radius, outline=(31, 41, 55)):
    x = float(x)
    y = float(y)
    if marker == "star":
        points = []
        for i in range(10):
            angle = -np.pi / 2 + i * np.pi / 5
            r = radius if i % 2 == 0 else radius * 0.45
            points.append((x + r * np.cos(angle), y + r * np.sin(angle)))
        draw.polygon(points, fill=color, outline=outline)
    elif marker == "square":
        draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=color, outline=outline)
    elif marker == "triangle":
        points = [(x, y - radius - 1), (x - radius - 1, y + radius), (x + radius + 1, y + radius)]
        draw.polygon(points, fill=color, outline=outline)
    else:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=outline)


def scale_rows(rows, left, top, width, height):
    xs = [row["x"] for row in rows]
    ys = [row["y"] for row in rows]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    pad_x = (max_x - min_x) * 0.08 + 1e-6
    pad_y = (max_y - min_y) * 0.08 + 1e-6
    min_x -= pad_x
    max_x += pad_x
    min_y -= pad_y
    max_y += pad_y
    scaled = []
    for row in rows:
        item = dict(row)
        item["px"] = left + (row["x"] - min_x) / (max_x - min_x) * width
        item["py"] = top + height - (row["y"] - min_y) / (max_y - min_y) * height
        scaled.append(item)
    return scaled


def plot_tsne(rows, path, title):
    width, height = 1500, 1120
    margin_x, margin_top, legend_h = 100, 95, 170
    panel_w = width - 2 * margin_x
    panel_h = height - margin_top - legend_h - 70
    image = Image.new("RGB", (width, height), (248, 249, 252))
    draw = ImageDraw.Draw(image)
    title_font = load_font(32, bold=True)
    label_font = load_font(24)
    small_font = load_font(20)

    title_w = draw.textlength(title, font=title_font)
    draw.text(((width - title_w) / 2, 35), title, fill=(20, 28, 39), font=title_font)
    left, top = margin_x, margin_top
    draw.rectangle((left, top, left + panel_w, top + panel_h), fill=(235, 238, 245), outline=(195, 203, 216))
    for gx in range(1, 6):
        x = left + gx * panel_w / 6
        draw.line((x, top, x, top + panel_h), fill=(255, 255, 255), width=2)
    for gy in range(1, 5):
        y = top + gy * panel_h / 5
        draw.line((left, y, left + panel_w, y), fill=(255, 255, 255), width=2)

    scaled = scale_rows(rows, left + 45, top + 45, panel_w - 90, panel_h - 90)
    marker_map = {"Audio": "triangle", "Video": "square", "Text label": "star"}
    radius_map = {"Audio": 7, "Video": 7, "Text label": 18}
    for modality in ("Audio", "Video", "Text label"):
        for row in scaled:
            if row["modality"] != modality:
                continue
            color = hex_to_rgb(COLORS.get(row["label"], "#333333"))
            draw_marker(draw, row["px"], row["py"], color, marker_map[modality], radius_map[modality])

    for row in scaled:
        if row["modality"] == "Text label":
            label = LABEL_NAMES.get(row["label"], row["label"])
            draw.text((row["px"] + 12, row["py"] - 12), label, fill=(17, 24, 39), font=small_font)

    legend_top = height - legend_h + 8
    x = margin_x
    draw.text((x, legend_top), "Modality", fill=(30, 30, 30), font=label_font)
    x += 140
    for modality in ("Text label", "Video", "Audio"):
        draw_marker(draw, x, legend_top + 17, (85, 85, 85), marker_map[modality], 10 if modality != "Text label" else 14)
        draw.text((x + 24, legend_top + 4), modality, fill=(30, 30, 30), font=small_font)
        x += 150

    x = margin_x
    y = legend_top + 65
    draw.text((x, y), "Emotion class", fill=(30, 30, 30), font=label_font)
    x += 180
    for label in sorted({row["label"] for row in rows}):
        color = hex_to_rgb(COLORS.get(label, "#333333"))
        draw.rectangle((x, y + 5, x + 24, y + 29), fill=color, outline=(31, 41, 55))
        draw.text((x + 34, y + 2), LABEL_NAMES.get(label, label), fill=(30, 30, 30), font=small_font)
        x += 150
    image.save(path)


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def summarize_split(dataset_dir):
    summary = {}
    for split in ("train", "test"):
        items = read_iemocap_items(dataset_dir, split)
        summary[split] = dict(Counter(item["label"] for item in items).most_common())
    return summary
