#!/usr/bin/env python3
"""Train V1 or V5 projection heads on identical cached IEMOCAP batches."""

import argparse
import json
import os
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from iemocap_common import (
    adapter_state_dict,
    apply_prior_to_logits,
    compute_iemocap_gaussian_prior,
    gaussian_confidence_gate,
    load_pretrained_model,
    project_features,
    project_prior_features,
    set_seed,
    set_trainable_heads_only,
)
from utils.volume import volume_computation3


def save_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def batch_prior_diagnostics(prior, label_ids):
    with torch.no_grad():
        n = prior.size(0)
        eye = torch.eye(n, dtype=torch.bool, device=prior.device)
        same = label_ids[:, None].eq(label_ids[None, :]) & ~eye
        different = ~label_ids[:, None].eq(label_ids[None, :])
        same_mean = prior[same].mean() if same.any() else prior.new_tensor(float("nan"))
        diff_mean = prior[different].mean() if different.any() else prior.new_tensor(float("nan"))
        return {
            "prior_same_class": float(same_mean.cpu()),
            "prior_different_class": float(diff_mean.cpu()),
            "prior_class_gap": float((same_mean - diff_mean).cpu()),
            "same_class_repulsion_weight": float((1.0 - same_mean).cpu()),
            "different_class_repulsion_weight": float((1.0 - diff_mean).cpu()),
        }


def compute_loss(model, final_feats, prior_feats, label_ids, args):
    feat_t, feat_v, feat_a = final_feats
    volume = volume_computation3(feat_t, feat_v, feat_a)
    temp = model.contra_temp.clamp(min=1e-3, max=1.0)
    logits_t2va = -volume / temp
    diagnostics = {}
    if args.version == "v5":
        prior_raw = compute_iemocap_gaussian_prior(
            list(prior_feats), tau=args.gaussian_tau, kl_reduction=args.prior_kl_reduction
        )
        prior = prior_raw
        if args.confidence_gate:
            prior = gaussian_confidence_gate(
                prior,
                list(prior_feats),
                temp=args.confidence_temp,
                power=args.confidence_power,
                floor=args.confidence_floor,
                subtract_random_baseline=args.confidence_random_baseline,
            )
        diagnostics.update({f"raw_{k}": v for k, v in batch_prior_diagnostics(prior_raw, label_ids).items()})
        diagnostics.update(batch_prior_diagnostics(prior, label_ids))
        logits_t2va = apply_prior_to_logits(logits_t2va, prior)
        logits_va2t = apply_prior_to_logits((-volume / temp).T, prior.T)
    else:
        logits_va2t = logits_t2va.T
    targets = torch.arange(feat_t.size(0), device=feat_t.device)
    loss = 0.5 * (
        F.cross_entropy(logits_t2va, targets, label_smoothing=args.label_smoothing)
        + F.cross_entropy(logits_va2t, targets, label_smoothing=args.label_smoothing)
    )
    return loss, diagnostics


def mean_records(records):
    keys = sorted({key for record in records for key in record if key not in {"step", "loss"}})
    result = {}
    for key in keys:
        values = [record[key] for record in records if key in record and np.isfinite(record[key])]
        if values:
            result[key] = float(np.mean(values))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["v1", "v5"], required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--class-anchor-cache", default=None)
    parser.add_argument("--pretrain-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-norm", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--gaussian-tau", type=float, default=0.5)
    parser.add_argument("--prior-kl-reduction", choices=["mean", "sum"], default="mean")
    parser.add_argument("--confidence-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--confidence-temp", type=float, default=0.07)
    parser.add_argument("--confidence-power", type=float, default=0.5)
    parser.add_argument("--confidence-floor", type=float, default=0.10)
    parser.add_argument("--confidence-random-baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for cached IEMOCAP training.")

    cache = torch.load(args.train_cache, map_location="cpu")
    n_samples = len(cache["ids"])
    label_names = sorted(set(cache["labels"]))
    label_to_id = {label: idx for idx, label in enumerate(label_names)}
    all_label_ids = torch.tensor([label_to_id[label] for label in cache["labels"]], dtype=torch.long)
    class_anchors = None
    all_anchor_rows = None
    if args.class_anchor_cache:
        class_anchors = torch.load(args.class_anchor_cache, map_location="cpu")
        anchor_label_to_row = {label: idx for idx, label in enumerate(class_anchors["labels"])}
        missing_labels = sorted(set(cache["labels"]) - set(anchor_label_to_row))
        if missing_labels:
            raise ValueError(f"Class anchor cache is missing labels: {missing_labels}")
        all_anchor_rows = torch.tensor(
            [anchor_label_to_row[label] for label in cache["labels"]], dtype=torch.long
        )
    model = load_pretrained_model(args.pretrain_dir, device, drop_audio_encoder=True)
    trainable = set_trainable_heads_only(model)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    config = {
        "settings": {
            key: value
            for key, value in vars(args).items()
            if key not in {"train_cache", "class_anchor_cache", "pretrain_dir", "output_dir"}
        },
        "samples": n_samples,
        "class_counts": dict(Counter(cache["labels"])),
        "label_to_id": label_to_id,
        "fairness_controls": {
            "frozen_encoder_cache": True,
            "batch_order_seeded_by_epoch": True,
            "same_pretrained_initialization": True,
            "labels_not_used_by_training_loss": args.class_anchor_cache is None,
            "controlled_repeated_class_anchor_stress_test": args.class_anchor_cache is not None,
        },
    }
    save_json(os.path.join(args.output_dir, "train_config.json"), config)

    history = []
    global_step = 0
    started = time.time()
    log_path = os.path.join(args.output_dir, "logs", "train_log.jsonl")
    with open(log_path, "w", encoding="utf-8") as log_handle:
        for epoch in range(1, args.epochs + 1):
            model.eval()
            model.contra_head_t.train()
            model.contra_head_v.train()
            model.contra_head_a.train()
            generator = torch.Generator().manual_seed(args.seed + epoch)
            order = torch.randperm(n_samples, generator=generator)
            usable = (n_samples // args.batch_size) * args.batch_size
            order = order[:usable]
            step_records = []
            for offset in range(0, usable, args.batch_size):
                index = order[offset : offset + args.batch_size]
                optimizer.zero_grad(set_to_none=True)
                if class_anchors is None:
                    text_final = cache["text_final"].index_select(0, index).to(device, non_blocking=True)
                else:
                    anchor_rows = all_anchor_rows.index_select(0, index)
                    text_final = class_anchors["text_final"].index_select(0, anchor_rows).to(device, non_blocking=True)
                video_final = cache["video_final"].index_select(0, index).to(device, non_blocking=True)
                audio = cache["audio"].index_select(0, index).to(device, non_blocking=True)
                final_feats = project_features(model, text_final, video_final, audio)
                prior_feats = None
                if args.version == "v5":
                    if class_anchors is None:
                        text_middle = cache["text_middle"].index_select(0, index).to(device, non_blocking=True)
                    else:
                        text_middle = class_anchors["text_middle"].index_select(0, anchor_rows).to(device, non_blocking=True)
                    video_middle = cache["video_middle"].index_select(0, index).to(device, non_blocking=True)
                    prior_feats = project_prior_features(model, text_middle, video_middle, audio)
                label_ids = all_label_ids.index_select(0, index).to(device)
                loss, diagnostics = compute_loss(model, final_feats, prior_feats, label_ids, args)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at epoch={epoch} step={global_step + 1}: {loss}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_norm)
                optimizer.step()
                global_step += 1
                record = {"loss": float(loss.detach().cpu()), **diagnostics}
                step_records.append(record)
                if global_step % args.log_steps == 0:
                    printable = {
                        "version": args.version,
                        "epoch": epoch,
                        "global_step": global_step,
                        "lr": optimizer.param_groups[0]["lr"],
                        "loss": record["loss"],
                        "elapsed_sec": round(time.time() - started, 2),
                        **{key: round(value, 6) for key, value in diagnostics.items()},
                    }
                    print(printable, flush=True)
                    log_handle.write(json.dumps(printable) + "\n")
                    log_handle.flush()
            scheduler.step()
            epoch_record = {
                "version": args.version,
                "epoch": epoch,
                "global_step": global_step,
                "lr": optimizer.param_groups[0]["lr"],
                "loss": float(np.mean([record["loss"] for record in step_records])),
                **mean_records(step_records),
            }
            history.append(epoch_record)
            print(f"epoch_summary={epoch_record}", flush=True)
            log_handle.write(json.dumps({"epoch_summary": epoch_record}) + "\n")
            log_handle.flush()
            if epoch % 5 == 0 or epoch == args.epochs:
                torch.save(
                    {
                        "version": f"{args.version}_iemocap_false_negative_v2",
                        "adapter": adapter_state_dict(model),
                        "epoch": epoch,
                        "global_step": global_step,
                        "settings": config["settings"],
                        "history": history,
                    },
                    os.path.join(args.output_dir, "checkpoints", f"epoch_{epoch:03d}.pt"),
                )

    last_path = os.path.join(args.output_dir, "checkpoints", "last.pt")
    torch.save(
        {
            "version": f"{args.version}_iemocap_false_negative_v2",
            "adapter": adapter_state_dict(model),
            "epoch": args.epochs,
            "global_step": global_step,
            "settings": config["settings"],
            "history": history,
        },
        last_path,
    )
    save_json(os.path.join(args.output_dir, "history.json"), history)
    print(f"Finished {args.version}.", flush=True)


if __name__ == "__main__":
    main()
