#!/usr/bin/env python3
"""Publication figure: empirical false-negative repulsion in GRAM vs SmoothGRAM."""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from scipy.stats import wilcoxon


NAVY = "#26354A"
MUTED = "#6C7888"
GRID = "#DCE2E8"
ORANGE = "#E69F00"
BLUE = "#4C78A8"
TEAL = "#1B9E77"
RED = "#D65A4A"
PANEL = "#F7F9FB"


def normalize(feature):
    return F.normalize(feature.float(), dim=-1)


def gaussian_prior(features, tau=0.5, eps=1e-6):
    pooled = [normalize(feature) for feature in features]
    stacked = torch.stack(pooled, dim=0)
    mu = stacked.mean(dim=0)
    var = stacked.var(dim=0, unbiased=False).clamp_min(eps)
    mu_i, mu_j = mu[:, None, :], mu[None, :, :]
    var_i, var_j = var[:, None, :], var[None, :, :]
    kl_ij = 0.5 * ((var_i / var_j) + (mu_j - mu_i).pow(2) / var_j - 1 + torch.log(var_j / var_i)).mean(-1)
    kl_ji = 0.5 * ((var_j / var_i) + (mu_i - mu_j).pow(2) / var_i - 1 + torch.log(var_i / var_j)).mean(-1)
    prior = torch.exp(-(0.5 * (kl_ij + kl_ji)) / tau).clamp(0, 1)
    prior.fill_diagonal_(1.0)
    return prior


def confidence_gate(prior, features, temp=0.07, power=0.5, floor=0.35):
    pooled = [normalize(feature) for feature in features]
    pair_confidences = []
    for i in range(len(pooled)):
        for j in range(i + 1, len(pooled)):
            logits = pooled[i] @ pooled[j].T / temp
            row_prob = F.softmax(logits, dim=1).diag()
            col_prob = F.softmax(logits, dim=0).diag()
            pair_confidences.append(torch.sqrt((row_prob * col_prob).clamp_min(0)))
    sample_conf = torch.stack(pair_confidences).mean(0).clamp(0, 1).pow(power)
    sample_conf = floor + (1 - floor) * sample_conf
    pair_conf = torch.sqrt(sample_conf[:, None] * sample_conf[None, :]).to(prior)
    gated = prior * pair_conf
    gated.fill_diagonal_(1.0)
    return gated


def bootstrap_ci(values, rng, repeats=3000):
    values = np.asarray(values, dtype=np.float64)
    means = np.empty(repeats, dtype=np.float64)
    for i in range(repeats):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return np.percentile(means, [2.5, 97.5])


def node(ax, xy, color, text, subtext=None, radius=0.075):
    ax.add_patch(Circle(xy, radius, facecolor=color, edgecolor="white", linewidth=2.2, zorder=5))
    ax.add_patch(Circle(xy, radius, facecolor="none", edgecolor=NAVY, linewidth=0.8, zorder=6))
    ax.text(xy[0], xy[1], text, ha="center", va="center", color="white", fontsize=10, fontweight="bold", zorder=7)
    if subtext:
        ax.text(xy[0], xy[1] - radius - 0.052, subtext, ha="center", va="top", color=MUTED, fontsize=8.5)


def link(ax, start, end, color, width, label, rad=0.0, dashed=False):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=12, connectionstyle=f"arc3,rad={rad}",
        linewidth=width, color=color, linestyle=(0, (4, 3)) if dashed else "solid", alpha=0.92, zorder=2,
    )
    ax.add_patch(patch)
    x = (start[0] + end[0]) / 2
    y = (start[1] + end[1]) / 2 + (0.06 if rad <= 0 else -0.02)
    ax.text(x, y, label, ha="center", va="center", fontsize=8.5, color=color,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.9), zorder=8)


def mechanism_panel(ax, smooth, same_mean, diff_mean):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    node(ax, (0.12, 0.52), ORANGE, "i", "anchor")
    node(ax, (0.48, 0.52), ORANGE, "i+", "paired positive")
    node(ax, (0.84, 0.73), ORANGE, "j", "same-class non-pair")
    node(ax, (0.84, 0.28), BLUE, "k", "different class")
    link(ax, (0.20, 0.52), (0.39, 0.52), TEAL, 2.6, "attract")
    if not smooth:
        link(ax, (0.20, 0.59), (0.76, 0.72), RED, 2.8, r"$W_{ij}=1.000$", rad=0.08)
        link(ax, (0.20, 0.45), (0.76, 0.29), RED, 2.8, r"$W_{ik}=1.000$", rad=-0.08)
        ax.text(0.50, 0.04, "Uniform repulsion treats false negatives as hard negatives",
                ha="center", color=NAVY, fontsize=9.5, fontweight="bold")
    else:
        link(ax, (0.20, 0.59), (0.76, 0.72), TEAL, 1.6,
             rf"$W_{{ij}}={1-same_mean:.3f}$", rad=0.08, dashed=True)
        link(ax, (0.20, 0.45), (0.76, 0.29), RED, 2.5,
             rf"$W_{{ik}}={1-diff_mean:.3f}$", rad=-0.08)
        ax.text(0.50, 0.04, "Semantic overlap selectively weakens false-negative repulsion",
                ha="center", color=NAVY, fontsize=9.5, fontweight="bold")


def style_distribution_axis(ax):
    ax.set_facecolor(PANEL)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.set_ylim(-0.008, 0.40)
    ax.set_yticks(np.arange(0, 0.41, 0.1))
    ax.set_ylabel(r"Softening strength  $1-W_{ij}=P_{ij}$", color=NAVY, fontsize=9.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--projected", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--test-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    projected = torch.load(args.projected, map_location="cpu")["v5"]
    with open(args.metrics, encoding="utf-8") as handle:
        metrics = json.load(handle)
    cache = torch.load(args.test_cache, map_location="cpu")
    id_to_index = {sample_id: idx for idx, sample_id in enumerate(cache["ids"])}
    selected = [id_to_index[sample_id] for sample_id in metrics["selected_ids"]]
    index = torch.tensor(selected, dtype=torch.long)
    features = [projected[key].index_select(0, index) for key in ("prior_T", "prior_V", "prior_A")]
    prior = confidence_gate(gaussian_prior(features, tau=0.5), features, temp=0.07, power=0.5, floor=0.35)

    labels = np.asarray([cache["labels"][idx] for idx in selected])
    same = labels[:, None] == labels[None, :]
    upper = np.triu(np.ones_like(same, dtype=bool), k=1)
    prior_np = prior.numpy()
    same_values = prior_np[same & upper]
    diff_values = prior_np[(~same) & upper]
    # Use one value per held-out anchor for inference, avoiding a test that
    # incorrectly treats the many overlapping sample pairs as independent.
    eye = np.eye(len(labels), dtype=bool)
    same_anchor = np.asarray([prior_np[i][same[i] & ~eye[i]].mean() for i in range(len(labels))])
    diff_anchor = np.asarray([prior_np[i][~same[i]].mean() for i in range(len(labels))])
    same_mean, diff_mean = float(same_anchor.mean()), float(diff_anchor.mean())
    rng = np.random.default_rng(2026)
    same_ci = bootstrap_ci(same_anchor, rng)
    diff_ci = bootstrap_ci(diff_anchor, rng)
    delta_anchor = same_anchor - diff_anchor
    delta_mean = float(delta_anchor.mean())
    delta_ci = bootstrap_ci(delta_anchor, rng)
    w_stat, p_value = wilcoxon(same_anchor, diff_anchor, alternative="greater")

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titleweight": "bold",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(13.8, 6.6), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.03, 1.0], hspace=0.18, wspace=0.16,
                          left=0.055, right=0.985, top=0.86, bottom=0.12)
    diag_l = fig.add_subplot(gs[0, 0])
    diag_r = fig.add_subplot(gs[0, 1])
    dist_l = fig.add_subplot(gs[1, 0])
    dist_r = fig.add_subplot(gs[1, 1])

    fig.text(0.5, 0.955, "False-negative repulsion: GRAM vs. SmoothGRAM",
             ha="center", va="top", fontsize=18, color=NAVY, fontweight="bold")
    fig.text(0.5, 0.912,
             "Held-out IEMOCAP test split · 120 samples · soft prior computed from V5 intermediate multimodal features",
             ha="center", va="top", fontsize=9.5, color=MUTED)
    fig.text(0.267, 0.875, "(a) Original GRAM", ha="center", fontsize=13, color=NAVY, fontweight="bold")
    fig.text(0.752, 0.875, "(b) SmoothGRAM", ha="center", fontsize=13, color=NAVY, fontweight="bold")
    fig.add_artist(plt.Line2D([0.502, 0.502], [0.10, 0.87], transform=fig.transFigure, color=GRID, linewidth=1.2))

    mechanism_panel(diag_l, False, same_mean, diff_mean)
    mechanism_panel(diag_r, True, same_mean, diff_mean)

    style_distribution_axis(dist_l)
    dist_l.bar([0, 1], [0, 0], width=0.58, color=[ORANGE, BLUE], edgecolor=NAVY, linewidth=0.7, zorder=3)
    dist_l.scatter([0, 1], [0, 0], s=34, color=[ORANGE, BLUE], edgecolor="white", linewidth=0.8, zorder=5)
    dist_l.set_xticks([0, 1], ["Same-class\nnon-pairs", "Different-class\nnon-pairs"])
    dist_l.text(0.5, 0.205, "All non-paired samples", ha="center", color=MUTED, fontsize=9)
    dist_l.text(0.5, 0.155, r"$P_{ij}=0$  →  $W_{ij}=1$", ha="center", color=NAVY, fontsize=14, fontweight="bold")
    dist_l.text(0.5, 0.105, "no adaptive softening", ha="center", color=RED, fontsize=9.5)

    style_distribution_axis(dist_r)
    parts = dist_r.violinplot([same_anchor, diff_anchor], positions=[0, 1], widths=0.68,
                              showmeans=False, showmedians=False, showextrema=False, bw_method=0.25)
    for body, color in zip(parts["bodies"], [ORANGE, BLUE]):
        body.set_facecolor(color)
        body.set_edgecolor(NAVY)
        body.set_linewidth(0.7)
        body.set_alpha(0.32)
    for x, values, color in [(0, same_anchor, ORANGE), (1, diff_anchor, BLUE)]:
        take = rng.choice(len(values), size=min(160, len(values)), replace=False)
        jitter = rng.normal(0, 0.075, size=len(take))
        dist_r.scatter(x + jitter, values[take], s=7, color=color, alpha=0.22, linewidth=0, zorder=2)
    means = [same_mean, diff_mean]
    cis = [same_ci, diff_ci]
    for x, mean, ci, color in zip([0, 1], means, cis, [ORANGE, BLUE]):
        dist_r.errorbar(x, mean, yerr=[[mean-ci[0]], [ci[1]-mean]], fmt="o", markersize=7,
                        color=NAVY, markerfacecolor=color, markeredgecolor="white", markeredgewidth=1,
                        capsize=4, linewidth=1.5, zorder=6)
        dist_r.text(x, mean + 0.035, f"mean {mean:.3f}", ha="center", fontsize=9, color=NAVY, fontweight="bold")
    dist_r.set_xticks([0, 1], ["Same-class\nnon-pairs", "Different-class\nnon-pairs"])
    dist_r.text(0.98, 0.95, f"mean Δ = {delta_mean:.3f}\npaired Wilcoxon $p$ < 10$^{{-6}}$" if p_value < 1e-6 else f"mean Δ = {delta_mean:.3f}\npaired $p$ = {p_value:.2g}",
                transform=dist_r.transAxes, ha="right", va="top", fontsize=8.8, color=NAVY,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRID, lw=0.8))

    fig.text(0.5, 0.035,
             "Same-class non-pairs receive stronger softening than different-class non-pairs, while GRAM applies uniform repulsion.",
             ha="center", color=NAVY, fontsize=10, fontweight="bold")

    base = os.path.join(args.output_dir, "iemocap_false_negative_repulsion_comparison")
    fig.savefig(base + ".png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(base + ".pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(base + ".svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    result = {
        "samples": len(selected),
        "classes": metrics["classes"],
        "same_class_pairs": int(len(same_values)),
        "different_class_pairs": int(len(diff_values)),
        "same_class_softening_mean": same_mean,
        "same_class_softening_ci95": same_ci.tolist(),
        "different_class_softening_mean": diff_mean,
        "different_class_softening_ci95": diff_ci.tolist(),
        "anchor_level_softening_delta_mean": delta_mean,
        "anchor_level_softening_delta_ci95": delta_ci.tolist(),
        "paired_wilcoxon_statistic": float(w_stat),
        "paired_wilcoxon_p_one_sided": float(p_value),
        "interpretation_scope": "Mechanism-level evidence for selective false-negative softening; not evidence of improved clustering.",
    }
    with open(base + "_stats.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, indent=2))
    print("Saved false-negative diagnostic outputs.")


if __name__ == "__main__":
    main()
