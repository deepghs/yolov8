"""Generate every figure used in the report.

Inputs (relative to this file):
  ../data/embed_quality_results.csv
  ../data/yolo_perspective_results.json

Outputs (relative to this file):
  ../figures/01_effective_rank.png
  ../figures/02_anisotropy_vs_uniformity.png
  ../figures/03_augmentation_invariance.png
  ../figures/04_detection_jaccard_spearman.png
  ../figures/05_family_scaling.png
  ../figures/06_multimetric_radar.png
  ../figures/07_cross_model_cka.png

Run:
  conda run -n yolov8 python generate_charts.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
FIG = HERE.parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Style: keep visuals consistent across all charts.
# ---------------------------------------------------------------------------

FAMILY_COLOR = {
    "v3u": "#7f7f7f",
    "v5u": "#1f77b4",
    "v8":  "#2ca02c",
    "v8-task": "#9467bd",
    "v9":  "#d62728",
    "v10": "#ff7f0e",
    "v11": "#17becf",
    "rtdetr": "#8c564b",
}

MODEL_ORDER = [
    ("yolov3-tinyu.pt", "v3u-tiny", "v3u"),
    ("yolov5nu.pt", "v5u-n", "v5u"),
    ("yolov5su.pt", "v5u-s", "v5u"),
    ("yolov5mu.pt", "v5u-m", "v5u"),
    ("yolov8n.pt", "v8-n", "v8"),
    ("yolov8s.pt", "v8-s", "v8"),
    ("yolov8m.pt", "v8-m", "v8"),
    ("yolov8n-cls.pt", "v8-n-cls", "v8-task"),
    ("yolov8n-seg.pt", "v8-n-seg", "v8-task"),
    ("yolov8n-pose.pt", "v8-n-pose", "v8-task"),
    ("yolov8n-obb.pt", "v8-n-obb", "v8-task"),
    ("yolov9t.pt", "v9-t", "v9"),
    ("yolov9s.pt", "v9-s", "v9"),
    ("yolov10n.pt", "v10-n", "v10"),
    ("yolov10s.pt", "v10-s", "v10"),
    ("yolo11n.pt", "v11-n", "v11"),
    ("yolo11s.pt", "v11-s", "v11"),
    ("yolo11m.pt", "v11-m", "v11"),
    ("rtdetr-l.pt", "rtdetr-l", "rtdetr"),
]

PRETTY = {n: short for n, short, _ in MODEL_ORDER}
FAMILY = {n: fam for n, _, fam in MODEL_ORDER}


def _load() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with open(DATA / "embed_quality_results.csv") as f:
        for row in csv.DictReader(f):
            for k, v in list(row.items()):
                if k != "weights":
                    try:
                        row[k] = float(v) if v != "" else None
                    except (ValueError, TypeError):
                        pass
            rows[row["weights"]] = row
    with open(DATA / "yolo_perspective_results.json") as f:
        per = json.load(f)
    cka = per.pop("_cka", None)
    for name, info in per.items():
        if name not in rows:
            rows[name] = {"weights": name}
        rows[name]["self_detection_nmi"] = info.get("self_detection_nmi")
        rows[name]["self_detection_probe_acc"] = info.get("self_detection_probe_acc")
        rows[name]["detection_jaccard_spearman"] = info.get("detection_jaccard_spearman")
        rows[name]["augmentation_invariance_margin"] = (
            info.get("augmentation", {}) or {}
        ).get("invariance_margin")
    return rows, cka


def _ordered(rows: dict, key: str, ascending: bool) -> list[tuple[str, float]]:
    pairs = [(n, rows[n].get(key)) for n in rows
             if rows[n].get(key) is not None]
    pairs.sort(key=lambda p: p[1], reverse=not ascending)
    return pairs


def _color(name: str) -> str:
    return FAMILY_COLOR[FAMILY.get(name, "v8")]


def _legend_handles():
    from matplotlib.patches import Patch
    return [Patch(facecolor=c, label=fam) for fam, c in FAMILY_COLOR.items()]


# ---------------------------------------------------------------------------
# 1. Effective rank
# ---------------------------------------------------------------------------

def fig_effective_rank(rows):
    pairs = _ordered(rows, "effective_rank", ascending=False)
    names = [p[0] for p in pairs]
    vals = [p[1] for p in pairs]
    dims = [int(rows[n]["d"]) for n in names]
    colors = [_color(n) for n in names]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(range(len(vals)), vals, color=colors, edgecolor="black", linewidth=0.4)
    for i, (v, d) in enumerate(zip(vals, dims)):
        ratio = v / d * 100
        ax.text(i, v + 0.5, f"{v:.0f}\n({ratio:.0f}% of D)",
                ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([PRETTY[n] for n in names], rotation=55, ha="right")
    ax.set_ylabel("effective rank  (exp(entropy of eigenvalue spectrum))")
    ax.set_title("Effective rank — how many dimensions actually carry signal\n"
                 "(higher = embedding spreads across more directions; bounded by D)")
    ax.legend(handles=_legend_handles(), title="family", loc="upper right",
              fontsize=8, ncol=4)
    ax.set_ylim(0, max(vals) * 1.18)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "01_effective_rank.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Anisotropy vs uniformity
# ---------------------------------------------------------------------------

def fig_aniso_vs_uniformity(rows):
    fig, ax = plt.subplots(figsize=(9, 7))
    for n in rows:
        a = rows[n].get("anisotropy_mean_cos")
        u = rows[n].get("uniformity_wang_isola")
        if a is None or u is None:
            continue
        ax.scatter(a, u, s=80, color=_color(n), edgecolor="black", linewidth=0.5,
                   zorder=3)
        ax.annotate(PRETTY[n], (a, u), xytext=(4, 4), textcoords="offset points",
                    fontsize=7.5)
    # quadrant guides
    ax.axhline(-1.0, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.axvline(0.7, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.text(0.55, -1.55, "well behaved\n(low aniso, low uniformity)",
            fontsize=8, color="darkgreen", ha="center")
    ax.text(0.95, -0.15, "collapsed / spiky\n(high aniso, near-zero uniformity)",
            fontsize=8, color="darkred", ha="center")
    ax.set_xlabel("anisotropy (mean off-diag cosine)  ←  lower is better")
    ax.set_ylabel("Wang–Isola uniformity loss  ←  lower (more negative) is better")
    ax.set_title("Geometric quality of the L2-normalised embedding\n"
                 "(top-right = embedding collapsed near a line)")
    ax.legend(handles=_legend_handles(), title="family", loc="lower left",
              fontsize=8, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "02_anisotropy_vs_uniformity.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Augmentation invariance
# ---------------------------------------------------------------------------

def fig_aug_invariance(rows):
    pairs = _ordered(rows, "augmentation_invariance_margin", ascending=False)
    names = [p[0] for p in pairs]
    vals = [p[1] for p in pairs]
    self_sims = [rows[n].get("augmentation", {}) for n in names]  # not present at csv level
    # Reload self / cross from the JSON for annotation
    with open(DATA / "yolo_perspective_results.json") as f:
        per = json.load(f)
    self_aug = [per.get(n, {}).get("augmentation", {}).get("self_aug_cos_mean") for n in names]
    cross = [per.get(n, {}).get("augmentation", {}).get("cross_image_cos_mean") for n in names]
    colors = [_color(n) for n in names]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(range(len(vals)), vals, color=colors, edgecolor="black", linewidth=0.4)
    for i, (v, s, c) in enumerate(zip(vals, self_aug, cross)):
        if s is not None and c is not None:
            ax.text(i, v + 0.005, f"{v:.2f}\n(s={s:.2f}\nc={c:.2f})",
                    ha="center", va="bottom", fontsize=6.3)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([PRETTY[n] for n in names], rotation=55, ha="right")
    ax.set_ylabel("invariance margin  =  cos(orig, augmented)  −  cos(orig_i, orig_j)")
    ax.set_title("Augmentation-invariance margin — primary 'YOLO content awareness'\n"
                 "(higher = embedding stays put under brightness / blur / rotate / JPEG, "
                 "but discriminates between distinct images)")
    ax.legend(handles=_legend_handles(), title="family", loc="upper right",
              fontsize=8, ncol=4)
    ax.set_ylim(0, max(vals) * 1.30)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "03_augmentation_invariance.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Detection-Jaccard Spearman
# ---------------------------------------------------------------------------

def fig_detection_jaccard(rows):
    pairs = []
    for n in rows:
        v = rows[n].get("detection_jaccard_spearman")
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        pairs.append((n, v))
    pairs.sort(key=lambda p: p[1], reverse=True)
    names = [p[0] for p in pairs]
    vals = [p[1] for p in pairs]
    colors = [_color(n) for n in names]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(range(len(vals)), vals, color=colors, edgecolor="black", linewidth=0.4)
    for i, v in enumerate(vals):
        ax.text(i, v + (0.005 if v >= 0 else -0.015), f"{v:+.2f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([PRETTY[n] for n in names], rotation=55, ha="right")
    ax.set_ylabel("Spearman ρ between cos(emb_i, emb_j) and Jaccard(detected_i, detected_j)")
    ax.set_title("Does cosine similarity track 'same set of detected objects'?\n"
                 "(positive = yes; negative = embedding contradicts the model's own detector)")
    ax.legend(handles=_legend_handles(), title="family", loc="upper right",
              fontsize=8, ncol=4)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "04_detection_jaccard_spearman.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Family scaling
# ---------------------------------------------------------------------------

def fig_family_scaling(rows):
    families_to_plot = {
        "v5u":  ["yolov5nu.pt", "yolov5su.pt", "yolov5mu.pt"],
        "v8":   ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
        "v9":   ["yolov9t.pt", "yolov9s.pt"],
        "v10":  ["yolov10n.pt", "yolov10s.pt"],
        "v11":  ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"],
    }
    panels = [
        ("effective_rank", "effective rank (higher better)", False),
        ("augmentation_invariance_margin", "augmentation-invariance margin (higher better)", False),
        ("self_detection_probe_acc", "self-detection probe acc (higher better)", False),
        ("anisotropy_mean_cos", "anisotropy mean cos (lower better)", True),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(20, 5))
    size_label = {0: "n/t", 1: "s", 2: "m"}
    for ax, (key, title, invert) in zip(axes, panels):
        for fam, members in families_to_plot.items():
            xs = []
            ys = []
            for i, m in enumerate(members):
                v = rows.get(m, {}).get(key)
                if v is None:
                    continue
                xs.append(i)
                ys.append(v)
            ax.plot(xs, ys, marker="o", linewidth=2, label=fam, color=FAMILY_COLOR[fam])
        ax.set_xticks(list(range(3)))
        ax.set_xticklabels([size_label[i] for i in range(3)])
        ax.set_xlabel("model size")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        if invert:
            ax.invert_yaxis()
    axes[0].legend(title="family", loc="best")
    fig.suptitle("Within-family scaling: does going from nano → small → medium help?", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "05_family_scaling.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Multi-metric radar (top performers vs cautionary tales)
# ---------------------------------------------------------------------------

def fig_radar(rows):
    chosen = [
        ("yolov8m.pt", "v8-m  (best overall)"),
        ("yolov5mu.pt", "v5u-m  (close 2nd, 768d)"),
        ("yolov5su.pt", "v5u-s  (best AugInv)"),
        ("rtdetr-l.pt", "rtdetr-l  (best Det-Jac)"),
        ("yolov9s.pt", "v9-s  (light + good)"),
        ("yolov3-tinyu.pt", "v3u-tiny  (collapsed)"),
        ("yolov8n-pose.pt", "v8-n-pose  (task-head harms)"),
        ("yolo11n.pt", "v11-n  (small dim curse)"),
    ]
    metrics = [
        ("isotropy_inv", "1 - aniso", False),  # higher better
        ("effective_rank_norm", "EffR / D", False),
        ("uniformity_inv", "−Wang–Isola", False),
        ("self_detection_probe_acc", "Det-Probe", False),
        ("detection_jaccard_spearman", "Det-Jac ρ", False),
        ("augmentation_invariance_margin", "AugInv", False),
    ]
    # Build a normalised matrix per metric across chosen rows.
    raw = {}
    for n, _ in chosen:
        r = rows[n]
        raw[n] = {
            "isotropy_inv": 1.0 - (r.get("anisotropy_mean_cos") or 1.0),
            "effective_rank_norm": (r.get("effective_rank") or 0) /
                                   max(1.0, float(r.get("d") or 1.0)),
            "uniformity_inv": -(r.get("uniformity_wang_isola") or 0.0),
            "self_detection_probe_acc": r.get("self_detection_probe_acc"),
            "detection_jaccard_spearman": r.get("detection_jaccard_spearman"),
            "augmentation_invariance_margin": r.get("augmentation_invariance_margin"),
        }
    # min-max normalise each metric across the chosen models so 0=worst, 1=best
    norm = {n: {} for n, _ in chosen}
    for key, _, _ in metrics:
        vals = [raw[n][key] for n, _ in chosen if raw[n][key] is not None]
        lo, hi = min(vals), max(vals)
        for n, _ in chosen:
            v = raw[n][key]
            norm[n][key] = (v - lo) / (hi - lo) if v is not None and hi > lo else 0.0

    # plot
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), subplot_kw=dict(polar=True))
    axes = axes.ravel()
    for ax, (name, label) in zip(axes, chosen):
        vals = [norm[name][k] for k, _, _ in metrics] + [norm[name][metrics[0][0]]]
        ax.plot(angles, vals, color=_color(name), linewidth=2)
        ax.fill(angles, vals, color=_color(name), alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m[1] for m in metrics], fontsize=8)
        ax.set_yticks([0.25, 0.5, 0.75])
        ax.set_yticklabels([])
        ax.set_ylim(0, 1)
        ax.set_title(label, fontsize=10, pad=14)
    fig.suptitle("Per-model radar — six normalised quality axes\n"
                 "(each axis is min-max scaled across the eight models shown)", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "06_multimetric_radar.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. CKA heatmap
# ---------------------------------------------------------------------------

def fig_cka(cka_block):
    if not cka_block:
        return
    names = cka_block["names"]
    mat = np.asarray(cka_block["matrix"])
    order = sorted(range(len(names)),
                   key=lambda i: [m[0] for m in MODEL_ORDER].index(names[i])
                   if names[i] in [m[0] for m in MODEL_ORDER] else 99)
    mat = mat[np.ix_(order, order)]
    names = [names[i] for i in order]

    fig, ax = plt.subplots(figsize=(11.5, 10))
    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels([PRETTY.get(n, n) for n in names], rotation=70, ha="right")
    ax.set_yticklabels([PRETTY.get(n, n) for n in names])
    for i in range(len(names)):
        for j in range(len(names)):
            col = "white" if mat[i, j] < 0.5 else "black"
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    color=col, fontsize=7)
    ax.set_title("Cross-model Linear CKA on Imagenette val (n=2000)\n"
                 "Higher = two models 'see' the image set with the same similarity structure")
    plt.colorbar(im, ax=ax, label="Linear CKA")
    fig.tight_layout()
    fig.savefig(FIG / "07_cross_model_cka.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    rows, cka_block = _load()
    fig_effective_rank(rows)
    fig_aniso_vs_uniformity(rows)
    fig_aug_invariance(rows)
    fig_detection_jaccard(rows)
    fig_family_scaling(rows)
    fig_radar(rows)
    fig_cka(cka_block)
    print(f"wrote 7 PNGs to {FIG}")


if __name__ == "__main__":
    main()
