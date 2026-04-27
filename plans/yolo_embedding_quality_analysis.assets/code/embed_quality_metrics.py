"""Embedding mathematical-quality benchmark suite.

For each Ultralytics pretrained checkpoint, embed a fixed image set and
compute a battery of unsupervised quality metrics on the resulting [N, D]
matrix. Metrics chosen from the embedding-evaluation literature:

  * anisotropy_mean_cos         - Ethayarajh 2019 / Mu & Viswanath 2018
                                  mean off-diagonal pairwise cosine of
                                  L2-normed vectors. Lower magnitude => more
                                  isotropic; ideal close to 0.
  * spectral_top1               - largest covariance eigenvalue / trace.
                                  Mu & Viswanath all-but-the-top concept.
                                  Lower => no single direction dominates.
  * spectral_top10              - top-10 eigenvalue mass.
  * effective_rank              - Roy & Vetterli 2007: exp(H(p)) where p_i =
                                  lambda_i / sum lambda_j. The "soft"
                                  rank; higher => more dimensions actually
                                  carry signal. Bounded by min(N-1, D).
  * stable_rank                 - ||X||_F^2 / ||X||_op^2 = trace / lambda_1.
                                  A noise-robust rank surrogate.
  * participation_ratio         - Recanatesi et al.: (sum lambda)^2 /
                                  sum lambda^2. "Effective dimensionality"
                                  used in computational neuroscience.
  * isoscore                    - Rudman et al. 2022: rescaled isotropy
                                  defect on [0, 1] with chance ~0.
                                  Higher => embedding fills the space
                                  uniformly.
  * uniformity_wang_isola       - Wang & Isola 2020 ICML: log E[exp(-2 d^2)]
                                  on the unit sphere. More negative =>
                                  more uniformly spread.
  * twonn_intrinsic_dim         - Facco et al. 2017 Sci Rep: ratio-based
                                  intrinsic-dimension estimator using only
                                  1st & 2nd nearest-neighbour distances.
                                  Reports the dimensionality of the local
                                  manifold the data lives on.
  * hubness_skewness_k10        - Radovanovic et al. 2010: skewness of the
                                  k-occurrence distribution. > 1.4 indicates
                                  problematic hubness (a few "popular"
                                  points dominate everyone's nearest list).
  * mean_norm_pre_l2            - mean L2 norm of raw embeddings before
                                  normalisation. Sanity / scale check.
  * knn1_acc_imagenette         - 1-NN leave-one-out cosine accuracy using
                                  the 10 Imagenette class labels. Not a
                                  pure mathematical metric but it tells us
                                  whether all that geometric quality
                                  translates to semantic separation.
  * silhouette_imagenette       - sklearn silhouette score with cosine
                                  metric, 10-class Imagenette labels.

Outputs a CSV plus a ranked summary printed to stdout.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from yolov8.embed import get_embedding


HERE = Path(__file__).parent
WEIGHTS_DIR = HERE / "weights"
IMAGENETTE_VAL = HERE / "imagenette2-160" / "val"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def collect_imagenette_val(root: Path, max_per_class: int | None = None
                           ) -> tuple[list[Path], np.ndarray, list[str]]:
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    paths: list[Path] = []
    labels: list[int] = []
    for ci, c in enumerate(classes):
        items = sorted((root / c).glob("*.JPEG"))
        if max_per_class:
            items = items[:max_per_class]
        paths.extend(items)
        labels.extend([ci] * len(items))
    return paths, np.asarray(labels, dtype=np.int64), classes


# ---------------------------------------------------------------------------
# Metric implementations
# ---------------------------------------------------------------------------

def _spectrum(X: np.ndarray) -> np.ndarray:
    """Return non-negative eigenvalues of the centred covariance, descending."""
    Xc = X - X.mean(axis=0, keepdims=True)
    # SVD is more numerically stable than np.linalg.eigh on covariance directly.
    s = np.linalg.svd(Xc, compute_uv=False)
    lam = (s ** 2) / max(1, X.shape[0] - 1)
    lam = np.sort(lam)[::-1]
    return lam


def anisotropy_mean_cos(X_unit: np.ndarray, max_pairs: int = 5_000_000) -> float:
    """Mean off-diagonal pairwise cosine of L2-normed vectors."""
    n = X_unit.shape[0]
    sim = X_unit @ X_unit.T
    iu = np.triu_indices(n, k=1)
    if iu[0].size > max_pairs:
        rng = np.random.default_rng(0)
        idx = rng.choice(iu[0].size, size=max_pairs, replace=False)
        return float(sim[iu[0][idx], iu[1][idx]].mean())
    return float(sim[iu].mean())


def spectral_concentration(lam: np.ndarray, k: int = 1) -> float:
    """Top-k eigenvalue mass / total. Mu&Viswanath all-but-the-top."""
    return float(lam[:k].sum() / max(lam.sum(), 1e-12))


def effective_rank(lam: np.ndarray) -> float:
    """exp(entropy(p)) where p_i = lambda_i / sum lambda_j (Roy & Vetterli 2007)."""
    p = lam / max(lam.sum(), 1e-12)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def stable_rank(lam: np.ndarray) -> float:
    """||X||_F^2 / ||X||_op^2 = sum / max."""
    return float(lam.sum() / max(lam[0], 1e-12))


def participation_ratio(lam: np.ndarray) -> float:
    """(sum lambda)^2 / sum lambda^2 (Recanatesi et al.)."""
    s = lam.sum()
    s2 = (lam * lam).sum()
    return float((s * s) / max(s2, 1e-12))


def isoscore(X: np.ndarray) -> float:
    """Rudman et al. 2022 IsoScore on [0, 1]; close to 1 = isotropic.

    Implementation follows Algorithm 1: PCA-rotate, normalise so that
    component-wise norms sum to D, then compare to all-ones vector.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    rotated = Xc @ Vt.T  # shape [N, D']
    D = rotated.shape[1]
    sigma = (rotated ** 2).sum(axis=0)  # per-component "energy"
    sigma = sigma / max(sigma.sum(), 1e-12) * D  # rescale to sum=D
    diff = sigma - 1.0
    iso_defect = np.sqrt((diff ** 2).sum()) / np.sqrt(2 * (D - 1)) if D > 1 else 0.0
    iso_score_raw = 1.0 - iso_defect ** 2  # paper uses 1 - delta(X)^2
    # Rescale chance baseline to ~0 (the paper subtracts 1/D, for high-D it's
    # negligible; we keep the un-rebased value so all models are on the same
    # axis).
    return float(max(0.0, min(1.0, iso_score_raw)))


def uniformity_wang_isola(X_unit: np.ndarray, t: float = 2.0,
                          max_pairs: int = 5_000_000) -> float:
    """log E[exp(-t ||x-y||^2)] on unit sphere; more negative = more uniform."""
    n = X_unit.shape[0]
    sim = X_unit @ X_unit.T
    iu = np.triu_indices(n, k=1)
    pair_sim = sim[iu]
    if pair_sim.size > max_pairs:
        rng = np.random.default_rng(0)
        pair_sim = pair_sim[rng.choice(pair_sim.size, size=max_pairs, replace=False)]
    pair_d2 = np.clip(2.0 - 2.0 * pair_sim, 0.0, 4.0)
    return float(np.log(np.exp(-t * pair_d2).mean() + 1e-30))


def twonn_intrinsic_dim(X: np.ndarray, fraction: float = 0.9) -> float:
    """Facco et al. 2017 TwoNN intrinsic dim. ``fraction`` drops the upper
    tail to be robust to outliers, as the paper recommends."""
    from scipy.spatial import cKDTree

    tree = cKDTree(X)
    d, _ = tree.query(X, k=3, workers=-1)  # [N, 3] distances; col0 = self
    r1 = d[:, 1]
    r2 = d[:, 2]
    valid = r1 > 0  # drop exact duplicates which break the ratio
    r1 = r1[valid]
    r2 = r2[valid]
    mu = r2 / np.maximum(r1, 1e-12)
    mu = np.sort(mu)
    n = mu.shape[0]
    keep = max(2, int(round(n * fraction)))
    mu_k = mu[:keep]
    Femp = (np.arange(1, keep + 1)) / n
    y = -np.log1p(-Femp)
    x = np.log(mu_k)
    # least-squares slope through the origin: argmin ||y - slope x||
    slope = float(np.sum(x * y) / max(np.sum(x * x), 1e-12))
    return slope


def hubness_skewness(X: np.ndarray, k: int = 10) -> float:
    """Skewness of k-occurrence count (Radovanovic et al. 2010)."""
    from scipy.spatial import cKDTree
    from scipy.stats import skew

    tree = cKDTree(X)
    _, idx = tree.query(X, k=k + 1, workers=-1)
    idx = idx[:, 1:]  # drop self
    counts = np.bincount(idx.flatten(), minlength=X.shape[0])
    return float(skew(counts))


def knn1_accuracy(X_unit: np.ndarray, labels: np.ndarray) -> float:
    """Cosine 1-NN leave-one-out accuracy."""
    sim = X_unit @ X_unit.T
    np.fill_diagonal(sim, -np.inf)
    nn = sim.argmax(axis=1)
    return float((labels[nn] == labels).mean())


def silhouette_cosine(X_unit: np.ndarray, labels: np.ndarray, max_n: int = 4000) -> float:
    """sklearn silhouette with cosine metric. Subsample if too large."""
    from sklearn.metrics import silhouette_score

    n = X_unit.shape[0]
    if n > max_n:
        rng = np.random.default_rng(0)
        keep = rng.choice(n, size=max_n, replace=False)
        X_unit = X_unit[keep]
        labels = labels[keep]
    return float(silhouette_score(X_unit, labels, metric="cosine"))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def embed_dataset(weights: Path, paths: list[Path], imgsz: int,
                  batch: int = 32) -> np.ndarray:
    """Compute [N, D] raw (un-normalised) embeddings for all images."""
    from ultralytics import RTDETR, YOLO

    cls = RTDETR if weights.stem.lower().startswith("rtdetr") else YOLO
    model = cls(str(weights))
    inner = model.model.to(DEVICE).float()
    inner.train(False)

    out: list[np.ndarray] = []
    t0 = time.perf_counter()
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        emb = get_embedding(model, [str(p) for p in chunk], imgsz=imgsz, device=DEVICE)
        out.append(emb.cpu().numpy().astype(np.float32))
    arr = np.concatenate(out, axis=0)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return arr


def imgsz_for(stem: str) -> int:
    s = stem.lower()
    if s.endswith("-cls"):
        return 224
    return 640


def compute_metrics(X: np.ndarray, labels: np.ndarray | None) -> dict:
    norms = np.linalg.norm(X, axis=1)
    X_unit = X / np.maximum(norms[:, None], 1e-12)
    lam = _spectrum(X)

    out = {
        "n": X.shape[0],
        "d": X.shape[1],
        "mean_norm_pre_l2": float(norms.mean()),
        "anisotropy_mean_cos": anisotropy_mean_cos(X_unit),
        "spectral_top1": spectral_concentration(lam, 1),
        "spectral_top10": spectral_concentration(lam, 10),
        "effective_rank": effective_rank(lam),
        "stable_rank": stable_rank(lam),
        "participation_ratio": participation_ratio(lam),
        "isoscore": isoscore(X),
        "uniformity_wang_isola": uniformity_wang_isola(X_unit),
        "twonn_intrinsic_dim": twonn_intrinsic_dim(X),
        "hubness_skewness_k10": hubness_skewness(X, k=10),
    }
    if labels is not None:
        out["knn1_acc_imagenette"] = knn1_accuracy(X_unit, labels)
        out["silhouette_imagenette"] = silhouette_cosine(X_unit, labels)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", default=None,
                    help="explicit list of .pt files (default: all under weights/)")
    ap.add_argument("--imgs-per-class", type=int, default=None,
                    help="cap N per class for speed (default: full val set)")
    ap.add_argument("--out-csv", default=str(HERE / "embed_quality_results.csv"))
    args = ap.parse_args()

    paths, labels, classes = collect_imagenette_val(IMAGENETTE_VAL, args.imgs_per_class)
    print(f"loaded {len(paths)} images across {len(classes)} classes")

    weights = sorted(WEIGHTS_DIR.glob("*.pt")) if args.weights is None \
        else [Path(w) for w in args.weights]

    rows = []
    for w in weights:
        sz = imgsz_for(w.stem)
        print(f"\n[{w.name:<22s}] embedding {len(paths)} imgs at {sz}px ...", flush=True)
        t0 = time.perf_counter()
        X = embed_dataset(w, paths, imgsz=sz)
        dt = time.perf_counter() - t0
        print(f"  embed shape={X.shape}  in {dt:.1f}s "
              f"({dt / len(paths) * 1000:.2f} ms/img)")

        m = compute_metrics(X, labels)
        m.update(weights=w.name, imgsz=sz, embed_seconds=round(dt, 1))
        rows.append(m)

        # one-line print per model
        keys = ["d", "anisotropy_mean_cos", "isoscore", "effective_rank",
                "stable_rank", "twonn_intrinsic_dim",
                "uniformity_wang_isola", "hubness_skewness_k10",
                "knn1_acc_imagenette", "silhouette_imagenette"]
        line = " | ".join(f"{k}={m[k]:.3f}" if isinstance(m[k], float) else f"{k}={m[k]}"
                          for k in keys if k in m)
        print(f"  {line}")

    # write CSV
    import csv
    cols = ["weights", "imgsz", "n", "d", "embed_seconds",
            "mean_norm_pre_l2", "anisotropy_mean_cos",
            "spectral_top1", "spectral_top10",
            "effective_rank", "stable_rank", "participation_ratio",
            "isoscore", "uniformity_wang_isola",
            "twonn_intrinsic_dim", "hubness_skewness_k10",
            "knn1_acc_imagenette", "silhouette_imagenette"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
