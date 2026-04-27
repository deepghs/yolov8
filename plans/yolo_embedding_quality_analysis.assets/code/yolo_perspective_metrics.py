"""YOLO-perspective embedding quality metrics.

Five metrics that specifically measure whether an embedding reflects what
the YOLO model itself "sees" - i.e. whether nearby vectors in cosine space
correspond to images the model would describe similarly:

  1. self_detection_nmi
       Run the model's full detection head on each image, take the top-1
       detected COCO class; k-means cluster the embeddings (k = number of
       distinct detected classes); report NMI between cluster IDs and
       detected class IDs. High = embedding clusters match the model's
       own world view.

  2. self_detection_probe_acc
       Treat top-1 detected class as a label; train a linear (logistic)
       probe on a 70/30 split of embeddings; report held-out accuracy.
       High = embedding is linearly separable along the dimensions the
       model itself cares about.

  3. detection_jaccard_spearman
       For each image collect the bag of detected class IDs (conf > 0.25).
       For all image pairs (i, j) compute Spearman rank correlation
       between cosine(emb_i, emb_j) and Jaccard(bag_i, bag_j). High =
       cosine similarity in embedding space tracks "do these images
       contain the same set of objects according to the model".

  4. augmentation_invariance_margin
       For each image generate K=4 photometric / mild-geometric
       augmentations, embed them, and measure mean cosine(orig, aug);
       compute mean cosine(orig_i, orig_j) for i!=j as the chance
       baseline. Margin = self_aug_sim - cross_image_sim. High = the
       embedding moves with content changes but stays put under
       non-content perturbations.

  5. cross_model_linear_cka
       For every pair of detection models compute Linear CKA on their
       embeddings of the same images (Kornblith et al. 2019). High CKA
       across pairs => the YOLO family agrees on similarity structure;
       outlier rows reveal models that "see the world" differently.

Pure cls model (yolov8n-cls.pt) is excluded from metrics 1-3 since it
emits ImageNet-1000 logits not COCO detections; included for 4-5.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from yolov8.embed import get_embedding


HERE = Path(__file__).parent
WEIGHTS_DIR = HERE / "weights"
IMAGENETTE_VAL = HERE / "imagenette2-160" / "val"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Helpers shared with embed_quality_metrics.py
# ---------------------------------------------------------------------------

def collect_paths(root: Path, max_per_class: int | None) -> list[Path]:
    classes = sorted(p for p in root.iterdir() if p.is_dir())
    out: list[Path] = []
    for c in classes:
        items = sorted(c.glob("*.JPEG"))
        if max_per_class:
            items = items[:max_per_class]
        out.extend(items)
    return out


def imgsz_for(stem: str) -> int:
    return 224 if stem.lower().endswith("-cls") else 640


def load_wrapper(weights: Path):
    from ultralytics import RTDETR, YOLO

    cls = RTDETR if weights.stem.lower().startswith("rtdetr") else YOLO
    return cls(str(weights))


def is_detection_model(weights: Path) -> bool:
    """True for COCO-detection-style models. cls is excluded; seg/pose/obb
    still detect bounding boxes plus their extra outputs, so we keep them."""
    s = weights.stem.lower()
    if s.endswith("-cls"):
        return False
    return True


def embed_all(weights: Path, paths: list[Path], imgsz: int,
              batch: int = 32, normalize: bool = True) -> np.ndarray:
    model = load_wrapper(weights)
    out: list[np.ndarray] = []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        e = get_embedding(model, [str(p) for p in chunk], imgsz=imgsz,
                          device=DEVICE, normalize=normalize)
        out.append(e.cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# Self-detection: run the model's detection head, collect class labels
# ---------------------------------------------------------------------------

def detect_all(weights: Path, paths: list[Path], imgsz: int, conf: float = 0.25,
               batch: int = 32) -> tuple[list[int | None], list[list[int]]]:
    """Run full detection. Return (top1_class_per_image, bag_of_classes_per_image).
    For images where nothing is detected above ``conf``, the top-1 entry is
    ``None`` and the bag is ``[]``.
    """
    model = load_wrapper(weights)
    top1: list[int | None] = []
    bag: list[list[int]] = []
    # ultralytics treats list[str] as a batch source.
    for i in range(0, len(paths), batch):
        chunk = [str(p) for p in paths[i:i + batch]]
        results = model(chunk, imgsz=imgsz, conf=conf, verbose=False, device=DEVICE)
        for r in results:
            if r.boxes is None or r.boxes.cls is None or len(r.boxes.cls) == 0:
                top1.append(None)
                bag.append([])
                continue
            classes = r.boxes.cls.detach().cpu().numpy().astype(np.int64)
            confs = r.boxes.conf.detach().cpu().numpy()
            # top-1 = highest-confidence class
            top1.append(int(classes[int(confs.argmax())]))
            bag.append(sorted(set(int(c) for c in classes)))
    return top1, bag


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def self_detection_nmi(emb: np.ndarray, top1: list[int | None]) -> float | None:
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score

    mask = np.asarray([t is not None for t in top1])
    if mask.sum() < 30:
        return None
    labels = np.asarray([t for t in top1 if t is not None], dtype=np.int64)
    classes = np.unique(labels)
    if len(classes) < 2:
        return None
    km = KMeans(n_clusters=len(classes), n_init=5, random_state=0)
    cluster_ids = km.fit_predict(emb[mask])
    return float(normalized_mutual_info_score(labels, cluster_ids))


def self_detection_probe_acc(emb: np.ndarray, top1: list[int | None],
                             min_per_class: int = 3) -> float | None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    mask = np.asarray([t is not None for t in top1])
    if mask.sum() < 30:
        return None
    X = emb[mask]
    y = np.asarray([t for t in top1 if t is not None], dtype=np.int64)
    # Drop classes with too few samples - logistic regression's stratified
    # split needs at least 2 per class.
    counts = np.bincount(y)
    keep_classes = np.where(counts >= min_per_class)[0]
    if len(keep_classes) < 2:
        return None
    keep = np.isin(y, keep_classes)
    if keep.sum() < 30:
        return None
    X = X[keep]
    y = y[keep]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                          random_state=0, stratify=y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(max_iter=2000, multi_class="auto",
                                 n_jobs=-1).fit(Xtr, ytr)
    return float(clf.score(Xte, yte))


def detection_jaccard_spearman(emb: np.ndarray, bags: list[list[int]],
                               max_pairs: int = 200_000) -> float | None:
    from scipy.stats import spearmanr

    n = emb.shape[0]
    rng = np.random.default_rng(0)
    pair_count = n * (n - 1) // 2
    if pair_count > max_pairs:
        # uniform random sample of pairs
        i = rng.integers(0, n, size=max_pairs)
        j = rng.integers(0, n, size=max_pairs)
        keep = i != j
        i = i[keep]
        j = j[keep]
    else:
        ij = np.asarray(np.triu_indices(n, k=1))
        i, j = ij[0], ij[1]
    cos = (emb[i] * emb[j]).sum(axis=1)  # already L2 normed
    sets_i = [set(bags[k]) for k in i]
    sets_j = [set(bags[k]) for k in j]
    jac = np.empty(len(i), dtype=np.float32)
    for k in range(len(i)):
        a, b = sets_i[k], sets_j[k]
        u = a | b
        jac[k] = len(a & b) / max(1, len(u)) if u else 0.0
    valid = (jac > 0) | (jac == 0)  # all
    if valid.sum() < 30:
        return None
    rho, _ = spearmanr(cos[valid], jac[valid])
    return float(rho)


# ---------------------------------------------------------------------------
# Augmentation invariance
# ---------------------------------------------------------------------------

def _augment_one(im_bgr: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    import cv2

    if kind == "brightness":
        factor = float(rng.uniform(0.65, 1.35))
        out = np.clip(im_bgr.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return out
    if kind == "blur":
        k = int(rng.choice([3, 5]))
        return cv2.GaussianBlur(im_bgr, (k, k), 0)
    if kind == "rotate":
        deg = float(rng.uniform(-8, 8))
        h, w = im_bgr.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
        return cv2.warpAffine(im_bgr, M, (w, h), borderValue=(114, 114, 114))
    if kind == "jpeg":
        q = int(rng.integers(35, 65))
        ok, enc = cv2.imencode(".jpg", im_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else im_bgr
    raise ValueError(kind)


def augmentation_invariance(weights: Path, paths: list[Path], imgsz: int,
                            n_imgs: int = 200, batch: int = 32) -> dict:
    """Mean cosine(orig, augmented) vs cross-image baseline."""
    import cv2

    rng = np.random.default_rng(0)
    sample = list(rng.choice(np.asarray(paths), size=min(n_imgs, len(paths)),
                             replace=False))
    raw = [cv2.imread(str(p)) for p in sample]  # variable HxW; we letterbox per-image
    kinds = ["brightness", "blur", "rotate", "jpeg"]

    model = load_wrapper(weights)

    def _embed_one_at_a_time(arrs: list[np.ndarray]) -> np.ndarray:
        # Images have different shapes; the public extractor accepts a single
        # ndarray (HWC) and letterboxes it for us, so we batch by stacking
        # only after letterbox. Easiest path: feed one-by-one.
        outs: list[np.ndarray] = []
        for im in arrs:
            e = get_embedding(model, im, imgsz=imgsz, device=DEVICE,
                              normalize=True)
            outs.append(e.cpu().numpy().astype(np.float32))
        return np.concatenate(outs, axis=0)

    orig = _embed_one_at_a_time(raw)
    self_sim = []
    for kind in kinds:
        aug = [_augment_one(im, kind, rng) for im in raw]
        e_aug = _embed_one_at_a_time(aug)
        # cosine between orig[i] and aug[i] (already unit-norm)
        self_sim.append((orig * e_aug).sum(axis=1).mean())
    self_sim_mean = float(np.mean(self_sim))

    # cross-image baseline: mean cosine between orig[i] and orig[j != i]
    sim = orig @ orig.T
    np.fill_diagonal(sim, np.nan)
    cross_sim_mean = float(np.nanmean(sim))

    return {
        "self_aug_cos_mean": self_sim_mean,
        "cross_image_cos_mean": cross_sim_mean,
        "invariance_margin": self_sim_mean - cross_sim_mean,
        "per_kind": {k: float(s) for k, s in zip(kinds, self_sim)},
    }


# ---------------------------------------------------------------------------
# Cross-model Linear CKA
# ---------------------------------------------------------------------------

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Centred Linear CKA between two [N, d_x] / [N, d_y] feature matrices
    (Kornblith et al. 2019). Symmetric, in [0, 1]."""
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    # ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    num = np.linalg.norm(Yc.T @ Xc, ord="fro") ** 2
    den = np.linalg.norm(Xc.T @ Xc, ord="fro") * np.linalg.norm(Yc.T @ Yc, ord="fro")
    return float(num / max(den, 1e-12))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgs-per-class", type=int, default=200)
    ap.add_argument("--aug-imgs", type=int, default=200,
                    help="number of images for the augmentation-invariance probe")
    ap.add_argument("--out-json", default=str(HERE / "yolo_perspective_results.json"))
    args = ap.parse_args()

    paths = collect_paths(IMAGENETTE_VAL, args.imgs_per_class)
    print(f"loaded {len(paths)} images")

    weights = sorted(WEIGHTS_DIR.glob("*.pt"))
    rows: dict[str, dict] = {}
    embed_cache: dict[str, np.ndarray] = {}

    for w in weights:
        sz = imgsz_for(w.stem)
        print(f"\n[{w.name}]  imgsz={sz}")
        t0 = time.perf_counter()
        emb = embed_all(w, paths, imgsz=sz, normalize=True)
        embed_cache[w.name] = emb
        print(f"  embed: shape={emb.shape}  {(time.perf_counter() - t0):.1f}s")

        info: dict = {"d": int(emb.shape[1]), "n": int(emb.shape[0])}

        if is_detection_model(w):
            t0 = time.perf_counter()
            top1, bag = detect_all(w, paths, imgsz=sz)
            print(f"  detect: top-1 set has {len({t for t in top1 if t is not None})} classes; "
                  f"avg objs/img = {np.mean([len(b) for b in bag]):.2f}; "
                  f"none-detected = {sum(1 for t in top1 if t is None)}; "
                  f"{(time.perf_counter() - t0):.1f}s")
            info["self_detection_nmi"] = self_detection_nmi(emb, top1)
            info["self_detection_probe_acc"] = self_detection_probe_acc(emb, top1)
            info["detection_jaccard_spearman"] = detection_jaccard_spearman(emb, bag)
        else:
            info["self_detection_nmi"] = None
            info["self_detection_probe_acc"] = None
            info["detection_jaccard_spearman"] = None

        # Augmentation invariance (works for any model that produces embedding)
        t0 = time.perf_counter()
        info["augmentation"] = augmentation_invariance(w, paths, imgsz=sz,
                                                       n_imgs=args.aug_imgs)
        print(f"  aug invariance margin = {info['augmentation']['invariance_margin']:.4f} "
              f"(self={info['augmentation']['self_aug_cos_mean']:.4f}, "
              f"cross={info['augmentation']['cross_image_cos_mean']:.4f}); "
              f"{(time.perf_counter() - t0):.1f}s")

        rows[w.name] = info

        # incremental snapshot in case of interruption
        with open(args.out_json, "w") as f:
            json.dump(rows, f, indent=2, default=lambda x: None)

    # ------------------------------------------------------------------
    # Cross-model Linear CKA
    # ------------------------------------------------------------------
    print("\n=== cross-model Linear CKA ===")
    names = sorted(embed_cache.keys())
    cka_mat = np.zeros((len(names), len(names)), dtype=np.float32)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j < i:
                cka_mat[i, j] = cka_mat[j, i]
                continue
            cka_mat[i, j] = linear_cka(embed_cache[a], embed_cache[b])
    rows["_cka"] = {
        "names": names,
        "matrix": cka_mat.tolist(),
    }

    with open(args.out_json, "w") as f:
        json.dump(rows, f, indent=2, default=lambda x: None)
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
