"""INT8 post-training quantization for trained YOLO workdirs.

Bakes the empirically-derived **Tier S** recipe from
``plans/YOLO-INT8-PTQ-CALIBRATION-RECIPE.md``:

* Calibrator: ``CalibrationMethod.Percentile`` with cutoff ``99.999``
  and ``CalibTensorRangeSymmetric=True``.
* Activations: symmetric ``QUInt8``; weights: per-channel symmetric
  ``QInt8``; ``reduce_range=True`` (mandatory on ORT 1.23 CPU EP).
* Quant format: ``QDQ``.
* Head exclusion: tail-60 of the ONNX graph filtered through a fixed
  set of non-Conv op types — covers v8/v9/v10/v11 (extended set is
  also harmless on v8/v11).
* Calibration data: ``N=128`` images sampled uniformly at random from
  the training pool, seeded for determinism.

The recipe was verified universally optimal across yolov8n / yolov8s /
yolov8m / yolov9s / yolo11n / yolo11s / yolov10n on COCO val2017
(95.7 %–98.8 % mAP50 retention vs FP32 ONNX) and on a private 4-class
finetuned dataset (96.0 % retention).

Outputs land under ``<workdir>/quant/``:

* ``int8/<name>_imgsz<sz>_int8.onnx`` — the deployable artifact
* ``onnx/<name>_imgsz<sz>_fp32.onnx`` and ``..._pre.onnx`` — the
  intermediate FP32 / pre-processed graphs (kept for re-quantization
  without re-exporting)
* ``calib_lists/random<N>_seed<S>.txt`` — the deterministic calib list
* ``eval.json`` — INT8 mAP50 / mAP50-95 / P / R / speed on the val
  split
* ``threshold.json`` — F1-vs-confidence curve recomputed from the INT8
  validator metrics (so the deployed model has its own optimal
  confidence threshold, distinct from the FP32 one)
* ``quant_args.json`` — exact recipe knobs used (for reproducibility)

The INT8 ONNX itself is fully self-describing — its
``metadata_props`` carries every standard ultralytics key plus a
``dghs.yolov8.quant.*`` namespace with the recipe and the
INT8-specific eval / threshold payload.

CLI::

    python -m yolov8.quantize -w runs/<task_name>

Library::

    >>> from yolov8.quantize import quantize_workdir
    >>> int8_path = quantize_workdir("runs/yolo_full_4label_yolo11n")
"""
from __future__ import annotations

import json
import os
import random
import shutil
import sys
import tempfile
import time
from functools import partial
from pathlib import Path
from typing import Any, List, Mapping, Optional

import click
import numpy as np
import onnx
import yaml
from ditk import logging
from ultralytics import YOLO, RTDETR

from .onnx import export_yolo_to_onnx
from .onnx._metadata import collect_metadata, attach_metadata
from .utils import (
    GLOBAL_CONTEXT_SETTINGS,
    compute_threshold_data,
    derive_model_meta_from_path,
)
from .utils import print_version as _origin_print_version


# ---------------------------------------------------------------------------
# Tier S recipe constants (do NOT change without re-running §7 experiments)
# ---------------------------------------------------------------------------

#: Number of trailing graph nodes scanned when picking the head-exclude set.
#: 60 covers v10's NMS-free postproc (TopK / GatherElements / GatherND /
#: Mod / Equal / ...); harmless on v8/v9/v11 whose tail is shorter.
HEAD_TAIL_K: int = 60

#: Op types we always keep in FP32 within the tail window. Mix of v8/v11
#: standard ops (Sigmoid/Softmax/Concat/Split/Reshape/Transpose/Sub/Add/
#: Div/Mul) plus v10's NMS-free postproc (TopK/GatherElements/GatherND/
#: ReduceMax/Tile/Unsqueeze/Sign/Equal/Not/Mod/Cast/And).
HEAD_EXCLUDE_OPS: frozenset = frozenset({
    'Sigmoid', 'Softmax', 'Concat', 'Split', 'Reshape', 'Transpose',
    'Sub', 'Add', 'Div', 'Mul',
    'TopK', 'GatherElements', 'GatherND', 'ReduceMax', 'Tile',
    'Unsqueeze', 'Sign', 'Equal', 'Not', 'Mod', 'Cast', 'And',
})

#: Tier S calibrator config knobs.
TIER_S = {
    'calibrator': 'Percentile',
    'percentile': 99.999,
    'symmetric': True,
    'per_channel': True,
    'reduce_range': True,
    'quant_format': 'QDQ',
    'weight_type': 'QInt8',
    'activation_type': 'QUInt8',
    'recipe_name': 'TierS-Percentile99999-sym',
}

#: Default calibration N (128 — N=500 buys <0.5 pp at 4× the cost).
DEFAULT_CALIB_N: int = 128
DEFAULT_CALIB_SEED: int = 0


# ---------------------------------------------------------------------------
# Data resolution: workdir → train/val image paths
# ---------------------------------------------------------------------------

class QuantizationConfigError(RuntimeError):
    """Raised when the workdir does not point at a usable dataset.

    Either ``args.yaml`` is missing, ``data:`` is empty, the resolved
    yaml does not exist, or the ``train``/``val`` paths inside don't
    resolve to a directory of images. The message names the missing
    artifact so the user can rerun with an explicit ``--data`` (or
    ``--train-images`` / ``--val-data``) override.
    """


def _read_args_yaml(workdir: Path) -> dict:
    """Load ``<workdir>/args.yaml`` and return its parsed dict.

    :raises QuantizationConfigError: if the file is missing or empty.
    """
    p = workdir / 'args.yaml'
    if not p.is_file():
        raise QuantizationConfigError(
            f'workdir args.yaml not found at {p!s} — '
            f'pass --data /path/to/data.yaml explicitly.'
        )
    with open(p, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise QuantizationConfigError(
            f'args.yaml at {p!s} did not parse to a dict.'
        )
    return cfg


def _resolve_split(data_yaml: Path, split: str) -> Path:
    """Resolve a ``train`` or ``val`` split entry to an absolute images dir.

    Defers to ultralytics' :func:`ultralytics.data.utils.check_det_dataset`
    so we accept exactly the same yaml dialects the trainer accepts —
    including the quirky case where ``train: ../train/images`` ends up
    pointing at ``<path>/train/images`` rather than the literal
    ``..``-relative dir.

    The split value can be either a directory of images or a ``.txt``
    manifest of image paths; in the latter case we walk to the parent
    of the first listed file and return that directory.

    :raises QuantizationConfigError: if the split cannot be resolved
        to an existing directory.
    """
    try:
        from ultralytics.data.utils import check_det_dataset
    except Exception as e:  # pragma: no cover
        raise QuantizationConfigError(
            f'ultralytics yaml resolver unavailable ({e}); '
            f'pass --train-images explicitly.'
        )
    info = check_det_dataset(str(data_yaml))
    val = info.get(split)
    if val is None:
        raise QuantizationConfigError(
            f'{data_yaml!s} has no resolvable "{split}:" entry.'
        )
    cand = Path(val)
    if cand.is_dir():
        return cand
    if cand.is_file() and cand.suffix == '.txt':
        with open(cand, 'r', encoding='utf-8') as f:
            first = next((ln.strip() for ln in f if ln.strip()), None)
        if first:
            base = Path(info.get('path') or data_yaml.parent)
            first_p = (base / first).resolve() \
                if not Path(first).is_absolute() else Path(first)
            if first_p.exists():
                return first_p.parent
    raise QuantizationConfigError(
        f'cannot resolve {split!r} to an images directory for {data_yaml!s} '
        f'(check_det_dataset returned {val!r}). '
        f'Pass --train-images explicitly.'
    )


def _list_images(images_dir: Path) -> List[Path]:
    """Return all common-format image files in ``images_dir`` (recursive
    over the immediate subtree, sorted for determinism)."""
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
    out: list[Path] = []
    for ext in exts:
        out.extend(images_dir.glob(ext))
    return sorted(out)


def build_calib_list(images_dir: Path, n: int, seed: int,
                     out_path: Path) -> List[Path]:
    """Sample ``n`` images uniformly at random from ``images_dir`` and
    write the absolute paths to ``out_path``. Deterministic given seed.

    :raises QuantizationConfigError: if fewer than ``n`` images exist.
    """
    pool = _list_images(images_dir)
    if len(pool) < n:
        raise QuantizationConfigError(
            f'only {len(pool)} images at {images_dir!s}; need at least {n}.'
        )
    rng = random.Random(seed)
    picked = rng.sample(pool, n)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(str(p) for p in picked) + '\n',
                        encoding='utf-8')
    return picked


# ---------------------------------------------------------------------------
# Pre-processing for ONNX RT calibrator: letterbox + BGR→RGB + /255 + NCHW
# ---------------------------------------------------------------------------

def _letterbox(im_bgr, sz: int = 640, color=(114, 114, 114)):
    import cv2  # local import: optional dep at call site
    h, w = im_bgr.shape[:2]
    r = min(sz / h, sz / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    if (nh, nw) != (h, w):
        im_bgr = cv2.resize(im_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (sz - nh) // 2
    bot = sz - nh - top
    left = (sz - nw) // 2
    right = sz - nw - left
    return cv2.copyMakeBorder(im_bgr, top, bot, left, right,
                              cv2.BORDER_CONSTANT, value=color)


def _preprocess(path: Path, sz: int = 640) -> np.ndarray:
    import cv2
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(_letterbox(bgr, sz),
                       cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(rgb, (2, 0, 1))[None, ...]


# ---------------------------------------------------------------------------
# ONNX-side helpers
# ---------------------------------------------------------------------------

def collect_head_excludes(pre_onnx_path: Path) -> List[str]:
    """Return the names of nodes inside the trailing ``HEAD_TAIL_K``
    window whose ``op_type`` is in :data:`HEAD_EXCLUDE_OPS` — these
    will be skipped by the QDQ inserter.

    :param pre_onnx_path: Path to the *pre-processed* ONNX (output of
        :func:`onnxruntime.quantization.shape_inference.quant_pre_process`).
    """
    proto = onnx.load(str(pre_onnx_path))
    nodes = list(proto.graph.node)
    tail = nodes[-HEAD_TAIL_K:]
    return [n.name for n in tail if n.op_type in HEAD_EXCLUDE_OPS]


def _read_metadata_props(onnx_path: Path) -> dict[str, str]:
    """Read ``metadata_props`` from an ONNX file as a plain dict."""
    proto = onnx.load(str(onnx_path))
    return {p.key: p.value for p in proto.metadata_props}


def _read_names_for_threshold(best_pt: Path, workdir: Path) -> dict:
    """Resolve a ``{class_idx: label}`` dict without triggering
    ult's ``YOLO.names`` property (which lazy-inits a predictor and
    crashes on hosts where CUDA enumeration is misconfigured).

    Order: (1) ``<workdir>/labels.json`` if present; (2) the inner
    ``BaseModel``'s ``.names`` attribute on the trained ``best.pt``,
    accessed via ``YOLO(best_pt).model.names`` which doesn't go
    through the predictor; (3) empty dict.
    """
    labels_p = workdir / 'labels.json'
    if labels_p.is_file():
        try:
            with open(labels_p, 'r', encoding='utf-8') as f:
                arr = json.load(f)
            if isinstance(arr, list):
                return {i: str(s) for i, s in enumerate(arr)}
            if isinstance(arr, dict):
                return {int(k): str(v) for k, v in arr.items()}
        except Exception:
            pass
    try:
        wrapper = YOLO(str(best_pt))
        inner = getattr(wrapper, 'model', None)
        if inner is not None and hasattr(inner, 'names') and inner.names:
            return {int(k): str(v) for k, v in dict(inner.names).items()}
    except Exception:
        pass
    return {}


# ---------------------------------------------------------------------------
# Tier S quantization (the actual quantize_static call)
# ---------------------------------------------------------------------------

def _tier_s_quantize(pre_onnx: Path, calib_paths: List[Path],
                     int8_out: Path, imgsz: int) -> None:
    """Run ``quantize_static`` with the Tier S configuration.

    Side-effects: writes a fresh INT8 ONNX to ``int8_out``. Uses ORT
    CPU EP for calibration. Hardcodes the recipe knobs documented in
    :data:`TIER_S` — DO NOT widen them in this call site; if a caller
    wants a different recipe, write a new function.
    """
    from onnxruntime.quantization import (
        CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.registry import (
        QDQRegistry, QLinearOpsRegistry,
    )
    import onnxruntime as ort

    sess = ort.InferenceSession(str(pre_onnx),
                                providers=['CPUExecutionProvider'])
    in_name = sess.get_inputs()[0].name
    del sess

    excluded = collect_head_excludes(pre_onnx)
    op_types = sorted(set(QLinearOpsRegistry) | set(QDQRegistry))

    class _Reader(CalibrationDataReader):
        def __init__(self, ps):
            self._it = iter(ps)

        def get_next(self):
            try:
                return {in_name: _preprocess(next(self._it), imgsz)}
            except StopIteration:
                return None

    quantize_static(
        str(pre_onnx), str(int8_out),
        calibration_data_reader=_Reader(calib_paths),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        reduce_range=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        calibrate_method=CalibrationMethod.Percentile,
        nodes_to_exclude=excluded,
        op_types_to_quantize=op_types,
        extra_options={
            'CalibPercentile': 99.999,
            'CalibTensorRangeSymmetric': True,
            'ActivationSymmetric': True,
            'WeightSymmetric': True,
        },
    )


# ---------------------------------------------------------------------------
# FP32 export + pre-process (with v10/v12 NMS-free fallback)
# ---------------------------------------------------------------------------

def _export_fp32_onnx(model_pt: Path, out_path: Path,
                      imgsz: int, opset_version: int) -> None:
    """Export ``model_pt`` to a non-dynamic FP32 ONNX at ``out_path``."""
    if out_path.is_file():
        return
    model_type, _ = derive_model_meta_from_path(model_pt)
    Wrapper = RTDETR if model_type == 'rtdetr' else YOLO
    wrapper = Wrapper(str(model_pt))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_yolo_to_onnx(
        wrapper, str(out_path),
        imgsz=imgsz, opset_version=opset_version,
        simplify=True, dynamic=False,
    )


def _pre_process(fp32_onnx: Path, pre_out: Path) -> None:
    """Run ``onnxruntime.quantization.shape_inference.quant_pre_process``.
    Falls back to ``skip_symbolic_shape=True`` for v10's NMS-free TopK
    that breaks ORT's symbolic shape inferencer.
    """
    from onnxruntime.quantization.shape_inference import quant_pre_process

    if pre_out.is_file():
        return
    pre_out.parent.mkdir(parents=True, exist_ok=True)
    try:
        quant_pre_process(str(fp32_onnx), str(pre_out),
                          skip_optimization=False,
                          skip_onnx_shape=False,
                          skip_symbolic_shape=False,
                          auto_merge=False)
    except Exception as e:
        logging.info(
            f'symbolic shape inference failed ({e}); '
            f'retrying with skip_symbolic_shape=True'
        )
        quant_pre_process(str(fp32_onnx), str(pre_out),
                          skip_symbolic_shape=True)


# ---------------------------------------------------------------------------
# Validation + threshold extraction (post-quantization)
# ---------------------------------------------------------------------------

def _validate_int8(int8_path: Path, data_yaml: Path, imgsz: int,
                   names: Optional[dict] = None
                   ) -> tuple[dict, Optional[dict]]:
    """Run ``model.val(...)`` on the INT8 ONNX and return
    ``(eval_dict, threshold_dict)``.

    ``threshold_dict`` is computed from the freshly-populated
    ``model.metrics`` via the lower-level :func:`_payload_from_metrics`
    so we never trigger ult's ``model.names`` property — that property
    lazy-initialises a predictor with device autodetect, which crashes
    on hosts where ``CUDA_VISIBLE_DEVICES`` masks devices that the
    runtime still sees.

    :param int8_path: INT8 ONNX to evaluate.
    :param data_yaml: dataset yaml; ``model.val`` reads splits from it.
    :param imgsz: square input size.
    :param names: optional ``{class_idx: label}`` dict; used for the
        per-class breakdown in the threshold payload. Defaults to
        empty dict (legacy ``f1_score``/``threshold`` keys still get
        filled).
    :returns: ``(eval_payload, threshold_payload)``.
        ``threshold_payload`` is ``None`` on ultralytics <8.1 (no
        ``f1_curve`` stored on the metric).
    """
    from .utils.threshold import _payload_from_metrics  # noqa: PLC0415

    model = YOLO(str(int8_path))
    metrics = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        device='cpu',
        batch=1,
        plots=False,
        verbose=False,
    )
    eval_payload = {
        'mAP50': float(metrics.box.map50),
        'mAP50_95': float(metrics.box.map),
        'precision': float(metrics.box.mp),
        'recall': float(metrics.box.mr),
        'speed_ms_per_img': dict(getattr(metrics, 'speed', {}) or {}),
    }
    # Use the lower-level helper directly so we can supply names
    # without touching the model.names property (which would trigger
    # predictor / device setup and fail on misconfigured CUDA hosts).
    threshold_payload = _payload_from_metrics(
        metrics, kind='box', names=names or {})
    return eval_payload, threshold_payload


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def quantize_workdir(
    workdir: str | os.PathLike,
    *,
    name: Optional[str] = None,
    data: Optional[str | os.PathLike] = None,
    train_images: Optional[str | os.PathLike] = None,
    calib_n: int = DEFAULT_CALIB_N,
    calib_seed: int = DEFAULT_CALIB_SEED,
    imgsz: Optional[int] = None,
    opset_version: int = 14,
    eval: bool = True,
    force: bool = False,
) -> Path:
    """Quantize the trained model in ``workdir`` to INT8 with the
    Tier S recipe, write all artifacts under ``<workdir>/quant/``,
    and return the path to the produced INT8 ONNX.

    :param workdir: Training workdir containing
        ``weights/best.pt`` and ``args.yaml``.
    :type workdir: str or os.PathLike
    :param name: Stem used for produced filenames; defaults to the
        workdir basename.
    :type name: str or None
    :param data: Override path to the dataset yaml. When ``None``,
        read from ``workdir/args.yaml`` ``data:`` field.
    :type data: str or os.PathLike or None
    :param train_images: Override path to the directory of training
        images used as the calibration pool. When ``None``, resolved
        from the dataset yaml's ``train:`` entry.
    :type train_images: str or os.PathLike or None
    :param calib_n: Number of calibration images. ``128`` is Tier S
        default and empirically sufficient.
    :type calib_n: int
    :param calib_seed: RNG seed for ``random.Random(seed).sample(...)``.
    :type calib_seed: int
    :param imgsz: Square input size. ``None`` = read from ``args.yaml``,
        falling back to ``640``.
    :type imgsz: int or None
    :param opset_version: ONNX opset for the FP32 export step.
    :type opset_version: int
    :param eval: When ``True`` (the default), runs ``model.val(...)`` on
        the INT8 ONNX and writes ``eval.json`` + ``threshold.json``.
        Setting ``False`` skips the val pass — useful when the user
        wants to rebuild only the INT8 graph fast and re-eval later.
    :type eval: bool
    :param force: When ``True``, ignore any existing
        ``quant/int8/*_int8.onnx`` and re-run the full pipeline. Default
        ``False`` reuses an existing artifact (calibration is the most
        expensive step at ~3-10 min depending on model size; cheap to
        skip).
    :type force: bool
    :returns: Absolute path to the produced INT8 ONNX.
    :rtype: pathlib.Path
    :raises QuantizationConfigError: if the workdir cannot be resolved
        to a usable dataset and no ``--data`` / ``--train-images``
        override was supplied.

    Example::

        >>> from yolov8.quantize import quantize_workdir
        >>> path = quantize_workdir("runs/yolo_full_4label_yolo11n")
        >>> path.name.endswith("_int8.onnx")
        True
    """
    workdir = Path(workdir).resolve()
    name = name or workdir.name
    quant_dir = workdir / 'quant'
    onnx_dir = quant_dir / 'onnx'
    int8_dir = quant_dir / 'int8'
    calib_dir = quant_dir / 'calib_lists'
    quant_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir.mkdir(parents=True, exist_ok=True)
    int8_dir.mkdir(parents=True, exist_ok=True)

    # ---- resolve config from args.yaml ----
    args_yaml = _read_args_yaml(workdir)
    if imgsz is None:
        imgsz = int(args_yaml.get('imgsz') or 640)

    if data is None:
        data = args_yaml.get('data')
        if not data:
            raise QuantizationConfigError(
                f'workdir args.yaml has no "data:" field — pass --data explicitly.'
            )
    data_yaml = Path(data).resolve()
    if not data_yaml.is_file():
        raise QuantizationConfigError(
            f'dataset yaml not found at {data_yaml!s}; '
            f'pass --data /path/to/data.yaml.'
        )

    if train_images is None:
        train_dir = _resolve_split(data_yaml, 'train')
    else:
        train_dir = Path(train_images).resolve()
        if not train_dir.is_dir():
            raise QuantizationConfigError(
                f'--train-images {train_dir!s} is not a directory.'
            )

    # ---- check checkpoint ----
    best_pt = workdir / 'weights' / 'best.pt'
    if not best_pt.is_file():
        raise QuantizationConfigError(
            f'{best_pt!s} not found.'
        )

    fp32_onnx = onnx_dir / f'{name}_imgsz{imgsz}_fp32.onnx'
    pre_onnx = onnx_dir / f'{name}_imgsz{imgsz}_pre.onnx'
    int8_onnx = int8_dir / f'{name}_imgsz{imgsz}_int8.onnx'
    calib_list_path = calib_dir / f'random{calib_n}_seed{calib_seed}.txt'
    eval_json = quant_dir / 'eval.json'
    threshold_json = quant_dir / 'threshold.json'
    quant_args_json = quant_dir / 'quant_args.json'

    logging.info(f'[quantize] workdir={workdir!s}, imgsz={imgsz}, '
                 f'calib_n={calib_n}, calib_seed={calib_seed}')

    # The expensive step is calibration (~3-10 min). Skip it if the
    # artifact already exists, unless the caller passed ``force=True``.
    # The eval / threshold / metadata / args.json steps below are
    # cheap (~5 min for val on a small val split, sub-second for
    # everything else) and are re-run unless their own outputs exist
    # — the per-step ``.is_file()`` checks already gate them.
    quantize_done = int8_onnx.is_file() and not force

    if quantize_done:
        logging.info(f'[quantize] reusing INT8 ONNX: {int8_onnx!s}')
    else:
        # ---- 1. FP32 ONNX export (uses yolov8.onnx with full metadata) ----
        t0 = time.perf_counter()
        _export_fp32_onnx(best_pt, fp32_onnx, imgsz, opset_version)
        logging.info(f'[quantize] fp32 onnx ready: {fp32_onnx!s} '
                     f'({fp32_onnx.stat().st_size/1e6:.1f} MB) '
                     f'in {time.perf_counter()-t0:.1f}s')

        # ---- 2. quant_pre_process (with v10 fallback) ----
        t0 = time.perf_counter()
        _pre_process(fp32_onnx, pre_onnx)
        logging.info(f'[quantize] pre-processed onnx: {pre_onnx!s} '
                     f'in {time.perf_counter()-t0:.1f}s')

        # ---- 3. build random calibration list ----
        calib_paths = build_calib_list(train_dir, calib_n, calib_seed,
                                       calib_list_path)
        logging.info(f'[quantize] calibration list: {len(calib_paths)} '
                     f'images -> {calib_list_path!s}')

        # ---- 4. Tier S quantize_static ----
        t0 = time.perf_counter()
        _tier_s_quantize(pre_onnx, calib_paths, int8_onnx, imgsz)
        logging.info(f'[quantize] INT8 onnx: {int8_onnx!s} '
                     f'({int8_onnx.stat().st_size/1e6:.1f} MB) '
                     f'in {time.perf_counter()-t0:.1f}s')

    # FP32 onnx is needed for metadata propagation below even on the
    # reuse path; build it lazily if it was wiped between runs.
    if not fp32_onnx.is_file():
        _export_fp32_onnx(best_pt, fp32_onnx, imgsz, opset_version)
    if not pre_onnx.is_file():
        _pre_process(fp32_onnx, pre_onnx)

    # ---- 5. evaluate INT8 + recompute threshold ----
    eval_payload: Optional[dict] = None
    threshold_payload: Optional[dict] = None
    if eval:
        eval_done = (eval_json.is_file() and threshold_json.is_file()
                     and not force)
        if eval_done:
            with open(eval_json, 'r', encoding='utf-8') as f:
                eval_payload = json.load(f)
            with open(threshold_json, 'r', encoding='utf-8') as f:
                threshold_payload = json.load(f)
            logging.info(f'[quantize] reusing eval/threshold JSONs: '
                         f'INT8 mAP50={eval_payload["mAP50"]:.4f}')
        else:
            # Resolve names from <workdir>/labels.json or the .pt's
            # inner BaseModel — never via ult's YOLO.names property
            # (lazy predictor init crashes on misconfigured CUDA hosts).
            names = _read_names_for_threshold(best_pt, workdir)
            t0 = time.perf_counter()
            eval_payload, threshold_payload = _validate_int8(
                int8_onnx, data_yaml, imgsz, names=names)
            logging.info(f'[quantize] INT8 mAP50={eval_payload["mAP50"]:.4f}'
                         f' mAP50-95={eval_payload["mAP50_95"]:.4f} '
                         f'in {time.perf_counter()-t0:.1f}s')
            with open(eval_json, 'w', encoding='utf-8') as f:
                json.dump(eval_payload, f, ensure_ascii=False, indent=4)
            if threshold_payload is not None:
                with open(threshold_json, 'w', encoding='utf-8') as f:
                    json.dump(threshold_payload, f, ensure_ascii=False, indent=4)

    # ---- 6. write recipe-config sidecar ----
    quant_args = {
        **TIER_S,
        'calib_n': calib_n,
        'calib_seed': calib_seed,
        'head_tail_k': HEAD_TAIL_K,
        'head_exclude_op_types': sorted(HEAD_EXCLUDE_OPS),
        'imgsz': imgsz,
        'opset_version': opset_version,
        'fp32_source': str(best_pt),
        'data_yaml': str(data_yaml),
        'train_images_dir': str(train_dir),
    }
    with open(quant_args_json, 'w', encoding='utf-8') as f:
        json.dump(quant_args, f, ensure_ascii=False, indent=4)

    # ---- 7. attach metadata to INT8 ONNX ----
    # Carry over the FP32 ONNX's metadata (which yolov8.onnx already
    # populated with the standard ult keys + dghs.yolov8.* namespace)
    # and add quant-specific keys + the INT8-specific threshold.
    md = _read_metadata_props(fp32_onnx)
    md['dghs.yolov8.exporter'] = 'yolov8.quantize'
    md['dghs.yolov8.quant.recipe'] = TIER_S['recipe_name']
    md['dghs.yolov8.quant.calibrator'] = TIER_S['calibrator']
    md['dghs.yolov8.quant.calib_percentile'] = str(TIER_S['percentile'])
    md['dghs.yolov8.quant.symmetric'] = '1' if TIER_S['symmetric'] else '0'
    md['dghs.yolov8.quant.per_channel'] = '1' if TIER_S['per_channel'] else '0'
    md['dghs.yolov8.quant.reduce_range'] = '1' if TIER_S['reduce_range'] else '0'
    md['dghs.yolov8.quant.quant_format'] = TIER_S['quant_format']
    md['dghs.yolov8.quant.calib_method'] = 'random'
    md['dghs.yolov8.quant.calib_n'] = str(calib_n)
    md['dghs.yolov8.quant.calib_seed'] = str(calib_seed)
    md['dghs.yolov8.quant.head_tail_k'] = str(HEAD_TAIL_K)
    md['dghs.yolov8.quant.head_exclude_count'] = str(
        len(collect_head_excludes(pre_onnx)))
    if eval_payload is not None:
        md['dghs.yolov8.quant.eval'] = json.dumps(eval_payload)
    # INT8-specific threshold supersedes the FP32 one in this graph's
    # metadata namespace; the FP32 threshold lives in the FP32 ONNX
    # and in <workdir>/threshold.json.
    if threshold_payload is not None:
        md['dghs.yolov8.threshold'] = json.dumps(threshold_payload,
                                                  ensure_ascii=False)
    attach_metadata(str(int8_onnx), md)
    logging.info(f'[quantize] attached metadata ({len(md)} keys) to '
                 f'{int8_onnx!s}')

    return int8_onnx


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

print_version = partial(_origin_print_version, 'quantize')


@click.command(context_settings={**GLOBAL_CONTEXT_SETTINGS})
@click.option('-v', '--version', is_flag=True,
              callback=print_version, expose_value=False, is_eager=True,
              help='Tier S INT8 PTQ for a trained workdir.')
@click.option('--workdir', '-w', 'workdir',
              type=click.Path(file_okay=False, exists=True), required=True,
              help='Training workdir (must contain weights/best.pt + args.yaml).',
              show_default=True)
@click.option('--name', '-n', 'name', type=str, default=None,
              help='Stem used in output filenames. Defaults to workdir basename.',
              show_default=True)
@click.option('--data', '-d', 'data', type=click.Path(dir_okay=False),
              default=None,
              help='Override dataset yaml path (otherwise read from args.yaml).',
              show_default=True)
@click.option('--train-images', 'train_images',
              type=click.Path(file_okay=False), default=None,
              help='Override the directory of training images used for calibration.',
              show_default=True)
@click.option('--calib-n', 'calib_n', type=int, default=DEFAULT_CALIB_N,
              help='Calibration set size. 128 is Tier S default; >500 not advised.',
              show_default=True)
@click.option('--calib-seed', 'calib_seed', type=int, default=DEFAULT_CALIB_SEED,
              help='RNG seed for the random calibration sampler.',
              show_default=True)
@click.option('--imgsz', 'imgsz', type=int, default=None,
              help='Square input size. Default: read from args.yaml (else 640).',
              show_default=True)
@click.option('--opset-version', 'opset_version', type=int, default=14,
              help='ONNX opset for the FP32 export step.', show_default=True)
@click.option('--no-eval', 'no_eval', is_flag=True, default=False,
              help='Skip the post-quantization val pass and threshold recompute.')
@click.option('--force', 'force', is_flag=True, default=False,
              help='Re-run quantization even if an INT8 ONNX already exists '
                   'in <workdir>/quant/int8/ (default: reuse).')
def cli(workdir: str, name: Optional[str], data: Optional[str],
        train_images: Optional[str], calib_n: int, calib_seed: int,
        imgsz: Optional[int], opset_version: int, no_eval: bool,
        force: bool):
    """Click entry point: quantize a workdir to INT8 with Tier S.

    All artifacts land under ``<workdir>/quant/``::

        quant/onnx/<name>_imgsz<sz>_fp32.onnx
        quant/onnx/<name>_imgsz<sz>_pre.onnx
        quant/int8/<name>_imgsz<sz>_int8.onnx          <-- deployable
        quant/calib_lists/random<N>_seed<S>.txt
        quant/eval.json
        quant/threshold.json
        quant/quant_args.json

    Example::

        >>> # CLI form (preferred):
        >>> #   python -m yolov8.quantize -w runs/yolo_full_4label_yolo11n
        >>> from yolov8.quantize import cli  # click callback
    """
    logging.try_init_root(logging.INFO)
    try:
        out = quantize_workdir(
            workdir, name=name, data=data, train_images=train_images,
            calib_n=calib_n, calib_seed=calib_seed, imgsz=imgsz,
            opset_version=opset_version, eval=not no_eval, force=force,
        )
    except QuantizationConfigError as e:
        click.echo(f'error: {e}', err=True)
        sys.exit(1)
    click.echo(f'INT8 ONNX written to {out!s}')


if __name__ == '__main__':
    cli()
