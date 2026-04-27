"""Pull F1-vs-confidence threshold information out of a freshly-trained
Ultralytics model.

Replaces the previous "render F1_curve.png and OCR its title" hack: the
arrays the plot is drawn from (``metrics.box.f1_curve`` and friends,
shape ``(num_classes, 1000)``, plus the confidence axis ``metrics.box.px``)
are still attached to the model's trainer/validator after
``model.train(...)`` returns. Reading them directly is exact (full float
precision) and gives us per-class data the OCR path threw away.

Three flavours of the same extraction are exposed:

* :func:`compute_threshold_data` — wrapper-flavoured. Reads
  ``model.trainer.validator.metrics`` and uses ``model.names``. Right
  call site after ``model.train()`` returns.
* :func:`compute_threshold_data_from_trainer` — trainer-flavoured.
  Reads ``trainer.validator.metrics`` directly and uses
  ``trainer.data['names']`` / ``trainer.model.names``. Right entry
  point inside an ``on_train_end`` callback — only place guaranteed to
  fire when patience exhausts mid-training, when training time-budget
  runs out, or when ultralytics is doing DDP via ``subprocess.run`` (in
  which case ``model.trainer`` in the main process never sees populated
  metrics).
* :func:`compute_threshold_data_from_validator_stats` — fallback used
  on ultralytics <8.1 where the ``Metric`` class doesn't store
  ``f1_curve`` / ``px`` (the data is computed inside
  ``ap_per_class`` and discarded). Recomputes the curves from the
  validator's accumulated ``stats`` list — same algorithm ultralytics
  uses internally for ``F1_curve.png``.

The serialised payload is forward/back-compatible: ``f1_score`` and
``threshold`` keep the legacy semantic (the aggregate "all classes"
best F1 and its argmax confidence — exactly what the OCR pulled out of
the plot title), and per-class info is added under a new ``per_class``
key that legacy readers can ignore.
"""
from typing import Any, Optional

import numpy as np

from ditk import logging


_OLD_ULT_THRESHOLD_HINT = (
    "ultralytics <8.1 does not store f1_curve / px on the Metric class, "
    "so threshold.json cannot be derived from the validator's metrics "
    "object. The fallback path needs validator.stats; if that is also "
    "empty (e.g. the trainer was wrapped or the validator was overridden) "
    "the only fix is to upgrade ultralytics to >=8.1, where Metric.update "
    "stores the curves directly."
)


def _payload_from_curves(f1_curve: np.ndarray,
                         px: np.ndarray,
                         ap_class_index: Optional[np.ndarray],
                         names: Optional[dict]) -> Optional[dict]:
    """Assemble the threshold-payload dict from already-computed curves.

    Shared between the three top-level entry points so all three
    produce a bit-identical JSON. Returns ``None`` when the inputs are
    empty.

    :param f1_curve: ``[num_classes, 1000]`` per-class F1 vs confidence.
    :type f1_curve: numpy.ndarray
    :param px: ``[1000]`` confidence axis.
    :type px: numpy.ndarray
    :param ap_class_index: Indices of classes that actually have data,
        same length as ``f1_curve``'s first axis. ``None`` skips the
        per-class breakdown.
    :type ap_class_index: numpy.ndarray or None
    :param names: ``{class_id: str}`` mapping for human-readable labels
        in the per-class breakdown. Defaults to stringified ints.
    :type names: dict or None
    :returns: A dict with ``f1_score`` / ``threshold`` (and optional
        ``per_class``) or ``None`` if the inputs are empty.
    :rtype: dict or None
    """
    f1_curve = np.asarray(f1_curve)
    px = np.asarray(px)
    if f1_curve.size == 0 or px.size == 0:
        return None

    mean_f1 = f1_curve.mean(axis=0)
    best_idx = int(mean_f1.argmax())
    payload: dict = {
        'f1_score': float(mean_f1[best_idx]),
        'threshold': float(px[best_idx]),
    }
    names = names or {}
    if ap_class_index is not None and len(ap_class_index) == f1_curve.shape[0]:
        per_class: dict = {}
        for k, cls_idx in enumerate(ap_class_index):
            row = f1_curve[k]
            j = int(row.argmax())
            label = names.get(int(cls_idx), str(int(cls_idx)))
            per_class[label] = {
                'f1_score': float(row[j]),
                'threshold': float(px[j]),
            }
        if per_class:
            payload['per_class'] = per_class
    return payload


def _payload_from_metrics(metrics: Any, *, kind: str, names: Optional[dict]) -> Optional[dict]:
    """Read pre-computed curves off a ``DetMetrics`` / ``SegmentMetrics``-like
    object, the path used by ultralytics >=8.1 where ``Metric.update``
    stores ``f1_curve`` / ``px`` as attributes.

    :param metrics: ``DetMetrics`` / ``SegmentMetrics``-style object
        with a ``.box`` (and optional ``.seg``) Metric subobject.
    :type metrics: ultralytics.utils.metrics.DetMetrics or
        ultralytics.utils.metrics.SegmentMetrics
    :param kind: ``'box'`` or ``'seg'``. Falls back to ``'box'`` if the
        requested attribute is missing.
    :type kind: str
    :param names: ``{class_id: str}`` mapping for the per-class
        breakdown.
    :type names: dict or None
    :returns: Payload dict, or ``None`` when the curves aren't stored
        on this Metric class (i.e. ultralytics is too old).
    :rtype: dict or None
    """
    src = getattr(metrics, kind, None) or getattr(metrics, 'box', None)
    if src is None:
        return None
    f1_curve = getattr(src, 'f1_curve', None)
    px = getattr(src, 'px', None)
    if f1_curve is None or px is None:
        return None
    ap_class_index = getattr(src, 'ap_class_index', None)
    return _payload_from_curves(f1_curve, px, ap_class_index, names)


def compute_threshold_data_from_validator_stats(validator: Any, *,
                                                names: Optional[dict] = None
                                                ) -> Optional[dict]:
    """Recompute the threshold payload from a validator's stored ``stats``.

    Ultralytics <8.1 computes the F1 curve inside
    :func:`ap_per_class` and discards it after picking the
    argmax-of-mean point. The raw stats — ``(tp, conf, pred_cls,
    target_cls)`` — are still in ``validator.stats`` after a validation
    pass, so we re-derive ``f1_curve`` / ``px`` ourselves using the
    same arithmetic ultralytics uses internally.

    :param validator: Object exposing a ``stats`` attribute as
        ultralytics' ``DetectionValidator`` does (a list of per-batch
        tuples).
    :type validator: object
    :param names: ``{class_id: str}`` mapping. Defaults to stringified
        ints when ``None``.
    :type names: dict or None
    :returns: Payload dict, or ``None`` if ``validator.stats`` is empty
        / mis-shaped or the resulting per-class arrays are degenerate.
    :rtype: dict or None

    Example::

        >>> # Inside an on_train_end callback when ultralytics is 8.0:
        >>> from yolov8.utils import compute_threshold_data_from_validator_stats
        >>> data = compute_threshold_data_from_validator_stats(
        ...     trainer.validator, names=trainer.data.get("names"))
        >>> assert data is None or "threshold" in data
    """
    import torch as _torch  # validator stats may be torch tensors

    stats = getattr(validator, 'stats', None)
    if not stats:
        return None
    try:
        # ult 8.0 detect/seg validators use list-of-tuples; concat each
        # column. Tensor columns are converted to numpy first.
        cols = list(zip(*stats))
        np_cols = []
        for col in cols:
            if not col:
                continue
            first = col[0]
            if isinstance(first, _torch.Tensor):
                np_cols.append(_torch.cat(list(col), 0).cpu().numpy())
            else:
                np_cols.append(np.concatenate(list(col), axis=0))
        if len(np_cols) < 4:
            return None
        # Detection stats: (tp, conf, pred_cls, target_cls). Segment
        # adds tp_mask up front, which we ignore for the bbox-level
        # F1 (that matches what compute_threshold_data does on newer
        # ult anyway: kind='box').
        if len(np_cols) >= 4 and np_cols[0].ndim == 2:
            # detection layout
            tp, conf, pred_cls, target_cls = np_cols[:4]
        elif len(np_cols) >= 5:
            # segment layout: tp_b, tp_m, conf, pred_cls, target_cls.
            # We only need the bbox tp.
            tp, conf, pred_cls, target_cls = np_cols[0], np_cols[2], np_cols[3], np_cols[4]
        else:
            return None
    except Exception as err:  # pragma: no cover - defensive
        logging.debug(f'validator.stats column extraction failed: {err!r}')
        return None

    eps = 1e-16
    px_axis = np.linspace(0, 1, 1000)
    # Sort by descending confidence (same as ap_per_class).
    order = np.argsort(-conf)
    tp, conf, pred_cls = tp[order], conf[order], pred_cls[order]
    unique_classes, n_per_class = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]
    if nc == 0:
        return None

    p_curve = np.zeros((nc, 1000))
    r_curve = np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        mask = pred_cls == c
        n_l = n_per_class[ci]
        n_p = int(mask.sum())
        if n_p == 0 or n_l == 0:
            continue
        fpc = (1 - tp[mask]).cumsum(0)
        tpc = tp[mask].cumsum(0)
        # Ultralytics uses tp[..., 0] = IoU=0.5 col when tp is 2D; we
        # follow the same convention.
        tpc0 = tpc[:, 0] if tpc.ndim == 2 else tpc
        fpc0 = fpc[:, 0] if fpc.ndim == 2 else fpc
        recall = tpc0 / (n_l + eps)
        precision = tpc0 / (tpc0 + fpc0 + eps)
        # ap_per_class interpolates against ``-px`` because conf
        # decreases as we descend through the sort. Mirror that.
        r_curve[ci] = np.interp(-px_axis, -conf[mask], recall, left=0)
        p_curve[ci] = np.interp(-px_axis, -conf[mask], precision, left=1)

    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    return _payload_from_curves(f1_curve, px_axis,
                                ap_class_index=unique_classes.astype(int),
                                names=names)


def compute_threshold_data(model: Any, *, kind: str = 'box') -> Optional[dict]:
    """Return the threshold.json payload for ``model``.

    Tries the modern path first (``model.trainer.validator.metrics``
    with stored ``f1_curve`` / ``px``); falls back to recomputing from
    ``validator.stats`` on ultralytics <8.1. Returns ``None`` and emits
    a single ``WARNING`` log when *neither* path has data — that's the
    "ultralytics too old" case the user-facing message points at.

    :param model: Trained ``ultralytics.YOLO`` / ``ultralytics.RTDETR``
        instance.
    :type model: ultralytics.YOLO or ultralytics.RTDETR
    :param kind: ``'box'`` for detection / RT-DETR, ``'seg'`` for
        segmentation (mask-level F1). Falls back to ``'box'`` when the
        requested namespace is missing.
    :type kind: str
    :returns: Payload dict shaped like::

            {
              "f1_score": float,        # max of mean(f1_curve, axis=0)
              "threshold": float,       # px at the argmax above
              "per_class": {            # additive; absent when per-class
                "<label>": {            #   info can't be derived
                  "f1_score": float,
                  "threshold": float,
                },
                ...
              }
            }

        or ``None`` when no usable curves were found.
    :rtype: dict or None

    Example::

        >>> from ultralytics import YOLO
        >>> from yolov8.utils import compute_threshold_data
        >>> model = YOLO("yolov8n.pt")
        >>> # Suppose a training run just completed:
        >>> # model.train(data="coco8.yaml", epochs=1)
        >>> data = compute_threshold_data(model)
        >>> data is None or set(data) >= {"f1_score", "threshold"}
        True
    """
    trainer = getattr(model, 'trainer', None)
    validator = getattr(trainer, 'validator', None) if trainer is not None else None
    metrics = getattr(validator, 'metrics', None)
    names = getattr(model, 'names', None) or {}

    if metrics is not None:
        payload = _payload_from_metrics(metrics, kind=kind, names=names)
        if payload is not None:
            return payload

    # Path 2: after a stand-alone validation (``model.val(...)``) the
    # ``DetMetrics``-like object lives directly on ``model.metrics``,
    # not under any trainer. This is the recovery path used by the
    # DDP fallback in :mod:`yolov8.train.object_detection` /
    # :mod:`~yolov8.train.segmentation`: ult's subprocess.run-based
    # DDP launcher discards user callbacks, so the parent's trainer
    # never sees populated metrics; we re-validate the best ckpt and
    # read the curves off ``model.metrics`` here.
    m_metrics = getattr(model, 'metrics', None)
    if m_metrics is not None and (hasattr(m_metrics, 'box') or hasattr(m_metrics, 'seg')):
        payload = _payload_from_metrics(m_metrics, kind=kind, names=names)
        if payload is not None:
            return payload

    if validator is not None:
        payload = compute_threshold_data_from_validator_stats(validator, names=names)
        if payload is not None:
            return payload

    if metrics is not None or validator is not None or m_metrics is not None:
        _warn_old_ultralytics()
    return None


def compute_threshold_data_from_trainer(trainer: Any, *, kind: str = 'box') -> Optional[dict]:
    """Trainer-flavoured variant of :func:`compute_threshold_data`,
    suited for use inside an ``on_train_end`` callback.

    The trainer object passed to such a callback has
    ``trainer.validator.metrics`` populated by ultralytics' final
    validation step (which runs after the training loop breaks for any
    reason — normal end, patience exhaustion, time budget, the
    ``self.stop`` flag), so this is the only entry point that reliably
    has metrics regardless of how training terminated.

    :param trainer: Ultralytics trainer instance. Must expose either
        ``trainer.validator.metrics`` (modern ult) or ``trainer.validator.stats``
        (the fallback path for ult <8.1).
    :type trainer: ultralytics.engine.trainer.BaseTrainer
    :param kind: ``'box'`` or ``'seg'``. Same semantics as
        :func:`compute_threshold_data`.
    :type kind: str
    :returns: Same payload dict as :func:`compute_threshold_data`, or
        ``None`` when neither metrics nor stats are populated.
    :rtype: dict or None

    Example::

        >>> def _on_train_end(trainer):
        ...     from yolov8.utils import compute_threshold_data_from_trainer
        ...     payload = compute_threshold_data_from_trainer(trainer)
        ...     if payload is not None:
        ...         ...  # write threshold.json
    """
    validator = getattr(trainer, 'validator', None)
    metrics = getattr(validator, 'metrics', None)

    names: Optional[dict] = None
    data = getattr(trainer, 'data', None)
    if isinstance(data, dict):
        names = data.get('names')
    if not names:
        inner = getattr(trainer, 'model', None)
        names = getattr(inner, 'names', None)
    names = names or {}

    if metrics is not None:
        payload = _payload_from_metrics(metrics, kind=kind, names=names)
        if payload is not None:
            return payload

    if validator is not None:
        payload = compute_threshold_data_from_validator_stats(validator, names=names)
        if payload is not None:
            return payload

    if metrics is not None or validator is not None:
        _warn_old_ultralytics()
    return None


def _warn_old_ultralytics() -> None:
    """Emit a single, descriptive warning when both extraction paths
    fail. The message names the installed ultralytics version and tells
    the user how to fix it.
    """
    try:
        import ultralytics
        ult_v = ultralytics.__version__
    except Exception:
        ult_v = '<unknown>'
    logging.warning(
        f"yolov8.utils.compute_threshold_data: cannot derive f1/threshold "
        f"on ultralytics {ult_v}. {_OLD_ULT_THRESHOLD_HINT}"
    )
