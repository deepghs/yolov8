"""Pull F1-vs-confidence threshold information out of a freshly-trained
Ultralytics model.

Replaces the previous "render F1_curve.png and OCR its title" hack: the
arrays the plot is drawn from (``metrics.box.f1_curve`` and friends,
shape ``(num_classes, 1000)``, plus the confidence axis ``metrics.box.px``)
are still attached to the model's trainer/validator after
``model.train(...)`` returns. Reading them directly is exact (full float
precision) and gives us per-class data that the OCR path threw away when
extracting the plot's "all classes <f1> at <thr>" title.

Two flavours of the same extraction are exposed:

* :func:`compute_threshold_data` (wrapper-flavoured) reads
  ``model.trainer.validator.metrics`` and uses ``model.names`` for class
  labels. This is the call site after ``model.train()`` returns.

* :func:`compute_threshold_data_from_trainer` (trainer-flavoured) reads
  ``trainer.validator.metrics`` directly and uses ``trainer.data['names']``
  / ``trainer.model.names``. This is the right entry point inside an
  ``on_train_end`` callback - the only place guaranteed to fire when
  patience exhausts mid-training, when training time-budget runs out,
  or when ultralytics is doing DDP via ``subprocess.run`` (in which case
  ``model.trainer`` in the main process never sees populated metrics).

The serialised payload is forward/back-compatible: ``f1_score`` and
``threshold`` keep the legacy semantic (the aggregate "all classes"
best F1 and its argmax confidence - exactly what the OCR pulled out of
the plot title), and per-class info is added under a new ``per_class``
key that legacy readers can ignore.
"""
from typing import Any, Optional

import numpy as np


def _payload_from_metrics(metrics: Any, *, kind: str, names: Optional[dict]) -> Optional[dict]:
    """Shared reducer: pick the right ``Metric`` subobject (``.box`` /
    ``.seg``), grab ``f1_curve`` / ``px``, and assemble the payload.

    Returns ``None`` when the requested arrays are missing or empty -
    callers should then skip writing ``threshold.json`` (same as the
    legacy OCR-failure path).
    """
    src = getattr(metrics, kind, None) or getattr(metrics, 'box', None)
    if src is None:
        return None

    f1_curve = getattr(src, 'f1_curve', None)
    px = getattr(src, 'px', None)
    if f1_curve is None or px is None:
        return None

    f1_curve = np.asarray(f1_curve)
    px = np.asarray(px)
    if f1_curve.size == 0 or px.size == 0:
        return None

    # Aggregate "all classes" best F1 / threshold - same definition the
    # ultralytics plotter uses for the F1 curve title, and therefore the
    # exact value the legacy OCR was approximating.
    mean_f1 = f1_curve.mean(axis=0)
    best_idx = int(mean_f1.argmax())

    payload: dict = {
        'f1_score': float(mean_f1[best_idx]),
        'threshold': float(px[best_idx]),
    }

    # Additive: per-class best F1 + threshold. ``ap_class_index`` is the
    # ordered list of class indices the validator actually saw, so the
    # k-th row of ``f1_curve`` corresponds to ``ap_class_index[k]``.
    ap_class_index = getattr(src, 'ap_class_index', None)
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


def compute_threshold_data(model: Any, *, kind: str = 'box') -> Optional[dict]:
    """Return the threshold.json payload for ``model`` (a trained
    ultralytics ``YOLO`` / ``RTDETR`` instance), or ``None`` if no
    metrics are populated.

    The metrics live on ``model.trainer.validator.metrics``. After a
    *normal* single-GPU training, this is populated by the trainer's
    final-validation step and stays attached. After a multi-GPU / DDP
    run - where ultralytics spawns a subprocess via ``subprocess.run`` -
    the main process's ``model.trainer`` is the same object that was
    created *before* the subprocess started training, so its
    ``.validator.metrics`` is empty. Use
    :func:`compute_threshold_data_from_trainer` from inside an
    ``on_train_end`` callback to capture metrics before the subprocess
    exits.
    """
    trainer = getattr(model, 'trainer', None)
    validator = getattr(trainer, 'validator', None) if trainer is not None else None
    metrics = getattr(validator, 'metrics', None)
    if metrics is None:
        return None
    names = getattr(model, 'names', None) or {}
    return _payload_from_metrics(metrics, kind=kind, names=names)


def compute_threshold_data_from_trainer(trainer: Any, *, kind: str = 'box') -> Optional[dict]:
    """Trainer-flavoured variant of :func:`compute_threshold_data`,
    suited for use inside an ``on_train_end`` callback.

    The trainer object passed to such a callback has
    ``trainer.validator.metrics`` populated by the trainer's
    final-validation step (which runs after the training loop breaks
    for any reason - normal end, patience exhaustion, time budget,
    ``self.stop`` flag), so this is the only entry point that reliably
    has metrics regardless of how training terminated. Class names come
    from ``trainer.data['names']`` with a fallback to
    ``trainer.model.names``.
    """
    validator = getattr(trainer, 'validator', None)
    metrics = getattr(validator, 'metrics', None)
    if metrics is None:
        return None

    names: Optional[dict] = None
    data = getattr(trainer, 'data', None)
    if isinstance(data, dict):
        names = data.get('names')
    if not names:
        inner = getattr(trainer, 'model', None)
        names = getattr(inner, 'names', None)
    return _payload_from_metrics(metrics, kind=kind, names=names or {})
