"""Pull F1-vs-confidence threshold information out of a freshly-trained
Ultralytics model.

Replaces the previous "render F1_curve.png and OCR its title" hack: the
arrays that the plot is drawn from (``metrics.box.f1_curve`` and friends,
shape ``(num_classes, 1000)``, plus the confidence axis ``metrics.box.px``)
are still attached to the model's trainer/validator after ``model.train(...)``
returns. Reading them directly is exact (full float precision) and gives
us per-class data that the OCR path threw away when extracting the
plot's "all classes <f1> at <thr>" title.

The serialised payload is forward/back-compatible: ``f1_score`` and
``threshold`` keep the legacy semantic (the aggregate "all classes"
best F1 and its argmax confidence — exactly what the OCR pulled out of
the plot title), and per-class info is added under a new ``per_class``
key that legacy readers can ignore.
"""
from typing import Any, Optional

import numpy as np


def compute_threshold_data(model: Any, *, kind: str = 'box') -> Optional[dict]:
    """Return the threshold.json payload for ``model`` (a trained
    ultralytics ``YOLO`` / ``RTDETR`` instance), or ``None`` if the data
    is not available.

    Parameters
    ----------
    model:
        The model object after ``model.train(...)`` has returned. Must
        have a populated ``model.trainer.validator.metrics`` (i.e. a
        validation pass actually ran during training).
    kind:
        Which Metric namespace to read — ``'box'`` for detection /
        RT-DETR (bbox-level F1) or ``'seg'`` for segmentation
        (mask-level F1, matches what the legacy code OCR'd out of
        ``MaskF1_curve.png``). Falls back to ``'box'`` if the requested
        namespace is missing.

    Returns
    -------
    A dict shaped like::

        {
          "f1_score": float,        # max of mean(f1_curve, axis=0)
          "threshold": float,       # px at the argmax above
          "per_class": {            # additive; absent if per-class
            "<label>": {            #   info can't be derived
              "f1_score": float,    #   per-class max F1
              "threshold": float,   #   per-class argmax confidence
            },
            ...
          }
        }

    or ``None`` if no usable curves were found (callers should then
    skip writing ``threshold.json`` — same as the legacy OCR-failure
    path).
    """
    trainer = getattr(model, 'trainer', None)
    validator = getattr(trainer, 'validator', None) if trainer is not None else None
    metrics = getattr(validator, 'metrics', None)
    if metrics is None:
        return None

    # Pick the right Metric subobject. ``DetMetrics`` only has ``.box``;
    # ``SegmentMetrics`` has both ``.box`` (bbox) and ``.seg`` (mask).
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

    # Aggregate "all classes" best F1 / threshold — same definition the
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
    names = getattr(model, 'names', None) or {}
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
