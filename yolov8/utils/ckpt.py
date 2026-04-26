"""Helpers for introspecting Ultralytics checkpoints.

The training entry points no longer write a ``model_type.json`` sidecar
into the work directory: the same information lives inside every
Ultralytics checkpoint and can be recovered after the fact, which
removes the previous 30-second delayed-write race condition.
"""
from typing import Tuple

import torch


def derive_model_meta(state_dict: dict) -> Tuple[str, str]:
    """Return ``(model_type, problem_type)`` derived from an Ultralytics
    checkpoint dict.

    Both fields are inferred from the **embedded model object's class name**
    (``type(state_dict['model']).__name__``), not from ``train_args``. The
    pickled model object is the most authoritative source: ``train_args``
    is a free-form string mapping that can name third-party / non-official
    base weights, but the class itself is what Ultralytics actually
    instantiated and trained on, so it never lies about the architecture.

    Class-name conventions used:

    - ``RTDETRDetectionModel``  → ``('rtdetr', 'detection')``
    - ``SegmentationModel``     → ``('yolo',   'segmentation')``
    - any other / unknown       → ``('yolo',   'detection')``  (default)

    The substring matches (``'RTDETR'`` and ``'Segmentation'``) are
    deliberately tolerant of subclasses such as ``CustomSegmentationModel``.
    """
    model = state_dict.get('model')
    cls_name = type(model).__name__ if model is not None else ''

    model_type = 'rtdetr' if 'RTDETR' in cls_name else 'yolo'
    problem_type = 'segmentation' if 'Segmentation' in cls_name else 'detection'

    return model_type, problem_type


def derive_model_meta_from_path(ckpt_path: str) -> Tuple[str, str]:
    """Convenience wrapper: load ``ckpt_path`` with torch and return
    :func:`derive_model_meta` of it. ``weights_only=False`` is required
    because Ultralytics checkpoints embed pickled model objects."""
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    return derive_model_meta(state_dict)
