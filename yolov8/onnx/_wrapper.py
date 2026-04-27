"""ONNX-traceable rewrites of ``BaseModel._predict_once``.

These are private helpers that the public exporters in
:mod:`yolov8.onnx.export` compose. They walk the inner model's layer
``ModuleList`` exactly the way Ultralytics' own
``BaseModel._predict_once`` / ``RTDETRDetectionModel.predict`` do, but
short-circuit the embed branch so ONNX tracing sees plain
``cat([adaptive_avg_pool2d(x).flatten(1), ...])`` instead of the
``unbind`` Ultralytics emits when called directly.

Only :class:`_PredictOnlyModel` and :class:`_DualHeadModel` are public
within the :mod:`yolov8.onnx` subpackage; nothing here is re-exported
from :mod:`yolov8.onnx`.
"""
from __future__ import annotations

import copy as _copy
import os
from typing import Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO, RTDETR
from ultralytics.nn.modules import C2f, Detect, RTDETRDecoder

from ..embed import resolve_inner
from ..utils import derive_model_meta_from_path


# ---------------------------------------------------------------------------
# Inner-model loading & export-mode preparation
# ---------------------------------------------------------------------------

def _resolve_for_export(model_or_path):
    """Resolve any of the accepted source types into the inner ``BaseModel``.

    :param model_or_path: A ``YOLO`` / ``RTDETR`` wrapper, a raw
        ``BaseModel`` instance, the path to a ``.pt`` checkpoint, or the
        path to a training workdir containing ``weights/best.pt``.
    :type model_or_path: ultralytics.YOLO or ultralytics.RTDETR or
        torch.nn.Module or str or os.PathLike
    :returns: ``(wrapper, inner_basemodel, workdir)`` where ``wrapper`` is
        the original wrapper if one was supplied (so we can read
        ``.task`` / ``.names`` for metadata), ``inner_basemodel`` is the
        ``BaseModel`` ready for further export prep, and ``workdir`` is
        the original directory if the source was a workdir path
        (otherwise ``None``).
    :rtype: tuple
    :raises FileNotFoundError: If a workdir path was given but
        ``weights/best.pt`` is absent.
    """
    if isinstance(model_or_path, (str, os.PathLike)):
        path = os.fspath(model_or_path)
        workdir = path if os.path.isdir(path) else None
        if workdir is not None:
            best_pt = os.path.join(workdir, "weights", "best.pt")
            if not os.path.isfile(best_pt):
                raise FileNotFoundError(f"workdir lacks weights/best.pt: {workdir}")
            path = best_pt
        model_type, _ = derive_model_meta_from_path(path)
        wrapper = YOLO(path) if model_type == "yolo" else RTDETR(path)
        return wrapper, resolve_inner(wrapper), workdir

    # Wrapper-style object: has ``.task`` and an inner ``.model`` with
    # its own ``predict`` (``ultralytics.YOLO`` / ``ultralytics.RTDETR``).
    if (hasattr(model_or_path, "task")
            and hasattr(model_or_path, "model")
            and hasattr(getattr(model_or_path, "model", None), "predict")):
        return model_or_path, resolve_inner(model_or_path), None

    # Raw inner BaseModel.
    return None, resolve_inner(model_or_path), None


def _prepare_for_export(inner: nn.Module, dynamic: bool) -> nn.Module:
    """Apply Ultralytics' own export-mode mutations on a deep copy.

    Replicates the head-mutation block from
    ``ultralytics.engine.exporter`` so the original model object stays
    untouched. Two graph-shape-affecting transformations are applied:

    * ``Detect`` / ``RTDETRDecoder`` heads are switched to
      ``export=True``, which drops the eval-mode ``(decoded, raw)`` tuple
      and emits a single tensor (or a tuple when the head genuinely has
      multiple outputs, like ``Segment``).
    * ``C2f`` blocks swap to ``forward_split`` for a cleaner ONNX graph.

    The other attributes (``dynamic`` / ``format`` / ``max_det``) are
    written defensively even though some old Ultralytics versions don't
    read them; they're plain Python attributes and setting them never
    breaks anything.

    :param inner: The inner ``BaseModel`` to mutate.
    :type inner: torch.nn.Module
    :param dynamic: Whether to emit dynamic batch / spatial axes; set on
        the head modules so they pick the right output reshape path.
    :type dynamic: bool
    :returns: A deep copy with the export-mode mutations applied.
    :rtype: torch.nn.Module
    """
    inner = _copy.deepcopy(inner)
    for p in inner.parameters():
        p.requires_grad = False
    inner.train(False)
    inner.float()
    if hasattr(inner, "fuse"):
        try:
            inner = inner.fuse()
        except Exception:
            pass  # custom heads sometimes raise; fusing is optional
    for m in inner.modules():
        if isinstance(m, (Detect, RTDETRDecoder)):
            m.dynamic = dynamic
            m.export = True
            m.format = "onnx"
            m.max_det = getattr(m, "max_det", 300)
        elif isinstance(m, C2f):
            m.forward = m.forward_split
    return inner


# ---------------------------------------------------------------------------
# ONNX-traceable forward wrappers
# ---------------------------------------------------------------------------

def _walk_and_run(inner: nn.Module, x: torch.Tensor,
                  embed_indices: Sequence[int] | None,
                  pooled: list[torch.Tensor] | None):
    """Walk ``inner.model[:-1]`` and return ``(head_input, head)``.

    Mirrors ``ultralytics.nn.tasks.BaseModel._predict_once`` minus the
    final head invocation. When ``pooled`` is non-``None`` and an index
    in ``embed_indices`` is hit, the layer's output is pooled and
    appended to ``pooled``.

    :param inner: Inner ``BaseModel`` after :func:`_prepare_for_export`.
    :type inner: torch.nn.Module
    :param x: Input tensor of shape ``[N, 3, H, W]``.
    :type x: torch.Tensor
    :param embed_indices: Layer indices whose pooled feature maps should
        be appended to ``pooled``. ``None`` means no embedding tap.
    :type embed_indices: Sequence[int] or None
    :param pooled: Mutable list to receive the pooled feature tensors.
        ``None`` to disable embedding extraction entirely.
    :type pooled: list or None
    :returns: ``(head_input, head_module)`` where ``head_input`` is what
        the final head module should be called with, and ``head_module``
        is ``inner.model[-1]``.
    :rtype: tuple
    """
    save = set(int(i) for i in inner.save) | (
        set(int(i) for i in embed_indices) if embed_indices else set())
    embed_set = set(int(i) for i in embed_indices) if embed_indices else set()
    y: list[torch.Tensor | None] = []
    last = inner.model[-1]

    for m in inner.model[:-1]:
        if m.f != -1:
            if isinstance(m.f, int):
                x = y[m.f]
            else:
                x = [x if j == -1 else y[j] for j in m.f]
        x = m(x)
        y.append(x if m.i in save else None)
        if pooled is not None and m.i in embed_set:
            pooled.append(F.adaptive_avg_pool2d(x, 1).flatten(1))

    if last.f != -1:
        if isinstance(last.f, int):
            head_in = y[last.f]
        else:
            head_in = [x if j == -1 else y[j] for j in last.f]
    else:
        head_in = x
    return head_in, last


class _PredictOnlyModel(nn.Module):
    """Inner-equivalent whose ``forward`` emits only the head output(s).

    For single-output heads this is one tensor; for ``Segment`` it's a
    ``(boxes, mask_proto)`` tuple.

    :param inner: Inner ``BaseModel`` after :func:`_prepare_for_export`.
    :type inner: torch.nn.Module
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self._head_takes_batch = isinstance(inner.model[-1], RTDETRDecoder)

    def forward(self, x: torch.Tensor):
        head_in, head = _walk_and_run(self.inner, x, None, None)
        return head(head_in, None) if self._head_takes_batch else head(head_in)


class _DualHeadModel(nn.Module):
    """Predict-only plus a pooled-embedding tail.

    The head's normal output(s) come first; the embedding is always the
    last element of the returned tuple, so the output naming is a strict
    superset of the single-head variant.

    :param inner: Inner ``BaseModel`` after :func:`_prepare_for_export`.
    :type inner: torch.nn.Module
    :param layer_indices: Which layer indices to pool & concat for the
        embedding output. Must be non-empty and strictly below the head
        index (``len(inner.model) - 1``).
    :type layer_indices: Sequence[int]
    :raises ValueError: If ``layer_indices`` is empty or overlaps the
        head.
    """

    def __init__(self, inner: nn.Module, layer_indices: Sequence[int]):
        super().__init__()
        self.inner = inner
        idx = sorted(set(int(i) for i in layer_indices))
        if not idx:
            raise ValueError("layer_indices must be non-empty")
        if max(idx) >= len(inner.model) - 1:
            raise ValueError(
                f"embedding layer {max(idx)} overlaps the head at "
                f"{len(inner.model) - 1}; pick an index strictly below it.")
        self.embed_indices = idx
        self._head_takes_batch = isinstance(inner.model[-1], RTDETRDecoder)

    def forward(self, x: torch.Tensor):
        pooled: list[torch.Tensor] = []
        head_in, head = _walk_and_run(self.inner, x, self.embed_indices, pooled)
        embedding = torch.cat(pooled, dim=1)
        pred = head(head_in, None) if self._head_takes_batch else head(head_in)
        if isinstance(pred, (tuple, list)):
            return (*pred, embedding)
        return (pred, embedding)
