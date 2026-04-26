"""ONNX export helpers.

Two functions:

  * :func:`export_yolo_to_onnx` - thin wrapper around ``model.export()``,
    used by :mod:`yolov8.export` for the standard "ship the trained
    detection / segmentation graph" pipeline.

  * :func:`export_yolo_to_onnx_with_embedding` - dual-head export. The
    resulting ONNX graph runs the model once and emits two outputs:

      - ``predictions``: whatever the original head normally returns under
        ``export=True`` (Detect/RTDETRDecoder/Segment/Pose/OBB...);
      - ``embedding``: the channel-wise concat of adaptive-avg-pooled
        feature maps at the requested layer indices.

    This re-implements the embed branch of
    ``ultralytics.nn.tasks.BaseModel._predict_once`` as a static
    ``nn.Module`` (see :class:`yolov8.embed.EmbedHead`) and combines it
    with the head, so the graph has no Python-side control flow that
    confuses ONNX tracing.
"""
from __future__ import annotations

import copy as _copy
import os.path
from shutil import SameFileError
from typing import Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from hbutils.system import copy
from ultralytics import YOLO, RTDETR
from ultralytics.nn.modules import C2f, Detect, RTDETRDecoder

from .embed import default_embed_indices, resolve_inner
from .utils import derive_model_meta_from_path


def export_yolo_to_onnx(workdir: str, onnx_filename, opset_version: int = 14,
                        no_optimize: bool = False):
    best_pt = os.path.join(workdir, 'weights', 'best.pt')
    model_type, _ = derive_model_meta_from_path(best_pt)
    if model_type == 'yolo':
        yolo = YOLO(best_pt)
    else:
        yolo = RTDETR(best_pt)

    if os.path.dirname(onnx_filename):
        os.makedirs(os.path.dirname(onnx_filename), exist_ok=True)

    _retval = yolo.export(format='onnx', dynamic=True, simplify=not no_optimize, opset=opset_version)
    _exported_onnx_file = _retval or (os.path.splitext(yolo.ckpt_path)[0] + '.onnx')
    try:
        copy(_exported_onnx_file, onnx_filename)
    except SameFileError:
        pass


# ---------------------------------------------------------------------------
# Dual-head (predictions + embedding) export
# ---------------------------------------------------------------------------

class _DualHeadModel(nn.Module):
    """Run the inner model up to the head; emit both the head's output and a
    pooled embedding at the requested layer indices.

    Walks the layer ``ModuleList`` exactly the way Ultralytics' own
    ``BaseModel._predict_once`` / ``RTDETRDetectionModel.predict`` do, but
    short-circuits the embed branch so ONNX tracing sees plain
    ``cat([adaptive_avg_pool2d(x).flatten(1), ...])`` instead of the dynamic
    ``unbind`` Ultralytics uses to split a batch into a tuple.

    The head is the last entry in ``inner.model``; for ``RTDETRDetectionModel``
    its forward signature is ``(x, batch=None)`` while everything else uses
    plain ``(x)``. Detect / RTDETRDecoder modules must be in export mode
    (``m.export = True``) before tracing - the caller is responsible for that
    so we don't mutate the original model object here.
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
        self.save = set(int(i) for i in inner.save) | set(idx)
        self._head_takes_batch = isinstance(inner.model[-1], RTDETRDecoder)

    def forward(self, x: torch.Tensor):
        y: list[torch.Tensor | None] = []
        feats: list[torch.Tensor] = []
        last = self.inner.model[-1]
        for m in self.inner.model[:-1]:
            if m.f != -1:
                if isinstance(m.f, int):
                    x = y[m.f]
                else:
                    x = [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
            if m.i in self.embed_indices:
                feats.append(F.adaptive_avg_pool2d(x, 1).flatten(1))
        embedding = torch.cat(feats, dim=1)

        if last.f != -1:
            if isinstance(last.f, int):
                head_in = y[last.f]
            else:
                head_in = [x if j == -1 else y[j] for j in last.f]
        else:
            head_in = x
        if self._head_takes_batch:
            pred = last(head_in, None)
        else:
            pred = last(head_in)
        # Detect heads in non-export mode return (decoded, raw_feats); when
        # ``m.export = True`` (set by the caller) they return a single tensor.
        # Be defensive anyway.
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        return pred, embedding


def _prepare_for_export(inner: nn.Module, dynamic: bool) -> nn.Module:
    """Mirror the head-mutation block from ``ultralytics.engine.exporter``
    (lines around ``m.export = True``) on a deep copy, so the caller's
    original model object stays untouched."""
    inner = _copy.deepcopy(inner)
    for p in inner.parameters():
        p.requires_grad = False
    inner.train(False)
    inner.float()
    if hasattr(inner, "fuse"):
        try:
            inner = inner.fuse()
        except Exception:
            # Some custom heads disable fuse via raise; that's fine, skip it.
            pass
    for m in inner.modules():
        if isinstance(m, (Detect, RTDETRDecoder)):
            m.dynamic = dynamic
            m.export = True
            m.format = "onnx"
            m.max_det = getattr(m, "max_det", 300)
        elif isinstance(m, C2f):
            # Cleaner ONNX graph; matches ultralytics' own exporter.
            m.forward = m.forward_split
    return inner


def _resolve_for_export(model_or_path: Union[str, "os.PathLike", object]) -> nn.Module:
    """Accept a path / wrapper / raw BaseModel and return the inner ``nn.Module``.

    Path strings are loaded through :class:`ultralytics.YOLO` or
    :class:`ultralytics.RTDETR` based on the embedded class name in the
    checkpoint, the same way :func:`export_yolo_to_onnx` does it.
    """
    if isinstance(model_or_path, (str, os.PathLike)):
        path = os.fspath(model_or_path)
        if os.path.isdir(path):
            best_pt = os.path.join(path, "weights", "best.pt")
            if not os.path.isfile(best_pt):
                raise FileNotFoundError(f"workdir lacks weights/best.pt: {path}")
            path = best_pt
        model_type, _ = derive_model_meta_from_path(path)
        wrapper = YOLO(path) if model_type == "yolo" else RTDETR(path)
        return resolve_inner(wrapper)
    return resolve_inner(model_or_path)


def export_yolo_to_onnx_with_embedding(
    model,
    onnx_filename: str,
    layer_indices: Sequence[int] | None = None,
    imgsz: int = 640,
    opset_version: int = 14,
    dynamic: bool = True,
    simplify: bool = True,
    batch: int = 1,
    device: Union[str, torch.device, None] = None,
) -> str:
    """Export an Ultralytics model to a dual-head ONNX (predictions + embedding).

    :param model:           ``ultralytics.YOLO``/``ultralytics.RTDETR``
                            wrapper, raw ``BaseModel``, ``.pt`` path, or a
                            workdir containing ``weights/best.pt``.
    :param onnx_filename:   destination path for the ``.onnx`` file.
    :param layer_indices:   layers to pool & concat for the embedding output.
                            ``None`` → ``[len(inner.model) - 2]``, matching
                            ``ultralytics.YOLO.embed()``.
    :param imgsz:           input letterbox size used for the dummy tracing
                            tensor.
    :param opset_version:   ONNX opset.
    :param dynamic:         expose dynamic batch / spatial axes.
    :param simplify:        run ``onnxsim`` after export when available.
    :param batch:           dummy-input batch size for tracing.
    :param device:          where to do the trace. ``None`` → match the
                            model's current device, falling back to CPU.

    :returns:               the ``onnx_filename`` that was written.
    """
    inner_raw = _resolve_for_export(model)
    if device is None:
        try:
            device = next(inner_raw.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    inner = _prepare_for_export(inner_raw, dynamic).to(device)
    layer_indices = list(layer_indices) if layer_indices is not None else default_embed_indices(inner)
    wrap = _DualHeadModel(inner, layer_indices).to(device)
    wrap.train(False)

    dummy = torch.zeros(batch, 3, imgsz, imgsz, device=device)
    # Two dry runs: the first triggers any lazy-init; the second ensures
    # output shapes are stable, mirroring ultralytics' own exporter pattern.
    with torch.inference_mode():
        for _ in range(2):
            _pred, _emb = wrap(dummy)
    pred_dim = list(_pred.shape)
    embed_dim = int(_emb.shape[-1])

    if os.path.dirname(onnx_filename):
        os.makedirs(os.path.dirname(onnx_filename), exist_ok=True)

    dyn_axes = None
    if dynamic:
        dyn_axes = {
            "images": {0: "batch", 2: "height", 3: "width"},
            "predictions": {0: "batch"},
            "embedding": {0: "batch"},
        }
        # Detect/Segment/Pose/OBB outputs an anchor dim that scales with
        # input HxW; declare it dynamic when the user asked for dynamic
        # spatial axes. RTDETRDecoder emits a fixed query count, so we leave
        # axis 2 alone for that case.
        if not isinstance(inner.model[-1], RTDETRDecoder) and len(pred_dim) >= 3:
            dyn_axes["predictions"][2] = "anchors"

    torch.onnx.export(
        wrap,
        dummy,
        onnx_filename,
        input_names=["images"],
        output_names=["predictions", "embedding"],
        opset_version=opset_version,
        dynamic_axes=dyn_axes,
        do_constant_folding=True,
    )

    if simplify:
        try:
            import onnx
            import onnxsim

            model_proto = onnx.load(onnx_filename)
            simplified, ok = onnxsim.simplify(model_proto)
            if ok:
                onnx.save(simplified, onnx_filename)
        except Exception:
            # onnxsim is best-effort; the un-simplified file is still valid.
            pass

    # Reattach metadata so callers can inspect what they got without
    # re-tracing the graph.
    export_yolo_to_onnx_with_embedding.last_pred_shape = tuple(pred_dim)
    export_yolo_to_onnx_with_embedding.last_embed_dim = embed_dim
    export_yolo_to_onnx_with_embedding.last_layer_indices = list(layer_indices)
    return onnx_filename
