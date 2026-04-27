"""ONNX export helpers.

Two exports, both writing directly to the user-given path via
``torch.onnx.export`` - no temp files, no copy-after-the-fact, no
dependence on the upstream exporter's internal layout choices:

  * :func:`export_yolo_to_onnx` - single-head export, equivalent to what
    ``YOLO.export(format='onnx')`` would have produced. Used by
    :mod:`yolov8.export` for the standard "ship the trained graph"
    pipeline.

  * :func:`export_yolo_to_onnx_with_embedding` - dual-head export. The
    head's normal output(s) come first under their usual names, and a
    pooled embedding tensor is appended as the trailing output.

Naming contract (verified against the default ``YOLO.export(format='onnx')``
on ultralytics 8.0.196 -> 8.3.105 and used by ``deepghs/imgutils``):

    input  name : ``images``                                ``[batch, 3, H, W]``
    head outputs:
        detect / pose / obb / classify / rtdetr ->          ``output0``
        segment                                 ->          ``output0`` + ``output1``
    dual-head adds:                                         ``embedding``

This means a dual-head ONNX is a strict superset of a single-head one:
``sess.run(['output0'], {'images': x})`` keeps working byte-for-byte
when consumers only need predictions (``imgutils.generic.yolo``).

Metadata. The produced ONNX preserves every ``metadata_props`` key the
upstream ultralytics exporter writes (``description`` / ``author`` /
``date`` / ``version`` / ``license`` / ``docs`` / ``stride`` / ``task``
/ ``batch`` / ``imgsz`` / ``names``) and adds a ``dghs.yolov8.*``
namespace recording the package version, the exporter used, the
input/output naming, and - for dual-head exports - the embedding's
layer indices and dimension. When the source is a training workdir
that contains a ``threshold.json`` (per-class F1 / threshold from the
validator), that file is embedded under ``dghs.yolov8.threshold`` so
the resulting ``.onnx`` is fully self-describing. Callers may inject
additional key/value pairs via ``extra_metadata=...``.
"""
from __future__ import annotations

import copy as _copy
import json
import os
import os.path
from datetime import datetime
from typing import Any, Mapping, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO, RTDETR
from ultralytics.nn.modules import C2f, Detect, RTDETRDecoder

from .config.meta import __VERSION__ as _DGHS_YOLOV8_VERSION
from .embed import default_embed_indices, resolve_inner
from .utils import derive_model_meta_from_path


# ---------------------------------------------------------------------------
# Inner-model loading & export-mode preparation
# ---------------------------------------------------------------------------

def _resolve_for_export(model_or_path):
    """Return ``(wrapper_or_None, inner_BaseModel, workdir_or_None)``.

    The wrapper is kept so we can read ``wrapper.task`` / ``wrapper.names``
    when collecting metadata, and the workdir is kept so we can pick up
    ``threshold.json`` automatically.
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
    # its own ``predict``. ``ultralytics.YOLO`` and ``ultralytics.RTDETR``
    # both fit this shape.
    if (hasattr(model_or_path, "task")
            and hasattr(model_or_path, "model")
            and hasattr(getattr(model_or_path, "model", None), "predict")):
        return model_or_path, resolve_inner(model_or_path), None

    # Raw inner BaseModel.
    return None, resolve_inner(model_or_path), None


def _prepare_for_export(inner: nn.Module, dynamic: bool) -> nn.Module:
    """Mirror the head-mutation block from ``ultralytics.engine.exporter``
    on a deep copy, so the caller's original model stays untouched.

    The graph-shape-affecting transformations:

    * ``Detect`` / ``RTDETRDecoder`` heads switch to ``export=True``,
      which drops the eval-mode ``(decoded, raw)`` tuple and emits a
      single tensor (or a tuple when the head genuinely has multiple
      outputs, like ``Segment``).
    * ``C2f`` blocks swap to ``forward_split`` for a cleaner ONNX graph.

    The other attributes (``dynamic`` / ``format`` / ``max_det``) are
    written defensively even though some old ultralytics versions don't
    read them - they're plain Python attributes and setting them never
    breaks anything.
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
# Wrappers that produce ONNX-traceable graphs
# ---------------------------------------------------------------------------

def _walk_and_run(inner: nn.Module, x: torch.Tensor,
                  embed_indices: Sequence[int] | None,
                  pooled: list[torch.Tensor] | None):
    """Walk the layer ``ModuleList`` (everything except the head) the
    same way ``BaseModel._predict_once`` does and return
    ``(head_input, head)``."""
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
    """Inner-equivalent whose forward emits only the head output(s).
    For single-output heads this is one tensor; for ``Segment`` it's
    a ``(boxes, mask_proto)`` tuple."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self._head_takes_batch = isinstance(inner.model[-1], RTDETRDecoder)

    def forward(self, x: torch.Tensor):
        head_in, head = _walk_and_run(self.inner, x, None, None)
        return head(head_in, None) if self._head_takes_batch else head(head_in)


class _DualHeadModel(nn.Module):
    """Predict-only + pooled-embedding tail. The head's normal output(s)
    come first; the embedding is always the last element of the
    returned tuple, so the output naming is a strict superset of the
    single-head variant."""

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


# ---------------------------------------------------------------------------
# Common export plumbing
# ---------------------------------------------------------------------------

def _build_dynamic_axes(output_names: Sequence[str],
                        head_dim_lens: Sequence[int],
                        is_rtdetr: bool) -> dict:
    axes: dict = {"images": {0: "batch", 2: "height", 3: "width"}}
    for n, ndim in zip(output_names, head_dim_lens):
        ax = {0: "batch"}
        if not is_rtdetr and n.startswith("output") and ndim >= 3:
            ax[2] = "anchors"
        axes[n] = ax
    if "embedding" in output_names:
        axes["embedding"] = {0: "batch"}
    return axes


def _maybe_simplify(onnx_filename: str) -> None:
    """Best-effort onnxsim pass. Leaves the un-simplified graph in place
    if onnx / onnxsim aren't installed or simplification fails."""
    try:
        import onnx
        import onnxsim

        model_proto = onnx.load(onnx_filename)
        simplified, ok = onnxsim.simplify(model_proto)
        if ok:
            onnx.save(simplified, onnx_filename)
    except Exception:
        return


def _export_via_torch(wrap: nn.Module,
                      dummy: torch.Tensor,
                      onnx_filename: str,
                      output_names: Sequence[str],
                      head_dim_lens: Sequence[int],
                      is_rtdetr: bool,
                      opset_version: int,
                      dynamic: bool,
                      simplify: bool) -> str:
    if os.path.dirname(onnx_filename):
        os.makedirs(os.path.dirname(onnx_filename), exist_ok=True)

    dyn_axes = _build_dynamic_axes(output_names, head_dim_lens, is_rtdetr) if dynamic else None
    torch.onnx.export(
        wrap,
        dummy,
        onnx_filename,
        input_names=["images"],
        output_names=list(output_names),
        opset_version=opset_version,
        dynamic_axes=dyn_axes,
        do_constant_folding=True,
    )
    if simplify:
        _maybe_simplify(onnx_filename)
    return onnx_filename


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

_HEAD_TO_TASK = {
    "RTDETRDecoder": "detect",
    "Segment": "segment",
    "Pose": "pose",
    "OBB": "obb",
    "Classify": "classify",
    "WorldDetect": "detect",
    "Detect": "detect",
}


def _task_from_head(inner: nn.Module) -> str:
    head_cls = type(inner.model[-1]).__name__
    for key, task in _HEAD_TO_TASK.items():
        if key in head_cls:
            return task
    return "detect"


def _stride_int(inner: nn.Module) -> int:
    s = getattr(inner, "stride", None)
    if s is None:
        return 32
    try:
        if hasattr(s, "max"):
            return int(s.max().item() if hasattr(s.max(), "item") else max(s))
        return int(max(s))
    except Exception:
        return 32


def _collect_metadata(
    wrapper,
    inner: nn.Module,
    imgsz: int,
    batch: int,
    *,
    exporter_name: str,
    output_names: Sequence[str],
    has_embedding: bool = False,
    layer_indices: Sequence[int] | None = None,
    embed_dim: int | None = None,
    workdir: str | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Build the metadata_props dict that mirrors what
    ``ultralytics.engine.exporter`` writes, plus a ``dghs.yolov8.*``
    namespace describing this export specifically."""
    try:
        from ultralytics import __version__ as ult_version
    except Exception:
        ult_version = "unknown"

    task = (getattr(wrapper, "task", None) if wrapper is not None else None) \
        or _task_from_head(inner)
    names = (getattr(wrapper, "names", None) if wrapper is not None else None) \
        or getattr(inner, "names", None) or {}

    sz = list(imgsz) if isinstance(imgsz, (list, tuple)) else [int(imgsz), int(imgsz)]

    md: dict[str, str] = {
        # === ultralytics.engine.exporter parity ===
        # Names and order match the dict that exporter._add_metadata builds
        # so any consumer relying on these keys (notably imgutils, which
        # reads ``imgsz`` via json.loads and ``names`` via ast literal)
        # keeps working unchanged.
        "description": f"Ultralytics {type(inner).__name__} model exported from yolov8",
        "author": "Ultralytics",
        "date": datetime.now().isoformat(),
        "version": str(ult_version),
        "license": "AGPL-3.0 License (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com",
        "stride": str(_stride_int(inner)),
        "task": task,
        "batch": str(batch),
        "imgsz": json.dumps(sz),         # imgutils does json.loads on this
        "names": repr(dict(names) if not isinstance(names, dict) else names),

        # === dghs/yolov8 namespace ===
        "dghs.yolov8.version": _DGHS_YOLOV8_VERSION,
        "dghs.yolov8.exporter": exporter_name,
        "dghs.yolov8.has_embedding": "1" if has_embedding else "0",
        "dghs.yolov8.input_name": "images",
        "dghs.yolov8.output_names": json.dumps(list(output_names)),
    }
    if has_embedding and layer_indices is not None:
        md["dghs.yolov8.embed_layer_indices"] = json.dumps(list(layer_indices))
    if embed_dim is not None:
        md["dghs.yolov8.embed_dim"] = str(int(embed_dim))

    # threshold.json from training - if present, embed it verbatim so the
    # ONNX is fully self-describing without a sidecar file. This matches
    # the legacy yolov8.export pipeline that used to ship threshold.json
    # alongside the .onnx in the export zip.
    if workdir is not None:
        thr_path = os.path.join(workdir, "threshold.json")
        if os.path.isfile(thr_path):
            try:
                with open(thr_path, "r", encoding="utf-8") as f:
                    md["dghs.yolov8.threshold"] = f.read()
            except Exception:
                pass

    if extra_metadata:
        for k, v in extra_metadata.items():
            if v is None:
                continue
            if isinstance(v, str):
                md[str(k)] = v
            elif isinstance(v, (dict, list, tuple)):
                md[str(k)] = json.dumps(v)
            else:
                md[str(k)] = str(v)
    return md


def _attach_metadata(onnx_filename: str, md: Mapping[str, str]) -> None:
    """Replace the ONNX file's ``metadata_props`` with ``md``. Called
    *after* :func:`_maybe_simplify` so simplifier rewrites can't drop
    metadata we just wrote."""
    import onnx

    proto = onnx.load(onnx_filename)
    while len(proto.metadata_props):
        proto.metadata_props.pop()
    for k, v in md.items():
        if v is None:
            continue
        e = proto.metadata_props.add()
        e.key = str(k)
        e.value = str(v)
    onnx.save(proto, onnx_filename)


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

def export_yolo_to_onnx(
    model,
    onnx_filename: str,
    *,
    imgsz: int = 640,
    opset_version: int = 14,
    dynamic: bool = True,
    simplify: bool = True,
    batch: int = 1,
    device: Union[str, torch.device, None] = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> str:
    """Export an Ultralytics model to a single-head ONNX (predictions only).

    Drop-in replacement for the previous wrapper around ``model.export()``
    with three improvements:

    * The graph is written **directly** to ``onnx_filename`` via
      ``torch.onnx.export`` - no temp file, no copy-after-the-fact.
    * Output node naming matches the default
      ``YOLO.export(format='onnx')`` exactly so consumers like
      ``deepghs/imgutils`` remain compatible.
    * The full ultralytics-style ``metadata_props`` is preserved (every
      key the upstream exporter would have written), and a
      ``dghs.yolov8.*`` namespace adds version, exporter, and I/O-name
      info; a workdir source auto-attaches its ``threshold.json``.

    :param model:           ``YOLO`` / ``RTDETR`` wrapper, raw
                            ``BaseModel``, ``.pt`` path, or training
                            workdir containing ``weights/best.pt``.
    :param onnx_filename:   destination path. Parent dirs created if
                            needed.
    :param imgsz:           input letterbox size for tracing.
    :param opset_version:   ONNX opset.
    :param dynamic:         expose dynamic batch / spatial / anchor axes.
    :param simplify:        run ``onnxsim`` after export when available.
    :param batch:           dummy-input batch size for tracing.
    :param device:          where to do the trace.
    :param extra_metadata:  optional ``{key: value}`` injected into the
                            ONNX ``metadata_props`` after the standard
                            keys; values are coerced to ``str`` (dicts /
                            lists go through ``json.dumps``).
    :returns:               the ``onnx_filename`` that was written.
    """
    wrapper, inner_raw, workdir = _resolve_for_export(model)
    if device is None:
        try:
            device = next(inner_raw.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    inner = _prepare_for_export(inner_raw, dynamic).to(device)
    wrap = _PredictOnlyModel(inner).to(device)
    wrap.train(False)
    is_rtdetr = isinstance(inner.model[-1], RTDETRDecoder)

    dummy = torch.zeros(batch, 3, imgsz, imgsz, device=device)
    with torch.inference_mode():
        for _ in range(2):
            out = wrap(dummy)
    head_outputs = list(out) if isinstance(out, (tuple, list)) else [out]
    output_names = [f"output{i}" for i in range(len(head_outputs))]
    head_dim_lens = [t.dim() for t in head_outputs]

    onnx_filename = _export_via_torch(
        wrap, dummy, onnx_filename, output_names, head_dim_lens,
        is_rtdetr, opset_version, dynamic, simplify,
    )
    md = _collect_metadata(
        wrapper, inner, imgsz, batch,
        exporter_name="export_yolo_to_onnx",
        output_names=output_names,
        workdir=workdir,
        extra_metadata=extra_metadata,
    )
    _attach_metadata(onnx_filename, md)
    return onnx_filename


def export_yolo_to_onnx_with_embedding(
    model,
    onnx_filename: str,
    *,
    layer_indices: Sequence[int] | None = None,
    imgsz: int = 640,
    opset_version: int = 14,
    dynamic: bool = True,
    simplify: bool = True,
    batch: int = 1,
    device: Union[str, torch.device, None] = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> str:
    """Export an Ultralytics model to a dual-head ONNX (predictions + embedding).

    Output names are a strict superset of :func:`export_yolo_to_onnx`'s.
    The head's usual ``output0`` (and ``output1`` for ``Segment``) come
    first so existing consumers like ``imgutils.generic.yolo`` keep
    working unchanged; ``embedding`` is appended as the trailing output.

    Numerical guarantee. The ``output0`` tensor produced by this dual-head
    ONNX is byte-equivalent to the ``output0`` produced by
    :func:`export_yolo_to_onnx` for the same model and the same input;
    this is verified end-to-end in ``test/test_embed.py``.

    Metadata. All the ``metadata_props`` :func:`export_yolo_to_onnx`
    writes are present here too, plus ``dghs.yolov8.has_embedding=1``,
    ``dghs.yolov8.embed_layer_indices`` and ``dghs.yolov8.embed_dim``.

    :param model:           ``YOLO`` / ``RTDETR`` wrapper, raw
                            ``BaseModel``, ``.pt`` path, or training
                            workdir.
    :param onnx_filename:   destination path.
    :param layer_indices:   layers to pool & concat for the embedding
                            output. ``None`` -> ``[len(inner.model) - 2]``,
                            matching ``ultralytics.YOLO.embed()``.
    :param imgsz:           input letterbox size for tracing.
    :param opset_version:   ONNX opset.
    :param dynamic:         expose dynamic batch / spatial / anchor axes.
    :param simplify:        run ``onnxsim`` after export when available.
    :param batch:           dummy-input batch size for tracing.
    :param device:          where to do the trace.
    :param extra_metadata:  optional ``{key: value}`` written into the
                            ONNX ``metadata_props`` after the standard
                            keys.
    :returns:               the ``onnx_filename`` that was written.
    """
    wrapper, inner_raw, workdir = _resolve_for_export(model)
    if device is None:
        try:
            device = next(inner_raw.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    inner = _prepare_for_export(inner_raw, dynamic).to(device)
    layer_indices = list(layer_indices) if layer_indices is not None \
        else default_embed_indices(inner)
    wrap = _DualHeadModel(inner, layer_indices).to(device)
    wrap.train(False)
    is_rtdetr = isinstance(inner.model[-1], RTDETRDecoder)

    dummy = torch.zeros(batch, 3, imgsz, imgsz, device=device)
    with torch.inference_mode():
        for _ in range(2):
            out = wrap(dummy)
    *head_outputs, embedding = out
    output_names = [f"output{i}" for i in range(len(head_outputs))] + ["embedding"]
    head_dim_lens = [t.dim() for t in head_outputs] + [embedding.dim()]
    embed_dim = int(embedding.shape[-1])

    onnx_filename = _export_via_torch(
        wrap, dummy, onnx_filename, output_names, head_dim_lens,
        is_rtdetr, opset_version, dynamic, simplify,
    )
    md = _collect_metadata(
        wrapper, inner, imgsz, batch,
        exporter_name="export_yolo_to_onnx_with_embedding",
        output_names=output_names,
        has_embedding=True,
        layer_indices=layer_indices,
        embed_dim=embed_dim,
        workdir=workdir,
        extra_metadata=extra_metadata,
    )
    _attach_metadata(onnx_filename, md)

    # Convenience: stash last-export info on the function object so callers
    # don't have to reload the .onnx to introspect what was emitted.
    export_yolo_to_onnx_with_embedding.last_pred_shapes = tuple(
        tuple(t.shape) for t in head_outputs)
    export_yolo_to_onnx_with_embedding.last_embed_dim = embed_dim
    export_yolo_to_onnx_with_embedding.last_layer_indices = list(layer_indices)
    export_yolo_to_onnx_with_embedding.last_output_names = list(output_names)
    return onnx_filename
