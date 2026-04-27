"""ONNX metadata writer.

Builds the ``metadata_props`` dict that mirrors what
``ultralytics.engine.exporter`` writes (so consumers like
``deepghs/imgutils`` keep working unchanged) and adds a
``dghs.yolov8.*`` namespace describing this export specifically.

Public surface (within :mod:`yolov8.onnx`):

* :func:`collect_metadata` — build the dict from a wrapper / inner pair
  plus optional workdir / extra-metadata.
* :func:`attach_metadata` — write that dict as ``metadata_props`` on an
  existing ONNX file.

Both are private to :mod:`yolov8.onnx`; downstream callers should use
the high-level :mod:`yolov8.onnx.export` functions instead.
"""
from __future__ import annotations

import json
import os.path
from datetime import datetime
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from ..config.meta import __VERSION__ as _DGHS_YOLOV8_VERSION


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
    """Infer the ultralytics task name from the head module class.

    :param inner: Inner ``BaseModel``.
    :type inner: torch.nn.Module
    :returns: ``"detect"`` / ``"segment"`` / ``"pose"`` / ``"obb"`` /
        ``"classify"``. Unknown heads default to ``"detect"``.
    :rtype: str
    """
    head_cls = type(inner.model[-1]).__name__
    for key, task in _HEAD_TO_TASK.items():
        if key in head_cls:
            return task
    return "detect"


def _stride_int(inner: nn.Module) -> int:
    """Return the model's max stride as an int, defaulting to 32 when
    the attribute is absent or unreadable.

    :param inner: Inner ``BaseModel``.
    :type inner: torch.nn.Module
    :returns: Max output stride (e.g. ``32`` for vanilla YOLO).
    :rtype: int
    """
    s = getattr(inner, "stride", None)
    if s is None:
        return 32
    try:
        if hasattr(s, "max"):
            return int(s.max().item() if hasattr(s.max(), "item") else max(s))
        return int(max(s))
    except Exception:
        return 32


def collect_metadata(
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
    """Build the ``metadata_props`` dict for an ONNX export.

    Mirrors what ``ultralytics.engine.exporter`` writes (every key the
    upstream exporter would have written) and adds a ``dghs.yolov8.*``
    namespace describing the package version, the exporter used, the
    I/O naming, and — for dual-head exports — the embedding's layer
    indices and dimension. When the source is a training workdir that
    contains a ``threshold.json``, that file is embedded under
    ``dghs.yolov8.threshold`` so the resulting ``.onnx`` is fully
    self-describing.

    :param wrapper: Original ``YOLO`` / ``RTDETR`` wrapper if available
        (for ``.task`` and ``.names``); ``None`` if a raw ``BaseModel``
        was passed in.
    :type wrapper: ultralytics.YOLO or ultralytics.RTDETR or None
    :param inner: The inner ``BaseModel`` used for stride / fallback
        ``.names`` / head-class introspection.
    :type inner: torch.nn.Module
    :param imgsz: Input letterbox size used for tracing.
    :type imgsz: int or Sequence[int]
    :param batch: Dummy-input batch size used for tracing.
    :type batch: int
    :param exporter_name: Name of the calling exporter, recorded under
        ``dghs.yolov8.exporter``.
    :type exporter_name: str
    :param output_names: List of output node names actually written into
        the graph. Used to populate ``dghs.yolov8.output_names``.
    :type output_names: Sequence[str]
    :param has_embedding: ``True`` if the export emitted an ``embedding``
        output. Sets ``dghs.yolov8.has_embedding`` and gates the embed
        layer / dim keys.
    :type has_embedding: bool
    :param layer_indices: Embed layer indices, recorded under
        ``dghs.yolov8.embed_layer_indices``. Ignored when
        ``has_embedding`` is ``False``.
    :type layer_indices: Sequence[int] or None
    :param embed_dim: Channel dimension of the embedding output,
        recorded under ``dghs.yolov8.embed_dim``.
    :type embed_dim: int or None
    :param workdir: Training workdir whose ``threshold.json`` should be
        slurped into ``dghs.yolov8.threshold``. ``None`` to skip.
    :type workdir: str or None
    :param extra_metadata: Caller-supplied additional key/value pairs.
        Dicts and lists go through ``json.dumps``; everything else
        through ``str()``. Keys that collide with ours are overwritten
        with the caller's value.
    :type extra_metadata: Mapping[str, Any] or None
    :returns: A dict suitable for :func:`attach_metadata`.
    :rtype: dict[str, str]
    """
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
        # Names and order match the dict that exporter._add_metadata
        # builds so any consumer relying on these keys (notably imgutils,
        # which reads ``imgsz`` via json.loads and ``names`` via
        # ast.literal_eval) keeps working unchanged.
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

    # threshold.json from training - embed it verbatim so the ONNX is
    # self-describing without a sidecar file. Matches the legacy
    # yolov8.export pipeline that used to ship threshold.json next to
    # the .onnx in the export zip.
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


def attach_metadata(onnx_filename: str, md: Mapping[str, str]) -> None:
    """Replace the ONNX file's ``metadata_props`` with ``md``.

    Should be called *after* any onnxsim simplification, because some
    onnxsim versions strip ``metadata_props`` during graph rewriting.

    :param onnx_filename: Path to the ``.onnx`` file to mutate in place.
    :type onnx_filename: str
    :param md: Mapping of key/value pairs to write. ``None`` values are
        skipped; everything else is coerced to ``str``.
    :type md: Mapping[str, str]
    """
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
