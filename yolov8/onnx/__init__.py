"""ONNX export and verification helpers for Ultralytics-trained models.

Subpackage layout:

* :mod:`yolov8.onnx.export` — the two public exporters
  :func:`~yolov8.onnx.export.export_yolo_to_onnx` (predict-only) and
  :func:`~yolov8.onnx.export.export_yolo_to_onnx_with_embedding`
  (dual-head: predictions + pooled embedding).
* :mod:`yolov8.onnx.consistency` — post-export numerical-parity check
  via :func:`~yolov8.onnx.consistency.compare_onnx_vs_torch`, plus the
  bundled sample image discovered by
  :func:`~yolov8.onnx.consistency.default_sample_image`.
* :mod:`yolov8.onnx._wrapper` and :mod:`yolov8.onnx._metadata` —
  private internals (ONNX-traceable forward wrappers and the metadata
  writer respectively).

The exporters are named to match the legacy single-module surface so
callers that imported ``from yolov8.onnx import export_yolo_to_onnx``
keep working unchanged.

Example::

    >>> from ultralytics import YOLO
    >>> from yolov8.onnx import export_yolo_to_onnx, compare_onnx_vs_torch
    >>> model = YOLO("yolov8n.pt")
    >>> export_yolo_to_onnx(model, "/tmp/yolov8n.onnx")
    '/tmp/yolov8n.onnx'
    >>> report = compare_onnx_vs_torch(model, "/tmp/yolov8n.onnx")
    >>> report["pass"]
    True
"""
from .consistency import compare_onnx_vs_torch, default_sample_image
from .export import export_yolo_to_onnx, export_yolo_to_onnx_with_embedding

__all__ = [
    "export_yolo_to_onnx",
    "export_yolo_to_onnx_with_embedding",
    "compare_onnx_vs_torch",
    "default_sample_image",
]
