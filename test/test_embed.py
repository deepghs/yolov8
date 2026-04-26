"""Lightweight tests for ``yolov8.embed`` and the dual-head ONNX export.

These tests use yaml-only model instantiation (random weights), so they run
without GPU and without downloading any pretrained ``.pt`` from the internet.
Numerical correctness is verified end-to-end (PyTorch <-> ONNX Runtime); the
absolute values don't matter, only that the two backends agree.
"""
from __future__ import annotations

import numpy as np
import onnxruntime as ort
import pytest
import torch
from ultralytics import RTDETR, YOLO

from yolov8.embed import (
    EmbedHead,
    default_embed_indices,
    get_embedding,
    resolve_inner,
)
from yolov8.onnx import export_yolo_to_onnx_with_embedding


# Picked to be small but valid for every family: every YOLO head strides at
# most 32x, so 96 leaves a 3x3 final feature map; RT-DETR's hybrid encoder
# needs at least 128.
_IMGSZ_YOLO = 96
_IMGSZ_RTDETR = 128


YOLO_YAMLS = [
    # v3u
    ("yolov3-tinyu.yaml", "detect"),
    # v5u
    ("yolov5nu.yaml", "detect"),
    # v8 + task variants
    ("yolov8n.yaml", "detect"),
    ("yolov8n-seg.yaml", "segment"),
    ("yolov8n-pose.yaml", "pose"),
    ("yolov8n-obb.yaml", "obb"),
    ("yolov8n-cls.yaml", "classify"),
    # v9 / v10 / v11
    ("yolov9t.yaml", "detect"),
    ("yolov10n.yaml", "detect"),
    ("yolo11n.yaml", "detect"),
]


@pytest.fixture(scope="module")
def cpu():
    return torch.device("cpu")


@pytest.mark.unittest
class TestResolveInner:
    def test_yolo_wrapper(self):
        model = YOLO("yolov8n.yaml")
        inner = resolve_inner(model)
        assert hasattr(inner, "predict")
        assert hasattr(inner, "model")
        assert len(inner.model) > 0

    def test_rtdetr_wrapper(self):
        model = RTDETR("rtdetr-l.yaml")
        inner = resolve_inner(model)
        assert hasattr(inner, "predict")
        assert hasattr(inner, "model")

    def test_raw_inner_passes_through(self):
        model = YOLO("yolov8n.yaml")
        inner = model.model
        assert resolve_inner(inner) is inner

    def test_default_indices(self):
        inner = YOLO("yolov8n.yaml").model
        assert default_embed_indices(inner) == [len(inner.model) - 2]


@pytest.mark.unittest
@pytest.mark.parametrize("yaml_name,task", YOLO_YAMLS)
class TestGetEmbeddingYOLO:
    def test_tensor_input_shape_and_dtype(self, yaml_name, task, cpu):
        model = YOLO(yaml_name)
        x = torch.randn(2, 3, _IMGSZ_YOLO, _IMGSZ_YOLO)
        emb = get_embedding(model, x, device=cpu)
        assert emb.ndim == 2
        assert emb.shape[0] == 2
        assert emb.dtype == torch.float32
        assert torch.isfinite(emb).all()
        assert model.task == task  # sanity: yaml -> task mapping holds

    def test_normalize_yields_unit_norm(self, yaml_name, task, cpu):
        model = YOLO(yaml_name)
        x = torch.randn(3, 3, _IMGSZ_YOLO, _IMGSZ_YOLO)
        emb = get_embedding(model, x, normalize=True, device=cpu)
        norms = emb.norm(p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_deterministic_under_inference_mode(self, yaml_name, task, cpu):
        model = YOLO(yaml_name)
        x = torch.randn(1, 3, _IMGSZ_YOLO, _IMGSZ_YOLO)
        a = get_embedding(model, x, device=cpu)
        b = get_embedding(model, x, device=cpu)
        assert torch.allclose(a, b, atol=1e-6)


@pytest.mark.unittest
class TestGetEmbeddingMisc:
    def test_rtdetr(self, cpu):
        model = RTDETR("rtdetr-l.yaml")
        x = torch.randn(1, 3, _IMGSZ_RTDETR, _IMGSZ_RTDETR)
        emb = get_embedding(model, x, device=cpu)
        assert emb.shape == (1, 256)

    def test_custom_layer_indices_concat(self, cpu):
        model = YOLO("yolov8n.yaml")
        x = torch.randn(1, 3, _IMGSZ_YOLO, _IMGSZ_YOLO)
        d_default = get_embedding(model, x, device=cpu).shape[-1]
        # Three explicit P3/P4/P5-ish layers; their concat is wider than the
        # single default layer.
        e_multi = get_embedding(model, x, layer_indices=[15, 18, 21], device=cpu)
        assert e_multi.shape[-1] > d_default

    def test_invalid_layer_index_raises(self, cpu):
        model = YOLO("yolov8n.yaml")
        x = torch.randn(1, 3, _IMGSZ_YOLO, _IMGSZ_YOLO)
        with pytest.raises(ValueError):
            get_embedding(model, x, layer_indices=[999], device=cpu)
        with pytest.raises(ValueError):
            get_embedding(model, x, layer_indices=[], device=cpu)

    def test_numpy_input(self, cpu):
        model = YOLO("yolov8n.yaml")
        # HWC BGR uint8, like cv2.imread output
        arr = np.random.randint(0, 256, size=(120, 160, 3), dtype=np.uint8)
        emb = get_embedding(model, arr, imgsz=_IMGSZ_YOLO, device=cpu)
        assert emb.shape[0] == 1


@pytest.mark.unittest
class TestEmbedHeadParity:
    """``EmbedHead`` must produce the same vector as ``inner.predict(x, embed=...)``."""

    def test_yolov8n_parity(self, cpu):
        model = YOLO("yolov8n.yaml").to(cpu)
        inner = model.model
        inner.train(False)
        x = torch.randn(1, 3, _IMGSZ_YOLO, _IMGSZ_YOLO)
        idx = default_embed_indices(inner)
        with torch.inference_mode():
            ref = inner.predict(x, embed=idx)
            ref = torch.stack(list(ref), dim=0) if isinstance(ref, (tuple, list)) else ref
            head = EmbedHead(inner, idx)
            out = head(x)
        assert out.shape == ref.shape
        assert torch.allclose(out, ref, atol=1e-5)

    def test_yolov8n_seg_parity(self, cpu):
        model = YOLO("yolov8n-seg.yaml").to(cpu)
        inner = model.model
        inner.train(False)
        x = torch.randn(1, 3, _IMGSZ_YOLO, _IMGSZ_YOLO)
        with torch.inference_mode():
            ref = torch.stack(list(inner.predict(x, embed=default_embed_indices(inner))), dim=0)
            out = EmbedHead(inner).forward(x)
        assert torch.allclose(out, ref, atol=1e-5)


@pytest.mark.unittest
class TestDualHeadONNX:
    """End-to-end: PyTorch dual-head wrapper output == onnxruntime output."""

    @pytest.mark.parametrize("yaml_name", [
        "yolov8n.yaml",
        "yolov8n-seg.yaml",
        "yolov8n-pose.yaml",
        "yolov8n-cls.yaml",
        "yolo11n.yaml",
        "yolov9t.yaml",
        "yolov10n.yaml",
        "yolov5nu.yaml",
    ])
    def test_yolo_dual_head(self, tmp_path, yaml_name, cpu):
        model = YOLO(yaml_name)
        out_path = tmp_path / "dual.onnx"
        export_yolo_to_onnx_with_embedding(
            model, str(out_path),
            imgsz=_IMGSZ_YOLO,
            opset_version=14,
            dynamic=False,
            simplify=False,
            device=cpu,
        )
        assert out_path.exists()

        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        names = [o.name for o in sess.get_outputs()]
        assert names == ["predictions", "embedding"]

        x_np = np.random.randn(1, 3, _IMGSZ_YOLO, _IMGSZ_YOLO).astype(np.float32)
        pred_onnx, emb_onnx = sess.run(["predictions", "embedding"], {"images": x_np})
        assert pred_onnx.shape[0] == 1
        assert emb_onnx.ndim == 2 and emb_onnx.shape[0] == 1
        assert np.isfinite(emb_onnx).all()

        # PyTorch reference using the public extractor must match the onnx
        # embedding column-for-column.
        x_t = torch.from_numpy(x_np)
        emb_torch = get_embedding(model, x_t, device=cpu).cpu().numpy()
        max_diff = float(np.abs(emb_torch - emb_onnx).max())
        assert max_diff < 1e-3, f"embedding mismatch: max|d|={max_diff}"

    def test_rtdetr_dual_head(self, tmp_path, cpu):
        model = RTDETR("rtdetr-l.yaml")
        out_path = tmp_path / "rtdetr_dual.onnx"
        export_yolo_to_onnx_with_embedding(
            model, str(out_path),
            imgsz=_IMGSZ_RTDETR,
            opset_version=16,  # RT-DETR uses ops better-supported in opset>=16
            dynamic=False,
            simplify=False,
            device=cpu,
        )
        assert out_path.exists()

        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        names = [o.name for o in sess.get_outputs()]
        assert names == ["predictions", "embedding"]

        x_np = np.random.randn(1, 3, _IMGSZ_RTDETR, _IMGSZ_RTDETR).astype(np.float32)
        pred_onnx, emb_onnx = sess.run(["predictions", "embedding"], {"images": x_np})
        assert emb_onnx.shape == (1, 256)
        assert np.isfinite(emb_onnx).all()

        x_t = torch.from_numpy(x_np)
        emb_torch = get_embedding(model, x_t, device=cpu).cpu().numpy()
        assert np.allclose(emb_torch, emb_onnx, atol=1e-3)
