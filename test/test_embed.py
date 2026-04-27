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
from yolov8.onnx import (
    export_yolo_to_onnx,
    export_yolo_to_onnx_with_embedding,
)


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
class TestPredictOnlyONNX:
    """``export_yolo_to_onnx`` writes a graph with the imgutils-compatible
    naming contract: input ``images``, output ``output0`` (and ``output1``
    for segmentation), full ultralytics-style metadata_props preserved
    plus a ``dghs.yolov8.*`` namespace."""

    @pytest.mark.parametrize("yaml_name,expected_outputs", [
        ("yolov8n.yaml", ["output0"]),
        ("yolov8n-seg.yaml", ["output0", "output1"]),
        ("yolov8n-pose.yaml", ["output0"]),
        ("yolov8n-cls.yaml", ["output0"]),
        ("yolo11n.yaml", ["output0"]),
        ("yolov9t.yaml", ["output0"]),
        ("yolov10n.yaml", ["output0"]),
        ("yolov5nu.yaml", ["output0"]),
    ])
    def test_yolo_predict_only(self, tmp_path, yaml_name, expected_outputs, cpu):
        import json as _json
        import onnx as _onnx

        model = YOLO(yaml_name)
        out_path = tmp_path / "predict_only.onnx"
        export_yolo_to_onnx(
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
        assert names == expected_outputs
        in_names = [i.name for i in sess.get_inputs()]
        assert in_names == ["images"]

        x_np = np.random.randn(1, 3, _IMGSZ_YOLO, _IMGSZ_YOLO).astype(np.float32)
        outputs = sess.run(expected_outputs, {"images": x_np})
        for o in outputs:
            assert np.isfinite(o).all()

        # Metadata sanity: every ult-style key the upstream exporter
        # writes is present, plus our dghs namespace.
        proto = _onnx.load(str(out_path))
        md = {p.key: p.value for p in proto.metadata_props}
        for k in ("description", "author", "date", "version", "license",
                  "docs", "stride", "task", "batch", "imgsz", "names"):
            assert k in md, f"ult metadata key '{k}' missing"
        assert "dghs.yolov8.version" in md
        assert md["dghs.yolov8.exporter"] == "export_yolo_to_onnx"
        assert md["dghs.yolov8.has_embedding"] == "0"
        # imgutils contract: ``imgsz`` is JSON-parseable; ``names`` is a
        # Python dict literal that imgutils parses with ast - we don't
        # call ast here, just check that it looks right.
        _json.loads(md["imgsz"])
        assert md["names"].startswith("{") and md["names"].endswith("}")


@pytest.mark.unittest
class TestDualHeadONNX:
    """End-to-end: dual-head ONNX has ``output0`` (+``output1`` for seg)
    plus ``embedding`` as the trailing output. The head outputs are
    numerically equivalent to the predict-only export."""

    @pytest.mark.parametrize("yaml_name,head_outputs", [
        ("yolov8n.yaml", ["output0"]),
        ("yolov8n-seg.yaml", ["output0", "output1"]),
        ("yolov8n-pose.yaml", ["output0"]),
        ("yolov8n-cls.yaml", ["output0"]),
        ("yolo11n.yaml", ["output0"]),
        ("yolov9t.yaml", ["output0"]),
        ("yolov10n.yaml", ["output0"]),
        ("yolov5nu.yaml", ["output0"]),
    ])
    def test_yolo_dual_head(self, tmp_path, yaml_name, head_outputs, cpu):
        import onnx as _onnx

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
        assert names == head_outputs + ["embedding"]
        assert [i.name for i in sess.get_inputs()] == ["images"]

        x_np = np.random.randn(1, 3, _IMGSZ_YOLO, _IMGSZ_YOLO).astype(np.float32)
        outs = sess.run(names, {"images": x_np})
        emb_onnx = outs[-1]
        assert emb_onnx.ndim == 2 and emb_onnx.shape[0] == 1
        assert np.isfinite(emb_onnx).all()

        # PyTorch reference must match column-for-column.
        x_t = torch.from_numpy(x_np)
        emb_torch = get_embedding(model, x_t, device=cpu).cpu().numpy()
        max_diff = float(np.abs(emb_torch - emb_onnx).max())
        assert max_diff < 1e-3, f"embedding mismatch: max|d|={max_diff}"

        proto = _onnx.load(str(out_path))
        md = {p.key: p.value for p in proto.metadata_props}
        assert md["dghs.yolov8.exporter"] == "export_yolo_to_onnx_with_embedding"
        assert md["dghs.yolov8.has_embedding"] == "1"
        assert md["dghs.yolov8.embed_dim"] == str(emb_onnx.shape[1])

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
        assert names == ["output0", "embedding"]

        x_np = np.random.randn(1, 3, _IMGSZ_RTDETR, _IMGSZ_RTDETR).astype(np.float32)
        pred_onnx, emb_onnx = sess.run(["output0", "embedding"], {"images": x_np})
        assert emb_onnx.shape == (1, 256)
        assert np.isfinite(emb_onnx).all()

        x_t = torch.from_numpy(x_np)
        emb_torch = get_embedding(model, x_t, device=cpu).cpu().numpy()
        assert np.allclose(emb_torch, emb_onnx, atol=1e-3)


@pytest.mark.unittest
class TestPredictParity:
    """Single-head and dual-head exports must produce numerically
    equivalent ``output0`` for the same model and input. This is the
    guarantee that lets imgutils consumers freely pick either flavour."""

    @pytest.mark.parametrize("yaml_name", [
        "yolov8n.yaml",
        "yolov8n-seg.yaml",
        "yolo11n.yaml",
    ])
    def test_output0_numerical_parity(self, tmp_path, yaml_name, cpu):
        # Need a deterministic init for the parity check, otherwise the two
        # exports each load a fresh randomly-initialised YAML model.
        torch.manual_seed(0)
        model_a = YOLO(yaml_name)
        torch.manual_seed(0)
        model_b = YOLO(yaml_name)

        a_path = tmp_path / "a_predict.onnx"
        b_path = tmp_path / "b_dual.onnx"
        export_yolo_to_onnx(
            model_a, str(a_path),
            imgsz=_IMGSZ_YOLO, opset_version=14,
            dynamic=False, simplify=False, device=cpu,
        )
        export_yolo_to_onnx_with_embedding(
            model_b, str(b_path),
            imgsz=_IMGSZ_YOLO, opset_version=14,
            dynamic=False, simplify=False, device=cpu,
        )

        sa = ort.InferenceSession(str(a_path), providers=["CPUExecutionProvider"])
        sb = ort.InferenceSession(str(b_path), providers=["CPUExecutionProvider"])

        x_np = np.random.randn(1, 3, _IMGSZ_YOLO, _IMGSZ_YOLO).astype(np.float32)
        out_a = sa.run(["output0"], {"images": x_np})[0]
        out_b = sb.run(["output0"], {"images": x_np})[0]
        assert out_a.shape == out_b.shape
        max_diff = float(np.abs(out_a - out_b).max())
        assert max_diff < 1e-3, f"output0 parity failed: max|d|={max_diff}"
