"""Numerical-parity check between the PyTorch model and an exported ONNX.

The export functions in :mod:`yolov8.onnx.export` accept ``verify=True``
(default) which routes through :func:`compare_onnx_vs_torch` to confirm
the produced ``.onnx`` reproduces the PyTorch model's outputs on the
sample image bundled in :mod:`yolov8.onnx.assets` (or any image the
caller supplies).

Standalone usage::

    >>> from yolov8.onnx import export_yolo_to_onnx, compare_onnx_vs_torch
    >>> from ultralytics import YOLO
    >>> model = YOLO("yolov8n.pt")
    >>> export_yolo_to_onnx(model, "/tmp/yolov8n.onnx", verify=False)
    >>> report = compare_onnx_vs_torch(model, "/tmp/yolov8n.onnx")
    >>> report["pass"]
    True

The comparison is on **head outputs only** (``output0`` and, for
segmentation, ``output1``). Pixel-level tolerance is set so the check
catches genuine graph mismatches but tolerates the small Resize /
Upsample numerical drift that some onnxruntime backends introduce.
"""
from __future__ import annotations

import os.path
from pathlib import Path
from typing import Any, Mapping, Sequence, Union

import numpy as np
import torch

from ..embed import _to_input_tensor


_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_DEFAULT_SAMPLE = _ASSETS_DIR / "sample_bus.jpg"


def default_sample_image() -> Path:
    """Return the path to the bundled sample image.

    :returns: Absolute path to ``yolov8/onnx/assets/sample_bus.jpg``.
    :rtype: pathlib.Path
    :raises FileNotFoundError: If the asset was removed from the
        installed package.

    Example::

        >>> from yolov8.onnx.consistency import default_sample_image
        >>> default_sample_image().exists()
        True
    """
    if not _DEFAULT_SAMPLE.is_file():
        raise FileNotFoundError(
            f"sample image not found at {_DEFAULT_SAMPLE}; "
            f"pass image=... to compare_onnx_vs_torch explicitly")
    return _DEFAULT_SAMPLE


def _torch_outputs(inner: torch.nn.Module, x: torch.Tensor,
                   embed_indices: Sequence[int] | None):
    """Run the PyTorch wrapper that the ONNX graph is traced from and
    return its output tensor(s) as a list of numpy arrays.

    Used internally by :func:`compare_onnx_vs_torch`; not part of the
    public API surface.
    """
    from ._wrapper import _DualHeadModel, _PredictOnlyModel, _prepare_for_export

    prepared = _prepare_for_export(inner, dynamic=True).to(x.device)
    if embed_indices is None:
        wrap = _PredictOnlyModel(prepared)
    else:
        wrap = _DualHeadModel(prepared, embed_indices)
    wrap.train(False)
    with torch.inference_mode():
        out = wrap(x)
    if isinstance(out, (tuple, list)):
        return [t.detach().cpu().numpy() for t in out]
    return [out.detach().cpu().numpy()]


def compare_onnx_vs_torch(
    model,
    onnx_path: Union[str, "os.PathLike"],
    *,
    image: Union[str, "os.PathLike", np.ndarray, torch.Tensor, None] = None,
    imgsz: int = 640,
    atol: float = 1e-3,
    cosine_threshold: float = 0.999,
    device: Union[str, torch.device, None] = None,
) -> dict[str, Any]:
    """Run the same input through PyTorch and onnxruntime, compare per-output.

    The check passes when *every* head output simultaneously satisfies
    ``max|diff| <= atol`` **or** ``cosine >= cosine_threshold`` —
    accepting numerical drift in the Resize / Upsample lowering that
    some onnxruntime backends pick up while still catching real graph
    mismatches.

    :param model: ``ultralytics.YOLO`` / ``ultralytics.RTDETR`` wrapper,
        raw ``BaseModel``, ``.pt`` path, or training workdir. Same
        accepted shapes as :func:`yolov8.onnx.export.export_yolo_to_onnx`.
    :type model: ultralytics.YOLO or ultralytics.RTDETR or
        torch.nn.Module or str or os.PathLike
    :param onnx_path: Path to the ``.onnx`` file produced by one of the
        exporters.
    :type onnx_path: str or os.PathLike
    :param image: Image to feed both backends. Can be a path, an HWC BGR
        ``ndarray`` (``cv2.imread`` style), a preprocessed
        ``[N,3,H,W]`` tensor, or ``None`` to use the bundled sample.
    :type image: str or os.PathLike or numpy.ndarray or torch.Tensor or None
    :param imgsz: Letterbox size when ``image`` is a path / ndarray.
        Ignored for tensor inputs.
    :type imgsz: int
    :param atol: Absolute tolerance for the max-difference check.
    :type atol: float
    :param cosine_threshold: Lower bound on per-output cosine similarity
        between flattened PyTorch and ONNX outputs.
    :type cosine_threshold: float
    :param device: Where to run PyTorch. ``None`` inherits from the
        model, falling back to CPU.
    :type device: str or torch.device or None
    :returns: A report dict with keys: ``pass`` (bool),
        ``output_names`` (list of str), ``per_output`` (list of dicts
        with ``name``, ``shape``, ``max_abs_diff``, ``cosine``,
        ``ok``), and ``image`` (the resolved image path or
        ``"<tensor>"``).
    :rtype: dict

    Example::

        >>> from yolov8.onnx import compare_onnx_vs_torch
        >>> report = compare_onnx_vs_torch("yolov8n.pt", "/tmp/yolov8n.onnx")
        >>> report["pass"]
        True
        >>> [o["name"] for o in report["per_output"]]
        ['output0']
    """
    import onnx
    import onnxruntime as ort

    from ._wrapper import _resolve_for_export

    wrapper, inner_raw, _wd = _resolve_for_export(model)
    if device is None:
        try:
            device = next(inner_raw.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    if image is None:
        image = default_sample_image()
    target_dtype = next(inner_raw.parameters()).dtype
    x_t = _to_input_tensor(image, imgsz, device, target_dtype)

    # Decide whether the ONNX has an embedding tail by reading the
    # output name list back from the file: dual-head exports always end
    # in ``embedding``, single-head ones don't.
    proto = onnx.load(str(onnx_path))
    onnx_output_names = [o.name for o in proto.graph.output]
    has_embedding = onnx_output_names and onnx_output_names[-1] == "embedding"

    # Recover embed layer indices from metadata so the PyTorch reference
    # walks the exact same graph the ONNX was traced from.
    embed_indices: list[int] | None = None
    if has_embedding:
        md = {p.key: p.value for p in proto.metadata_props}
        try:
            import json as _json
            embed_indices = _json.loads(md.get("dghs.yolov8.embed_layer_indices", "[]"))
        except Exception:
            embed_indices = None
        if not embed_indices:
            from ..embed import default_embed_indices
            embed_indices = default_embed_indices(inner_raw)

    torch_outs = _torch_outputs(inner_raw, x_t, embed_indices)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_outs = sess.run(onnx_output_names, {"images": x_t.cpu().numpy().astype(np.float32)})

    per_output = []
    overall_ok = True
    for name, t_out, o_out in zip(onnx_output_names, torch_outs, onnx_outs):
        t_out = np.asarray(t_out)
        o_out = np.asarray(o_out)
        if t_out.shape != o_out.shape:
            per_output.append(dict(
                name=name, shape=tuple(o_out.shape),
                max_abs_diff=float("inf"), cosine=0.0, ok=False,
                error=f"shape mismatch: torch={t_out.shape} onnx={o_out.shape}",
            ))
            overall_ok = False
            continue
        diff = np.abs(t_out - o_out)
        max_d = float(diff.max())
        a = t_out.ravel().astype(np.float64)
        b = o_out.ravel().astype(np.float64)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        cos = float(a @ b / denom) if denom > 0 else 1.0
        ok = (max_d <= atol) or (cos >= cosine_threshold)
        per_output.append(dict(
            name=name, shape=tuple(o_out.shape),
            max_abs_diff=max_d, cosine=cos, ok=ok,
        ))
        if not ok:
            overall_ok = False

    return {
        "pass": bool(overall_ok),
        "output_names": list(onnx_output_names),
        "per_output": per_output,
        "image": str(image) if not isinstance(image, torch.Tensor) else "<tensor>",
    }
