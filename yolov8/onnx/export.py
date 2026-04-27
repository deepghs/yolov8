"""Public ONNX export entry points.

Two functions, both writing directly to the user-given path via
``torch.onnx.export``:

* :func:`export_yolo_to_onnx` — single-head export, equivalent to the
  default ``YOLO.export(format='onnx')`` shape but with deterministic
  output path and full metadata preservation.
* :func:`export_yolo_to_onnx_with_embedding` — dual-head export. The
  head's normal output(s) come first under their usual names, and a
  pooled embedding tensor is appended as the trailing output.

Naming contract (verified against ``YOLO.export(format='onnx')`` on
ultralytics 8.0.196 → 8.4.41 and used verbatim by
``deepghs/imgutils``):

* input name: ``images``, shape ``[batch, 3, H, W]``
* head outputs:

  - detect / pose / obb / classify / rtdetr → ``output0``
  - segment → ``output0`` + ``output1``

* dual-head appends ``embedding`` as the trailing output

A dual-head ONNX is therefore a strict superset of a single-head one:
``sess.run(['output0'], {'images': x})`` keeps working when consumers
only need predictions.

Metadata. Both exports preserve every ``metadata_props`` key the
upstream ultralytics exporter writes (``description`` / ``author`` /
``date`` / ``version`` / ``license`` / ``docs`` / ``stride`` / ``task``
/ ``batch`` / ``imgsz`` / ``names``) and add a ``dghs.yolov8.*``
namespace recording the package version, the exporter used, the I/O
naming, and — for dual-head — the embedding's layer indices and
dimension. When the source is a workdir with a ``threshold.json`` it
is embedded under ``dghs.yolov8.threshold`` so the resulting ``.onnx``
is fully self-describing. Callers may inject extra pairs via
``extra_metadata=``.
"""
from __future__ import annotations

import os
import os.path
from typing import Any, Mapping, Sequence, Union

import torch
import torch.nn as nn
from ultralytics.nn.modules import RTDETRDecoder

from ..embed import default_embed_indices
from ._metadata import attach_metadata, collect_metadata
from ._wrapper import (
    _DualHeadModel,
    _PredictOnlyModel,
    _prepare_for_export,
    _resolve_for_export,
)


def _build_dynamic_axes(output_names: Sequence[str],
                        head_dim_lens: Sequence[int],
                        is_rtdetr: bool) -> dict:
    """Construct the ``dynamic_axes`` argument for ``torch.onnx.export``.

    Declares a dynamic batch on every output, a dynamic anchor dim on
    head outputs of detection-style heads (where the anchor count
    scales with ``H/stride * W/stride``), and a dynamic batch on the
    embedding when present. RT-DETR's fixed query count (300) means
    only its batch axis is dynamic.

    :param output_names: Output node names.
    :type output_names: Sequence[str]
    :param head_dim_lens: ``len(t.shape)`` for each head output, in the
        same order as ``output_names`` (used to detect 3D anchor-style
        outputs).
    :type head_dim_lens: Sequence[int]
    :param is_rtdetr: ``True`` when the head is an
        :class:`ultralytics.nn.modules.RTDETRDecoder`.
    :type is_rtdetr: bool
    :returns: A dict suitable for ``torch.onnx.export``'s
        ``dynamic_axes`` keyword.
    :rtype: dict
    """
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
    """Best-effort ``onnxsim.simplify`` pass.

    Leaves the un-simplified graph in place if onnx / onnxsim aren't
    installed or simplification fails. Must be called *before*
    :func:`yolov8.onnx._metadata.attach_metadata` since some onnxsim
    versions strip ``metadata_props`` during graph rewriting.

    :param onnx_filename: Path to the ``.onnx`` file to simplify.
    :type onnx_filename: str
    """
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
    """Run ``torch.onnx.export`` with our naming/dynamic-axes conventions.

    Private helper shared by both exporters. Caller is responsible for
    metadata attachment afterwards.

    :param wrap: The ONNX-traceable wrapper (:class:`_PredictOnlyModel`
        or :class:`_DualHeadModel`).
    :type wrap: torch.nn.Module
    :param dummy: Tracing input.
    :type dummy: torch.Tensor
    :param onnx_filename: Destination path.
    :type onnx_filename: str
    :param output_names: Names to assign each model output, in order.
    :type output_names: Sequence[str]
    :param head_dim_lens: Number of dims per head output (for dynamic
        axes).
    :type head_dim_lens: Sequence[int]
    :param is_rtdetr: Whether the head is RT-DETR (affects anchor-axis
        decision).
    :type is_rtdetr: bool
    :param opset_version: ONNX opset.
    :type opset_version: int
    :param dynamic: Whether to declare dynamic batch / spatial / anchor
        axes.
    :type dynamic: bool
    :param simplify: Whether to run onnxsim afterwards.
    :type simplify: bool
    :returns: ``onnx_filename`` (echoed for chaining).
    :rtype: str
    """
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


def _has_trained_weights(wrapper, workdir) -> bool:
    """Best-effort heuristic: did the model come from a trained checkpoint?

    The ONNX consistency check (:func:`yolov8.onnx.consistency.compare_onnx_vs_torch`)
    compares per-output tensors with strict tolerance, which is fine for
    trained models but noisy on random-init weights — head outputs are
    arbitrary numbers and a single Resize / Upsample lowering quirk can
    push ``max_abs_diff`` to multiple units.

    Treat the model as "trained" when:

    * the source was a workdir (always contains a finished best.pt), or
    * the wrapper has a non-empty ``pt_path`` (set when constructed
      from a ``.pt`` path or one of the ``ultralytics/assets`` names),
      or
    * the wrapper's ``ckpt`` dict contains actual checkpoint contents
      (a ``model`` key). Older ultralytics releases used
      ``wrapper.ckpt is None`` as the yaml-init signal; 8.3.105+
      changed that to ``wrapper.ckpt == {}``, so we test both shapes.

    A wrapper built from a YAML config (``YOLO('yolov8n.yaml')``)
    therefore reports ``False`` and ``verify`` is skipped silently —
    parity tests against arbitrary random-init outputs are noisy and
    not meaningful.

    :param wrapper: The high-level wrapper resolved by
        :func:`_resolve_for_export`. May be ``None`` for raw
        ``BaseModel`` inputs.
    :param workdir: The original workdir if the source was a directory,
        otherwise ``None``.
    :returns: ``True`` if we believe the weights are trained.
    :rtype: bool
    """
    if workdir is not None:
        return True
    if wrapper is None:
        return False
    pt_path = getattr(wrapper, "pt_path", None)
    if pt_path:
        return True
    ckpt = getattr(wrapper, "ckpt", None)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return True
    return False


def _resolve_device(inner: nn.Module, device) -> torch.device:
    """Coerce a ``device`` argument to ``torch.device``.

    :param inner: Inner ``BaseModel``; used as a fallback when
        ``device`` is ``None``.
    :type inner: torch.nn.Module
    :param device: User-supplied device, may be ``None``.
    :returns: A concrete ``torch.device``.
    :rtype: torch.device
    """
    if device is None:
        try:
            return next(inner.parameters()).device
        except StopIteration:
            return torch.device("cpu")
    return torch.device(device)


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
    verify: bool = True,
) -> str:
    """Export an Ultralytics model to a single-head ONNX (predictions only).

    Drop-in replacement for the previous wrapper around
    ``model.export()`` with three improvements:

    * The graph is written **directly** to ``onnx_filename`` via
      ``torch.onnx.export`` — no temp file, no copy-after-the-fact.
    * Output node naming matches the default
      ``YOLO.export(format='onnx')`` exactly so consumers like
      ``deepghs/imgutils`` remain compatible.
    * The full ultralytics-style ``metadata_props`` is preserved (every
      key the upstream exporter would have written), and a
      ``dghs.yolov8.*`` namespace adds version / exporter / I/O-name
      info; a workdir source auto-attaches its ``threshold.json``.

    :param model: ``YOLO`` / ``RTDETR`` wrapper, raw ``BaseModel``,
        ``.pt`` path, or training workdir containing
        ``weights/best.pt``.
    :type model: ultralytics.YOLO or ultralytics.RTDETR or
        torch.nn.Module or str or os.PathLike
    :param onnx_filename: Destination path. Parent directories are
        created if needed.
    :type onnx_filename: str
    :param imgsz: Input letterbox size used for the dummy tracing
        tensor.
    :type imgsz: int
    :param opset_version: ONNX opset.
    :type opset_version: int
    :param dynamic: Expose dynamic batch / spatial / anchor axes.
    :type dynamic: bool
    :param simplify: Run ``onnxsim`` after export when available.
    :type simplify: bool
    :param batch: Dummy-input batch size for tracing.
    :type batch: int
    :param device: Where to do the trace. ``None`` inherits from the
        model, falling back to CPU.
    :type device: str or torch.device or None
    :param extra_metadata: Optional ``{key: value}`` injected into the
        ONNX ``metadata_props`` after the standard keys; values are
        coerced to ``str`` (dicts / lists go through ``json.dumps``).
    :type extra_metadata: Mapping[str, Any] or None
    :param verify: When ``True`` (the default) run
        :func:`yolov8.onnx.consistency.compare_onnx_vs_torch` on the
        bundled sample image and raise ``RuntimeError`` if PyTorch and
        onnxruntime disagree. Pass ``False`` to skip verification (e.g.
        when the trace device disagrees with the available
        onnxruntime providers).
    :type verify: bool
    :returns: The ``onnx_filename`` that was written.
    :rtype: str
    :raises RuntimeError: When ``verify=True`` and the PyTorch / ONNX
        outputs disagree beyond the consistency-checker tolerances.

    Example::

        >>> from ultralytics import YOLO
        >>> from yolov8.onnx import export_yolo_to_onnx
        >>> model = YOLO("yolov8n.pt")
        >>> export_yolo_to_onnx(model, "yolov8n.onnx", imgsz=640)
        'yolov8n.onnx'
    """
    wrapper, inner_raw, workdir = _resolve_for_export(model)
    device = _resolve_device(inner_raw, device)

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
    md = collect_metadata(
        wrapper, inner, imgsz, batch,
        exporter_name="export_yolo_to_onnx",
        output_names=output_names,
        workdir=workdir,
        extra_metadata=extra_metadata,
    )
    attach_metadata(onnx_filename, md)

    if verify and _has_trained_weights(wrapper, workdir):
        # Lazy import to avoid pulling onnxruntime at module import time
        # for callers who never enable verification.
        from .consistency import compare_onnx_vs_torch

        report = compare_onnx_vs_torch(model, onnx_filename, imgsz=imgsz, device=device)
        if not report["pass"]:
            raise RuntimeError(
                f"onnx export verification failed for {onnx_filename}: "
                f"{report['per_output']}")
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
    verify: bool = True,
) -> str:
    """Export an Ultralytics model to a dual-head ONNX (predictions + embedding).

    Output names are a strict superset of :func:`export_yolo_to_onnx`'s.
    The head's usual ``output0`` (and ``output1`` for ``Segment``) come
    first so existing consumers like ``imgutils.generic.yolo`` keep
    working unchanged; ``embedding`` is appended as the trailing output.

    Numerical guarantee: the ``output0`` tensor produced by this
    dual-head ONNX is byte-equivalent to the ``output0`` produced by
    :func:`export_yolo_to_onnx` for the same model and input — verified
    end-to-end in ``test/test_embed.py``.

    Metadata: all the keys :func:`export_yolo_to_onnx` writes are
    present here too, plus ``dghs.yolov8.has_embedding=1``,
    ``dghs.yolov8.embed_layer_indices`` and ``dghs.yolov8.embed_dim``.

    :param model: ``YOLO`` / ``RTDETR`` wrapper, raw ``BaseModel``,
        ``.pt`` path, or training workdir.
    :type model: ultralytics.YOLO or ultralytics.RTDETR or
        torch.nn.Module or str or os.PathLike
    :param onnx_filename: Destination path.
    :type onnx_filename: str
    :param layer_indices: Layers to pool & concat for the embedding
        output. ``None`` → ``[len(inner.model) - 2]``, matching
        ``ultralytics.YOLO.embed()``.
    :type layer_indices: Sequence[int] or None
    :param imgsz: Input letterbox size for tracing.
    :type imgsz: int
    :param opset_version: ONNX opset.
    :type opset_version: int
    :param dynamic: Expose dynamic batch / spatial / anchor axes.
    :type dynamic: bool
    :param simplify: Run ``onnxsim`` after export when available.
    :type simplify: bool
    :param batch: Dummy-input batch size for tracing.
    :type batch: int
    :param device: Where to do the trace.
    :type device: str or torch.device or None
    :param extra_metadata: Optional extra ``metadata_props`` entries.
    :type extra_metadata: Mapping[str, Any] or None
    :param verify: When ``True`` (default) run
        :func:`yolov8.onnx.consistency.compare_onnx_vs_torch` post-export.
    :type verify: bool
    :returns: The ``onnx_filename`` that was written.
    :rtype: str
    :raises RuntimeError: When ``verify=True`` and outputs disagree.

    Example::

        >>> from ultralytics import YOLO
        >>> from yolov8.onnx import export_yolo_to_onnx_with_embedding
        >>> model = YOLO("yolov8n.pt")
        >>> export_yolo_to_onnx_with_embedding(model, "yolov8n_embed.onnx")
        'yolov8n_embed.onnx'
    """
    wrapper, inner_raw, workdir = _resolve_for_export(model)
    device = _resolve_device(inner_raw, device)

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
    md = collect_metadata(
        wrapper, inner, imgsz, batch,
        exporter_name="export_yolo_to_onnx_with_embedding",
        output_names=output_names,
        has_embedding=True,
        layer_indices=layer_indices,
        embed_dim=embed_dim,
        workdir=workdir,
        extra_metadata=extra_metadata,
    )
    attach_metadata(onnx_filename, md)

    # Convenience: stash last-export info on the function object so
    # callers don't have to reload the .onnx to introspect what was
    # emitted.
    export_yolo_to_onnx_with_embedding.last_pred_shapes = tuple(
        tuple(t.shape) for t in head_outputs)
    export_yolo_to_onnx_with_embedding.last_embed_dim = embed_dim
    export_yolo_to_onnx_with_embedding.last_layer_indices = list(layer_indices)
    export_yolo_to_onnx_with_embedding.last_output_names = list(output_names)

    if verify and _has_trained_weights(wrapper, workdir):
        from .consistency import compare_onnx_vs_torch

        report = compare_onnx_vs_torch(model, onnx_filename, imgsz=imgsz, device=device)
        if not report["pass"]:
            raise RuntimeError(
                f"onnx export verification failed for {onnx_filename}: "
                f"{report['per_output']}")
    return onnx_filename
