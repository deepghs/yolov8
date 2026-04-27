"""Universal image-embedding extractor for any Ultralytics-supported model.

The Ultralytics ``BaseModel._predict_once`` (and the task-specific
``RTDETRDetectionModel.predict`` etc.) carries an ``embed=[layer_idx, ...]``
parameter: it walks the network up to ``max(embed)``, applies
``adaptive_avg_pool2d -> flatten`` on every requested layer's feature map,
concatenates them along the channel dim, and returns one 1D vector per batch
item. This module wraps that machinery into a single ``get_embedding`` call
that accepts whatever the user already has in hand:

  * an ``ultralytics.YOLO`` / ``ultralytics.RTDETR`` wrapper
  * a raw ``BaseModel`` / ``DetectionModel`` / ... instance
  * any object exposing ``.model`` whose value is the inner ``nn.Module``

and whatever input form is convenient:

  * a path / list of paths to image files
  * a ``numpy.ndarray`` ``[H, W, 3]`` (BGR uint8) or batch ``[N, H, W, 3]``
  * a preprocessed ``torch.Tensor`` ``[N, 3, H, W]`` (float32, 0-1, RGB)

The default embedding layer is the second-to-last index of the inner
``ModuleList`` (``len(inner.model) - 2``), which matches Ultralytics'
high-level ``YOLO.embed(source)`` default. Pass ``layer_indices`` to combine
multiple scales (channel-wise concat).
"""
from __future__ import annotations

import os
from typing import Iterable, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ImageLike = Union[str, os.PathLike, np.ndarray, torch.Tensor, "list", "tuple"]


# ---------------------------------------------------------------------------
# Inner-model resolution
# ---------------------------------------------------------------------------

def resolve_inner(model) -> nn.Module:
    """Return the underlying ``BaseModel`` ``nn.Module`` from any
    Ultralytics wrapper or raw model.

    Both ``ultralytics.YOLO`` and ``ultralytics.RTDETR`` expose the
    inner network as ``wrapper.model`` (an ``nn.Module``). Raw
    ``BaseModel`` subclasses *also* have a ``.model`` attribute, but
    theirs is the ``nn.ModuleList`` of layers, not an ``nn.Module``.
    We disambiguate by inspecting whether the candidate has its own
    ``predict`` method - the inner ``BaseModel`` does, the layer
    ``ModuleList`` does not.

    :param model: Any of ``ultralytics.YOLO``, ``ultralytics.RTDETR``,
        a raw ``BaseModel``-style instance, or anything exposing
        ``.model`` whose value satisfies the protocol described above.
    :type model: ultralytics.YOLO or ultralytics.RTDETR or
        torch.nn.Module
    :returns: The inner ``BaseModel`` instance (an ``nn.Module``).
    :rtype: torch.nn.Module
    :raises TypeError: When ``model`` does not look like any of the
        accepted shapes.

    Example::

        >>> from ultralytics import YOLO
        >>> from yolov8.embed import resolve_inner
        >>> wrapper = YOLO("yolov8n.pt")
        >>> inner = resolve_inner(wrapper)
        >>> hasattr(inner, "predict")
        True
    """
    candidate = getattr(model, "model", model)
    # ``BaseModel`` (and all Ultralytics task models) define ``predict``;
    # plain ``nn.ModuleList`` does not.
    if hasattr(candidate, "predict") and hasattr(candidate, "model"):
        return candidate
    # ``model`` was already the inner BaseModel
    if hasattr(model, "predict") and hasattr(model, "model"):
        return model
    raise TypeError(
        f"Cannot resolve inner BaseModel from {type(model).__name__}; "
        f"expected ultralytics.YOLO/RTDETR wrapper or a BaseModel subclass."
    )


def default_embed_indices(inner: nn.Module) -> list[int]:
    """Return the default embed-layer index list: the head's
    immediately-preceding layer.

    Matches the default of ``ultralytics.engine.model.Model.embed`` -
    the last neck layer, which after pooling gives a single
    ``[B, channels]`` feature vector.

    :param inner: Inner ``BaseModel`` (use :func:`resolve_inner` to get
        one from a wrapper).
    :type inner: torch.nn.Module
    :returns: ``[len(inner.model) - 2]``.
    :rtype: list[int]

    Example::

        >>> from ultralytics import YOLO
        >>> from yolov8.embed import default_embed_indices, resolve_inner
        >>> default_embed_indices(resolve_inner(YOLO("yolov8n.pt")))
        [21]
    """
    return [len(inner.model) - 2]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _letterbox_bgr(im: np.ndarray, new_shape: int, color=(114, 114, 114)) -> np.ndarray:
    """Aspect-preserving resize + pad to a square ``new_shape``. Mirrors the
    LetterBox transform Ultralytics uses for prediction so the tensor we feed
    in matches what ``YOLO.predict`` would feed."""
    import cv2  # heavy dep, lazy import

    h0, w0 = im.shape[:2]
    r = min(new_shape / h0, new_shape / w0)
    nh, nw = int(round(h0 * r)), int(round(w0 * r))
    if (nh, nw) != (h0, w0):
        im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new_shape - nh) // 2
    left = (new_shape - nw) // 2
    bottom = new_shape - nh - top
    right = new_shape - nw - left
    return cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def _read_image_bgr(path: Union[str, os.PathLike]) -> np.ndarray:
    import cv2

    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return bgr


def _to_input_tensor(source: ImageLike, imgsz: int, device: torch.device,
                     dtype: torch.dtype) -> torch.Tensor:
    """Coerce ``source`` to an ``[N, 3, imgsz, imgsz]`` float tensor on
    ``device`` with values in ``[0, 1]`` and channel order RGB."""
    import cv2

    if isinstance(source, torch.Tensor):
        t = source
        if t.ndim == 3:
            t = t.unsqueeze(0)
        if t.ndim != 4 or t.shape[1] != 3:
            raise ValueError(f"expected tensor of shape [N,3,H,W], got {tuple(t.shape)}")
        return t.to(device=device, dtype=dtype)

    if isinstance(source, np.ndarray):
        arr = source
        # Accept HWC or NHWC arrays, both BGR uint8 by Ultralytics convention.
        if arr.ndim == 3:
            arr = arr[None, ...]
        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise ValueError(f"expected ndarray of shape [H,W,3] or [N,H,W,3], got {arr.shape}")
        items = [_letterbox_bgr(im, imgsz) for im in arr]
    else:
        if isinstance(source, (str, os.PathLike)):
            paths: Sequence = [source]
        elif isinstance(source, (list, tuple)):
            paths = source
        else:
            raise TypeError(f"unsupported source type: {type(source).__name__}")
        items = [_letterbox_bgr(_read_image_bgr(p), imgsz) for p in paths]

    rgb = np.stack([cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in items], axis=0)
    t = torch.from_numpy(rgb).to(device=device, dtype=dtype).div_(255.0)
    return t.permute(0, 3, 1, 2).contiguous()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@torch.inference_mode()
def get_embedding(model,
                  source: ImageLike,
                  layer_indices: Iterable[int] | None = None,
                  imgsz: int = 640,
                  normalize: bool = False,
                  device: torch.device | str | None = None) -> torch.Tensor:
    """Return ``[N, D]`` image embeddings from any Ultralytics model object.

    The output dim ``D`` equals the sum of channels of the requested
    layers' feature maps after ``adaptive_avg_pool2d``. For RT-DETR /
    WorldModel / classification heads this works without any
    special-casing — internally it routes through :class:`EmbedHead`,
    which re-implements the embed branch of
    ``BaseModel._predict_once`` and is therefore independent of the
    ultralytics ``predict(..., embed=...)`` keyword (added in ~8.1).

    :param model: ``ultralytics.YOLO`` / ``ultralytics.RTDETR`` wrapper,
        or a raw ``BaseModel`` instance.
    :type model: ultralytics.YOLO or ultralytics.RTDETR or
        torch.nn.Module
    :param source: An image path, a list of paths, an ``np.ndarray``
        (HWC or NHWC BGR uint8), or a preprocessed ``torch.Tensor``
        of shape ``[N, 3, H, W]``.
    :type source: str or os.PathLike or numpy.ndarray or torch.Tensor
        or list or tuple
    :param layer_indices: Layers to pool & concat for the embedding.
        ``None`` selects the second-to-last layer (the same default
        as ``ultralytics.YOLO.embed()``).
    :type layer_indices: Iterable[int] or None
    :param imgsz: Input letterbox size when ``source`` is a path /
        ndarray. Ignored for tensor inputs.
    :type imgsz: int
    :param normalize: When ``True``, L2-normalise each row of the
        output (``||row||₂ == 1``).
    :type normalize: bool
    :param device: Where to run. ``None`` keeps the model's current
        device.
    :type device: torch.device or str or None
    :returns: Tensor of shape ``[N, D]`` with one row per input image.
    :rtype: torch.Tensor
    :raises ValueError: If ``layer_indices`` is empty or out of range.

    Example::

        >>> from ultralytics import YOLO
        >>> from yolov8.embed import get_embedding
        >>> model = YOLO("yolov8n.pt")
        >>> emb = get_embedding(model, "bus.jpg", normalize=True)
        >>> emb.shape, float(emb.norm(dim=-1)[0])
        (torch.Size([1, 256]), 1.0)
    """
    inner = resolve_inner(model)
    if device is not None:
        inner = inner.to(torch.device(device))
    inner.train(False)

    target_device = next(inner.parameters()).device
    target_dtype = next(inner.parameters()).dtype

    if layer_indices is None:
        embed = default_embed_indices(inner)
    else:
        embed = sorted(set(int(i) for i in layer_indices))
        if not embed:
            raise ValueError("layer_indices must be non-empty")
        if max(embed) >= len(inner.model):
            raise ValueError(
                f"layer index {max(embed)} >= model depth {len(inner.model)}; "
                f"valid range is [0, {len(inner.model) - 1}]")

    x = _to_input_tensor(source, imgsz, target_device, target_dtype)
    # We deliberately do not call ``inner.predict(x, embed=embed)`` even
    # though that is the supported public API on recent ultralytics builds:
    # the ``embed=`` keyword was only added in ultralytics ~8.1.x and is
    # missing on 8.0.x (used by the legacy roboflow publish path). Routing
    # through :class:`EmbedHead` re-implements the same pool-and-concat as
    # ``BaseModel._predict_once`` does, so we get identical numerics across
    # every ultralytics release in our supported range without introducing
    # a version probe.
    head = EmbedHead(inner, embed)
    emb = head(x)

    if normalize:
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
    return emb


class EmbedHead(nn.Module):
    """Static, traceable rewrite of the embed branch of
    ``BaseModel._predict_once``.

    Used internally by
    :func:`yolov8.onnx.export_yolo_to_onnx_with_embedding` to provide
    the embedding side of the dual-head ONNX, and exposed here so
    callers who want a pure-PyTorch embedding-only graph (no head)
    can reuse it without rebuilding the layer-walk loop themselves.

    :param inner: Inner ``BaseModel`` instance. Use
        :func:`resolve_inner` to obtain one from a wrapper.
    :type inner: torch.nn.Module
    :param layer_indices: Layer indices to pool & concat. ``None``
        falls back to :func:`default_embed_indices`.
    :type layer_indices: Sequence[int] or None

    Example::

        >>> from ultralytics import YOLO
        >>> from yolov8.embed import EmbedHead, resolve_inner
        >>> import torch
        >>> inner = resolve_inner(YOLO("yolov8n.pt"))
        >>> head = EmbedHead(inner)
        >>> head(torch.randn(1, 3, 640, 640)).shape
        torch.Size([1, 256])
    """

    def __init__(self, inner: nn.Module, layer_indices: Sequence[int] | None = None):
        super().__init__()
        self.inner = inner
        idx = sorted(set(int(i) for i in (layer_indices or default_embed_indices(inner))))
        self.embed_indices = idx
        self.max_idx = max(idx)
        self.save = set(int(i) for i in inner.save) | set(idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y: list[torch.Tensor | None] = []
        feats: list[torch.Tensor] = []
        for m in self.inner.model:
            if m.i > self.max_idx:
                break
            if m.f != -1:
                if isinstance(m.f, int):
                    x = y[m.f]
                else:
                    x = [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
            if m.i in self.embed_indices:
                feats.append(F.adaptive_avg_pool2d(x, 1).flatten(1))
        return torch.cat(feats, dim=1)
