import json
import os.path
from typing import Optional, Union

from ditk import logging
from ultralytics import YOLO

from ..utils import compute_threshold_data
from ._threshold_callback import make_on_train_end_threshold_writer
from ._threshold_recovery import recover_threshold_via_val


def train_segmentation(workdir: str, train_cfg: str, level: str = 's', yversion: Union[int, str] = 8,
                       max_epochs: int = 200, batch: int = 16, pretrained: Optional[str] = None, **kwargs):
    """Train an Ultralytics segmentation model into ``workdir``.

    Same shape as :func:`yolov8.train.train_object_detection` but
    picks ``yolov{N}{level}-seg.pt`` for the pretrained checkpoint and
    captures mask-level F1 curves (``kind='seg'``) for the threshold
    JSON. RT-DETR is not supported here - use detection for RT-DETR.

    :param workdir: Run output directory. See
        :func:`train_object_detection` for the full layout
        conventions.
    :type workdir: str
    :param train_cfg: Dataset YAML path or directory containing
        ``data.yaml`` / ``data.yml``.
    :type train_cfg: str
    :param level: Model size suffix (``"n"`` / ``"s"`` / ``"m"`` /
        ``"l"`` / ``"x"``).
    :type level: str
    :param yversion: YOLO family selector (``8`` / ``11`` / ...).
    :type yversion: int or str
    :param max_epochs: Maximum epochs.
    :type max_epochs: int
    :param batch: Per-device batch size.
    :type batch: int
    :param pretrained: Override the pretrained-checkpoint source.
    :type pretrained: str or None
    :param kwargs: Forwarded to ``model.train(...)``.

    Example::

        >>> from yolov8.train import train_segmentation
        >>> train_segmentation(
        ...     workdir="runs/seg_smoke",
        ...     train_cfg="coco8-seg.yaml",
        ...     level="n",
        ...     max_epochs=3,
        ...     batch=4,
        ... )
    """
    logging.try_init_root(logging.INFO)

    # Load a pretrained YOLO model (recommended for training)
    previous_pt = os.path.join(workdir, 'weights', 'last.pt')
    if pretrained and os.path.isdir(pretrained):
        pretrained = os.path.join(pretrained, 'weights', 'best.pt')
    if yversion in {11, '11', 12, '12'}:
        model = YOLO(pretrained or f'yolo{yversion}{level}-seg.pt')
    else:
        model = YOLO(pretrained or f'yolov{yversion}{level}-seg.pt')
    resume = os.path.exists(previous_pt)
    workdir = os.path.abspath(workdir)
    logging.info(f'Workdir: {workdir!r}')
    os.makedirs(workdir, exist_ok=True)

    if os.path.isdir(train_cfg):
        if os.path.exists(os.path.join(train_cfg, 'data.yaml')):
            train_cfg = os.path.join(train_cfg, 'data.yaml')
        elif os.path.exists(os.path.join(train_cfg, 'data.yml')):
            train_cfg = os.path.join(train_cfg, 'data.yml')
        else:
            raise IsADirectoryError(f'train_cfg {train_cfg} is a directory, please given a configuration file.')

    # See yolov8/train/object_detection.py for why we use a callback
    # instead of a post-``model.train()`` block. ``kind='seg'`` matches
    # what the legacy OCR path read out of ``MaskF1_curve.png``; the
    # underlying helper falls back to bbox-level curves when mask
    # metrics aren't available.
    model.add_callback('on_train_end',
                       make_on_train_end_threshold_writer(workdir, kind='seg'))

    # Train the model using the 'coco128.yaml' dataset for 3 epochs
    model.train(
        data=train_cfg,
        epochs=max_epochs,
        batch=batch,
        name=os.path.basename(workdir),
        project=os.path.dirname(workdir),
        save=True,
        plots=True,
        exist_ok=True,
        resume=resume,
        **kwargs
    )

    # In-memory fallback for callback bypass; no-op when the callback
    # already wrote the file. See yolov8/train/object_detection.py
    # for the DDP-recovery rationale.
    threshold_path = os.path.join(workdir, 'threshold.json')
    if not os.path.exists(threshold_path):
        try:
            threshold_data = compute_threshold_data(model, kind='seg')
        except Exception as err:
            logging.warning(f'compute_threshold_data failed: {err!r}; '
                            f'will try DDP recovery instead')
            threshold_data = None
        if threshold_data is not None:
            logging.info(f'Writing F1 / threshold metadata to {threshold_path!r}')
            with open(threshold_path, 'w') as f:
                json.dump(threshold_data, f, ensure_ascii=False, indent=4)

    # DDP recovery (see object_detection.py for full rationale).
    recover_threshold_via_val(
        workdir, train_cfg,
        kind='seg', is_rtdetr=False,
        val_device=kwargs.get('device'),
    )
