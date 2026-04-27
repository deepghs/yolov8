import json
import os.path
from typing import Optional, Union

from ditk import logging
from ultralytics import YOLO, RTDETR

from ..utils import compute_threshold_data
from ._threshold_callback import make_on_train_end_threshold_writer


def train_object_detection(workdir: str, train_cfg: str, level: str = 's', yversion: Union[int, str] = 8,
                           max_epochs: int = 200, batch: int = 16, pretrained: Optional[str] = None, **kwargs):
    logging.try_init_root(logging.INFO)

    # Load a pretrained YOLO model (recommended for training)
    previous_pt = os.path.join(workdir, 'weights', 'last.pt')
    if pretrained and os.path.isdir(pretrained):
        pretrained = os.path.join(pretrained, 'weights', 'best.pt')
    if yversion in {11, '11', 12, '12'}:
        model = YOLO(pretrained or f'yolo{yversion}{level}.pt')
    elif isinstance(yversion, str) and yversion.lower() == 'rtdetr':
        model = RTDETR(pretrained or f'rtdetr-{level}.pt')
    else:
        model = YOLO(pretrained or f'yolov{yversion}{level}.pt')
    resume = os.path.exists(previous_pt)
    workdir = os.path.abspath(workdir)
    os.makedirs(workdir, exist_ok=True)

    if os.path.isdir(train_cfg):
        if os.path.exists(os.path.join(train_cfg, 'data.yaml')):
            train_cfg = os.path.join(train_cfg, 'data.yaml')
        elif os.path.exists(os.path.join(train_cfg, 'data.yml')):
            train_cfg = os.path.join(train_cfg, 'data.yml')
        else:
            raise IsADirectoryError(f'train_cfg {train_cfg} is a directory, please given a configuration file.')

    # Register an on_train_end callback that writes threshold.json from
    # inside the trainer process. This is the only call site that
    # reliably has populated metrics: it fires after the trainer's final
    # validation pass regardless of how training terminates (normal end,
    # patience exhaustion, time budget) and runs in the trainer's own
    # process, so it works for multi-GPU / DDP runs where the main
    # process never sees the metrics. See _threshold_callback.py.
    model.add_callback('on_train_end',
                       make_on_train_end_threshold_writer(workdir, kind='box'))

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

    # Fallback: if the on_train_end callback did not produce
    # threshold.json (e.g. the user wired up a custom trainer that
    # bypasses the callback), try the post-train read once more. This is
    # a no-op when the callback already wrote the file.
    threshold_path = os.path.join(workdir, 'threshold.json')
    if not os.path.exists(threshold_path):
        try:
            threshold_data = compute_threshold_data(model, kind='box')
        except Exception as err:
            logging.warning(f'compute_threshold_data failed: {err!r}; skipping threshold.json')
            threshold_data = None
        if threshold_data is not None:
            logging.info(f'Writing F1 / threshold metadata to {threshold_path!r}')
            with open(threshold_path, 'w') as f:
                json.dump(threshold_data, f, ensure_ascii=False, indent=4)
