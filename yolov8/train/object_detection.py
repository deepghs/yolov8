import json
import os.path
from typing import Optional, Union

from ditk import logging
from ultralytics import YOLO, RTDETR


def train_object_detection(workdir: str, train_cfg: str, level: str = 's', yversion: Union[int, str] = 8,
                           max_epochs: int = 200, batch: int = 16, pretrained: Optional[str] = None, **kwargs):
    logging.try_init_root(logging.INFO)

    # Load a pretrained YOLO model (recommended for training)
    previous_pt = os.path.join(workdir, 'weights', 'last.pt')
    if pretrained and os.path.isdir(pretrained):
        pretrained = os.path.join(pretrained, 'weights', 'best.pt')
    if yversion in {11, '11', 12, '12'}:
        model = YOLO(pretrained or f'yolo{yversion}{level}.pt')
        model_type = 'yolo'
    elif isinstance(yversion, str) and yversion.lower() == 'rtdetr':
        model = RTDETR(pretrained or f'rtdetr-{level}.pt')
        model_type = 'rtdetr'
    else:
        model = YOLO(pretrained or f'yolov{yversion}{level}.pt')
        model_type = 'yolo'
    resume = os.path.exists(previous_pt)
    workdir = os.path.abspath(workdir)
    os.makedirs(workdir, exist_ok=True)
    with open(os.path.join(workdir, 'model_type.json'), 'w') as f:
        json.dump({
            'model_type': model_type,
        }, f)

    if os.path.isdir(train_cfg):
        if os.path.exists(os.path.join(train_cfg, 'data.yaml')):
            train_cfg = os.path.join(train_cfg, 'data.yaml')
        elif os.path.exists(os.path.join(train_cfg, 'data.yml')):
            train_cfg = os.path.join(train_cfg, 'data.yml')
        else:
            raise IsADirectoryError(f'train_cfg {train_cfg} is a directory, please given a configuration file.')

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
