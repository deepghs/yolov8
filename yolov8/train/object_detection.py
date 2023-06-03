import os.path

from ditk import logging
from ultralytics import YOLO


def train_object_detection(workdir: str, train_cfg: str, level: str = 's',
                           max_epochs: int = 200, **kwargs):
    logging.try_init_root(logging.INFO)

    # Load a pretrained YOLO model (recommended for training)
    previous_pt = os.path.join(workdir, 'weights', 'last.pt')
    if os.path.exists(previous_pt):
        logging.info(f'Initialize model from yolov8{level}.pt')
        model, resume = YOLO(f'yolov8{level}.pt'), True
    else:
        logging.info(f'Resume previous ckpt from {previous_pt!r}')
        logging.warn(f'Resumed model cannot be published to roboflow!!!')
        model, resume = YOLO(previous_pt), True
    workdir = os.path.abspath(workdir)

    # Train the model using the 'coco128.yaml' dataset for 3 epochs
    model.train(
        data=train_cfg, epochs=max_epochs,
        name=os.path.basename(os.path.basename(workdir)),
        project=os.path.dirname(os.path.basename(workdir)),
        save=True, plots=True,
        exist_ok=True, resume=resume,
        **kwargs
    )
