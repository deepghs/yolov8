import os.path

from ditk import logging
from ultralytics import YOLO


def train_object_detection(workdir: str, train_cfg: str, level: str = 's',
                           max_epochs: int = 200, **kwargs):
    logging.try_init_root(logging.INFO)

    # Load a pretrained YOLO model (recommended for training)
    previous_pt = os.path.join(workdir, 'weights', 'last.pt')
    model = YOLO(f'yolov8{level}.pt')
    resume = os.path.exists(previous_pt)
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
