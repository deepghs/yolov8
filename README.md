# yolov8

YOLOv8 wrapper for quicking train model

## Installation

```shell
git clone https://github.com/deepghs/yolov8.git
cd yolov8

# Main install: training (yolov8/v9/v10/v11/v12, rtdetr) + huggingface publish + onnx export
pip install -r requirements.txt
pip install -r requirements-onnx.txt

# Optional: enable `python -m yolov8.publish roboflow ...`.
# Roboflow is no longer a primary feature here; this extra is provided
# for legacy use and pins ultralytics back to 8.0.196 for SDK compatibility.
pip install -r requirements-roboflow.txt
```

## Training

```python
import os
import re

from yolov8.train import train_object_detection

if __name__ == '__main__':
    # directory to your dataset
    # about the dataset format: https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format
    dataset_dir = 'dir/to/your/dataset'

    # use the available models
    # LEVELs:
    # - for yolov8/v10/11: n/s/m/l/x
    # - for yolov9: t/s/m/l/x
    # - for rtdetr: l/x
    # YVERSIONs:
    # - 8: yolov8 model (see: https://docs.ultralytics.com/models/yolov8/)
    # - 9: yolov8 model (see: https://docs.ultralytics.com/models/yolov9/)
    # - 10: yolov10 model (see: https://docs.ultralytics.com/models/yolov10/)
    # - 11: yolo11 model (see: https://docs.ultralytics.com/models/yolo11/)
    # - rtdetr: rtdetr model (https://docs.ultralytics.com/models/rtdetr/)
    # Attention: if you just installed the requirements with `requirements.txt`,
    #            all of yolov8/v9/v10/v11/v12 and rtdetr are available out of the box.
    level = os.environ.get('LEVEL', 's')
    yversion = os.environ.get('YVERSION', '8') or '8'
    if re.fullmatch(r'^\d+$', yversion):
        yversion = int(yversion)
    if isinstance(yversion, int) and yversion != 8:
        suffix = f'_yv{yversion}'
    elif yversion == 'rtdetr':  # rtdetrs are really shitty, not recommend to use that
        suffix = '_rtdetr'
    else:
        suffix = ''

    # start training
    train_object_detection(
        f'runs/your_training_task_{level}{suffix}',  # your training dir
        train_cfg=os.path.join(dataset_dir, 'data.yaml'),
        level=level,
        yversion=yversion,
        max_epochs=100,
        patience=1000,

        # more kwargs are available here
        # see: https://docs.ultralytics.com/modes/train/#train-settings
    )

```

After the trainings, your training archives will be saved at directory `runs/your_training_task_xxx_xxx`.

## Publishing to Roboflow (legacy / optional)

> Roboflow integration is no longer a primary feature of this project and is
> kept here for legacy users. It supports yolov8 models only and requires the
> optional `requirements-roboflow.txt` extra (which pins ultralytics back to
> 8.0.196 for SDK compatibility).

```shell
# install the optional extra first
pip install -r requirements-roboflow.txt

# your api key of roboflow platform
export ROBOFLOW_APIKEY=raxxxxxxxxxxxxxxxxxC

python -m yolov8.publish roboflow \
  -w runs/your_training_task_xxx_xxx \
  -p your/project_name \
  -v 233
```

You can check the guide with the `--help` option.

## Publishing to HuggingFace Repository

```shell
# your hf token
export HF_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

python -m yolov8.publish huggingface \
  -w runs/your_training_task_xxx_xxx \
  -r your/hf_repository
```

You can check the guide with the `--help` option.

After you uploaded your models to huggingface repository, you can create a list in README of that repository.

```shell
python -m yolov8.list -r your/hf_repository
```

## Try My Models

You can quickly utilize your trained yolo model with [dghs-imgutils](https://github.com/deepghs/imgutils)
library.

```python
# pip install dghs-imgutils>=0.6.0
# export HF_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
import matplotlib.pyplot as plt

from imgutils.detect import detection_visualize
from imgutils.generic import yolo_predict

detection = yolo_predict(
    image='your/image.jpg',

    # use models from the repository
    repo_id='your/hf_repository',
    model_name='your_training_task_xxx_xxx',

    # you can use the recommended threshold in repository
    conf_threshold=0.25,
    iou_threshold=0.7,
)

print(f'Detections:\n{detection!r}')

plt.imshow(detection_visualize(
    image='your/image.jpg',
    detection=detection,
))
```

Or simply start a gradio demo for that

```python
# pip install dghs-imgutils[demo]>=0.6.0
# export HF_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
from imgutils.generic import YOLOModel

# quickly launch a gradio demo
YOLOModel(repo_id='your/hf_repository').launch_demo(
    default_model_name='your_training_task_xxx_xxx',

    # you can use the recommended threshold in repository
    default_conf_threshold=0.25,
    default_iou_threshold=0.7,

    # gradio servers, kwargs supported
    server_name=None,
    server_port=7860,
)
```

