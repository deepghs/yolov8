# dghs-yolov8

A thin, opinionated wrapper around
[Ultralytics](https://github.com/ultralytics/ultralytics) that streamlines the
training → export → publish pipeline used by the
[deepghs](https://github.com/deepghs) team. It keeps a single Python entry
point for training, takes care of resuming, automatically writes the metadata
needed for downstream tooling, and ships CLIs for ONNX export and publishing
trained models to HuggingFace (and, optionally, Roboflow).

> Heads up: this repository is **not** published to PyPI. Always install from
> a local clone using the `requirements*.txt` files described below.

## Table of contents

- [Features](#features)
- [Installation](#installation)
- [Training](#training)
  - [Object detection](#object-detection)
  - [Instance segmentation](#instance-segmentation)
  - [Choosing a model family / size](#choosing-a-model-family--size)
  - [What ends up in the work directory](#what-ends-up-in-the-work-directory)
  - [Resuming a run](#resuming-a-run)
- [Exporting to ONNX](#exporting-to-onnx)
- [Publishing to HuggingFace](#publishing-to-huggingface)
- [Aggregating a model table on a HuggingFace repo](#aggregating-a-model-table-on-a-huggingface-repo)
- [Publishing to Roboflow (legacy / optional)](#publishing-to-roboflow-legacy--optional)
- [Using a published model](#using-a-published-model)
- [Repository layout](#repository-layout)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Features

- One-line training entry for **object detection** and **instance
  segmentation** with `train_object_detection` / `train_segmentation`.
- Supports the full Ultralytics line: **YOLOv8 / v9 / v10 / v11 / v12** and
  **RT-DETR**, selectable through plain function arguments or env vars.
- **Auto-resume** when `weights/last.pt` is present in the work directory —
  re-running the same training script picks up where it left off.
- Captures `model_type.json` and `problem_type` metadata reliably, surviving
  Ultralytics' early work-directory wipe.
- `python -m yolov8.export` packages a training run into a portable archive
  (best checkpoint, ONNX, training curves, confusion matrix, anonymised
  TensorBoard logs).
- `python -m yolov8.publish huggingface` uploads the archive to a HuggingFace
  model repo with one command, including an OCR'd recommended F1 threshold.
- `python -m yolov8.list` regenerates a comparison table in the HuggingFace
  repo's `README.md`, in place, without disturbing the rest of the page.
- Optional Roboflow publish path kept for legacy users (see the dedicated
  section below).

## Installation

```shell
git clone https://github.com/deepghs/yolov8.git
cd yolov8

# Main install: training (yolov8/v9/v10/v11/v12, rtdetr) +
# huggingface publish + onnx export.
pip install -r requirements.txt
pip install -r requirements-onnx.txt

# Optional extra: enable `python -m yolov8.publish roboflow ...`.
# Roboflow is no longer a primary feature here; this set is kept for legacy
# use and pins ultralytics back to 8.0.196 for SDK compatibility, so install
# it only if you actually need the Roboflow path.
pip install -r requirements-roboflow.txt
```

GPU is required for any meaningful training. ONNX export and publishing run
fine on CPU.

## Training

The training entry points are plain Python functions. Both accept arbitrary
extra keyword arguments and forward them to Ultralytics' `model.train(...)`,
so anything the
[Ultralytics training settings](https://docs.ultralytics.com/modes/train/#train-settings)
documents is available — `imgsz`, `lr0`, `optimizer`, `device`, `workers`,
`augment`, etc.

Datasets follow the standard Ultralytics layout. See the
[YOLO format reference](https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format)
for the directory structure and `data.yaml` schema. If you point the wrapper
at a directory, it will look for `data.yaml` (or `data.yml`) inside it.

### Object detection

```python
import os
import re

from yolov8.train import train_object_detection

if __name__ == '__main__':
    # Path to your dataset root. Either a directory containing data.yaml
    # (or data.yml) or the data.yaml file itself works.
    # Dataset format reference:
    #   https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format
    dataset_dir = 'dir/to/your/dataset'

    # Pick a model family (YVERSION) and size (LEVEL).
    #
    # LEVEL — model scale, controls capacity vs speed:
    #   - yolov8 / v10 / v11 / v12: n / s / m / l / x
    #   - yolov9                  : t / s / m / l / x
    #   - rtdetr                  : l / x
    #
    # YVERSION — which model family to use:
    #   - 8       — YOLOv8   (https://docs.ultralytics.com/models/yolov8/)
    #   - 9       — YOLOv9   (https://docs.ultralytics.com/models/yolov9/)
    #   - 10      — YOLOv10  (https://docs.ultralytics.com/models/yolov10/)
    #   - 11      — YOLO11   (https://docs.ultralytics.com/models/yolo11/)
    #   - 12      — YOLO12
    #   - 'rtdetr' — RT-DETR (https://docs.ultralytics.com/models/rtdetr/)
    #
    # All of these are available out of the box once you have installed
    # `requirements.txt` (the previous `requirements-raw.txt` split is gone).
    level = os.environ.get('LEVEL', 's')
    yversion = os.environ.get('YVERSION', '8') or '8'
    if re.fullmatch(r'^\d+$', yversion):
        yversion = int(yversion)

    # Build a suffix so different families do not overwrite each other under runs/.
    if isinstance(yversion, int) and yversion != 8:
        suffix = f'_yv{yversion}'
    elif yversion == 'rtdetr':  # RT-DETR results have been disappointing in our setting; not recommended.
        suffix = '_rtdetr'
    else:
        suffix = ''

    # Kick off training. Outputs go to runs/<task_name>/ — see
    # "What ends up in the work directory" below for details. If a previous
    # run left a weights/last.pt behind, training resumes automatically.
    train_object_detection(
        # Work directory. Becomes runs/<basename>/ on disk; the training run's
        # `name` is the basename and `project` is its parent directory.
        f'runs/your_training_task_{level}{suffix}',

        # Either a data.yaml path, a data.yml path, or a directory containing one.
        train_cfg=os.path.join(dataset_dir, 'data.yaml'),

        level=level,           # see LEVEL above
        yversion=yversion,     # see YVERSION above
        max_epochs=100,        # forwarded to Ultralytics as `epochs`
        patience=1000,         # early-stopping patience; high value ≈ disabled

        # Anything else here is forwarded straight to model.train(...).
        # Examples:
        #   batch=16,
        #   imgsz=640,
        #   device=0,
        #   workers=8,
        #   optimizer='AdamW',
        # Full reference:
        #   https://docs.ultralytics.com/modes/train/#train-settings
    )
```

### Instance segmentation

The segmentation entry point mirrors detection. The only differences worth
noting: the segmentation pretrained weights have a `-seg` suffix
(`yolov8s-seg.pt`, `yolo11s-seg.pt`, etc.), and Ultralytics emits both
`Box*` and `Mask*` curves in the work directory. The wrapper handles both
sides automatically.

```python
import os

from yolov8.train import train_segmentation

if __name__ == '__main__':
    dataset_dir = 'dir/to/your/segmentation/dataset'

    train_segmentation(
        'runs/your_seg_task_s',
        train_cfg=os.path.join(dataset_dir, 'data.yaml'),
        level='s',
        yversion=8,        # also supports 9/10/11/12
        max_epochs=100,
        patience=1000,
        # imgsz=640, batch=16, device=0, ...
    )
```

> RT-DETR has no segmentation variant, so `train_segmentation` does not
> accept `yversion='rtdetr'`.

### Choosing a model family / size

A few rules of thumb when picking `(yversion, level)`:

- **Smaller models (`n`/`t`/`s`)** train faster, run faster, need less VRAM,
  and tend to be enough when classes are visually distinct.
- **Larger models (`m`/`l`/`x`)** help on cluttered scenes, fine-grained
  classes, or small objects, at the cost of throughput and memory.
- **Newer families** (v10/v11/v12) generally have better accuracy/latency
  tradeoffs than v8 at the same `level`. v8 is still a reasonable default
  if you need the broadest tooling compatibility.
- For the Roboflow publish path you must stay on **YOLOv8**; see the legacy
  Roboflow section.

### What ends up in the work directory

After training, `runs/your_training_task_xxx/` will contain (Ultralytics'
own outputs plus a few files this wrapper writes):

```
runs/your_training_task_xxx/
├── weights/
│   ├── best.pt                    best checkpoint by validation metric
│   └── last.pt                    most recent checkpoint (resume marker)
├── results.csv                    per-epoch metrics
├── results.png                    metrics plot
├── F1_curve.png  P_curve.png  R_curve.png  PR_curve.png
│                                  detection-task curves (or Box*/Mask*
│                                  variants for segmentation)
├── confusion_matrix.png
├── confusion_matrix_normalized.png
├── labels.jpg                     dataset label distribution
├── labels_correlogram.jpg
├── events.out.tfevents.*          TensorBoard logs
└── model_type.json                {"model_type": "yolo"|"rtdetr",
                                    "problem_type": "detection"|"segmentation"}
                                    — written by this wrapper
```

`model_type.json` is intentionally written **30 seconds after training
starts**, because Ultralytics wipes the work directory at training start.
Don't be alarmed by the brief gap; the export and publish flows depend on
this file being present.

### Resuming a run

If `weights/last.pt` exists when you call `train_object_detection` /
`train_segmentation` again with the same work directory, the wrapper sets
Ultralytics' `resume=True` for you. Just re-run the same script — no extra
flags required. To start fresh instead, point the function at a new work
directory or remove the existing one.

## Exporting to ONNX

`yolov8.export` packages a finished training run into a tidy bundle and
emits an ONNX file alongside the PyTorch checkpoint:

```shell
python -m yolov8.export -w runs/your_training_task_xxx
# Optional flags:
#   -n / --name           override the output basename (defaults to the workdir name)
#   --opset_version 14    ONNX opset (default 14)
```

What the export does, beyond just running `yolo.export()`:

- Anonymises the training-time host paths in `train_args` (`data`, `project`,
  `model`) using SHA3, so published checkpoints don't leak local directory
  layouts.
- Reads `F1_curve.png` (or `MaskF1_curve.png` for segmentation), OCRs the
  best F1 score and its threshold, and writes `threshold.json`. Downstream
  inference tooling reads this to suggest a sensible `conf_threshold`.
- Anonymises the hostname in TensorBoard event filenames before bundling.
- Produces a zip at `runs/<task>/export/<task>.zip` containing the
  checkpoint, ONNX, plots, csv, labels, and metadata.

You usually don't need to invoke this manually — `python -m yolov8.publish
huggingface` runs the same export pipeline before uploading. Use it directly
when you want a self-contained zip without publishing.

## Publishing to HuggingFace

```shell
# Your HuggingFace access token (must have write access to the target repo).
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

python -m yolov8.publish huggingface \
  -w runs/your_training_task_xxx_xxx \
  -r your_namespace/your_hf_repository
# Other flags:
#   -n / --name             override the per-model directory inside the repo
#   -R / --revision         target branch (default: main)
#   --opset_version 14      forwarded to ONNX export
```

This will, in a single commit:

1. Run the export pipeline above (with the same anonymisation guarantees).
2. Create the HuggingFace model repo if it does not exist.
3. Upload everything under `<name>/...` inside the repo:
   - `<name>/model.pt`        — anonymised PyTorch checkpoint
   - `<name>/model.onnx`      — exported ONNX
   - `<name>/labels.json`     — class index → label name
   - `<name>/threshold.json`  — `{f1_score, threshold}` (when OCR succeeded)
   - `<name>/model_type.json` — `{model_type, problem_type}`
   - Plots, results.csv, anonymised TensorBoard event files

You can publish many models into the same repository — each lives in its
own subdirectory, and the next section explains how to keep an index of
them in the repo's README.

## Aggregating a model table on a HuggingFace repo

Once a few models are uploaded, you can regenerate the comparison table at
the top of the HuggingFace repository's `README.md`:

```shell
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

python -m yolov8.list -r your_namespace/your_hf_repository
# Optional:
#   -R / --revision         target branch (default: main)
```

The command:

- Walks every `*/model.pt` in the repo and recomputes FLOPS / parameters,
  pulls the recorded F1 score / threshold, links to F1 and confusion-matrix
  plots, and lists the labels (with a `labels.json` link if there are many).
- Sorts rows by recency (newest first).
- Writes the result back into `README.md`. If the existing README already
  contains a markdown table whose header has `Model / FLOPS / Params /
  Labels`, that one block is replaced in place; everything else in the
  README (descriptions, badges, prose) is preserved. If no such table is
  present, the table is written at the top.

This makes it cheap to keep a public model "zoo" page up to date as you
publish new variants.

## Publishing to Roboflow (legacy / optional)

> Roboflow integration is no longer a primary feature of this project and
> is kept here for legacy users. It supports YOLOv8 models only and
> requires the optional `requirements-roboflow.txt` extra, which pins
> `ultralytics==8.0.196` for SDK compatibility — installing it will
> downgrade `ultralytics` from the main range. The `roboflow` import is
> done lazily inside the subcommand, so you can use the rest of the CLI
> without ever installing the extra.

```shell
# Install the optional extra first (downgrades ultralytics to 8.0.196).
pip install -r requirements-roboflow.txt

# Roboflow API key.
export ROBOFLOW_APIKEY=raxxxxxxxxxxxxxxxxxC

python -m yolov8.publish roboflow \
  -w runs/your_training_task_xxx_xxx \
  -p your_workspace/your_project \
  -v 233
```

Pass `--help` to either subcommand to see the full option list.

## Using a published model

Once a model is on HuggingFace, you can load and run it directly with
[dghs-imgutils](https://github.com/deepghs/imgutils) — no extra setup
required.

```python
# pip install dghs-imgutils>=0.6.0
# export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
import matplotlib.pyplot as plt

from imgutils.detect import detection_visualize
from imgutils.generic import yolo_predict

detection = yolo_predict(
    image='your/image.jpg',

    # Where to fetch the model from.
    repo_id='your_namespace/your_hf_repository',
    model_name='your_training_task_xxx_xxx',

    # Reasonable defaults; if the repo has a threshold.json for this model,
    # consider using its recommended threshold instead.
    conf_threshold=0.25,
    iou_threshold=0.7,
)

print(f'Detections:\n{detection!r}')

plt.imshow(detection_visualize(
    image='your/image.jpg',
    detection=detection,
))
```

Or launch a one-line Gradio demo for a whole repo, with a model picker:

```python
# pip install dghs-imgutils[demo]>=0.6.0
# export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
from imgutils.generic import YOLOModel

YOLOModel(repo_id='your_namespace/your_hf_repository').launch_demo(
    default_model_name='your_training_task_xxx_xxx',

    default_conf_threshold=0.25,
    default_iou_threshold=0.7,

    # Standard Gradio server kwargs.
    server_name=None,
    server_port=7860,
)
```

## Repository layout

```
yolov8/                   the actual Python package
├── train/
│   ├── object_detection.py    train_object_detection(...)
│   └── segmentation.py        train_segmentation(...)
├── export.py                  workdir → bundle (with anonymisation)
├── onnx.py                    YOLO/RTDETR → ONNX (dynamic, simplified)
├── publish.py                 `python -m yolov8.publish {huggingface,roboflow}`
├── list.py                    `python -m yolov8.list` — refresh HF README table
├── config/meta.py             single source of truth for version/author
└── utils/                     small utilities (CLI helpers, F1-curve OCR, etc.)

requirements.txt               main runtime deps (ultralytics<=8.3.105)
requirements-onnx.txt          ONNX export deps (always recommended)
requirements-roboflow.txt      optional, for the legacy Roboflow publish path
requirements-doc.txt           docs build deps
requirements-test.txt          test deps
```

For an architectural deep dive, hidden constraints, and AI-agent operating
rules, see [`CLAUDE.md`](./CLAUDE.md) (also exposed as `AGENTS.md`).

## Acknowledgements

- Built on top of [Ultralytics](https://github.com/ultralytics/ultralytics).
- Designed to slot into the [deepghs](https://github.com/deepghs) workflow
  alongside [dghs-imgutils](https://github.com/deepghs/imgutils).

## License

Apache License 2.0 — see [`LICENSE`](./LICENSE).
