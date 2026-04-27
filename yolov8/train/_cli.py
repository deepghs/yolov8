"""Click-based CLI for the two training entry points.

Exposes ``yolo-train`` flavoured subcommands so the user can launch
training with a single ``--train-cfg`` and let convention pick
everything else (yolo11s, ``runs/<dataset>_yolo11s`` workdir, sensible
ult defaults). All extra Ultralytics ``model.train(...)`` kwargs are
expressible via repeatable ``-p key=value`` flags whose values are
JSON-decoded when possible (see
:func:`yolov8.utils.cli.parse_hyperparams`).

The Python function entry points (:func:`train_object_detection` and
:func:`train_segmentation`) are deliberately untouched - the CLI is a
thin layer on top so existing scripts (``test_train_full_label4.py``
et al.) keep working.
"""
from __future__ import annotations

import os
import os.path
from functools import partial
from typing import Optional, Tuple, Union

import click
from ditk import logging

from ..utils import (
    GLOBAL_CONTEXT_SETTINGS,
    hyperparam_callback_factory,
    parse_yversion,
    yolo_train_param_schema,
)
from ..utils import print_version as _origin_print_version

#: Type-checked ``-p key=value`` callback bound to the Ultralytics
#: train schema. Built once at import time so the CLI doesn't pay the
#: schema-construction cost per invocation.
_TRAIN_HP_CALLBACK = hyperparam_callback_factory(yolo_train_param_schema())
from .object_detection import train_object_detection
from .segmentation import train_segmentation


print_version = partial(_origin_print_version, 'train')


_LEVEL_CHOICES = click.Choice(['n', 's', 'm', 'l', 'x', 't'], case_sensitive=False)
_DEFAULT_LEVEL = 's'
_DEFAULT_YVERSION = '11'


def _resolve_workdir(workdir: Optional[str], train_cfg: str,
                     yversion: Union[int, str], level: str) -> str:
    """Convention-over-configuration default workdir.

    Returns ``workdir`` verbatim when supplied. Otherwise builds
    ``runs/<dataset_basename>_yolo<yversion><level>``. The dataset
    basename comes from the ``train_cfg`` path:

    * ``coco8.yaml``                 -> ``coco8``
    * ``/foo/bar/data.yaml``         -> ``bar``  (the parent dir name,
      because ``data.yaml`` is a generic Ultralytics filename)
    * ``/foo/bar/`` (a directory)    -> ``bar``

    :param workdir: Caller-supplied workdir override; ``None`` to
        auto-derive.
    :type workdir: str or None
    :param train_cfg: Dataset YAML path or directory.
    :type train_cfg: str
    :param yversion: Already-normalised yversion (e.g. ``11`` or
        ``"rtdetr"``).
    :type yversion: int or str
    :param level: Model size suffix (``"n"`` / ``"s"`` / ...).
    :type level: str
    :returns: A path under ``runs/`` ready to hand to the trainer.
    :rtype: str
    """
    if workdir:
        return workdir
    cfg_path = os.path.normpath(train_cfg)
    if os.path.isdir(cfg_path):
        base = os.path.basename(cfg_path) or cfg_path
    else:
        stem = os.path.splitext(os.path.basename(cfg_path))[0]
        if stem.lower() in {"data", "dataset"}:
            base = os.path.basename(os.path.dirname(cfg_path)) or stem
        else:
            base = stem
    return os.path.join("runs", f"{base}_yolo{yversion}{level}")


def _common_train_options(func):
    """Shared ``@click.option`` stack for both detect and segment.

    Decorator-stacking helper: keeps the option list in one place so
    the two subcommands stay parameter-compatible. Order matches the
    function signature of :func:`train_object_detection` /
    :func:`train_segmentation` for readability.
    """
    decorators = [
        click.option('--train-cfg', '-d', 'train_cfg',
                     type=str, required=True,
                     help='Dataset YAML path, or a directory containing '
                          'data.yaml / data.yml. The only required argument; '
                          'everything else has a sensible default.'),
        click.option('--workdir', '-w', 'workdir',
                     type=click.Path(file_okay=False), default=None,
                     show_default='runs/<dataset>_yolo<yv><level>',
                     help='Run output directory. Defaults to '
                          'runs/<dataset_basename>_yolo<yversion><level>.'),
        click.option('--level', '-l', 'level',
                     type=_LEVEL_CHOICES, default=_DEFAULT_LEVEL,
                     show_default=True,
                     help='Model size suffix.'),
        click.option('--yversion', '-y', 'yversion',
                     type=str, default=_DEFAULT_YVERSION, show_default=True,
                     help='YOLO family selector. Integer (8 / 9 / 10 / 11 / '
                          '12) or "rtdetr" (detection only).'),
        click.option('--max-epochs', '-e', 'max_epochs',
                     type=int, default=200, show_default=True,
                     help='Maximum training epochs. Patience may end '
                          'training earlier (see -p patience=N).'),
        click.option('--batch', '-b', 'batch',
                     type=int, default=16, show_default=True,
                     help='Per-device batch size.'),
        click.option('--pretrained', 'pretrained',
                     type=str, default=None,
                     help='Override the pretrained checkpoint source. '
                          'Either a .pt path or a previous workdir '
                          '(uses <dir>/weights/best.pt). Default: the '
                          'canonical Ultralytics name for '
                          '<yversion>+<level>.'),
        click.option('-p', '--hyperparam', 'hyperparams',
                     multiple=True, callback=_TRAIN_HP_CALLBACK,
                     metavar='KEY=VALUE',
                     help='Extra train kwargs as KEY=VALUE (repeatable). '
                          'Type-checked against the Ultralytics train '
                          'schema (yolov8.utils.yolo_train_param_schema): '
                          'patience/epochs/seed/... are int, '
                          'mosaic/mixup/hsv_*/... are floats in [0,1], '
                          'cos_lr/save/verbose/... are bool, etc. Lists '
                          'are comma-separated (no brackets), each '
                          'element re-typed by the inner spec. Examples: '
                          '-p patience=20 -p imgsz=1280 '
                          '-p device=0,1,2,3 -p cos_lr=true '
                          '-p optimizer=AdamW. Unknown keys fall back '
                          'to JSON auto-detect.'),
    ]
    for dec in reversed(decorators):
        func = dec(func)
    return func


@click.group(context_settings={**GLOBAL_CONTEXT_SETTINGS})
@click.option('-v', '--version', is_flag=True,
              callback=print_version, expose_value=False, is_eager=True,
              help='Print version banner and exit.')
def cli():
    """Click entry point for ``python -m yolov8.train``.

    Two subcommands - ``detect`` and ``segment`` - mirror the two
    library functions :func:`yolov8.train.train_object_detection` and
    :func:`yolov8.train.train_segmentation`. They share the same
    convention-over-configuration knobs:

    * default model: ``yolo11s``
    * default workdir: ``runs/<dataset>_yolo<yv><level>``
    * extra Ultralytics ``model.train(...)`` kwargs via repeatable
      ``-p KEY=VALUE`` (JSON-decoded)

    Example::

        # Smallest case: just point at a dataset.
        python -m yolov8.train detect -d coco8.yaml

        # Ten-epoch run with v8m, custom batch and patience.
        python -m yolov8.train detect -d /data/foo/data.yaml \\
            -y 8 -l m -b 32 -e 10 -p patience=3 -p imgsz=640

        # Multi-GPU training (forwarded as ``device``):
        python -m yolov8.train segment -d /data/seg/data.yaml \\
            -p device=[0,1,2,3] -p batch=64 -p imgsz=896
    """


@cli.command('detect',
             context_settings={**GLOBAL_CONTEXT_SETTINGS},
             help='Train an Ultralytics detection model (YOLO or RT-DETR).')
@_common_train_options
def detect_cmd(train_cfg: str, workdir: Optional[str], level: str,
               yversion: str, max_epochs: int, batch: int,
               pretrained: Optional[str], hyperparams: dict):
    """Click subcommand: train detection.

    Forwards directly to :func:`train_object_detection` after
    auto-deriving ``workdir`` and normalising ``yversion``.
    """
    yv = parse_yversion(yversion)
    target_workdir = _resolve_workdir(workdir, train_cfg, yv, level)
    logging.try_init_root(logging.INFO)
    logging.info(f'workdir   : {target_workdir!r}')
    logging.info(f'train_cfg : {train_cfg!r}')
    logging.info(f'model     : yolo{yv}{level}')
    if hyperparams:
        logging.info(f'extra     : {hyperparams}')
    train_object_detection(
        target_workdir, train_cfg=train_cfg,
        level=level, yversion=yv,
        max_epochs=max_epochs, batch=batch,
        pretrained=pretrained,
        **hyperparams,
    )


@cli.command('segment',
             context_settings={**GLOBAL_CONTEXT_SETTINGS},
             help='Train an Ultralytics segmentation model.')
@_common_train_options
def segment_cmd(train_cfg: str, workdir: Optional[str], level: str,
                yversion: str, max_epochs: int, batch: int,
                pretrained: Optional[str], hyperparams: dict):
    """Click subcommand: train segmentation.

    Forwards directly to :func:`train_segmentation` after
    auto-deriving ``workdir`` and normalising ``yversion``. RT-DETR is
    not supported here.
    """
    yv = parse_yversion(yversion)
    if isinstance(yv, str) and yv.lower() == 'rtdetr':
        raise click.BadParameter(
            'RT-DETR has no segmentation variant; use the detect '
            'subcommand instead.')
    target_workdir = _resolve_workdir(workdir, train_cfg, yv, level)
    logging.try_init_root(logging.INFO)
    logging.info(f'workdir   : {target_workdir!r}')
    logging.info(f'train_cfg : {train_cfg!r}')
    logging.info(f'model     : yolo{yv}{level}-seg')
    if hyperparams:
        logging.info(f'extra     : {hyperparams}')
    train_segmentation(
        target_workdir, train_cfg=train_cfg,
        level=level, yversion=yv,
        max_epochs=max_epochs, batch=batch,
        pretrained=pretrained,
        **hyperparams,
    )
