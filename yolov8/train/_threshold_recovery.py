"""DDP-aware ``threshold.json`` recovery.

Ultralytics' multi-GPU trainer launches its workers via
``subprocess.run`` with a generated temp ``.py`` file
(``ultralytics.utils.dist.generate_ddp_file``). That file rebuilds the
trainer purely from ``vars(trainer.args)`` and *does not* propagate
user-attached callbacks - so the ``on_train_end`` writer registered
on the parent's ``model.callbacks`` never fires inside the worker
process. After ``subprocess.run`` returns, the parent's
``model.trainer.validator.metrics`` is still the empty stub created
by ``BaseTrainer.__init__`` (the worker process operated on its own
fresh trainer).

This module provides a recovery path: load ``weights/best.pt`` and
run a fresh in-process ``model.val(...)`` to repopulate the metrics,
then write ``threshold.json`` from those.

The single-GPU code path is unaffected (``_do_train`` runs
in-process so the on_train_end callback fires normally and the
recovery routine no-ops because the file already exists).
"""
from __future__ import annotations

import json
import os.path
from typing import Any, Optional

from ditk import logging

from ..utils import compute_threshold_data


def recover_threshold_via_val(
    workdir: str,
    train_cfg: str,
    *,
    kind: str = 'box',
    is_rtdetr: bool = False,
    val_device: Any = None,
) -> Optional[dict]:
    """Re-validate ``<workdir>/weights/best.pt`` in-process and write
    ``<workdir>/threshold.json`` from the resulting metrics.

    No-op when the file already exists or ``best.pt`` is missing.

    :param workdir: The training run directory. Must contain
        ``weights/best.pt`` for recovery to be possible.
    :type workdir: str
    :param train_cfg: Dataset YAML used for the validation pass; must
        match the one training used.
    :type train_cfg: str
    :param kind: ``'box'`` for detection / RT-DETR, ``'seg'`` for
        segmentation.
    :type kind: str
    :param is_rtdetr: ``True`` to load the ckpt via
        :class:`ultralytics.RTDETR`. Detection / segmentation use
        :class:`ultralytics.YOLO`.
    :type is_rtdetr: bool
    :param val_device: Device for the validation pass. Lists are
        collapsed to their first element because val runs on a single
        device in-process; passing a list would re-trigger
        ``subprocess.run`` and re-introduce the same callback-loss
        problem.
    :type val_device: int or str or list or None
    :returns: The threshold payload that was written, or ``None`` when
        recovery wasn't needed / wasn't possible.
    :rtype: dict or None
    """
    threshold_path = os.path.join(workdir, 'threshold.json')
    if os.path.exists(threshold_path):
        return None

    best_pt = os.path.join(workdir, 'weights', 'best.pt')
    if not os.path.isfile(best_pt):
        logging.warning(
            f'recovery skipped: {best_pt!r} does not exist; '
            f'threshold.json will not be written'
        )
        return None

    # val() is in-process; collapse a multi-GPU device list to the
    # first device so we don't re-launch DDP for the validation step.
    if isinstance(val_device, (list, tuple)) and val_device:
        val_device = val_device[0]

    logging.info(
        f'threshold.json missing after training (likely multi-GPU/DDP '
        f'discarded the on_train_end callback); re-validating '
        f'{best_pt!r} in-process to capture metrics'
    )
    try:
        from ultralytics import RTDETR, YOLO

        cls = RTDETR if is_rtdetr else YOLO
        fresh = cls(best_pt)
        # ``verbose=False`` keeps the recovery log quiet; ``plots=False``
        # avoids re-rendering the F1_curve.png the trainer already
        # produced.
        val_kwargs: dict = {'data': train_cfg, 'verbose': False, 'plots': False}
        if val_device is not None:
            val_kwargs['device'] = val_device
        fresh.val(**val_kwargs)
        data = compute_threshold_data(fresh, kind=kind)
    except Exception as err:
        logging.warning(f'recovery validation failed: {err!r}')
        return None

    if data is None:
        logging.warning(
            'recovery validation produced no usable metrics; '
            'threshold.json will not be written'
        )
        return None

    try:
        os.makedirs(os.path.dirname(threshold_path) or '.', exist_ok=True)
        with open(threshold_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logging.info(
            f'recovery: wrote {threshold_path!r} '
            f'(f1={data["f1_score"]:.4f}, threshold={data["threshold"]:.4f})'
        )
    except Exception as err:
        logging.warning(f'recovery threshold write failed: {err!r}')
        return None
    return data
