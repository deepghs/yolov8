"""Shared ``on_train_end`` callback that writes ``threshold.json`` from
inside the trainer process.

Why a callback rather than a post-``model.train()`` block:

* When ``patience`` exhausts, ultralytics breaks the training loop and
  re-validates ``best.pt`` to populate ``trainer.validator.metrics``,
  and then runs ``on_train_end`` callbacks - so the callback fires
  with valid metrics even on early-stop.
* When training is multi-GPU, the trainer object lives in a child
  process spawned by ``subprocess.run``; the main process's
  ``model.trainer.validator.metrics`` is the empty stub created before
  the subprocess started training, and the post-``train()`` extraction
  silently no-ops. The callback runs in the child and writes the file
  to disk before the child exits.
* When the user hits Ctrl-C the callback does not fire either way, but
  that's outside our threat model: a real interrupt also leaves
  ``best.pt`` half-baked.

The callback is idempotent: invoking it a second time would simply
overwrite the file with the same payload.
"""
from __future__ import annotations

import json
import os.path
from typing import Any, Callable

from ditk import logging

from ..utils import compute_threshold_data_from_trainer


def make_on_train_end_threshold_writer(workdir: str, *, kind: str) -> Callable[[Any], None]:
    """Build an ``on_train_end(trainer)`` callback that writes
    ``<workdir>/threshold.json`` from ``trainer.validator.metrics``.

    ``kind`` selects ``'box'`` / ``'seg'`` to match the legacy semantics
    of detection / segmentation training.
    """
    target = os.path.join(workdir, 'threshold.json')

    def _on_train_end(trainer: Any) -> None:
        try:
            data = compute_threshold_data_from_trainer(trainer, kind=kind)
        except Exception as err:  # never let a callback failure break training
            logging.warning(f'on_train_end threshold compute failed: {err!r}')
            return
        if data is None:
            logging.info('on_train_end: validator metrics empty, threshold.json not written')
            return
        try:
            os.makedirs(os.path.dirname(target) or '.', exist_ok=True)
            with open(target, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            logging.info(f'on_train_end: wrote {target!r} (f1={data["f1_score"]:.4f}, '
                         f'threshold={data["threshold"]:.4f})')
        except Exception as err:
            logging.warning(f'on_train_end threshold write failed: {err!r}')

    return _on_train_end
