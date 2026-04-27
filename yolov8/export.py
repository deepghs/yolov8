import glob
import json
import os
import re
import shutil
import zipfile
from functools import partial
from typing import Optional, List, Tuple

import click
import torch
from ditk import logging
from hbutils.encoding import sha3
from ultralytics import YOLO, RTDETR

from .onnx import export_yolo_to_onnx, export_yolo_to_onnx_with_embedding
from .utils import GLOBAL_CONTEXT_SETTINGS, derive_model_meta
from .utils import print_version as _origin_print_version

_KNOWN_FILES = [
    'confusion_matrix.png',
    'confusion_matrix_normalized.png',
    'labels.jpg',
    'labels_correlogram.jpg',
    'F1_curve.png',
    'P_curve.png',
    'PR_curve.png',
    'R_curve.png',
    'BoxF1_curve.png',
    'BoxP_curve.png',
    'BoxPR_curve.png',
    'BoxR_curve.png',
    'MaskF1_curve.png',
    'MaskP_curve.png',
    'MaskPR_curve.png',
    'MaskR_curve.png',
    'model_artifacts.json',
    'results.csv',
    'results.png',
]
_LOG_FILE_PATTERN = re.compile(r'^events\.out\.tfevents\.(?P<timestamp>\d+)\.(?P<machine>[^.]+)\.(?P<extra>[\s\S]+)$')


def export_model_from_workdir(workdir, export_dir, name: Optional[str] = None,
                              opset_version: int = 14, logfile_anonymous: bool = True,
                              with_embedding: bool = False,
                              embedding_layer_indices: Optional[list] = None
                              ) -> List[Tuple[str, str]]:
    """Materialise an upload-ready bundle from a finished training workdir.

    Walks ``workdir``, copies + anonymises ``weights/best.pt``, derives
    ``(model_type, problem_type)`` from the embedded class name,
    exports an ONNX (and optionally a dual-head ONNX with embedding),
    and copies any of the well-known plot / metrics files into
    ``export_dir``. The returned manifest is what
    :mod:`yolov8.publish` uploads to HuggingFace.

    The ``best.pt`` copy has ``train_args.data`` / ``project`` /
    ``model`` paths replaced with ``sha3-224`` digests so the
    published artefact doesn't leak the trainer's local filesystem
    layout.

    :param workdir: Source training directory containing
        ``weights/best.pt``.
    :type workdir: str
    :param export_dir: Destination directory for the staged files.
        Created if missing.
    :type export_dir: str
    :param name: Display name embedded in the staged filenames
        (``<name>_model.pt``, etc.). ``None`` falls back to the
        workdir's basename.
    :type name: str or None
    :param opset_version: ONNX opset for the produced ``.onnx`` file.
    :type opset_version: int
    :param logfile_anonymous: When ``True`` (the default), tensorboard
        ``events.out.tfevents.*`` filenames have their machine
        component sha3-anonymised before staging.
    :type logfile_anonymous: bool
    :param with_embedding: When ``True``, also export a dual-head ONNX
        named ``<name>_model_with_embedding.onnx`` with an additional
        ``embedding`` output suitable for retrieval / dedup / FAISS.
    :type with_embedding: bool
    :param embedding_layer_indices: Forwarded to
        :func:`yolov8.onnx.export_yolo_to_onnx_with_embedding` when
        ``with_embedding`` is ``True``. ``None`` defaults to
        Ultralytics' second-to-last layer.
    :type embedding_layer_indices: list[int] or None
    :returns: Manifest as a list of ``(local_path, repo_path)``
        tuples, in upload order.
    :rtype: list[tuple[str, str]]

    Example::

        >>> from yolov8.export import export_model_from_workdir
        >>> manifest = export_model_from_workdir(
        ...     workdir="runs/some_train",
        ...     export_dir="/tmp/export",
        ...     name="my_model",
        ...     with_embedding=True,
        ... )
        >>> [r for _, r in manifest][:2]
        ['model.pt', 'labels.json']
    """
    name = name or os.path.basename(os.path.abspath(workdir))
    os.makedirs(export_dir, exist_ok=True)

    files = []

    best_pt = os.path.join(workdir, 'weights', 'best.pt')
    best_pt_exp = os.path.join(export_dir, f'{name}_model.pt')
    logging.info(f'Copying best pt {best_pt!r} to {best_pt_exp!r}')
    state_dict = torch.load(best_pt, map_location='cpu', weights_only=False)
    # Derive (model_type, problem_type) from the checkpoint's embedded model
    # object. Done before the train_args anonymisation below for clarity —
    # the class object is untouched by that anonymisation either way.
    model_type, problem_type = derive_model_meta(state_dict)
    model_type_info = {'model_type': model_type, 'problem_type': problem_type}
    if 'train_args' in state_dict:
        if state_dict['train_args']['data']:
            state_dict['train_args']['data'] = sha3(state_dict['train_args']['data'].encode(), n=224)
        if state_dict['train_args']['project']:
            state_dict['train_args']['project'] = sha3(state_dict['train_args']['project'].encode(), n=224)
        if state_dict['train_args']['model'] and \
                ('/' in state_dict['train_args']['model'] or '\\' in state_dict['train_args']['model']):
            state_dict['train_args']['model'] = sha3(state_dict['train_args']['model'].encode(), n=224)
    torch.save(state_dict, best_pt_exp)
    # shutil.copy(best_pt, best_pt_exp)
    files.append((best_pt_exp, 'model.pt'))

    if model_type == 'yolo':
        names_map = YOLO(best_pt).names
    else:
        names_map = RTDETR(best_pt).names
    labels = [names_map[i] for i in range(len(names_map))]
    with open(os.path.join(workdir, 'labels.json'), 'w') as f:
        json.dump(labels, f, ensure_ascii=False, indent=4)
    files.append((os.path.join(workdir, 'labels.json'), 'labels.json'))
    # model_type.json is no longer written into the workdir — its content is
    # derived from the checkpoint and materialised here purely for the upload
    # bundle so the on-HF artifact layout stays unchanged.
    mt_path = os.path.join(export_dir, f'{name}_model_type.json')
    with open(mt_path, 'w') as f:
        json.dump(model_type_info, f, ensure_ascii=False, indent=4)
    files.append((mt_path, 'model_type.json'))

    # threshold.json is now produced at training time directly from the
    # validator's per-class arrays (yolov8.utils.compute_threshold_data).
    # If it is present in the workdir, ship it; otherwise — same fate as
    # the legacy OCR-failure case — silently skip.
    threshold_local = os.path.join(workdir, 'threshold.json')
    if os.path.exists(threshold_local):
        files.append((threshold_local, 'threshold.json'))

    best_onnx_exp = os.path.join(export_dir, f'{name}_model.onnx')
    logging.info(f'Export onnx model to {best_onnx_exp!r}')
    export_yolo_to_onnx(workdir, best_onnx_exp, opset_version=opset_version)
    files.append((best_onnx_exp, 'model.onnx'))

    if with_embedding:
        # Optional dual-head ONNX with an additional ``embedding``
        # output. Same checkpoint, produced via the embedding-aware
        # exporter so consumers wanting both retrieval features and
        # detection can use a single graph.
        embed_onnx_exp = os.path.join(export_dir, f'{name}_model_with_embedding.onnx')
        logging.info(f'Export onnx model with embedding to {embed_onnx_exp!r}')
        export_yolo_to_onnx_with_embedding(
            workdir, embed_onnx_exp,
            opset_version=opset_version,
            layer_indices=embedding_layer_indices,
        )
        files.append((embed_onnx_exp, 'model_with_embedding.onnx'))

    for f in _KNOWN_FILES:
        src = os.path.join(workdir, f)
        if os.path.exists(src):
            dst = os.path.join(export_dir, f'{name}_{f}')
            logging.info(f'Copying {src!r} to {dst!r}')
            shutil.copy(src, dst)
            files.append((dst, f))

    for logfile in glob.glob(os.path.join(workdir, 'events.out.tfevents.*')):
        logging.info(f'Tensorboard file {logfile!r} found.')
        matching = _LOG_FILE_PATTERN.fullmatch(os.path.basename(logfile))
        assert matching, f'Log file {logfile!r}\'s name not match with pattern {_LOG_FILE_PATTERN.pattern}.'

        timestamp = matching.group('timestamp')
        machine = matching.group('machine')
        if logfile_anonymous:
            machine = sha3(machine.encode(), n=224)
        extra = matching.group('extra')

        final_name = f'events.out.tfevents.{timestamp}.{machine}.{extra}'
        files.append((logfile, final_name))

    return files


print_version = partial(_origin_print_version, 'export')


@click.command(context_settings={**GLOBAL_CONTEXT_SETTINGS})
@click.option('-v', '--version', is_flag=True,
              callback=print_version, expose_value=False, is_eager=True,
              help="Utils with exporting best pts.")
@click.option('--workdir', '-w', 'workdir', type=click.Path(file_okay=False, exists=True), required=True,
              help='Work directory of the training.', show_default=True)
@click.option('--name', '-n', 'name', type=str, default=None,
              help='Name of the checkpoint. Default is the basename of the work directory.', show_default=True)
@click.option('--opset_version', 'opset_version', type=int, default=14,
              help='Version of OP set.', show_default=True)
@click.option('--with-embedding/--no-embedding', 'with_embedding', default=False,
              show_default=True,
              help='Also export a second ONNX (model_with_embedding.onnx) '
                   'whose graph emits the predictions plus a pooled image '
                   'embedding suitable for retrieval / dedup / FAISS.')
def cli(workdir: str, name: Optional[str], opset_version: int = 14,
        with_embedding: bool = False):
    """Click entry point: stage a workdir into ``<workdir>/export/`` and zip it.

    Calls :func:`export_model_from_workdir` and packs every produced
    file into ``<workdir>/export/<name>.zip``. CLI form::

        python -m yolov8.export -w runs/my_train [--with-embedding]

    :param workdir: Source training directory containing
        ``weights/best.pt``.
    :type workdir: str
    :param name: Display name for the bundle. Defaults to the
        workdir's basename.
    :type name: str or None
    :param opset_version: ONNX opset.
    :type opset_version: int
    :param with_embedding: Also stage ``model_with_embedding.onnx``
        with an ``embedding`` output for retrieval / dedup / FAISS.
    :type with_embedding: bool

    Example::

        >>> # CLI form (preferred):
        >>> #   python -m yolov8.export -w runs/some_train --with-embedding
        >>> from yolov8.export import cli  # click callback
    """
    logging.try_init_root(logging.INFO)
    export_dir = os.path.join(workdir, 'export')
    files = export_model_from_workdir(workdir, export_dir, name,
                                       opset_version=opset_version,
                                       with_embedding=with_embedding)

    zip_file = os.path.join(export_dir, f'{name}.zip')
    logging.info(f'Packing all the above file to archive {zip_file!r}')
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file, inner_name in files:
            zf.write(file, inner_name)


if __name__ == '__main__':
    cli()
