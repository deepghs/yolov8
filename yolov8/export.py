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


# Path-typed fields we sha3-anonymise before writing the published
# best.pt copy. Every access is *defensive* — missing dicts, missing
# keys, non-string values, or strings that don't actually look like
# paths are silently passed through. The helper below walks
# ``state_dict['train_args']`` and ``state_dict['model'].args`` /
# ``state_dict['ema'].args`` so the inner model object's frozen-at-
# train-time copy of the same paths gets scrubbed too — that copy is
# what slipped through the original ``train_args``-only path and led
# to ``/data/yolov8/runs`` / ``/nfs/...`` showing up in published
# bundles.
_ANON_FIELDS_TRAIN_ARGS = ('data', 'project', 'model')
_ANON_FIELDS_MODEL_INNER = ('data', 'project', 'save_dir', 'model')


def _anonymise_path_field(container, field: str) -> None:
    """If ``container`` (dict or attribute-bearing object) has
    ``field`` set to a string that looks like a filesystem path,
    replace it with the sha3-224 hash of that string. Anything else
    (missing field, ``None``, empty string, non-path string) is left
    alone.

    Operates in place on the in-memory state_dict copy; the on-disk
    file is never touched (the caller saves the mutated copy to a
    different path).
    """
    if container is None:
        return
    is_dict = isinstance(container, dict)
    if is_dict:
        if field not in container:
            return
        v = container[field]
    else:
        if not hasattr(container, field):
            return
        v = getattr(container, field)
    if not isinstance(v, str) or not v:
        return
    if '/' not in v and '\\' not in v:
        return  # not a path-like string (e.g. 'yolo11n.pt' bare model name)
    digest = sha3(v.encode(), n=224)
    if is_dict:
        container[field] = digest
    else:
        try:
            setattr(container, field, digest)
        except Exception:
            # Frozen object / restricted setattr — leave it; we still
            # got the dict-side anon, which is what readers normally
            # consult.
            pass


def _anonymise_state_dict_paths(state_dict) -> None:
    """Walk a freshly-loaded torch state_dict and sha3-anonymise every
    known path-typed field on the top-level ``train_args`` dict and
    on the inner model object's ``.args`` (and the EMA wrapper's, if
    present). All accesses are defensive — fields that don't exist
    are simply skipped.

    This is called on an *in-memory* copy of ``best.pt`` only; the
    on-disk training checkpoint is never touched. The mutated copy
    is then written to ``export_dir`` for upload.
    """
    train_args = state_dict.get('train_args')
    if isinstance(train_args, dict):
        for f in _ANON_FIELDS_TRAIN_ARGS:
            _anonymise_path_field(train_args, f)

    inner = state_dict.get('model')
    if inner is not None:
        inner_args = getattr(inner, 'args', None)
        for f in _ANON_FIELDS_MODEL_INNER:
            _anonymise_path_field(inner_args, f)

    # Newer ult preserves an EMA copy on best.pt sometimes. The EMA
    # wrapper has its own ``ema.args`` (or ``ema.module.args``) that
    # mirrors the inner model's args. Best-effort scan.
    ema = state_dict.get('ema')
    if ema is not None:
        for path_obj in (getattr(ema, 'args', None),
                         getattr(getattr(ema, 'module', None), 'args', None)):
            for f in _ANON_FIELDS_MODEL_INNER:
                _anonymise_path_field(path_obj, f)


def export_model_from_workdir(workdir, export_dir, name: Optional[str] = None,
                              opset_version: int = 14, logfile_anonymous: bool = True,
                              with_embedding: bool = False,
                              embedding_layer_indices: Optional[list] = None,
                              with_int8: bool = False,
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
    :param with_int8: When ``True``, also include the deployable INT8
        ONNX produced by :func:`yolov8.quantize.quantize_workdir`. If
        ``<workdir>/quant/int8/`` does not yet contain a quantized
        artifact, runs the Tier S PTQ pipeline first; otherwise
        simply ships the existing artifact and its sidecar
        ``eval.json`` / ``threshold.json``. The published files end
        up at ``model.int8.onnx`` / ``eval_int8.json`` /
        ``threshold_int8.json`` inside the upload bundle.
    :type with_int8: bool
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
    # object. Done before the path anonymisation below for clarity — the
    # class object is untouched by that anonymisation either way.
    model_type, problem_type = derive_model_meta(state_dict)
    model_type_info = {'model_type': model_type, 'problem_type': problem_type}
    # Anonymise every known path-typed field across train_args + inner model
    # args + EMA args. Defensive: each access tolerates missing fields /
    # different container shapes (dict vs attribute-bearing object). The
    # original best.pt on disk is *not* mutated — we only mutate the
    # in-memory state_dict and write it to ``best_pt_exp``.
    _anonymise_state_dict_paths(state_dict)
    torch.save(state_dict, best_pt_exp)
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

    if with_int8:
        # Optional deployable INT8 artifact produced by Tier S PTQ.
        # The quantize pipeline lives separately in :mod:`yolov8.quantize`
        # to keep the import surface lean (it pulls onnxruntime + opencv).
        # Imported lazily so an ``export -w ...`` without ``--with-int8``
        # remains usable even if those deps aren't installed.
        from .quantize import quantize_workdir  # noqa: PLC0415

        quant_dir = os.path.join(workdir, 'quant')
        int8_dir = os.path.join(quant_dir, 'int8')
        existing = (
            sorted(glob.glob(os.path.join(int8_dir, '*_int8.onnx')))
            if os.path.isdir(int8_dir) else []
        )
        if existing:
            int8_src = existing[0]
            logging.info(f'Reusing existing INT8 ONNX: {int8_src!r}')
        else:
            logging.info('No INT8 ONNX in workdir; running Tier S PTQ now.')
            int8_src = str(quantize_workdir(workdir))

        int8_exp = os.path.join(export_dir, f'{name}_model.int8.onnx')
        shutil.copy(int8_src, int8_exp)
        files.append((int8_exp, 'model.int8.onnx'))

        # Ship the INT8-specific eval + threshold sidecars too. The
        # threshold is also inside the .onnx metadata_props, but the
        # JSON sidecar is what aggregator tools (e.g. yolov8.list)
        # currently read — keep both paths working.
        eval_src = os.path.join(quant_dir, 'eval.json')
        if os.path.isfile(eval_src):
            eval_exp = os.path.join(export_dir, f'{name}_eval_int8.json')
            shutil.copy(eval_src, eval_exp)
            files.append((eval_exp, 'eval_int8.json'))
        thr_src = os.path.join(quant_dir, 'threshold.json')
        if os.path.isfile(thr_src):
            thr_exp = os.path.join(export_dir, f'{name}_threshold_int8.json')
            shutil.copy(thr_src, thr_exp)
            files.append((thr_exp, 'threshold_int8.json'))
        qa_src = os.path.join(quant_dir, 'quant_args.json')
        if os.path.isfile(qa_src):
            # Local quant_args.json keeps the original paths for the
            # operator's debugging convenience; the published copy goes
            # through anonymize_quant_args so we don't leak the
            # trainer's filesystem layout via the HF artifact.
            from .quantize import anonymize_quant_args  # noqa: PLC0415
            with open(qa_src, 'r', encoding='utf-8') as f:
                qa_local = json.load(f)
            qa_pub = anonymize_quant_args(qa_local)
            qa_exp = os.path.join(export_dir, f'{name}_quant_args.json')
            with open(qa_exp, 'w', encoding='utf-8') as f:
                json.dump(qa_pub, f, ensure_ascii=False, indent=4)
            files.append((qa_exp, 'quant_args.json'))

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
@click.option('--with-int8/--no-int8', 'with_int8', default=False,
              show_default=True,
              help='Also pack the deployable INT8 ONNX (Tier S PTQ). '
                   'Reuses <workdir>/quant/ if already populated; '
                   'otherwise runs the quantization pipeline first.')
def cli(workdir: str, name: Optional[str], opset_version: int = 14,
        with_embedding: bool = False, with_int8: bool = False):
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
    :param with_int8: Also stage the deployable INT8 ONNX
        (``model.int8.onnx``) and its sidecars (``eval_int8.json`` /
        ``threshold_int8.json`` / ``quant_args.json``). Reuses an
        existing ``<workdir>/quant/int8/*_int8.onnx`` if present;
        otherwise invokes :func:`yolov8.quantize.quantize_workdir`
        with default Tier S knobs.
    :type with_int8: bool

    Example::

        >>> # CLI form (preferred):
        >>> #   python -m yolov8.export -w runs/some_train --with-embedding --with-int8
        >>> from yolov8.export import cli  # click callback
    """
    logging.try_init_root(logging.INFO)
    export_dir = os.path.join(workdir, 'export')
    files = export_model_from_workdir(workdir, export_dir, name,
                                       opset_version=opset_version,
                                       with_embedding=with_embedding,
                                       with_int8=with_int8)

    zip_file = os.path.join(export_dir, f'{name}.zip')
    logging.info(f'Packing all the above file to archive {zip_file!r}')
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file, inner_name in files:
            zf.write(file, inner_name)


if __name__ == '__main__':
    cli()
