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

from .onnx import export_yolo_to_onnx
from .utils import GLOBAL_CONTEXT_SETTINGS, get_f1_and_threshold_from_image
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
                              opset_version: int = 14, logfile_anonymous: bool = True) -> List[Tuple[str, str]]:
    name = name or os.path.basename(os.path.abspath(workdir))
    os.makedirs(export_dir, exist_ok=True)

    files = []

    best_pt = os.path.join(workdir, 'weights', 'best.pt')
    best_pt_exp = os.path.join(export_dir, f'{name}_model.pt')
    logging.info(f'Copying best pt {best_pt!r} to {best_pt_exp!r}')
    state_dict = torch.load(best_pt)
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

    model_type = 'yolo'
    problem_type = 'detection'
    if os.path.exists(os.path.join(workdir, 'model_type.json')):
        with open(os.path.join(workdir, 'model_type.json'), 'r') as f:
            model_type_info = json.load(f)
            model_type = model_type_info['model_type']
            problem_type = model_type_info.get('problem_type', 'detection')
    if model_type == 'yolo':
        names_map = YOLO(best_pt).names
    else:
        names_map = RTDETR(best_pt).names
    labels = [names_map[i] for i in range(len(names_map))]
    with open(os.path.join(workdir, 'labels.json'), 'w') as f:
        json.dump(labels, f, ensure_ascii=False, indent=4)
    files.append((os.path.join(workdir, 'labels.json'), 'labels.json'))
    with open(os.path.join(workdir, 'model_type.json'), 'w') as f:
        json.dump(model_type_info, f, ensure_ascii=False, indent=4)
    files.append((os.path.join(workdir, 'model_type.json'), 'model_type.json'))

    threshold, max_f1_score = None, None
    if problem_type == 'detection' and os.path.exists(os.path.join(workdir, 'F1_curve.png')):
        threshold, max_f1_score = get_f1_and_threshold_from_image(os.path.join(workdir, 'F1_curve.png'))
    elif problem_type == 'segmentation' and os.path.exists(os.path.join(workdir, 'MaskF1_curve.png')):
        threshold, max_f1_score = get_f1_and_threshold_from_image(os.path.join(workdir, 'MaskF1_curve.png'))
    if threshold is not None and max_f1_score is not None:
        with open(os.path.join(workdir, 'threshold.json'), 'w') as f:
            json.dump({
                'f1_score': max_f1_score,
                'threshold': threshold,
            }, f, ensure_ascii=False, indent=4)
        files.append((os.path.join(workdir, 'threshold.json'), 'threshold.json'))

    best_onnx_exp = os.path.join(export_dir, f'{name}_model.onnx')
    logging.info(f'Export onnx model to {best_onnx_exp!r}')
    export_yolo_to_onnx(workdir, best_onnx_exp, opset_version=opset_version)
    files.append((best_onnx_exp, 'model.onnx'))

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
def cli(workdir: str, name: Optional[str], opset_version: int = 14):
    logging.try_init_root(logging.INFO)
    export_dir = os.path.join(workdir, 'export')
    files = export_model_from_workdir(workdir, export_dir, name, opset_version=opset_version)

    zip_file = os.path.join(export_dir, f'{name}.zip')
    logging.info(f'Packing all the above file to archive {zip_file!r}')
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file, inner_name in files:
            zf.write(file, inner_name)


if __name__ == '__main__':
    cli()
