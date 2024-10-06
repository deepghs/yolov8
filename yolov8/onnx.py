import json
import os.path
from shutil import SameFileError

from hbutils.system import copy
from ultralytics import YOLO, RTDETR


def export_yolo_to_onnx(workdir: str, onnx_filename, opset_version: int = 14,
                        no_optimize: bool = False):
    model_type = 'yolo'
    if os.path.exists(os.path.join(workdir, 'model_type.json')):
        with open(os.path.join(workdir, 'model_type.json'), 'r') as f:
            model_type = json.load(f)['model_type']
    if model_type == 'yolo':
        yolo = YOLO(os.path.join(workdir, 'weights', 'best.pt'))
    else:
        yolo = RTDETR(os.path.join(workdir, 'weights', 'best.pt'))

    if os.path.dirname(onnx_filename):
        os.makedirs(os.path.dirname(onnx_filename), exist_ok=True)

    _retval = yolo.export(format='onnx', dynamic=True, simplify=not no_optimize, opset=opset_version)
    _exported_onnx_file = _retval or (os.path.splitext(yolo.ckpt_path)[0] + '.onnx')
    try:
        copy(_exported_onnx_file, onnx_filename)
    except SameFileError:
        pass
