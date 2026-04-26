import os.path
from shutil import SameFileError

from hbutils.system import copy
from ultralytics import YOLO, RTDETR

from .utils import derive_model_meta_from_path


def export_yolo_to_onnx(workdir: str, onnx_filename, opset_version: int = 14,
                        no_optimize: bool = False):
    best_pt = os.path.join(workdir, 'weights', 'best.pt')
    model_type, _ = derive_model_meta_from_path(best_pt)
    if model_type == 'yolo':
        yolo = YOLO(best_pt)
    else:
        yolo = RTDETR(best_pt)

    if os.path.dirname(onnx_filename):
        os.makedirs(os.path.dirname(onnx_filename), exist_ok=True)

    _retval = yolo.export(format='onnx', dynamic=True, simplify=not no_optimize, opset=opset_version)
    _exported_onnx_file = _retval or (os.path.splitext(yolo.ckpt_path)[0] + '.onnx')
    try:
        copy(_exported_onnx_file, onnx_filename)
    except SameFileError:
        pass
