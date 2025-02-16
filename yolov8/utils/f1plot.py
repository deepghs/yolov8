import re
from typing import Optional, Tuple

from ditk import logging
from imgutils.data import ImageTyping
from imgutils.ocr import ocr


def get_f1_and_threshold_from_image(image: ImageTyping) -> Tuple[Optional[float], Optional[float]]:
    threshold, max_f1_score = None, None
    for _, label, _ in ocr(image):
        logging.info(f'Scanning OCR result: {label!s}')
        matching = re.fullmatch(r'^\s*a+l{2,}\s+c+l+a+s+s+(e+s+)?\s+(?P<f1>\d+(\.+\d+)?)\s+'
                                r'a+t+\s+(?P<threshold>\d+(\.+\d+)?)\s*$', label)
        if matching:
            threshold = float(re.sub(r'\.+', '.', matching.group('threshold')))
            max_f1_score = float(re.sub(r'\.+', '.', matching.group('f1')))
            break
    return threshold, max_f1_score
