import os.path
import time

import cv2
import numpy as np
import torch
from PySide6.QtCore import QThread, Signal
from pathlib import Path

from yolocode.yolov8.YOLOv8Thread import YOLOv8Thread
from yolocode.yolov8.data import load_inference_source
from yolocode.yolov8.data.augment import classify_transforms, LetterBox
from yolocode.yolov8.data.utils import IMG_FORMATS, VID_FORMATS
from yolocode.yolov8.engine.predictor import STREAM_WARNING
from yolocode.yolov8.engine.results import Results
from models.common import AutoBackend
from yolocode.yolov8.utils import callbacks, ops, LOGGER, colorstr, MACOS, WINDOWS
from collections import defaultdict
from yolocode.yolov5.utils.general import increment_path
from yolocode.yolov8.utils.checks import check_imgsz
from yolocode.yolov8.utils.torch_utils import select_device


class YOLOv11Thread(YOLOv8Thread):
    def __init__(self):
        super(YOLOv11Thread, self).__init__()
