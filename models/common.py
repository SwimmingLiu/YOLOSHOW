import ast
import contextlib
import importlib
import json
import math
import numpy as np
import os
import pandas as pd
import platform
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import zipfile
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path
from PIL import Image
from torch.cuda import amp
from torch.nn.modules.utils import _pair
from torchvision.ops import DeformConv2d
from urllib.parse import urlparse
from typing import Optional
from IPython.display import display
import cv2
from utils import glo

try:
    import ultralytics
    assert hasattr(ultralytics, "__version__")  # verify package is not directory
except (ImportError, AssertionError):
    import os
    os.system("pip install -U ultralytics")
    import ultralytics

yoloname = glo.get_value('yoloname')
yoloname1 = glo.get_value('yoloname1')
yoloname2 = glo.get_value('yoloname2')

yolo_name = ((str(yoloname1) if yoloname1 else '') + (str(yoloname2) if str(
    yoloname2) else '')) if yoloname1 or yoloname2 else yoloname


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class Conv(nn.Module):
        # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
        default_act = nn.SiLU()  # default activation

        def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
            super().__init__()
            self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
            self.bn = nn.BatchNorm2d(c2)
            self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        def forward(self, x):
            return self.act(self.bn(self.conv(x)))

        def forward_fuse(self, x):
            return self.act(self.conv(x))
class DWConv(Conv):
    # Depth-wise convolution
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)
class DWConvTranspose2d(nn.ConvTranspose2d):
    # Depth-wise transpose convolution
    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))
class TransformerLayer(nn.Module):
        # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
        def __init__(self, c, num_heads):
            super().__init__()
            self.q = nn.Linear(c, c, bias=False)
            self.k = nn.Linear(c, c, bias=False)
            self.v = nn.Linear(c, c, bias=False)
            self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
            self.fc1 = nn.Linear(c, c, bias=False)
            self.fc2 = nn.Linear(c, c, bias=False)

        def forward(self, x):
            x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
            x = self.fc2(self.fc1(x)) + x
            return x
class TransformerBlock(nn.Module):
        # Vision Transformer https://arxiv.org/abs/2010.11929
        def __init__(self, c1, c2, num_heads, num_layers):
            super().__init__()
            self.conv = None
            if c1 != c2:
                self.conv = Conv(c1, c2)
            self.linear = nn.Linear(c2, c2)  # learnable position embedding
            self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
            self.c2 = c2

        def forward(self, x):
            if self.conv is not None:
                x = self.conv(x)
            b, _, w, h = x.shape
            p = x.flatten(2).permute(2, 0, 1)
            return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)
class Bottleneck(nn.Module):
        # Standard bottleneck
        def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
            super().__init__()
            c_ = int(c2 * e)  # hidden channels
            self.cv1 = Conv(c1, c_, 1, 1)
            self.cv2 = Conv(c_, c2, 3, 1, g=g)
            self.add = shortcut and c1 == c2

        def forward(self, x):
            return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
class BottleneckCSP(nn.Module):
        # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
        def __init__(self, c1, c2, n=1, shortcut=True, g=1,
                     e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
            super().__init__()
            c_ = int(c2 * e)  # hidden channels
            self.cv1 = Conv(c1, c_, 1, 1)
            self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
            self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
            self.cv4 = Conv(2 * c_, c2, 1, 1)
            self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
            self.act = nn.SiLU()
            self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

        def forward(self, x):
            y1 = self.cv3(self.m(self.cv1(x)))
            y2 = self.cv2(x)
            return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))
class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))
class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))
class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))
class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)
class Contract(nn.Module):
        # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
        def __init__(self, gain=2):
            super().__init__()
            self.gain = gain

        def forward(self, x):
            b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
            s = self.gain
            x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
            x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
            return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)
class Expand(nn.Module):
        # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
        def __init__(self, gain=2):
            super().__init__()
            self.gain = gain

        def forward(self, x):
            b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
            s = self.gain
            x = x.view(b, s, s, c // s ** 2, h, w)  # x(1,2,2,16,80,80)
            x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
            return x.view(b, c // s ** 2, h * s, w * s)  # x(1,16,160,160)
class Concat(nn.Module):
        # Concatenate a list of tensors along dimension
        def __init__(self, dimension=1):
            super().__init__()
            self.d = dimension

        def forward(self, x):
            return torch.cat(x, self.d)
class Proto(nn.Module):
        # YOLOv5 mask Proto module for segmentation models
        def __init__(self, c1, c_=256, c2=32):  # ch_in, number of protos, number of masks
            super().__init__()
            self.cv1 = Conv(c1, c_, k=3)
            self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
            self.cv2 = Conv(c_, c_, k=3)
            self.cv3 = Conv(c_, c2)

        def forward(self, x):
            return self.cv3(self.cv2(self.upsample(self.cv1(x))))
class ImplicitA(nn.Module):
    def __init__(self, channel, mean=0., std=.02):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit + x
class ImplicitM(nn.Module):
    def __init__(self, channel, mean=1., std=.02):
        super(ImplicitM, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit * x
### --- YOLOv5 Code --- ###
if "yolov5" in yolo_name:
    from yolocode.yolov8.utils.plotting import Annotator, colors, save_one_box
    from yolocode.yolov5.utils import TryExcept
    from yolocode.yolov5.utils.dataloaders import exif_transpose, letterbox
    from yolocode.yolov5.utils.general import (
        LOGGER,
        ROOT,
        Profile,
        check_requirements,
        check_suffix,
        check_version,
        colorstr,
        increment_path,
        is_jupyter,
        make_divisible,
        non_max_suppression,
        scale_boxes,
        xywh2xyxy,
        xyxy2xywh,
        yaml_load,
    )
    from yolocode.yolov5.utils.torch_utils import copy_attr, smart_inference_mode
    from models.experimental import attempt_download_YOLOV5, \
        attempt_load_YOLOv5  # scoped to avoid circular import

    class CrossConv(nn.Module):
        # Cross Convolution Downsample
        def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
            # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
            super().__init__()
            c_ = int(c2 * e)  # hidden channels
            self.cv1 = Conv(c1, c_, (1, k), (1, s))
            self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
            self.add = shortcut and c1 == c2

        def forward(self, x):
            return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    class C3(nn.Module):
        # CSP Bottleneck with 3 convolutions
        def __init__(self, c1, c2, n=1, shortcut=True, g=1,
                     e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
            super().__init__()
            c_ = int(c2 * e)  # hidden channels
            self.cv1 = Conv(c1, c_, 1, 1)
            self.cv2 = Conv(c1, c_, 1, 1)
            self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
            self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

        def forward(self, x):
            return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
    class C3x(C3):
        # C3 module with cross-convolutions
        def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
            super().__init__(c1, c2, n, shortcut, g, e)
            c_ = int(c2 * e)
            self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)))
    class C3TR(C3):
        # C3 module with TransformerBlock()
        def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
            super().__init__(c1, c2, n, shortcut, g, e)
            c_ = int(c2 * e)
            self.m = TransformerBlock(c_, c_, 4, n)
    class C3SPP(C3):
        # C3 module with SPP()
        def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
            super().__init__(c1, c2, n, shortcut, g, e)
            c_ = int(c2 * e)
            self.m = SPP(c_, c_, k)
    class C3Ghost(C3):
        # C3 module with GhostBottleneck()
        def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
            super().__init__(c1, c2, n, shortcut, g, e)
            c_ = int(c2 * e)  # hidden channels
            self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))
    class GhostBottleneck(nn.Module):
        # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
        def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
            super().__init__()
            c_ = c2 // 2
            self.conv = nn.Sequential(
                GhostConv(c1, c_, 1, 1),  # pw
                DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                GhostConv(c_, c2, 1, 1, act=False),
            )  # pw-linear
            self.shortcut = (
                nn.Sequential(DWConv(c1, c1, k, s, act=False),
                              Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
            )

        def forward(self, x):
            return self.conv(x) + self.shortcut(x)
    class Classify(nn.Module):
        # YOLOv5 classification head, i.e. x(b,c1,20,20) to x(b,c2)
        def __init__(
                self, c1, c2, k=1, s=1, p=None, g=1, dropout_p=0.0
        ):  # ch_in, ch_out, kernel, stride, padding, groups, dropout probability
            super().__init__()
            c_ = 1280  # efficientnet_b0 size
            self.conv = Conv(c1, c_, k, s, autopad(k, p), g)
            self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
            self.drop = nn.Dropout(p=dropout_p, inplace=True)
            self.linear = nn.Linear(c_, c2)  # to x(b,c2)

        def forward(self, x):
            if isinstance(x, list):
                x = torch.cat(x, 1)
            return self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
    class DetectMultiBackend_YOLOv5(nn.Module):
        # YOLOv5 MultiBackend class for python inference on various backends
        def __init__(self, weights="yolov5s.pt", device=torch.device("cpu"), dnn=False, data=None, fp16=False,
                     fuse=True, parent_workpath=None):
            # Usage:
            #   PyTorch:              weights = *.pt
            #   TorchScript:                    *.torchscript
            #   ONNX Runtime:                   *.onnx
            #   ONNX OpenCV DNN:                *.onnx --dnn
            #   OpenVINO:                       *_openvino_model
            #   CoreML:                         *.mlmodel
            #   TensorRT:                       *.engine
            #   TensorFlow SavedModel:          *_saved_model
            #   TensorFlow GraphDef:            *.pb
            #   TensorFlow Lite:                *.tflite
            #   TensorFlow Edge TPU:            *_edgetpu.tflite
            #   PaddlePaddle:                   *_paddle_model
            super().__init__()
            w = str(weights[0] if isinstance(weights, list) else weights)
            pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, triton = self._model_type(
                w)
            fp16 &= pt or jit or onnx or engine or triton  # FP16
            nhwc = coreml or saved_model or pb or tflite or edgetpu  # BHWC formats (vs torch BCWH)
            stride = 32  # default stride
            cuda = torch.cuda.is_available() and device.type != "cpu"  # use CUDA
            if not (pt or triton):
                w = attempt_download_YOLOV5(w)  # download if not local

            if pt:  # PyTorch
                model = attempt_load_YOLOv5(weights if isinstance(weights, list) else w, device=device, inplace=True,
                                            fuse=fuse)
                stride = max(int(model.stride.max()), 32)  # model stride
                names = model.module.names if hasattr(model, "module") else model.names  # get class names
                model.half() if fp16 else model.float()
                self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
            elif jit:  # TorchScript
                LOGGER.info(f"Loading {w} for TorchScript inference...")
                extra_files = {"config.txt": ""}  # model metadata
                model = torch.jit.load(w, _extra_files=extra_files, map_location=device)
                model.half() if fp16 else model.float()
                if extra_files["config.txt"]:  # load metadata dict
                    d = json.loads(
                        extra_files["config.txt"],
                        object_hook=lambda d: {int(k) if k.isdigit() else k: v for k, v in d.items()},
                    )
                    stride, names = int(d["stride"]), d["names"]
            elif dnn:  # ONNX OpenCV DNN
                LOGGER.info(f"Loading {w} for ONNX OpenCV DNN inference...")
                check_requirements("opencv-python>=4.5.4")
                net = cv2.dnn.readNetFromONNX(w)
            elif onnx:  # ONNX Runtime
                LOGGER.info(f"Loading {w} for ONNX Runtime inference...")
                check_requirements(("onnx", "onnxruntime-gpu" if cuda else "onnxruntime"))
                import onnxruntime

                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]
                session = onnxruntime.InferenceSession(w, providers=providers)
                output_names = [x.name for x in session.get_outputs()]
                meta = session.get_modelmeta().custom_metadata_map  # metadata
                if "stride" in meta:
                    stride, names = int(meta["stride"]), eval(meta["names"])
            elif xml:  # OpenVINO
                LOGGER.info(f"Loading {w} for OpenVINO inference...")
                check_requirements("openvino>=2023.0")  # requires openvino-dev: https://pypi.org/project/openvino-dev/
                from openvino.runtime import Core, Layout, get_batch

                core = Core()
                if not Path(w).is_file():  # if not *.xml
                    w = next(Path(w).glob("*.xml"))  # get *.xml file from *_openvino_model dir
                ov_model = core.read_model(model=w, weights=Path(w).with_suffix(".bin"))
                if ov_model.get_parameters()[0].get_layout().empty:
                    ov_model.get_parameters()[0].set_layout(Layout("NCHW"))
                batch_dim = get_batch(ov_model)
                if batch_dim.is_static:
                    batch_size = batch_dim.get_length()
                ov_compiled_model = core.compile_model(ov_model,
                                                       device_name="AUTO")  # AUTO selects best available device
                stride, names = self._load_metadata(Path(w).with_suffix(".yaml"))  # load metadata
            elif engine:  # TensorRT
                LOGGER.info(f"Loading {w} for TensorRT inference...")
                import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download

                check_version(trt.__version__, "7.0.0", hard=True)  # require tensorrt>=7.0.0
                if device.type == "cpu":
                    device = torch.device("cuda:0")
                Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
                logger = trt.Logger(trt.Logger.INFO)
                with open(w, "rb") as f, trt.Runtime(logger) as runtime:
                    model = runtime.deserialize_cuda_engine(f.read())
                context = model.create_execution_context()
                bindings = OrderedDict()
                output_names = []
                fp16 = False  # default updated below
                dynamic = False
                for i in range(model.num_bindings):
                    name = model.get_binding_name(i)
                    dtype = trt.nptype(model.get_binding_dtype(i))
                    if model.binding_is_input(i):
                        if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                            dynamic = True
                            context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
                        if dtype == np.float16:
                            fp16 = True
                    else:  # output
                        output_names.append(name)
                    shape = tuple(context.get_binding_shape(i))
                    im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                    bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
                binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
                batch_size = bindings["images"].shape[0]  # if dynamic, this is instead max batch size
            elif coreml:  # CoreML
                LOGGER.info(f"Loading {w} for CoreML inference...")
                import coremltools as ct

                model = ct.models.MLModel(w)
            elif saved_model:  # TF SavedModel
                LOGGER.info(f"Loading {w} for TensorFlow SavedModel inference...")
                import tensorflow as tf

                keras = False  # assume TF1 saved_model
                model = tf.keras.models.load_model(w) if keras else tf.saved_model.load(w)
            elif pb:  # GraphDef https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
                LOGGER.info(f"Loading {w} for TensorFlow GraphDef inference...")
                import tensorflow as tf

                def wrap_frozen_graph(gd, inputs, outputs):
                    x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped
                    ge = x.graph.as_graph_element
                    return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.map_structure(ge, outputs))

                def gd_outputs(gd):
                    name_list, input_list = [], []
                    for node in gd.node:  # tensorflow.core.framework.node_def_pb2.NodeDef
                        name_list.append(node.name)
                        input_list.extend(node.input)
                    return sorted(f"{x}:0" for x in list(set(name_list) - set(input_list)) if not x.startswith("NoOp"))

                gd = tf.Graph().as_graph_def()  # TF GraphDef
                with open(w, "rb") as f:
                    gd.ParseFromString(f.read())
                frozen_func = wrap_frozen_graph(gd, inputs="x:0", outputs=gd_outputs(gd))
            elif tflite or edgetpu:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
                try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
                    from tflite_runtime.interpreter import Interpreter, load_delegate
                except ImportError:
                    import tensorflow as tf

                    Interpreter, load_delegate = (
                        tf.lite.Interpreter,
                        tf.lite.experimental.load_delegate,
                    )
                if edgetpu:  # TF Edge TPU https://coral.ai/software/#edgetpu-runtime
                    LOGGER.info(f"Loading {w} for TensorFlow Lite Edge TPU inference...")
                    delegate = {"Linux": "libedgetpu.so.1", "Darwin": "libedgetpu.1.dylib", "Windows": "edgetpu.dll"}[
                        platform.system()
                    ]
                    interpreter = Interpreter(model_path=w, experimental_delegates=[load_delegate(delegate)])
                else:  # TFLite
                    LOGGER.info(f"Loading {w} for TensorFlow Lite inference...")
                    interpreter = Interpreter(model_path=w)  # load TFLite model
                interpreter.allocate_tensors()  # allocate
                input_details = interpreter.get_input_details()  # inputs
                output_details = interpreter.get_output_details()  # outputs
                # load metadata
                with contextlib.suppress(zipfile.BadZipFile):
                    with zipfile.ZipFile(w, "r") as model:
                        meta_file = model.namelist()[0]
                        meta = ast.literal_eval(model.read(meta_file).decode("utf-8"))
                        stride, names = int(meta["stride"]), meta["names"]
            elif tfjs:  # TF.js
                raise NotImplementedError("ERROR: YOLOv5 TF.js inference is not supported")
            elif paddle:  # PaddlePaddle
                LOGGER.info(f"Loading {w} for PaddlePaddle inference...")
                check_requirements("paddlepaddle-gpu" if cuda else "paddlepaddle")
                import paddle.inference as pdi

                if not Path(w).is_file():  # if not *.pdmodel
                    w = next(Path(w).rglob("*.pdmodel"))  # get *.pdmodel file from *_paddle_model dir
                weights = Path(w).with_suffix(".pdiparams")
                config = pdi.Config(str(w), str(weights))
                if cuda:
                    config.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=0)
                predictor = pdi.create_predictor(config)
                input_handle = predictor.get_input_handle(predictor.get_input_names()[0])
                output_names = predictor.get_output_names()
            elif triton:  # NVIDIA Triton Inference Server
                LOGGER.info(f"Using {w} as Triton Inference Server...")
                check_requirements("tritonclient[all]")
                from utils.triton import TritonRemoteModel

                model = TritonRemoteModel(url=w)
                nhwc = model.runtime.startswith("tensorflow")
            else:
                raise NotImplementedError(f"ERROR: {w} is not a supported format")

            # class names
            if "names" not in locals():
                names = yaml_load(data)["names"] if data else {i: f"class{i}" for i in range(999)}
            if names[0] == "n01440764" and len(names) == 1000:  # ImageNet
                names = yaml_load(ROOT / "data/ImageNet.yaml")["names"]  # human-readable names

            self.__dict__.update(locals())  # assign all variables to self

        def forward(self, im, augment=False, visualize=False):
            # YOLOv5 MultiBackend inference
            b, ch, h, w = im.shape  # batch, channel, height, width
            if self.fp16 and im.dtype != torch.float16:
                im = im.half()  # to FP16
            if self.nhwc:
                im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)

            if self.pt:  # PyTorch
                y = self.model(im, augment=augment, visualize=visualize) if augment or visualize else self.model(im)
            elif self.jit:  # TorchScript
                y = self.model(im)
            elif self.dnn:  # ONNX OpenCV DNN
                im = im.cpu().numpy()  # torch to numpy
                self.net.setInput(im)
                y = self.net.forward()
            elif self.onnx:  # ONNX Runtime
                im = im.cpu().numpy()  # torch to numpy
                y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
            elif self.xml:  # OpenVINO
                im = im.cpu().numpy()  # FP32
                y = list(self.ov_compiled_model(im).values())
            elif self.engine:  # TensorRT
                if self.dynamic and im.shape != self.bindings["images"].shape:
                    i = self.model.get_binding_index("images")
                    self.context.set_binding_shape(i, im.shape)  # reshape if dynamic
                    self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)
                    for name in self.output_names:
                        i = self.model.get_binding_index(name)
                        self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))
                s = self.bindings["images"].shape
                assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
                self.binding_addrs["images"] = int(im.data_ptr())
                self.context.execute_v2(list(self.binding_addrs.values()))
                y = [self.bindings[x].data for x in sorted(self.output_names)]
            elif self.coreml:  # CoreML
                im = im.cpu().numpy()
                im = Image.fromarray((im[0] * 255).astype("uint8"))
                # im = im.resize((192, 320), Image.BILINEAR)
                y = self.model.predict({"image": im})  # coordinates are xywh normalized
                if "confidence" in y:
                    box = xywh2xyxy(y["coordinates"] * [[w, h, w, h]])  # xyxy pixels
                    conf, cls = y["confidence"].max(1), y["confidence"].argmax(1).astype(np.float)
                    y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
                else:
                    y = list(reversed(y.values()))  # reversed for segmentation models (pred, proto)
            elif self.paddle:  # PaddlePaddle
                im = im.cpu().numpy().astype(np.float32)
                self.input_handle.copy_from_cpu(im)
                self.predictor.run()
                y = [self.predictor.get_output_handle(x).copy_to_cpu() for x in self.output_names]
            elif self.triton:  # NVIDIA Triton Inference Server
                y = self.model(im)
            else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
                im = im.cpu().numpy()
                if self.saved_model:  # SavedModel
                    y = self.model(im, training=False) if self.keras else self.model(im)
                elif self.pb:  # GraphDef
                    y = self.frozen_func(x=self.tf.constant(im))
                else:  # Lite or Edge TPU
                    input = self.input_details[0]
                    int8 = input["dtype"] == np.uint8  # is TFLite quantized uint8 model
                    if int8:
                        scale, zero_point = input["quantization"]
                        im = (im / scale + zero_point).astype(np.uint8)  # de-scale
                    self.interpreter.set_tensor(input["index"], im)
                    self.interpreter.invoke()
                    y = []
                    for output in self.output_details:
                        x = self.interpreter.get_tensor(output["index"])
                        if int8:
                            scale, zero_point = output["quantization"]
                            x = (x.astype(np.float32) - zero_point) * scale  # re-scale
                        y.append(x)
                y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
                y[0][..., :4] *= [w, h, w, h]  # xywh normalized to pixels

            if isinstance(y, (list, tuple)):
                return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
            else:
                return self.from_numpy(y)

        def from_numpy(self, x):
            return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

        def warmup(self, imgsz=(1, 3, 640, 640)):
            # Warmup model by running inference once
            warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton
            if any(warmup_types) and (self.device.type != "cpu" or self.triton):
                im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
                for _ in range(2 if self.jit else 1):  #
                    self.forward(im)  # warmup

        @staticmethod
        def _model_type(p="path/to/model.pt"):
            # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
            # types = [pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle]
            from yolocode.yolov5.export import export_formats
            from yolocode.yolov5.utils.downloads import is_url

            sf = list(export_formats().Suffix)  # export suffixes
            if not is_url(p, check=False):
                check_suffix(p, sf)  # checks
            url = urlparse(p)  # if url may be Triton inference server
            types = [s in Path(p).name for s in sf]
            types[8] &= not types[9]  # tflite &= not edgetpu
            triton = not any(types) and all([any(s in url.scheme for s in ["http", "grpc"]), url.netloc])
            return types + [triton]

        @staticmethod
        def _load_metadata(f=Path("path/to/meta.yaml")):
            # Load metadata from meta.yaml if it exists
            if f.exists():
                d = yaml_load(f)
                return d["stride"], d["names"]  # assign stride, names
            return None, None
    class AutoShape(nn.Module):
        # YOLOv5 input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
        conf = 0.25  # NMS confidence threshold
        iou = 0.45  # NMS IoU threshold
        agnostic = False  # NMS class-agnostic
        multi_label = False  # NMS multiple labels per box
        classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
        max_det = 1000  # maximum number of detections per image
        amp = False  # Automatic Mixed Precision (AMP) inference

        def __init__(self, model, verbose=True):
            super().__init__()
            if verbose:
                LOGGER.info("Adding AutoShape... ")
            copy_attr(self, model, include=("yaml", "nc", "hyp", "names", "stride", "abc"),
                      exclude=())  # copy attributes
            self.dmb = isinstance(model, DetectMultiBackend_YOLOv5)  # DetectMultiBackend() instance
            self.pt = not self.dmb or model.pt  # PyTorch model
            self.model = model.eval()
            if self.pt:
                m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
                m.inplace = False  # Detect.inplace=False for safe multithread inference
                m.export = True  # do not output loss values

        def _apply(self, fn):
            # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
            self = super()._apply(fn)
            if self.pt:
                m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
                m.stride = fn(m.stride)
                m.grid = list(map(fn, m.grid))
                if isinstance(m.anchor_grid, list):
                    m.anchor_grid = list(map(fn, m.anchor_grid))
            return self

        @smart_inference_mode()
        def forward(self, ims, size=640, augment=False, profile=False):
            # Inference from various sources. For size(height=640, width=1280), RGB images example inputs are:
            #   file:        ims = 'data/images/zidane.jpg'  # str or PosixPath
            #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
            #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
            #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
            #   numpy:           = np.zeros((640,1280,3))  # HWC
            #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
            #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

            dt = (Profile(), Profile(), Profile())
            with dt[0]:
                if isinstance(size, int):  # expand
                    size = (size, size)
                p = next(self.model.parameters()) if self.pt else torch.empty(1, device=self.model.device)  # param
                autocast = self.amp and (p.device.type != "cpu")  # Automatic Mixed Precision (AMP) inference
                if isinstance(ims, torch.Tensor):  # torch
                    with amp.autocast(autocast):
                        return self.model(ims.to(p.device).type_as(p), augment=augment)  # inference

                # Pre-process
                n, ims = (len(ims), list(ims)) if isinstance(ims, (list, tuple)) else (
                    1, [ims])  # number, list of images
                shape0, shape1, files = [], [], []  # image and inference shapes, filenames
                for i, im in enumerate(ims):
                    f = f"image{i}"  # filename
                    if isinstance(im, (str, Path)):  # filename or uri
                        im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith("http") else im), im
                        im = np.asarray(exif_transpose(im))
                    elif isinstance(im, Image.Image):  # PIL Image
                        im, f = np.asarray(exif_transpose(im)), getattr(im, "filename", f) or f
                    files.append(Path(f).with_suffix(".jpg").name)
                    if im.shape[0] < 5:  # image in CHW
                        im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
                    im = im[..., :3] if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)  # enforce 3ch input
                    s = im.shape[:2]  # HWC
                    shape0.append(s)  # image shape
                    g = max(size) / max(s)  # gain
                    shape1.append([int(y * g) for y in s])
                    ims[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
                shape1 = [make_divisible(x, self.stride) for x in np.array(shape1).max(0)]  # inf shape
                x = [letterbox(im, shape1, auto=False)[0] for im in ims]  # pad
                x = np.ascontiguousarray(np.array(x).transpose((0, 3, 1, 2)))  # stack and BHWC to BCHW
                x = torch.from_numpy(x).to(p.device).type_as(p) / 255  # uint8 to fp16/32

            with amp.autocast(autocast):
                # Inference
                with dt[1]:
                    y = self.model(x, augment=augment)  # forward

                # Post-process
                with dt[2]:
                    y = non_max_suppression(
                        y if self.dmb else y[0],
                        self.conf,
                        self.iou,
                        self.classes,
                        self.agnostic,
                        self.multi_label,
                        max_det=self.max_det,
                    )  # NMS
                    for i in range(n):
                        scale_boxes(shape1, y[i][:, :4], shape0[i])

                return Detections(ims, y, files, dt, self.names, x.shape)
    class Detections:
        # YOLOv5 detections class for inference results
        def __init__(self, ims, pred, files, times=(0, 0, 0), names=None, shape=None):
            super().__init__()
            d = pred[0].device  # device
            gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in ims]  # normalizations
            self.ims = ims  # list of images as numpy arrays
            self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
            self.names = names  # class names
            self.files = files  # image filenames
            self.times = times  # profiling times
            self.xyxy = pred  # xyxy pixels
            self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
            self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
            self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
            self.n = len(self.pred)  # number of images (batch size)
            self.t = tuple(x.t / self.n * 1e3 for x in times)  # timestamps (ms)
            self.s = tuple(shape)  # inference BCHW shape

        def _run(self, pprint=False, show=False, save=False, crop=False, render=False, labels=True, save_dir=Path("")):
            s, crops = "", []
            for i, (im, pred) in enumerate(zip(self.ims, self.pred)):
                s += f"\nimage {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} "  # string
                if pred.shape[0]:
                    for c in pred[:, -1].unique():
                        n = (pred[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    s = s.rstrip(", ")
                    if show or save or render or crop:
                        annotator = Annotator(im, example=str(self.names))
                        for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
                            label = f"{self.names[int(cls)]} {conf:.2f}"
                            if crop:
                                file = save_dir / "crops" / self.names[int(cls)] / self.files[i] if save else None
                                crops.append(
                                    {
                                        "box": box,
                                        "conf": conf,
                                        "cls": cls,
                                        "label": label,
                                        "im": save_one_box(box, im, file=file, save=save),
                                    }
                                )
                            else:  # all others
                                annotator.box_label(box, label if labels else "", color=colors(cls))
                        im = annotator.im
                else:
                    s += "(no detections)"

                im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
                if show:
                    if is_jupyter():
                        from IPython.display import display

                        display(im)
                    else:
                        im.show(self.files[i])
                if save:
                    f = self.files[i]
                    im.save(save_dir / f)  # save
                    if i == self.n - 1:
                        LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
                if render:
                    self.ims[i] = np.asarray(im)
            if pprint:
                s = s.lstrip("\n")
                return f"{s}\nSpeed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {self.s}" % self.t
            if crop:
                if save:
                    LOGGER.info(f"Saved results to {save_dir}\n")
                return crops

        @TryExcept("Showing images is not supported in this environment")
        def show(self, labels=True):
            self._run(show=True, labels=labels)  # show results

        def save(self, labels=True, save_dir="runs/detect/exp", exist_ok=False):
            save_dir = increment_path(save_dir, exist_ok, mkdir=True)  # increment save_dir
            self._run(save=True, labels=labels, save_dir=save_dir)  # save results

        def crop(self, save=True, save_dir="runs/detect/exp", exist_ok=False):
            save_dir = increment_path(save_dir, exist_ok, mkdir=True) if save else None
            return self._run(crop=True, save=save, save_dir=save_dir)  # crop results

        def render(self, labels=True):
            self._run(render=True, labels=labels)  # render results
            return self.ims

        def pandas(self):
            # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
            new = copy(self)  # return copy
            ca = "xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"  # xyxy columns
            cb = "xcenter", "ycenter", "width", "height", "confidence", "class", "name"  # xywh columns
            for k, c in zip(["xyxy", "xyxyn", "xywh", "xywhn"], [ca, ca, cb, cb]):
                a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in
                     getattr(self, k)]  # update
                setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
            return new

        def tolist(self):
            # return a list of Detections objects, i.e. 'for result in results.tolist():'
            r = range(self.n)  # iterable
            return [
                Detections(
                    [self.ims[i]],
                    [self.pred[i]],
                    [self.files[i]],
                    self.times,
                    self.names,
                    self.s,
                )
                for i in r
            ]

        def print(self):
            LOGGER.info(self.__str__())

        def __len__(self):  # override len(results)
            return self.n

        def __str__(self):  # override print(results)
            return self._run(pprint=True)  # print results

        def __repr__(self):
            return f"YOLOv5 {self.__class__} instance\n" + self.__str__()
### --- YOLOv5 Code --- ###

### --- YOLOv7 Code --- ###
if "yolov7" in yolo_name:
    from yolocode.yolov7.utils.datasets import letterbox
    from yolocode.yolov7.utils.general import non_max_suppression, make_divisible, scale_coords, increment_path, \
        xyxy2xywh
    from yolocode.yolov7.utils.plots import color_list, plot_one_box
    from yolocode.yolov7.utils.torch_utils import time_synchronized

    class MP(nn.Module):
        def __init__(self, k=2):
            super(MP, self).__init__()
            self.m = nn.MaxPool2d(kernel_size=k, stride=k)

        def forward(self, x):
            return self.m(x)
    class SP(nn.Module):
        def __init__(self, k=3, s=1):
            super(SP, self).__init__()
            self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=k // 2)

        def forward(self, x):
            return self.m(x)
    class ReOrg(nn.Module):
        def __init__(self):
            super(ReOrg, self).__init__()

        def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
            return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
    class Chuncat(nn.Module):
        def __init__(self, dimension=1):
            super(Chuncat, self).__init__()
            self.d = dimension

        def forward(self, x):
            x1 = []
            x2 = []
            for xi in x:
                xi1, xi2 = xi.chunk(2, self.d)
                x1.append(xi1)
                x2.append(xi2)
            return torch.cat(x1 + x2, self.d)
    class Shortcut(nn.Module):
        def __init__(self, dimension=0):
            super(Shortcut, self).__init__()
            self.d = dimension

        def forward(self, x):
            return x[0] + x[1]
    class Foldcut(nn.Module):
        def __init__(self, dimension=0):
            super(Foldcut, self).__init__()
            self.d = dimension

        def forward(self, x):
            x1, x2 = x.chunk(2, self.d)
            return x1 + x2
    class RobustConv(nn.Module):
        # Robust convolution (use high kernel size 7-11 for: downsampling and other layers). Train for 300 - 450 epochs.
        def __init__(self, c1, c2, k=7, s=1, p=None, g=1, act=True,
                     layer_scale_init_value=1e-6):  # ch_in, ch_out, kernel, stride, padding, groups
            super(RobustConv, self).__init__()
            self.conv_dw = Conv(c1, c1, k=k, s=s, p=p, g=c1, act=act)
            self.conv1x1 = nn.Conv2d(c1, c2, 1, 1, 0, groups=1, bias=True)
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(c2)) if layer_scale_init_value > 0 else None

        def forward(self, x):
            x = x.to(memory_format=torch.channels_last)
            x = self.conv1x1(self.conv_dw(x))
            if self.gamma is not None:
                x = x.mul(self.gamma.reshape(1, -1, 1, 1))
            return x
    class RobustConv2(nn.Module):
        # Robust convolution 2 (use [32, 5, 2] or [32, 7, 4] or [32, 11, 8] for one of the paths in CSP).
        def __init__(self, c1, c2, k=7, s=4, p=None, g=1, act=True,
                     layer_scale_init_value=1e-6):  # ch_in, ch_out, kernel, stride, padding, groups
            super(RobustConv2, self).__init__()
            self.conv_strided = Conv(c1, c1, k=k, s=s, p=p, g=c1, act=act)
            self.conv_deconv = nn.ConvTranspose2d(in_channels=c1, out_channels=c2, kernel_size=s, stride=s,
                                                  padding=0, bias=True, dilation=1, groups=1
                                                  )
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(c2)) if layer_scale_init_value > 0 else None

        def forward(self, x):
            x = self.conv_deconv(self.conv_strided(x))
            if self.gamma is not None:
                x = x.mul(self.gamma.reshape(1, -1, 1, 1))
            return x
    def DWConv(c1, c2, k=1, s=1, act=True):
        # Depthwise convolution
        return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)
    class Stem(nn.Module):
        # Stem
        def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
            super(Stem, self).__init__()
            c_ = int(c2 / 2)  # hidden channels
            self.cv1 = Conv(c1, c_, 3, 2)
            self.cv2 = Conv(c_, c_, 1, 1)
            self.cv3 = Conv(c_, c_, 3, 2)
            self.pool = torch.nn.MaxPool2d(2, stride=2)
            self.cv4 = Conv(2 * c_, c2, 1, 1)

        def forward(self, x):
            x = self.cv1(x)
            return self.cv4(torch.cat((self.cv3(self.cv2(x)), self.pool(x)), dim=1))
    class DownC(nn.Module):
        # Spatial pyramid pooling layer used in YOLOv3-SPP
        def __init__(self, c1, c2, n=1, k=2):
            super(DownC, self).__init__()
            c_ = int(c1)  # hidden channels
            self.cv1 = Conv(c1, c_, 1, 1)
            self.cv2 = Conv(c_, c2 // 2, 3, k)
            self.cv3 = Conv(c1, c2 // 2, 1, 1)
            self.mp = nn.MaxPool2d(kernel_size=k, stride=k)

        def forward(self, x):
            return torch.cat((self.cv2(self.cv1(x)), self.cv3(self.mp(x))), dim=1)
    class Res(nn.Module):
        # ResNet bottleneck
        def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
            super(Res, self).__init__()
            c_ = int(c2 * e)  # hidden channels
            self.cv1 = Conv(c1, c_, 1, 1)
            self.cv2 = Conv(c_, c_, 3, 1, g=g)
            self.cv3 = Conv(c_, c2, 1, 1)
            self.add = shortcut and c1 == c2

        def forward(self, x):
            return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))
    class ResX(Res):
        # ResNet bottleneck
        def __init__(self, c1, c2, shortcut=True, g=32, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
            super().__init__(c1, c2, shortcut, g, e)
            c_ = int(c2 * e)  # hidden channels
    class Ghost(nn.Module):
        # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
        def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
            super(Ghost, self).__init__()
            c_ = c2 // 2
            self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),  # pw
                                      DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                      GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
            self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                          Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

        def forward(self, x):
            return self.conv(x) + self.shortcut(x)
    class SPPCSPC(nn.Module):
        # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
        def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
            super(SPPCSPC, self).__init__()
            c_ = int(2 * c2 * e)  # hidden channels
            self.cv1 = Conv(c1, c_, 1, 1)
            self.cv2 = Conv(c1, c_, 1, 1)
            self.cv3 = Conv(c_, c_, 3, 1)
            self.cv4 = Conv(c_, c_, 1, 1)
            self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
            self.cv5 = Conv(4 * c_, c_, 1, 1)
            self.cv6 = Conv(c_, c_, 3, 1)
            self.cv7 = Conv(2 * c_, c2, 1, 1)

        def forward(self, x):
            x1 = self.cv4(self.cv3(self.cv1(x)))
            y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
            y2 = self.cv2(x)
            return self.cv7(torch.cat((y1, y2), dim=1))
    class GhostSPPCSPC(SPPCSPC):
        # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
        def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
            super().__init__(c1, c2, n, shortcut, g, e, k)
            c_ = int(2 * c2 * e)  # hidden channels
            self.cv1 = GhostConv(c1, c_, 1, 1)
            self.cv2 = GhostConv(c1, c_, 1, 1)
            self.cv3 = GhostConv(c_, c_, 3, 1)
            self.cv4 = GhostConv(c_, c_, 1, 1)
            self.cv5 = GhostConv(4 * c_, c_, 1, 1)
            self.cv6 = GhostConv(c_, c_, 3, 1)
            self.cv7 = GhostConv(2 * c_, c2, 1, 1)
    class GhostStem(Stem):
        # Stem
        def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
            super().__init__(c1, c2, k, s, p, g, act)
            c_ = int(c2 / 2)  # hidden channels
            self.cv1 = GhostConv(c1, c_, 3, 2)
            self.cv2 = GhostConv(c_, c_, 1, 1)
            self.cv3 = GhostConv(c_, c_, 3, 2)
            self.cv4 = GhostConv(2 * c_, c2, 1, 1)
    class BottleneckCSPA(nn.Module):
        # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
        def __init__(self, c1, c2, n=1, shortcut=True, g=1,
                     e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
            super(BottleneckCSPA, self).__init__()
            c_ = int(c2 * e)  # hidden channels
            self.cv1 = Conv(c1, c_, 1, 1)
            self.cv2 = Conv(c1, c_, 1, 1)
            self.cv3 = Conv(2 * c_, c2, 1, 1)
            self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

        def forward(self, x):
            y1 = self.m(self.cv1(x))
            y2 = self.cv2(x)
            return self.cv3(torch.cat((y1, y2), dim=1))
    class BottleneckCSPB(nn.Module):
        # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
        def __init__(self, c1, c2, n=1, shortcut=False, g=1,
                     e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
            super(BottleneckCSPB, self).__init__()
            c_ = int(c2)  # hidden channels
            self.cv1 = Conv(c1, c_, 1, 1)
            self.cv2 = Conv(c_, c_, 1, 1)
            self.cv3 = Conv(2 * c_, c2, 1, 1)
            self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

        def forward(self, x):
            x1 = self.cv1(x)
            y1 = self.m(x1)
            y2 = self.cv2(x1)
            return self.cv3(torch.cat((y1, y2), dim=1))
    class BottleneckCSPC(nn.Module):
        # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
        def __init__(self, c1, c2, n=1, shortcut=True, g=1,
                     e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
            super(BottleneckCSPC, self).__init__()
            c_ = int(c2 * e)  # hidden channels
            self.cv1 = Conv(c1, c_, 1, 1)
            self.cv2 = Conv(c1, c_, 1, 1)
            self.cv3 = Conv(c_, c_, 1, 1)
            self.cv4 = Conv(2 * c_, c2, 1, 1)
            self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

        def forward(self, x):
            y1 = self.cv3(self.m(self.cv1(x)))
            y2 = self.cv2(x)
            return self.cv4(torch.cat((y1, y2), dim=1))
    class ResCSPA(BottleneckCSPA):
        # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
        def __init__(self, c1, c2, n=1, shortcut=True, g=1,
                     e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
            super().__init__(c1, c2, n, shortcut, g, e)
            c_ = int(c2 * e)  # hidden channels
            self.m = nn.Sequential(*[Res(c_, c_, shortcut, g, e=0.5) for _ in range(n)])
    class ResCSPB(BottleneckCSPB):
        # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
        def __init__(self, c1, c2, n=1, shortcut=True, g=1,
                     e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
            super().__init__(c1, c2, n, shortcut, g, e)
            c_ = int(c2)  # hidden channels
            self.m = nn.Sequential(*[Res(c_, c_, shortcut, g, e=0.5) for _ in range(n)])
    class ResCSPC(BottleneckCSPC):
        # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
        def __init__(self, c1, c2, n=1, shortcut=True, g=1,
                     e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
            super().__init__(c1, c2, n, shortcut, g, e)
            c_ = int(c2 * e)  # hidden channels
            self.m = nn.Sequential(*[Res(c_, c_, shortcut, g, e=0.5) for _ in range(n)])
    class ResXCSPA(ResCSPA):
        # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
        def __init__(self, c1, c2, n=1, shortcut=True, g=32,
                     e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
            super().__init__(c1, c2, n, shortcut, g, e)
            c_ = int(c2 * e)  # hidden channels
            self.m = nn.Sequential(*[Res(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
    class ResXCSPB(ResCSPB):
        # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
        def __init__(self, c1, c2, n=1, shortcut=True, g=32,
                     e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
            super().__init__(c1, c2, n, shortcut, g, e)
            c_ = int(c2)  # hidden channels
            self.m = nn.Sequential(*[Res(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
    class ResXCSPC(ResCSPC):
        # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
        def __init__(self, c1, c2, n=1, shortcut=True, g=32,
                     e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
            super().__init__(c1, c2, n, shortcut, g, e)
            c_ = int(c2 * e)  # hidden channels
            self.m = nn.Sequential(*[Res(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
    class GhostCSPA(BottleneckCSPA):
        # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
        def __init__(self, c1, c2, n=1, shortcut=True, g=1,
                     e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
            super().__init__(c1, c2, n, shortcut, g, e)
            c_ = int(c2 * e)  # hidden channels
            self.m = nn.Sequential(*[Ghost(c_, c_) for _ in range(n)])
    class GhostCSPB(BottleneckCSPB):
        # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
        def __init__(self, c1, c2, n=1, shortcut=True, g=1,
                     e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
            super().__init__(c1, c2, n, shortcut, g, e)
            c_ = int(c2)  # hidden channels
            self.m = nn.Sequential(*[Ghost(c_, c_) for _ in range(n)])
    class GhostCSPC(BottleneckCSPC):
        # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
        def __init__(self, c1, c2, n=1, shortcut=True, g=1,
                     e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
            super().__init__(c1, c2, n, shortcut, g, e)
            c_ = int(c2 * e)  # hidden channels
            self.m = nn.Sequential(*[Ghost(c_, c_) for _ in range(n)])

    class RepConv(nn.Module):
        # Represented convolution
        # https://arxiv.org/abs/2101.03697

        def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, deploy=False):
            super(RepConv, self).__init__()

            self.deploy = deploy
            self.groups = g
            self.in_channels = c1
            self.out_channels = c2

            assert k == 3
            assert autopad(k, p) == 1

            padding_11 = autopad(k, p) - k // 2

            self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

            if deploy:
                self.rbr_reparam = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)

            else:
                self.rbr_identity = (nn.BatchNorm2d(num_features=c1) if c2 == c1 and s == 1 else None)

                self.rbr_dense = nn.Sequential(
                    nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
                    nn.BatchNorm2d(num_features=c2),
                )

                self.rbr_1x1 = nn.Sequential(
                    nn.Conv2d(c1, c2, 1, s, padding_11, groups=g, bias=False),
                    nn.BatchNorm2d(num_features=c2),
                )

        def forward(self, inputs):
            if hasattr(self, "rbr_reparam"):
                return self.act(self.rbr_reparam(inputs))

            if self.rbr_identity is None:
                id_out = 0
            else:
                id_out = self.rbr_identity(inputs)

            return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

        def get_equivalent_kernel_bias(self):
            kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
            kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
            kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
            return (
                kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
                bias3x3 + bias1x1 + biasid,
            )

        def _pad_1x1_to_3x3_tensor(self, kernel1x1):
            if kernel1x1 is None:
                return 0
            else:
                return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

        def _fuse_bn_tensor(self, branch):
            if branch is None:
                return 0, 0
            if isinstance(branch, nn.Sequential):
                kernel = branch[0].weight
                running_mean = branch[1].running_mean
                running_var = branch[1].running_var
                gamma = branch[1].weight
                beta = branch[1].bias
                eps = branch[1].eps
            else:
                assert isinstance(branch, nn.BatchNorm2d)
                if not hasattr(self, "id_tensor"):
                    input_dim = self.in_channels // self.groups
                    kernel_value = np.zeros(
                        (self.in_channels, input_dim, 3, 3), dtype=np.float32
                    )
                    for i in range(self.in_channels):
                        kernel_value[i, i % input_dim, 1, 1] = 1
                    self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
                kernel = self.id_tensor
                running_mean = branch.running_mean
                running_var = branch.running_var
                gamma = branch.weight
                beta = branch.bias
                eps = branch.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return kernel * t, beta - running_mean * gamma / std

        def repvgg_convert(self):
            kernel, bias = self.get_equivalent_kernel_bias()
            return (
                kernel.detach().cpu().numpy(),
                bias.detach().cpu().numpy(),
            )

        def fuse_conv_bn(self, conv, bn):

            std = (bn.running_var + bn.eps).sqrt()
            bias = bn.bias - bn.running_mean * bn.weight / std

            t = (bn.weight / std).reshape(-1, 1, 1, 1)
            weights = conv.weight * t

            bn = nn.Identity()
            conv = nn.Conv2d(in_channels=conv.in_channels,
                             out_channels=conv.out_channels,
                             kernel_size=conv.kernel_size,
                             stride=conv.stride,
                             padding=conv.padding,
                             dilation=conv.dilation,
                             groups=conv.groups,
                             bias=True,
                             padding_mode=conv.padding_mode)

            conv.weight = torch.nn.Parameter(weights)
            conv.bias = torch.nn.Parameter(bias)
            return conv

        def fuse_repvgg_block(self):
            if self.deploy:
                return
            print(f"RepConv.fuse_repvgg_block")

            self.rbr_dense = self.fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])

            self.rbr_1x1 = self.fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
            rbr_1x1_bias = self.rbr_1x1.bias
            weight_1x1_expanded = torch.nn.functional.pad(self.rbr_1x1.weight, [1, 1, 1, 1])

            # Fuse self.rbr_identity
            if (isinstance(self.rbr_identity, nn.BatchNorm2d) or isinstance(self.rbr_identity,
                                                                            nn.modules.batchnorm.SyncBatchNorm)):
                # print(f"fuse: rbr_identity == BatchNorm2d or SyncBatchNorm")
                identity_conv_1x1 = nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=self.groups,
                    bias=False)
                identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.to(self.rbr_1x1.weight.data.device)
                identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.squeeze().squeeze()
                # print(f" identity_conv_1x1.weight = {identity_conv_1x1.weight.shape}")
                identity_conv_1x1.weight.data.fill_(0.0)
                identity_conv_1x1.weight.data.fill_diagonal_(1.0)
                identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.unsqueeze(2).unsqueeze(3)
                # print(f" identity_conv_1x1.weight = {identity_conv_1x1.weight.shape}")

                identity_conv_1x1 = self.fuse_conv_bn(identity_conv_1x1, self.rbr_identity)
                bias_identity_expanded = identity_conv_1x1.bias
                weight_identity_expanded = torch.nn.functional.pad(identity_conv_1x1.weight, [1, 1, 1, 1])
            else:
                # print(f"fuse: rbr_identity != BatchNorm2d, rbr_identity = {self.rbr_identity}")
                bias_identity_expanded = torch.nn.Parameter(torch.zeros_like(rbr_1x1_bias))
                weight_identity_expanded = torch.nn.Parameter(torch.zeros_like(weight_1x1_expanded))

            # print(f"self.rbr_1x1.weight = {self.rbr_1x1.weight.shape}, ")
            # print(f"weight_1x1_expanded = {weight_1x1_expanded.shape}, ")
            # print(f"self.rbr_dense.weight = {self.rbr_dense.weight.shape}, ")

            self.rbr_dense.weight = torch.nn.Parameter(
                self.rbr_dense.weight + weight_1x1_expanded + weight_identity_expanded)
            self.rbr_dense.bias = torch.nn.Parameter(self.rbr_dense.bias + rbr_1x1_bias + bias_identity_expanded)

            self.rbr_reparam = self.rbr_dense
            self.deploy = True

            if self.rbr_identity is not None:
                del self.rbr_identity
                self.rbr_identity = None

            if self.rbr_1x1 is not None:
                del self.rbr_1x1
                self.rbr_1x1 = None

            if self.rbr_dense is not None:
                del self.rbr_dense
                self.rbr_dense = None
    class RepBottleneck(Bottleneck):
        # Standard bottleneck
        def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
            super().__init__(c1, c2, shortcut=True, g=1, e=0.5)
            c_ = int(c2 * e)  # hidden channels
            self.cv2 = RepConv(c_, c2, 3, 1, g=g)
    class RepBottleneckCSPA(BottleneckCSPA):
        # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
        def __init__(self, c1, c2, n=1, shortcut=True, g=1,
                     e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
            super().__init__(c1, c2, n, shortcut, g, e)
            c_ = int(c2 * e)  # hidden channels
            self.m = nn.Sequential(*[RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
    class RepBottleneckCSPB(BottleneckCSPB):
        # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
        def __init__(self, c1, c2, n=1, shortcut=False, g=1,
                     e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
            super().__init__(c1, c2, n, shortcut, g, e)
            c_ = int(c2)  # hidden channels
            self.m = nn.Sequential(*[RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
    class RepBottleneckCSPC(BottleneckCSPC):
        # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
        def __init__(self, c1, c2, n=1, shortcut=True, g=1,
                     e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
            super().__init__(c1, c2, n, shortcut, g, e)
            c_ = int(c2 * e)  # hidden channels
            self.m = nn.Sequential(*[RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
    class RepRes(Res):
        # Standard bottleneck
        def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
            super().__init__(c1, c2, shortcut, g, e)
            c_ = int(c2 * e)  # hidden channels
            self.cv2 = RepConv(c_, c_, 3, 1, g=g)
    class RepResCSPA(ResCSPA):
        # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
        def __init__(self, c1, c2, n=1, shortcut=True, g=1,
                     e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
            super().__init__(c1, c2, n, shortcut, g, e)
            c_ = int(c2 * e)  # hidden channels
            self.m = nn.Sequential(*[RepRes(c_, c_, shortcut, g, e=0.5) for _ in range(n)])
    class RepResCSPB(ResCSPB):
        # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
        def __init__(self, c1, c2, n=1, shortcut=False, g=1,
                     e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
            super().__init__(c1, c2, n, shortcut, g, e)
            c_ = int(c2)  # hidden channels
            self.m = nn.Sequential(*[RepRes(c_, c_, shortcut, g, e=0.5) for _ in range(n)])
    class RepResCSPC(ResCSPC):
        # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
        def __init__(self, c1, c2, n=1, shortcut=True, g=1,
                     e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
            super().__init__(c1, c2, n, shortcut, g, e)
            c_ = int(c2 * e)  # hidden channels
            self.m = nn.Sequential(*[RepRes(c_, c_, shortcut, g, e=0.5) for _ in range(n)])
    class RepResX(ResX):
        # Standard bottleneck
        def __init__(self, c1, c2, shortcut=True, g=32, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
            super().__init__(c1, c2, shortcut, g, e)
            c_ = int(c2 * e)  # hidden channels
            self.cv2 = RepConv(c_, c_, 3, 1, g=g)
    class RepResXCSPA(ResXCSPA):
        # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
        def __init__(self, c1, c2, n=1, shortcut=True, g=32,
                     e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
            super().__init__(c1, c2, n, shortcut, g, e)
            c_ = int(c2 * e)  # hidden channels
            self.m = nn.Sequential(*[RepResX(c_, c_, shortcut, g, e=0.5) for _ in range(n)])
    class RepResXCSPB(ResXCSPB):
        # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
        def __init__(self, c1, c2, n=1, shortcut=False, g=32,
                     e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
            super().__init__(c1, c2, n, shortcut, g, e)
            c_ = int(c2)  # hidden channels
            self.m = nn.Sequential(*[RepResX(c_, c_, shortcut, g, e=0.5) for _ in range(n)])
    class RepResXCSPC(ResXCSPC):
        # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
        def __init__(self, c1, c2, n=1, shortcut=True, g=32,
                     e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
            super().__init__(c1, c2, n, shortcut, g, e)
            c_ = int(c2 * e)  # hidden channels
            self.m = nn.Sequential(*[RepResX(c_, c_, shortcut, g, e=0.5) for _ in range(n)])
    class NMS(nn.Module):
        # Non-Maximum Suppression (NMS) module
        conf = 0.25  # confidence threshold
        iou = 0.45  # IoU threshold
        classes = None  # (optional list) filter by class

        def __init__(self):
            super(NMS, self).__init__()

        def forward(self, x):
            return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)
    class autoShape(nn.Module):
        # input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
        conf = 0.25  # NMS confidence threshold
        iou = 0.45  # NMS IoU threshold
        classes = None  # (optional list) filter by class

        def __init__(self, model):
            super(autoShape, self).__init__()
            self.model = model.eval()

        def autoshape(self):
            print('autoShape already enabled, skipping... ')  # model already converted to model.autoshape()
            return self

        @torch.no_grad()
        def forward(self, imgs, size=640, augment=False, profile=False):
            # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
            #   filename:   imgs = 'data/samples/zidane.jpg'
            #   URI:             = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg'
            #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
            #   PIL:             = Image.open('image.jpg')  # HWC x(640,1280,3)
            #   numpy:           = np.zeros((640,1280,3))  # HWC
            #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
            #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

            t = [time_synchronized()]
            p = next(self.model.parameters())  # for device and type
            if isinstance(imgs, torch.Tensor):  # torch
                with amp.autocast(enabled=p.device.type != 'cpu'):
                    return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

            # Pre-process
            n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
            shape0, shape1, files = [], [], []  # image and inference shapes, filenames
            for i, im in enumerate(imgs):
                f = f'image{i}'  # filename
                if isinstance(im, str):  # filename or uri
                    im, f = np.asarray(
                        Image.open(requests.get(im, stream=True).raw if im.startswith('http') else im)), im
                elif isinstance(im, Image.Image):  # PIL Image
                    im, f = np.asarray(im), getattr(im, 'filename', f) or f
                files.append(Path(f).with_suffix('.jpg').name)
                if im.shape[0] < 5:  # image in CHW
                    im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
                im = im[:, :, :3] if im.ndim == 3 else np.tile(im[:, :, None], 3)  # enforce 3ch input
                s = im.shape[:2]  # HWC
                shape0.append(s)  # image shape
                g = (size / max(s))  # gain
                shape1.append([y * g for y in s])
                imgs[i] = im  # update
            shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
            x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
            x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
            x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
            x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32
            t.append(time_synchronized())

            with amp.autocast(enabled=p.device.type != 'cpu'):
                # Inference
                y = self.model(x, augment, profile)[0]  # forward
                t.append(time_synchronized())

                # Post-process
                y = non_max_suppression(y, conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)  # NMS
                for i in range(n):
                    scale_coords(shape1, y[i][:, :4], shape0[i])

                t.append(time_synchronized())
                return Detections(imgs, y, files, t, self.names, x.shape)
    class Detections:
        # detections class for YOLOv5 inference results
        def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
            super(Detections, self).__init__()
            d = pred[0].device  # device
            gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in
                  imgs]  # normalizations
            self.imgs = imgs  # list of images as numpy arrays
            self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
            self.names = names  # class names
            self.files = files  # image filenames
            self.xyxy = pred  # xyxy pixels
            self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
            self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
            self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
            self.n = len(self.pred)  # number of images (batch size)
            self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
            self.s = shape  # inference BCHW shape

        def display(self, pprint=False, show=False, save=False, render=False, save_dir=''):
            colors = color_list()
            for i, (img, pred) in enumerate(zip(self.imgs, self.pred)):
                str = f'image {i + 1}/{len(self.pred)}: {img.shape[0]}x{img.shape[1]} '
                if pred is not None:
                    for c in pred[:, -1].unique():
                        n = (pred[:, -1] == c).sum()  # detections per class
                        str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    if show or save or render:
                        for *box, conf, cls in pred:  # xyxy, confidence, class
                            label = f'{self.names[int(cls)]} {conf:.2f}'
                            plot_one_box(box, img, label=label, color=colors[int(cls) % 10])
                img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img  # from np
                if pprint:
                    print(str.rstrip(', '))
                if show:
                    img.show(self.files[i])  # show
                if save:
                    f = self.files[i]
                    img.save(Path(save_dir) / f)  # save
                    print(f"{'Saved' * (i == 0)} {f}", end=',' if i < self.n - 1 else f' to {save_dir}\n')
                if render:
                    self.imgs[i] = np.asarray(img)

        def print(self):
            self.display(pprint=True)  # print results
            print(
                f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' % self.t)

        def show(self):
            self.display(show=True)  # show results

        def save(self, save_dir='runs/hub/exp'):
            save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp')  # increment save_dir
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            self.display(save=True, save_dir=save_dir)  # save results

        def render(self):
            self.display(render=True)  # render results
            return self.imgs

        def pandas(self):
            # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
            new = copy(self)  # return copy
            ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
            cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
            for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
                a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in
                     getattr(self, k)]  # update
                setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
            return new

        def tolist(self):
            # return a list of Detections objects, i.e. 'for result in results.tolist():'
            x = [Detections([self.imgs[i]], [self.pred[i]], self.names, self.s) for i in range(self.n)]
            for d in x:
                for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                    setattr(d, k, getattr(d, k)[0])  # pop out of list
            return x

        def __len__(self):
            return self.n
    class Classify(nn.Module):
        # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
        def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
            super(Classify, self).__init__()
            self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
            self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
            self.flat = nn.Flatten()

        def forward(self, x):
            z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
            return self.flat(self.conv(z))  # flatten to x(b,c2)
    def transI_fusebn(kernel, bn):
        gamma = bn.weight
        std = (bn.running_var + bn.eps).sqrt()
        return kernel * ((gamma / std).reshape(-1, 1, 1, 1)), bn.bias - bn.running_mean * gamma / std
    class ConvBN(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size,
                     stride=1, padding=0, dilation=1, groups=1, deploy=False, nonlinear=None):
            super().__init__()
            if nonlinear is None:
                self.nonlinear = nn.Identity()
            else:
                self.nonlinear = nonlinear
            if deploy:
                self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
            else:
                self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
                self.bn = nn.BatchNorm2d(num_features=out_channels)

        def forward(self, x):
            if hasattr(self, 'bn'):
                return self.nonlinear(self.bn(self.conv(x)))
            else:
                return self.nonlinear(self.conv(x))

        def switch_to_deploy(self):
            kernel, bias = transI_fusebn(self.conv.weight, self.bn)
            conv = nn.Conv2d(in_channels=self.conv.in_channels, out_channels=self.conv.out_channels,
                             kernel_size=self.conv.kernel_size,
                             stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation,
                             groups=self.conv.groups, bias=True)
            conv.weight.data = kernel
            conv.bias.data = bias
            for para in self.parameters():
                para.detach_()
            self.__delattr__('conv')
            self.__delattr__('bn')
            self.conv = conv
    class OREPA_3x3_RepConv(nn.Module):

        def __init__(self, in_channels, out_channels, kernel_size,
                     stride=1, padding=0, dilation=1, groups=1,
                     internal_channels_1x1_3x3=None,
                     deploy=False, nonlinear=None, single_init=False):
            super(OREPA_3x3_RepConv, self).__init__()
            self.deploy = deploy

            if nonlinear is None:
                self.nonlinear = nn.Identity()
            else:
                self.nonlinear = nonlinear

            self.kernel_size = kernel_size
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.groups = groups
            assert padding == kernel_size // 2

            self.stride = stride
            self.padding = padding
            self.dilation = dilation

            self.branch_counter = 0

            self.weight_rbr_origin = nn.Parameter(
                torch.Tensor(out_channels, int(in_channels / self.groups), kernel_size, kernel_size))
            nn.init.kaiming_uniform_(self.weight_rbr_origin, a=math.sqrt(1.0))
            self.branch_counter += 1

            if groups < out_channels:
                self.weight_rbr_avg_conv = nn.Parameter(
                    torch.Tensor(out_channels, int(in_channels / self.groups), 1, 1))
                self.weight_rbr_pfir_conv = nn.Parameter(
                    torch.Tensor(out_channels, int(in_channels / self.groups), 1, 1))
                nn.init.kaiming_uniform_(self.weight_rbr_avg_conv, a=1.0)
                nn.init.kaiming_uniform_(self.weight_rbr_pfir_conv, a=1.0)
                self.weight_rbr_avg_conv.data
                self.weight_rbr_pfir_conv.data
                self.register_buffer('weight_rbr_avg_avg',
                                     torch.ones(kernel_size, kernel_size).mul(1.0 / kernel_size / kernel_size))
                self.branch_counter += 1

            else:
                raise NotImplementedError
            self.branch_counter += 1

            if internal_channels_1x1_3x3 is None:
                internal_channels_1x1_3x3 = in_channels if groups < out_channels else 2 * in_channels  # For mobilenet, it is better to have 2X internal channels

            if internal_channels_1x1_3x3 == in_channels:
                self.weight_rbr_1x1_kxk_idconv1 = nn.Parameter(
                    torch.zeros(in_channels, int(in_channels / self.groups), 1, 1))
                id_value = np.zeros((in_channels, int(in_channels / self.groups), 1, 1))
                for i in range(in_channels):
                    id_value[i, i % int(in_channels / self.groups), 0, 0] = 1
                id_tensor = torch.from_numpy(id_value).type_as(self.weight_rbr_1x1_kxk_idconv1)
                self.register_buffer('id_tensor', id_tensor)

            else:
                self.weight_rbr_1x1_kxk_conv1 = nn.Parameter(
                    torch.Tensor(internal_channels_1x1_3x3, int(in_channels / self.groups), 1, 1))
                nn.init.kaiming_uniform_(self.weight_rbr_1x1_kxk_conv1, a=math.sqrt(1.0))
            self.weight_rbr_1x1_kxk_conv2 = nn.Parameter(
                torch.Tensor(out_channels, int(internal_channels_1x1_3x3 / self.groups), kernel_size, kernel_size))
            nn.init.kaiming_uniform_(self.weight_rbr_1x1_kxk_conv2, a=math.sqrt(1.0))
            self.branch_counter += 1

            expand_ratio = 8
            self.weight_rbr_gconv_dw = nn.Parameter(
                torch.Tensor(in_channels * expand_ratio, 1, kernel_size, kernel_size))
            self.weight_rbr_gconv_pw = nn.Parameter(torch.Tensor(out_channels, in_channels * expand_ratio, 1, 1))
            nn.init.kaiming_uniform_(self.weight_rbr_gconv_dw, a=math.sqrt(1.0))
            nn.init.kaiming_uniform_(self.weight_rbr_gconv_pw, a=math.sqrt(1.0))
            self.branch_counter += 1

            if out_channels == in_channels and stride == 1:
                self.branch_counter += 1

            self.vector = nn.Parameter(torch.Tensor(self.branch_counter, self.out_channels))
            self.bn = nn.BatchNorm2d(out_channels)

            self.fre_init()

            nn.init.constant_(self.vector[0, :], 0.25)  # origin
            nn.init.constant_(self.vector[1, :], 0.25)  # avg
            nn.init.constant_(self.vector[2, :], 0.0)  # prior
            nn.init.constant_(self.vector[3, :], 0.5)  # 1x1_kxk
            nn.init.constant_(self.vector[4, :], 0.5)  # dws_conv

        def fre_init(self):
            prior_tensor = torch.Tensor(self.out_channels, self.kernel_size, self.kernel_size)
            half_fg = self.out_channels / 2
            for i in range(self.out_channels):
                for h in range(3):
                    for w in range(3):
                        if i < half_fg:
                            prior_tensor[i, h, w] = math.cos(math.pi * (h + 0.5) * (i + 1) / 3)
                        else:
                            prior_tensor[i, h, w] = math.cos(math.pi * (w + 0.5) * (i + 1 - half_fg) / 3)

            self.register_buffer('weight_rbr_prior', prior_tensor)

        def weight_gen(self):

            weight_rbr_origin = torch.einsum('oihw,o->oihw', self.weight_rbr_origin, self.vector[0, :])

            weight_rbr_avg = torch.einsum('oihw,o->oihw', torch.einsum('oihw,hw->oihw', self.weight_rbr_avg_conv,
                                                                       self.weight_rbr_avg_avg), self.vector[1, :])

            weight_rbr_pfir = torch.einsum('oihw,o->oihw', torch.einsum('oihw,ohw->oihw', self.weight_rbr_pfir_conv,
                                                                        self.weight_rbr_prior), self.vector[2, :])

            weight_rbr_1x1_kxk_conv1 = None
            if hasattr(self, 'weight_rbr_1x1_kxk_idconv1'):
                weight_rbr_1x1_kxk_conv1 = (self.weight_rbr_1x1_kxk_idconv1 + self.id_tensor).squeeze()
            elif hasattr(self, 'weight_rbr_1x1_kxk_conv1'):
                weight_rbr_1x1_kxk_conv1 = self.weight_rbr_1x1_kxk_conv1.squeeze()
            else:
                raise NotImplementedError
            weight_rbr_1x1_kxk_conv2 = self.weight_rbr_1x1_kxk_conv2

            if self.groups > 1:
                g = self.groups
                t, ig = weight_rbr_1x1_kxk_conv1.size()
                o, tg, h, w = weight_rbr_1x1_kxk_conv2.size()
                weight_rbr_1x1_kxk_conv1 = weight_rbr_1x1_kxk_conv1.view(g, int(t / g), ig)
                weight_rbr_1x1_kxk_conv2 = weight_rbr_1x1_kxk_conv2.view(g, int(o / g), tg, h, w)
                weight_rbr_1x1_kxk = torch.einsum('gti,gothw->goihw', weight_rbr_1x1_kxk_conv1,
                                                  weight_rbr_1x1_kxk_conv2).view(o, ig, h, w)
            else:
                weight_rbr_1x1_kxk = torch.einsum('ti,othw->oihw', weight_rbr_1x1_kxk_conv1, weight_rbr_1x1_kxk_conv2)

            weight_rbr_1x1_kxk = torch.einsum('oihw,o->oihw', weight_rbr_1x1_kxk, self.vector[3, :])

            weight_rbr_gconv = self.dwsc2full(self.weight_rbr_gconv_dw, self.weight_rbr_gconv_pw, self.in_channels)
            weight_rbr_gconv = torch.einsum('oihw,o->oihw', weight_rbr_gconv, self.vector[4, :])

            weight = weight_rbr_origin + weight_rbr_avg + weight_rbr_1x1_kxk + weight_rbr_pfir + weight_rbr_gconv

            return weight

        def dwsc2full(self, weight_dw, weight_pw, groups):

            t, ig, h, w = weight_dw.size()
            o, _, _, _ = weight_pw.size()
            tg = int(t / groups)
            i = int(ig * groups)
            weight_dw = weight_dw.view(groups, tg, ig, h, w)
            weight_pw = weight_pw.squeeze().view(o, groups, tg)

            weight_dsc = torch.einsum('gtihw,ogt->ogihw', weight_dw, weight_pw)
            return weight_dsc.view(o, i, h, w)

        def forward(self, inputs):
            weight = self.weight_gen()
            out = F.conv2d(inputs, weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation,
                           groups=self.groups)

            return self.nonlinear(self.bn(out))
    class RepConv_OREPA(nn.Module):

        def __init__(self, c1, c2, k=3, s=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False,
                     use_se=False, nonlinear=nn.SiLU()):
            super(RepConv_OREPA, self).__init__()
            self.deploy = deploy
            self.groups = groups
            self.in_channels = c1
            self.out_channels = c2

            self.padding = padding
            self.dilation = dilation
            self.groups = groups

            assert k == 3
            assert padding == 1

            padding_11 = padding - k // 2

            if nonlinear is None:
                self.nonlinearity = nn.Identity()
            else:
                self.nonlinearity = nonlinear

            if use_se:
                self.se = SEBlock(self.out_channels, internal_neurons=self.out_channels // 16)
            else:
                self.se = nn.Identity()

            if deploy:
                self.rbr_reparam = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                             kernel_size=k, stride=s,
                                             padding=padding, dilation=dilation, groups=groups, bias=True,
                                             padding_mode=padding_mode)

            else:
                self.rbr_identity = nn.BatchNorm2d(
                    num_features=self.in_channels) if self.out_channels == self.in_channels and s == 1 else None
                self.rbr_dense = OREPA_3x3_RepConv(in_channels=self.in_channels, out_channels=self.out_channels,
                                                   kernel_size=k, stride=s, padding=padding, groups=groups, dilation=1)
                self.rbr_1x1 = ConvBN(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1,
                                      stride=s, padding=padding_11, groups=groups, dilation=1)
                print('RepVGG Block, identity = ', self.rbr_identity)

        def forward(self, inputs):
            if hasattr(self, 'rbr_reparam'):
                return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

            if self.rbr_identity is None:
                id_out = 0
            else:
                id_out = self.rbr_identity(inputs)

            out1 = self.rbr_dense(inputs)
            out2 = self.rbr_1x1(inputs)
            out3 = id_out
            out = out1 + out2 + out3

            return self.nonlinearity(self.se(out))

        #   Optional. This improves the accuracy and facilitates quantization.
        #   1.  Cancel the original weight decay on rbr_dense.conv.weight and rbr_1x1.conv.weight.
        #   2.  Use like this.
        #       loss = criterion(....)
        #       for every RepVGGBlock blk:
        #           loss += weight_decay_coefficient * 0.5 * blk.get_cust_L2()
        #       optimizer.zero_grad()
        #       loss.backward()

        # Not used for OREPA
        def get_custom_L2(self):
            K3 = self.rbr_dense.weight_gen()
            K1 = self.rbr_1x1.conv.weight
            t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(
                -1, 1, 1, 1).detach()
            t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt())).reshape(-1, 1,
                                                                                                                 1,
                                                                                                                 1).detach()

            l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2,
                                                1:2] ** 2).sum()  # The L2 loss of the "circle" of weights in 3x3 kernel. Use regular L2 on them.
            eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1  # The equivalent resultant central point of 3x3 kernel.
            l2_loss_eq_kernel = (eq_kernel ** 2 / (
                    t3 ** 2 + t1 ** 2)).sum()  # Normalize for an L2 coefficient comparable to regular L2.
            return l2_loss_eq_kernel + l2_loss_circle

        def get_equivalent_kernel_bias(self):
            kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
            kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
            kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
            return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

        def _pad_1x1_to_3x3_tensor(self, kernel1x1):
            if kernel1x1 is None:
                return 0
            else:
                return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

        def _fuse_bn_tensor(self, branch):
            if branch is None:
                return 0, 0
            if not isinstance(branch, nn.BatchNorm2d):
                if isinstance(branch, OREPA_3x3_RepConv):
                    kernel = branch.weight_gen()
                elif isinstance(branch, ConvBN):
                    kernel = branch.conv.weight
                else:
                    raise NotImplementedError
                running_mean = branch.bn.running_mean
                running_var = branch.bn.running_var
                gamma = branch.bn.weight
                beta = branch.bn.bias
                eps = branch.bn.eps
            else:
                if not hasattr(self, 'id_tensor'):
                    input_dim = self.in_channels // self.groups
                    kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                    for i in range(self.in_channels):
                        kernel_value[i, i % input_dim, 1, 1] = 1
                    self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
                kernel = self.id_tensor
                running_mean = branch.running_mean
                running_var = branch.running_var
                gamma = branch.weight
                beta = branch.bias
                eps = branch.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return kernel * t, beta - running_mean * gamma / std

        def switch_to_deploy(self):
            if hasattr(self, 'rbr_reparam'):
                return
            print(f"RepConv_OREPA.switch_to_deploy")
            kernel, bias = self.get_equivalent_kernel_bias()
            self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.in_channels,
                                         out_channels=self.rbr_dense.out_channels,
                                         kernel_size=self.rbr_dense.kernel_size, stride=self.rbr_dense.stride,
                                         padding=self.rbr_dense.padding, dilation=self.rbr_dense.dilation,
                                         groups=self.rbr_dense.groups, bias=True)
            self.rbr_reparam.weight.data = kernel
            self.rbr_reparam.bias.data = bias
            for para in self.parameters():
                para.detach_()
            self.__delattr__('rbr_dense')
            self.__delattr__('rbr_1x1')
            if hasattr(self, 'rbr_identity'):
                self.__delattr__('rbr_identity')
    class WindowAttention(nn.Module):

        def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

            super().__init__()
            self.dim = dim
            self.window_size = window_size  # Wh, Ww
            self.num_heads = num_heads
            head_dim = dim // num_heads
            self.scale = qk_scale or head_dim ** -0.5

            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.attn_drop = nn.Dropout(attn_drop)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)

            nn.init.normal_(self.relative_position_bias_table, std=.02)
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, x, mask=None):

            B_, N, C = x.shape
            qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))

            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

            if mask is not None:
                nW = mask.shape[0]
                attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)
                attn = self.softmax(attn)
            else:
                attn = self.softmax(attn)

            attn = self.attn_drop(attn)

            # print(attn.dtype, v.dtype)
            try:
                x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
            except:
                # print(attn.dtype, v.dtype)
                x = (attn.half() @ v).transpose(1, 2).reshape(B_, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x
    class Mlp(nn.Module):

        def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.):
            super().__init__()
            out_features = out_features or in_features
            hidden_features = hidden_features or in_features
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.act = act_layer()
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.drop = nn.Dropout(drop)

        def forward(self, x):
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            return x
    def window_partition(x, window_size):

        B, H, W, C = x.shape
        assert H % window_size == 0, 'feature map h and w can not divide by window size'
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows
    def window_reverse(windows, window_size, H, W):

        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x
    class SwinTransformerLayer(nn.Module):

        def __init__(self, dim, num_heads, window_size=8, shift_size=0,
                     mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                     act_layer=nn.SiLU, norm_layer=nn.LayerNorm):
            super().__init__()
            self.dim = dim
            self.num_heads = num_heads
            self.window_size = window_size
            self.shift_size = shift_size
            self.mlp_ratio = mlp_ratio
            # if min(self.input_resolution) <= self.window_size:
            #     # if window size is larger than input resolution, we don't partition windows
            #     self.shift_size = 0
            #     self.window_size = min(self.input_resolution)
            assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

            self.norm1 = norm_layer(dim)
            self.attn = WindowAttention(
                dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        def create_mask(self, H, W):
            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

            return attn_mask

        def forward(self, x):
            # reshape x[b c h w] to x[b l c]
            _, _, H_, W_ = x.shape

            Padding = False
            if min(H_, W_) < self.window_size or H_ % self.window_size != 0 or W_ % self.window_size != 0:
                Padding = True
                # print(f'img_size {min(H_, W_)} is less than (or not divided by) window_size {self.window_size}, Padding.')
                pad_r = (self.window_size - W_ % self.window_size) % self.window_size
                pad_b = (self.window_size - H_ % self.window_size) % self.window_size
                x = F.pad(x, (0, pad_r, 0, pad_b))

            # print('2', x.shape)
            B, C, H, W = x.shape
            L = H * W
            x = x.permute(0, 2, 3, 1).contiguous().view(B, L, C)  # b, L, c

            # create mask from init to forward
            if self.shift_size > 0:
                attn_mask = self.create_mask(H, W).to(x.device)
            else:
                attn_mask = None

            shortcut = x
            x = self.norm1(x)
            x = x.view(B, H, W, C)

            # cyclic shift
            if self.shift_size > 0:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            else:
                shifted_x = x

            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

            # W-MSA/SW-MSA
            attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

            # merge windows
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

            # reverse cyclic shift
            if self.shift_size > 0:
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = shifted_x
            x = x.view(B, H * W, C)

            # FFN
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))

            x = x.permute(0, 2, 1).contiguous().view(-1, C, H, W)  # b c h w

            if Padding:
                x = x[:, :, :H_, :W_]  # reverse padding

            return x
    class SwinTransformerBlock(nn.Module):
        def __init__(self, c1, c2, num_heads, num_layers, window_size=8):
            super().__init__()
            self.conv = None
            if c1 != c2:
                self.conv = Conv(c1, c2)

            # remove input_resolution
            self.blocks = nn.Sequential(*[SwinTransformerLayer(dim=c2, num_heads=num_heads, window_size=window_size,
                                                               shift_size=0 if (i % 2 == 0) else window_size // 2) for i
                                          in range(num_layers)])

        def forward(self, x):
            if self.conv is not None:
                x = self.conv(x)
            x = self.blocks(x)
            return x
    class STCSPA(nn.Module):
        # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
        def __init__(self, c1, c2, n=1, shortcut=True, g=1,
                     e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
            super(STCSPA, self).__init__()
            c_ = int(c2 * e)  # hidden channels
            self.cv1 = Conv(c1, c_, 1, 1)
            self.cv2 = Conv(c1, c_, 1, 1)
            self.cv3 = Conv(2 * c_, c2, 1, 1)
            num_heads = c_ // 32
            self.m = SwinTransformerBlock(c_, c_, num_heads, n)
            # self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

        def forward(self, x):
            y1 = self.m(self.cv1(x))
            y2 = self.cv2(x)
            return self.cv3(torch.cat((y1, y2), dim=1))
    class STCSPB(nn.Module):
        # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
        def __init__(self, c1, c2, n=1, shortcut=False, g=1,
                     e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
            super(STCSPB, self).__init__()
            c_ = int(c2)  # hidden channels
            self.cv1 = Conv(c1, c_, 1, 1)
            self.cv2 = Conv(c_, c_, 1, 1)
            self.cv3 = Conv(2 * c_, c2, 1, 1)
            num_heads = c_ // 32
            self.m = SwinTransformerBlock(c_, c_, num_heads, n)
            # self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

        def forward(self, x):
            x1 = self.cv1(x)
            y1 = self.m(x1)
            y2 = self.cv2(x1)
            return self.cv3(torch.cat((y1, y2), dim=1))
    class STCSPC(nn.Module):
        # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
        def __init__(self, c1, c2, n=1, shortcut=True, g=1,
                     e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
            super(STCSPC, self).__init__()
            c_ = int(c2 * e)  # hidden channels
            self.cv1 = Conv(c1, c_, 1, 1)
            self.cv2 = Conv(c1, c_, 1, 1)
            self.cv3 = Conv(c_, c_, 1, 1)
            self.cv4 = Conv(2 * c_, c2, 1, 1)
            num_heads = c_ // 32
            self.m = SwinTransformerBlock(c_, c_, num_heads, n)
            # self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

        def forward(self, x):
            y1 = self.cv3(self.m(self.cv1(x)))
            y2 = self.cv2(x)
            return self.cv4(torch.cat((y1, y2), dim=1))
    class WindowAttention_v2(nn.Module):

        def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                     pretrained_window_size=[0, 0]):

            super().__init__()
            self.dim = dim
            self.window_size = window_size  # Wh, Ww
            self.pretrained_window_size = pretrained_window_size
            self.num_heads = num_heads

            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

            # mlp to generate continuous relative position bias
            self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(512, num_heads, bias=False))

            # get relative_coords_table
            relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
            relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
            relative_coords_table = torch.stack(
                torch.meshgrid([relative_coords_h,
                                relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
            if pretrained_window_size[0] > 0:
                relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
                relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
            else:
                relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
                relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
            relative_coords_table *= 8  # normalize to -8, 8
            relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
                torch.abs(relative_coords_table) + 1.0) / np.log2(8)

            self.register_buffer("relative_coords_table", relative_coords_table)

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            self.qkv = nn.Linear(dim, dim * 3, bias=False)
            if qkv_bias:
                self.q_bias = nn.Parameter(torch.zeros(dim))
                self.v_bias = nn.Parameter(torch.zeros(dim))
            else:
                self.q_bias = None
                self.v_bias = None
            self.attn_drop = nn.Dropout(attn_drop)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, x, mask=None):

            B_, N, C = x.shape
            qkv_bias = None
            if self.q_bias is not None:
                qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

            # cosine attention
            attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
            logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01))).exp()
            attn = attn * logit_scale

            relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
            relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
            attn = attn + relative_position_bias.unsqueeze(0)

            if mask is not None:
                nW = mask.shape[0]
                attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)
                attn = self.softmax(attn)
            else:
                attn = self.softmax(attn)

            attn = self.attn_drop(attn)

            try:
                x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
            except:
                x = (attn.half() @ v).transpose(1, 2).reshape(B_, N, C)

            x = self.proj(x)
            x = self.proj_drop(x)
            return x

        def extra_repr(self) -> str:
            return f'dim={self.dim}, window_size={self.window_size}, ' \
                   f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'

        def flops(self, N):
            # calculate flops for 1 window with token length of N
            flops = 0
            # qkv = self.qkv(x)
            flops += N * self.dim * 3 * self.dim
            # attn = (q @ k.transpose(-2, -1))
            flops += self.num_heads * N * (self.dim // self.num_heads) * N
            #  x = (attn @ v)
            flops += self.num_heads * N * N * (self.dim // self.num_heads)
            # x = self.proj(x)
            flops += N * self.dim * self.dim
            return flops
    class Mlp_v2(nn.Module):
        def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.):
            super().__init__()
            out_features = out_features or in_features
            hidden_features = hidden_features or in_features
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.act = act_layer()
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.drop = nn.Dropout(drop)

        def forward(self, x):
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            return x
    def window_partition_v2(x, window_size):

        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows
    def window_reverse_v2(windows, window_size, H, W):

        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x
    class SwinTransformerLayer_v2(nn.Module):

        def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                     mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                     act_layer=nn.SiLU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
            super().__init__()
            self.dim = dim
            # self.input_resolution = input_resolution
            self.num_heads = num_heads
            self.window_size = window_size
            self.shift_size = shift_size
            self.mlp_ratio = mlp_ratio
            # if min(self.input_resolution) <= self.window_size:
            #    # if window size is larger than input resolution, we don't partition windows
            #    self.shift_size = 0
            #    self.window_size = min(self.input_resolution)
            assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

            self.norm1 = norm_layer(dim)
            self.attn = WindowAttention_v2(
                dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
                qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                pretrained_window_size=(pretrained_window_size, pretrained_window_size))

            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp_v2(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        def create_mask(self, H, W):
            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

            return attn_mask

        def forward(self, x):
            # reshape x[b c h w] to x[b l c]
            _, _, H_, W_ = x.shape

            Padding = False
            if min(H_, W_) < self.window_size or H_ % self.window_size != 0 or W_ % self.window_size != 0:
                Padding = True
                # print(f'img_size {min(H_, W_)} is less than (or not divided by) window_size {self.window_size}, Padding.')
                pad_r = (self.window_size - W_ % self.window_size) % self.window_size
                pad_b = (self.window_size - H_ % self.window_size) % self.window_size
                x = F.pad(x, (0, pad_r, 0, pad_b))

            # print('2', x.shape)
            B, C, H, W = x.shape
            L = H * W
            x = x.permute(0, 2, 3, 1).contiguous().view(B, L, C)  # b, L, c

            # create mask from init to forward
            if self.shift_size > 0:
                attn_mask = self.create_mask(H, W).to(x.device)
            else:
                attn_mask = None

            shortcut = x
            x = x.view(B, H, W, C)

            # cyclic shift
            if self.shift_size > 0:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            else:
                shifted_x = x

            # partition windows
            x_windows = window_partition_v2(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

            # W-MSA/SW-MSA
            attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

            # merge windows
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
            shifted_x = window_reverse_v2(attn_windows, self.window_size, H, W)  # B H' W' C

            # reverse cyclic shift
            if self.shift_size > 0:
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = shifted_x
            x = x.view(B, H * W, C)
            x = shortcut + self.drop_path(self.norm1(x))

            # FFN
            x = x + self.drop_path(self.norm2(self.mlp(x)))
            x = x.permute(0, 2, 1).contiguous().view(-1, C, H, W)  # b c h w

            if Padding:
                x = x[:, :, :H_, :W_]  # reverse padding

            return x

        def extra_repr(self) -> str:
            return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
                   f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

        def flops(self):
            flops = 0
            H, W = self.input_resolution
            # norm1
            flops += self.dim * H * W
            # W-MSA/SW-MSA
            nW = H * W / self.window_size / self.window_size
            flops += nW * self.attn.flops(self.window_size * self.window_size)
            # mlp
            flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
            # norm2
            flops += self.dim * H * W
            return flops
    class SwinTransformer2Block(nn.Module):
        def __init__(self, c1, c2, num_heads, num_layers, window_size=7):
            super().__init__()
            self.conv = None
            if c1 != c2:
                self.conv = Conv(c1, c2)

            # remove input_resolution
            self.blocks = nn.Sequential(*[SwinTransformerLayer_v2(dim=c2, num_heads=num_heads, window_size=window_size,
                                                                  shift_size=0 if (i % 2 == 0) else window_size // 2)
                                          for i in range(num_layers)])

        def forward(self, x):
            if self.conv is not None:
                x = self.conv(x)
            x = self.blocks(x)
            return x
    class ST2CSPA(nn.Module):
        # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
        def __init__(self, c1, c2, n=1, shortcut=True, g=1,
                     e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
            super(ST2CSPA, self).__init__()
            c_ = int(c2 * e)  # hidden channels
            self.cv1 = Conv(c1, c_, 1, 1)
            self.cv2 = Conv(c1, c_, 1, 1)
            self.cv3 = Conv(2 * c_, c2, 1, 1)
            num_heads = c_ // 32
            self.m = SwinTransformer2Block(c_, c_, num_heads, n)
            # self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

        def forward(self, x):
            y1 = self.m(self.cv1(x))
            y2 = self.cv2(x)
            return self.cv3(torch.cat((y1, y2), dim=1))
    class ST2CSPB(nn.Module):
        # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
        def __init__(self, c1, c2, n=1, shortcut=False, g=1,
                     e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
            super(ST2CSPB, self).__init__()
            c_ = int(c2)  # hidden channels
            self.cv1 = Conv(c1, c_, 1, 1)
            self.cv2 = Conv(c_, c_, 1, 1)
            self.cv3 = Conv(2 * c_, c2, 1, 1)
            num_heads = c_ // 32
            self.m = SwinTransformer2Block(c_, c_, num_heads, n)
            # self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

        def forward(self, x):
            x1 = self.cv1(x)
            y1 = self.m(x1)
            y2 = self.cv2(x1)
            return self.cv3(torch.cat((y1, y2), dim=1))
    class ST2CSPC(nn.Module):
        # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
        def __init__(self, c1, c2, n=1, shortcut=True, g=1,
                     e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
            super(ST2CSPC, self).__init__()
            c_ = int(c2 * e)  # hidden channels
            self.cv1 = Conv(c1, c_, 1, 1)
            self.cv2 = Conv(c1, c_, 1, 1)
            self.cv3 = Conv(c_, c_, 1, 1)
            self.cv4 = Conv(2 * c_, c2, 1, 1)
            num_heads = c_ // 32
            self.m = SwinTransformer2Block(c_, c_, num_heads, n)
            # self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

        def forward(self, x):
            y1 = self.cv3(self.m(self.cv1(x)))
            y2 = self.cv2(x)
            return self.cv4(torch.cat((y1, y2), dim=1))
### --- YOLOv7 Code --- ###

### --- YOLOv8 Code --- ###
if "yolov8" in yolo_name or "rtdetr" in yolo_name or "yolov10" in yolo_name:
    from ultralytics.utils import ARM64, IS_JETSON, IS_RASPBERRYPI, LINUX, LOGGER, ROOT, yaml_load
    from ultralytics.utils.checks import check_requirements, check_suffix, check_version, check_yaml
    from ultralytics.utils.downloads import attempt_download_asset, is_url
    def check_class_names(names):
        """
        Check class names.

        Map imagenet class codes to human-readable names if required. Convert lists to dicts.
        """
        if isinstance(names, list):  # names is a list
            names = dict(enumerate(names))  # convert to dict
        if isinstance(names, dict):
            # Convert 1) string keys to int, i.e. '0' to 0, and non-string values to strings, i.e. True to 'True'
            names = {int(k): str(v) for k, v in names.items()}
            n = len(names)
            if max(names.keys()) >= n:
                raise KeyError(
                    f"{n}-class dataset requires class indices 0-{n - 1}, but you have invalid class indices "
                    f"{min(names.keys())}-{max(names.keys())} defined in your dataset YAML."
                )
            if isinstance(names[0], str) and names[0].startswith("n0"):  # imagenet class codes, i.e. 'n01440764'
                names_map = yaml_load(ROOT / "cfg/datasets/ImageNet.yaml")["map"]  # human-readable names
                names = {k: names_map[v] for k, v in names.items()}
        return names
    def default_class_names(data=None):
        """Applies default class names to an input YAML file or returns numerical class names."""
        if data:
            with contextlib.suppress(Exception):
                return yaml_load(check_yaml(data))["names"]
        return {i: f"class{i}" for i in range(999)}  # return default if above errors
    class AutoBackend(nn.Module):
        """
        Handles dynamic backend selection for running inference using Ultralytics YOLO models.

        The AutoBackend class is designed to provide an abstraction layer for various inference engines. It supports a wide
        range of formats, each with specific naming conventions as outlined below:

            Supported Formats and Naming Conventions:
                | Format                | File Suffix      |
                |-----------------------|------------------|
                | PyTorch               | *.pt             |
                | TorchScript           | *.torchscript    |
                | ONNX Runtime          | *.onnx           |
                | ONNX OpenCV DNN       | *.onnx (dnn=True)|
                | OpenVINO              | *openvino_model/ |
                | CoreML                | *.mlpackage      |
                | TensorRT              | *.engine         |
                | TensorFlow SavedModel | *_saved_model    |
                | TensorFlow GraphDef   | *.pb             |
                | TensorFlow Lite       | *.tflite         |
                | TensorFlow Edge TPU   | *_edgetpu.tflite |
                | PaddlePaddle          | *_paddle_model   |
                | NCNN                  | *_ncnn_model     |

        This class offers dynamic backend switching capabilities based on the input model format, making it easier to deploy
        models across various platforms.
        """

        @torch.no_grad()
        def __init__(
                self,
                weights="yolov8n.pt",
                device=torch.device("cpu"),
                dnn=False,
                data=None,
                fp16=False,
                batch=1,
                fuse=True,
                verbose=True,
        ):
            """
            Initialize the AutoBackend for inference.

            Args:
                weights (str): Path to the model weights file. Defaults to 'yolov8n.pt'.
                device (torch.device): Device to run the model on. Defaults to CPU.
                dnn (bool): Use OpenCV DNN module for ONNX inference. Defaults to False.
                data (str | Path | optional): Path to the additional data.yaml file containing class names. Optional.
                fp16 (bool): Enable half-precision inference. Supported only on specific backends. Defaults to False.
                batch (int): Batch-size to assume for inference.
                fuse (bool): Fuse Conv2D + BatchNorm layers for optimization. Defaults to True.
                verbose (bool): Enable verbose logging. Defaults to True.
            """
            super().__init__()
            w = str(weights[0] if isinstance(weights, list) else weights)
            nn_module = isinstance(weights, torch.nn.Module)
            (
                pt,
                jit,
                onnx,
                xml,
                engine,
                coreml,
                saved_model,
                pb,
                tflite,
                edgetpu,
                tfjs,
                paddle,
                ncnn,
                triton,
            ) = self._model_type(w)
            fp16 &= pt or jit or onnx or xml or engine or nn_module or triton  # FP16
            nhwc = coreml or saved_model or pb or tflite or edgetpu  # BHWC formats (vs torch BCWH)
            stride = 32  # default stride
            model, metadata = None, None

            # Set device
            cuda = torch.cuda.is_available() and device.type != "cpu"  # use CUDA
            if cuda and not any([nn_module, pt, jit, engine, onnx]):  # GPU dataloader formats
                device = torch.device("cpu")
                cuda = False

            # Download if not local
            if not (pt or triton or nn_module):
                w = attempt_download_asset(w)

            # In-memory PyTorch model
            if nn_module:
                model = weights.to(device)
                if fuse:
                    model = model.fuse(verbose=verbose)
                if hasattr(model, "kpt_shape"):
                    kpt_shape = model.kpt_shape  # pose-only
                stride = max(int(model.stride.max()), 32)  # model stride
                names = model.module.names if hasattr(model, "module") else model.names  # get class names
                model.half() if fp16 else model.float()
                self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
                pt = True

            # PyTorch
            elif pt:
                from ultralytics.nn.tasks import attempt_load_weights

                model = attempt_load_weights(
                    weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse
                )
                if hasattr(model, "kpt_shape"):
                    kpt_shape = model.kpt_shape  # pose-only
                stride = max(int(model.stride.max()), 32)  # model stride
                names = model.module.names if hasattr(model, "module") else model.names  # get class names
                model.half() if fp16 else model.float()
                self.model = model  # explicitly assign for to(), cpu(), cuda(), half()

            # TorchScript
            elif jit:
                LOGGER.info(f"Loading {w} for TorchScript inference...")
                extra_files = {"config.txt": ""}  # model metadata
                model = torch.jit.load(w, _extra_files=extra_files, map_location=device)
                model.half() if fp16 else model.float()
                if extra_files["config.txt"]:  # load metadata dict
                    metadata = json.loads(extra_files["config.txt"], object_hook=lambda x: dict(x.items()))

            # ONNX OpenCV DNN
            elif dnn:
                LOGGER.info(f"Loading {w} for ONNX OpenCV DNN inference...")
                check_requirements("opencv-python>=4.5.4")
                net = cv2.dnn.readNetFromONNX(w)

            # ONNX Runtime
            elif onnx:
                LOGGER.info(f"Loading {w} for ONNX Runtime inference...")
                check_requirements(("onnx", "onnxruntime-gpu" if cuda else "onnxruntime"))
                if IS_RASPBERRYPI or IS_JETSON:
                    # Fix 'numpy.linalg._umath_linalg' has no attribute '_ilp64' for TF SavedModel on RPi and Jetson
                    check_requirements("numpy==1.23.5")
                import onnxruntime

                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]
                session = onnxruntime.InferenceSession(w, providers=providers)
                output_names = [x.name for x in session.get_outputs()]
                metadata = session.get_modelmeta().custom_metadata_map

            # OpenVINO
            elif xml:
                LOGGER.info(f"Loading {w} for OpenVINO inference...")
                check_requirements("openvino>=2024.0.0")
                import openvino as ov

                core = ov.Core()
                w = Path(w)
                if not w.is_file():  # if not *.xml
                    w = next(w.glob("*.xml"))  # get *.xml file from *_openvino_model dir
                ov_model = core.read_model(model=str(w), weights=w.with_suffix(".bin"))
                if ov_model.get_parameters()[0].get_layout().empty:
                    ov_model.get_parameters()[0].set_layout(ov.Layout("NCHW"))

                # OpenVINO inference modes are 'LATENCY', 'THROUGHPUT' (not recommended), or 'CUMULATIVE_THROUGHPUT'
                inference_mode = "CUMULATIVE_THROUGHPUT" if batch > 1 else "LATENCY"
                LOGGER.info(f"Using OpenVINO {inference_mode} mode for batch={batch} inference...")
                ov_compiled_model = core.compile_model(
                    ov_model,
                    device_name="AUTO",  # AUTO selects best available device, do not modify
                    config={"PERFORMANCE_HINT": inference_mode},
                )
                input_name = ov_compiled_model.input().get_any_name()
                metadata = w.parent / "metadata.yaml"

            # TensorRT
            elif engine:
                LOGGER.info(f"Loading {w} for TensorRT inference...")
                try:
                    import tensorrt as trt  # noqa https://developer.nvidia.com/nvidia-tensorrt-download
                except ImportError:
                    if LINUX:
                        check_requirements("tensorrt>7.0.0,<=10.1.0")
                    import tensorrt as trt  # noqa
                check_version(trt.__version__, ">=7.0.0", hard=True)
                check_version(trt.__version__, "<=10.1.0", msg="https://github.com/ultralytics/ultralytics/pull/14239")
                if device.type == "cpu":
                    device = torch.device("cuda:0")
                Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
                logger = trt.Logger(trt.Logger.INFO)
                # Read file
                with open(w, "rb") as f, trt.Runtime(logger) as runtime:
                    try:
                        meta_len = int.from_bytes(f.read(4), byteorder="little")  # read metadata length
                        metadata = json.loads(f.read(meta_len).decode("utf-8"))  # read metadata
                    except UnicodeDecodeError:
                        f.seek(0)  # engine file may lack embedded Ultralytics metadata
                    model = runtime.deserialize_cuda_engine(f.read())  # read engine

                # Model context
                try:
                    context = model.create_execution_context()
                except Exception as e:  # model is None
                    LOGGER.error(f"ERROR: TensorRT model exported with a different version than {trt.__version__}\n")
                    raise e

                bindings = OrderedDict()
                output_names = []
                fp16 = False  # default updated below
                dynamic = False
                is_trt10 = not hasattr(model, "num_bindings")
                num = range(model.num_io_tensors) if is_trt10 else range(model.num_bindings)
                for i in num:
                    if is_trt10:
                        name = model.get_tensor_name(i)
                        dtype = trt.nptype(model.get_tensor_dtype(name))
                        is_input = model.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                        if is_input:
                            if -1 in tuple(model.get_tensor_shape(name)):
                                dynamic = True
                                context.set_input_shape(name, tuple(model.get_tensor_profile_shape(name, 0)[1]))
                                if dtype == np.float16:
                                    fp16 = True
                        else:
                            output_names.append(name)
                        shape = tuple(context.get_tensor_shape(name))
                    else:  # TensorRT < 10.0
                        name = model.get_binding_name(i)
                        dtype = trt.nptype(model.get_binding_dtype(i))
                        is_input = model.binding_is_input(i)
                        if model.binding_is_input(i):
                            if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                                dynamic = True
                                context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[1]))
                            if dtype == np.float16:
                                fp16 = True
                        else:
                            output_names.append(name)
                        shape = tuple(context.get_binding_shape(i))
                    im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                    bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
                binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
                batch_size = bindings["images"].shape[0]  # if dynamic, this is instead max batch size

            # CoreML
            elif coreml:
                LOGGER.info(f"Loading {w} for CoreML inference...")
                import coremltools as ct

                model = ct.models.MLModel(w)
                metadata = dict(model.user_defined_metadata)

            # TF SavedModel
            elif saved_model:
                LOGGER.info(f"Loading {w} for TensorFlow SavedModel inference...")
                import tensorflow as tf

                keras = False  # assume TF1 saved_model
                model = tf.keras.models.load_model(w) if keras else tf.saved_model.load(w)
                metadata = Path(w) / "metadata.yaml"

            # TF GraphDef
            elif pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
                LOGGER.info(f"Loading {w} for TensorFlow GraphDef inference...")
                import tensorflow as tf

                from ultralytics.engine.exporter import gd_outputs

                def wrap_frozen_graph(gd, inputs, outputs):
                    """Wrap frozen graphs for deployment."""
                    x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped
                    ge = x.graph.as_graph_element
                    return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.map_structure(ge, outputs))

                gd = tf.Graph().as_graph_def()  # TF GraphDef
                with open(w, "rb") as f:
                    gd.ParseFromString(f.read())
                frozen_func = wrap_frozen_graph(gd, inputs="x:0", outputs=gd_outputs(gd))
                with contextlib.suppress(StopIteration):  # find metadata in SavedModel alongside GraphDef
                    metadata = next(Path(w).resolve().parent.rglob(f"{Path(w).stem}_saved_model*/metadata.yaml"))

            # TFLite or TFLite Edge TPU
            elif tflite or edgetpu:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
                try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
                    from tflite_runtime.interpreter import Interpreter, load_delegate
                except ImportError:
                    import tensorflow as tf

                    Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate
                if edgetpu:  # TF Edge TPU https://coral.ai/software/#edgetpu-runtime
                    LOGGER.info(f"Loading {w} for TensorFlow Lite Edge TPU inference...")
                    delegate = {"Linux": "libedgetpu.so.1", "Darwin": "libedgetpu.1.dylib", "Windows": "edgetpu.dll"}[
                        platform.system()
                    ]
                    interpreter = Interpreter(model_path=w, experimental_delegates=[load_delegate(delegate)])
                else:  # TFLite
                    LOGGER.info(f"Loading {w} for TensorFlow Lite inference...")
                    interpreter = Interpreter(model_path=w)  # load TFLite model
                interpreter.allocate_tensors()  # allocate
                input_details = interpreter.get_input_details()  # inputs
                output_details = interpreter.get_output_details()  # outputs
                # Load metadata
                with contextlib.suppress(zipfile.BadZipFile):
                    with zipfile.ZipFile(w, "r") as model:
                        meta_file = model.namelist()[0]
                        metadata = ast.literal_eval(model.read(meta_file).decode("utf-8"))

            # TF.js
            elif tfjs:
                raise NotImplementedError("YOLOv8 TF.js inference is not currently supported.")

            # PaddlePaddle
            elif paddle:
                LOGGER.info(f"Loading {w} for PaddlePaddle inference...")
                check_requirements("paddlepaddle-gpu" if cuda else "paddlepaddle")
                import paddle.inference as pdi  # noqa

                w = Path(w)
                if not w.is_file():  # if not *.pdmodel
                    w = next(w.rglob("*.pdmodel"))  # get *.pdmodel file from *_paddle_model dir
                config = pdi.Config(str(w), str(w.with_suffix(".pdiparams")))
                if cuda:
                    config.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=0)
                predictor = pdi.create_predictor(config)
                input_handle = predictor.get_input_handle(predictor.get_input_names()[0])
                output_names = predictor.get_output_names()
                metadata = w.parents[1] / "metadata.yaml"

            # NCNN
            elif ncnn:
                LOGGER.info(f"Loading {w} for NCNN inference...")
                check_requirements("git+https://github.com/Tencent/ncnn.git" if ARM64 else "ncnn")  # requires NCNN
                import ncnn as pyncnn

                net = pyncnn.Net()
                net.opt.use_vulkan_compute = cuda
                w = Path(w)
                if not w.is_file():  # if not *.param
                    w = next(w.glob("*.param"))  # get *.param file from *_ncnn_model dir
                net.load_param(str(w))
                net.load_model(str(w.with_suffix(".bin")))
                metadata = w.parent / "metadata.yaml"

            # NVIDIA Triton Inference Server
            elif triton:
                check_requirements("tritonclient[all]")
                from ultralytics.utils.triton import TritonRemoteModel

                model = TritonRemoteModel(w)

            # Any other format (unsupported)
            else:
                from ultralytics.engine.exporter import export_formats

                raise TypeError(
                    f"model='{w}' is not a supported model format. Ultralytics supports: {export_formats()['Format']}\n"
                    f"See https://docs.ultralytics.com/modes/predict for help."
                )

            # Load external metadata YAML
            if isinstance(metadata, (str, Path)) and Path(metadata).exists():
                metadata = yaml_load(metadata)
            if metadata and isinstance(metadata, dict):
                for k, v in metadata.items():
                    if k in {"stride", "batch"}:
                        metadata[k] = int(v)
                    elif k in {"imgsz", "names", "kpt_shape"} and isinstance(v, str):
                        metadata[k] = eval(v)
                stride = metadata["stride"]
                task = metadata["task"]
                batch = metadata["batch"]
                imgsz = metadata["imgsz"]
                names = metadata["names"]
                kpt_shape = metadata.get("kpt_shape")
            elif not (pt or triton or nn_module):
                LOGGER.warning(f"WARNING ⚠️ Metadata not found for 'model={weights}'")

            # Check names
            if "names" not in locals():  # names missing
                names = default_class_names(data)
            names = check_class_names(names)

            # Disable gradients
            if pt:
                for p in model.parameters():
                    p.requires_grad = False

            self.__dict__.update(locals())  # assign all variables to self

        def forward(self, im, augment=False, visualize=False, embed=None):
            """
            Runs inference on the YOLOv8 MultiBackend model.

            Args:
                im (torch.Tensor): The image tensor to perform inference on.
                augment (bool): whether to perform data augmentation during inference, defaults to False
                visualize (bool): whether to visualize the output predictions, defaults to False
                embed (list, optional): A list of feature vectors/embeddings to return.

            Returns:
                (tuple): Tuple containing the raw output tensor, and processed output for visualization (if visualize=True)
            """
            b, ch, h, w = im.shape  # batch, channel, height, width
            if self.fp16 and im.dtype != torch.float16:
                im = im.half()  # to FP16
            if self.nhwc:
                im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)

            # PyTorch
            if self.pt or self.nn_module:
                y = self.model(im, augment=augment, visualize=visualize, embed=embed)

            # TorchScript
            elif self.jit:
                y = self.model(im)

            # ONNX OpenCV DNN
            elif self.dnn:
                im = im.cpu().numpy()  # torch to numpy
                self.net.setInput(im)
                y = self.net.forward()

            # ONNX Runtime
            elif self.onnx:
                im = im.cpu().numpy()  # torch to numpy
                y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})

            # OpenVINO
            elif self.xml:
                im = im.cpu().numpy()  # FP32

                if self.inference_mode in {"THROUGHPUT", "CUMULATIVE_THROUGHPUT"}:  # optimized for larger batch-sizes
                    n = im.shape[0]  # number of images in batch
                    results = [None] * n  # preallocate list with None to match the number of images

                    def callback(request, userdata):
                        """Places result in preallocated list using userdata index."""
                        results[userdata] = request.results

                    # Create AsyncInferQueue, set the callback and start asynchronous inference for each input image
                    async_queue = self.ov.runtime.AsyncInferQueue(self.ov_compiled_model)
                    async_queue.set_callback(callback)
                    for i in range(n):
                        # Start async inference with userdata=i to specify the position in results list
                        async_queue.start_async(inputs={self.input_name: im[i: i + 1]},
                                                userdata=i)  # keep image as BCHW
                    async_queue.wait_all()  # wait for all inference requests to complete
                    y = np.concatenate([list(r.values())[0] for r in results])

                else:  # inference_mode = "LATENCY", optimized for fastest first result at batch-size 1
                    y = list(self.ov_compiled_model(im).values())

            # TensorRT
            elif self.engine:
                if self.dynamic or im.shape != self.bindings["images"].shape:
                    if self.is_trt10:
                        self.context.set_input_shape("images", im.shape)
                        self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)
                        for name in self.output_names:
                            self.bindings[name].data.resize_(tuple(self.context.get_tensor_shape(name)))
                    else:
                        i = self.model.get_binding_index("images")
                        self.context.set_binding_shape(i, im.shape)
                        self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)
                        for name in self.output_names:
                            i = self.model.get_binding_index(name)
                            self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))

                s = self.bindings["images"].shape
                assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
                self.binding_addrs["images"] = int(im.data_ptr())
                self.context.execute_v2(list(self.binding_addrs.values()))
                y = [self.bindings[x].data for x in sorted(self.output_names)]

            # CoreML
            elif self.coreml:
                im = im[0].cpu().numpy()
                im_pil = Image.fromarray((im * 255).astype("uint8"))
                # im = im.resize((192, 320), Image.BILINEAR)
                y = self.model.predict({"image": im_pil})  # coordinates are xywh normalized
                if "confidence" in y:
                    raise TypeError(
                        "Ultralytics only supports inference of non-pipelined CoreML models exported with "
                        f"'nms=False', but 'model={w}' has an NMS pipeline created by an 'nms=True' export."
                    )
                    # TODO: CoreML NMS inference handling
                    # from ultralytics.utils.ops import xywh2xyxy
                    # box = xywh2xyxy(y['coordinates'] * [[w, h, w, h]])  # xyxy pixels
                    # conf, cls = y['confidence'].max(1), y['confidence'].argmax(1).astype(np.float32)
                    # y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
                elif len(y) == 1:  # classification model
                    y = list(y.values())
                elif len(y) == 2:  # segmentation model
                    y = list(reversed(y.values()))  # reversed for segmentation models (pred, proto)

            # PaddlePaddle
            elif self.paddle:
                im = im.cpu().numpy().astype(np.float32)
                self.input_handle.copy_from_cpu(im)
                self.predictor.run()
                y = [self.predictor.get_output_handle(x).copy_to_cpu() for x in self.output_names]

            # NCNN
            elif self.ncnn:
                mat_in = self.pyncnn.Mat(im[0].cpu().numpy())
                with self.net.create_extractor() as ex:
                    ex.input(self.net.input_names()[0], mat_in)
                    # WARNING: 'output_names' sorted as a temporary fix for https://github.com/pnnx/pnnx/issues/130
                    y = [np.array(ex.extract(x)[1])[None] for x in sorted(self.net.output_names())]

            # NVIDIA Triton Inference Server
            elif self.triton:
                im = im.cpu().numpy()  # torch to numpy
                y = self.model(im)

            # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
            else:
                im = im.cpu().numpy()
                if self.saved_model:  # SavedModel
                    y = self.model(im, training=False) if self.keras else self.model(im)
                    if not isinstance(y, list):
                        y = [y]
                elif self.pb:  # GraphDef
                    y = self.frozen_func(x=self.tf.constant(im))
                else:  # Lite or Edge TPU
                    details = self.input_details[0]
                    is_int = details["dtype"] in {np.int8, np.int16}  # is TFLite quantized int8 or int16 model
                    if is_int:
                        scale, zero_point = details["quantization"]
                        im = (im / scale + zero_point).astype(details["dtype"])  # de-scale
                    self.interpreter.set_tensor(details["index"], im)
                    self.interpreter.invoke()
                    y = []
                    for output in self.output_details:
                        x = self.interpreter.get_tensor(output["index"])
                        if is_int:
                            scale, zero_point = output["quantization"]
                            x = (x.astype(np.float32) - zero_point) * scale  # re-scale
                        if x.ndim == 3:  # if task is not classification, excluding masks (ndim=4) as well
                            # Denormalize xywh by image size. See https://github.com/ultralytics/ultralytics/pull/1695
                            # xywh are normalized in TFLite/EdgeTPU to mitigate quantization error of integer models
                            if x.shape[-1] == 6:  # end-to-end model
                                x[:, :, [0, 2]] *= w
                                x[:, :, [1, 3]] *= h
                            else:
                                x[:, [0, 2]] *= w
                                x[:, [1, 3]] *= h
                        y.append(x)
                # TF segment fixes: export is reversed vs ONNX export and protos are transposed
                if len(y) == 2:  # segment with (det, proto) output order reversed
                    if len(y[1].shape) != 4:
                        y = list(reversed(y))  # should be y = (1, 116, 8400), (1, 160, 160, 32)
                    if y[1].shape[-1] == 6:  # end-to-end model
                        y = [y[1]]
                    else:
                        y[1] = np.transpose(y[1], (0, 3, 1, 2))  # should be y = (1, 116, 8400), (1, 32, 160, 160)
                y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]

            # for x in y:
            #     print(type(x), len(x)) if isinstance(x, (list, tuple)) else print(type(x), x.shape)  # debug shapes
            if isinstance(y, (list, tuple)):
                if len(self.names) == 999 and (self.task == "segment" or len(y) == 2):  # segments and names not defined
                    ip, ib = (0, 1) if len(y[0].shape) == 4 else (1, 0)  # index of protos, boxes
                    nc = y[ib].shape[1] - y[ip].shape[3] - 4  # y = (1, 160, 160, 32), (1, 116, 8400)
                    self.names = {i: f"class{i}" for i in range(nc)}
                return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
            else:
                return self.from_numpy(y)

        def from_numpy(self, x):
            """
            Convert a numpy array to a tensor.

            Args:
                x (np.ndarray): The array to be converted.

            Returns:
                (torch.Tensor): The converted tensor
            """
            return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x

        def warmup(self, imgsz=(1, 3, 640, 640)):
            """
            Warm up the model by running one forward pass with a dummy input.

            Args:
                imgsz (tuple): The shape of the dummy input tensor in the format (batch_size, channels, height, width)
            """
            import torchvision  # noqa (import here so torchvision import time not recorded in postprocess time)

            warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton, self.nn_module
            if any(warmup_types) and (self.device.type != "cpu" or self.triton):
                im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
                for _ in range(2 if self.jit else 1):
                    self.forward(im)  # warmup

        @staticmethod
        def _model_type(p="path/to/model.pt"):
            """
            Takes a path to a model file and returns the model type. Possibles types are pt, jit, onnx, xml, engine, coreml,
            saved_model, pb, tflite, edgetpu, tfjs, ncnn or paddle.

            Args:
                p: path to the model file. Defaults to path/to/model.pt

            Examples:
                >>> model = AutoBackend(weights="path/to/model.onnx")
                >>> model_type = model._model_type()  # returns "onnx"
            """
            from ultralytics.engine.exporter import export_formats

            sf = export_formats()["Suffix"]  # export suffixes
            if not is_url(p) and not isinstance(p, str):
                check_suffix(p, sf)  # checks
            name = Path(p).name
            types = [s in name for s in sf]
            types[5] |= name.endswith(".mlmodel")  # retain support for older Apple CoreML *.mlmodel formats
            types[8] &= not types[9]  # tflite &= not edgetpu
            if any(types):
                triton = False
            else:
                from urllib.parse import urlsplit

                url = urlsplit(p)
                triton = bool(url.netloc) and bool(url.path) and url.scheme in {"http", "grpc"}

            return types + [triton]
### --- YOLOv8 Code --- ###

### --- YOLOv9 Code --- ###
if "yolov9" in yolo_name:
    from yolocode.yolov9.utils import TryExcept
    from yolocode.yolov9.utils.dataloaders import exif_transpose, letterbox
    from yolocode.yolov9.utils.general import (LOGGER, ROOT, Profile, check_requirements, check_suffix, check_version,
                                               colorstr,
                                               increment_path, is_notebook, make_divisible, non_max_suppression,
                                               scale_boxes,
                                               xywh2xyxy, xyxy2xywh, yaml_load)
    from yolocode.yolov9.utils.plots import Annotator, colors, save_one_box
    from yolocode.yolov9.utils.torch_utils import copy_attr, smart_inference_mode

    class AConv(nn.Module):
        def __init__(self, c1, c2):  # ch_in, ch_out, shortcut, kernels, groups, expand
            super().__init__()
            self.cv1 = Conv(c1, c2, 3, 2, 1)

        def forward(self, x):
            x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
            return self.cv1(x)
    class ADown(nn.Module):
        def __init__(self, c1, c2):  # ch_in, ch_out, shortcut, kernels, groups, expand
            super().__init__()
            self.c = c2 // 2
            self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
            self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

        def forward(self, x):
            x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
            x1, x2 = x.chunk(2, 1)
            x1 = self.cv1(x1)
            x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
            x2 = self.cv2(x2)
            return torch.cat((x1, x2), 1)
    class RepConvN(nn.Module):
        """RepConv is a basic rep-style block, including training and deploy status
        This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
        """
        default_act = nn.SiLU()  # default activation

        def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
            super().__init__()
            assert k == 3 and p == 1
            self.g = g
            self.c1 = c1
            self.c2 = c2
            self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

            self.bn = None
            self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
            self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

        def forward_fuse(self, x):
            """Forward process"""
            return self.act(self.conv(x))

        def forward(self, x):
            """Forward process"""
            id_out = 0 if self.bn is None else self.bn(x)
            return self.act(self.conv1(x) + self.conv2(x) + id_out)

        def get_equivalent_kernel_bias(self):
            kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
            kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
            kernelid, biasid = self._fuse_bn_tensor(self.bn)
            return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

        def _avg_to_3x3_tensor(self, avgp):
            channels = self.c1
            groups = self.g
            kernel_size = avgp.kernel_size
            input_dim = channels // groups
            k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
            k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
            return k

        def _pad_1x1_to_3x3_tensor(self, kernel1x1):
            if kernel1x1 is None:
                return 0
            else:
                return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

        def _fuse_bn_tensor(self, branch):
            if branch is None:
                return 0, 0
            if isinstance(branch, Conv):
                kernel = branch.conv.weight
                running_mean = branch.bn.running_mean
                running_var = branch.bn.running_var
                gamma = branch.bn.weight
                beta = branch.bn.bias
                eps = branch.bn.eps
            elif isinstance(branch, nn.BatchNorm2d):
                if not hasattr(self, 'id_tensor'):
                    input_dim = self.c1 // self.g
                    kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                    for i in range(self.c1):
                        kernel_value[i, i % input_dim, 1, 1] = 1
                    self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
                kernel = self.id_tensor
                running_mean = branch.running_mean
                running_var = branch.running_var
                gamma = branch.weight
                beta = branch.bias
                eps = branch.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return kernel * t, beta - running_mean * gamma / std

        def fuse_convs(self):
            if hasattr(self, 'conv'):
                return
            kernel, bias = self.get_equivalent_kernel_bias()
            self.conv = nn.Conv2d(in_channels=self.conv1.conv.in_channels,
                                  out_channels=self.conv1.conv.out_channels,
                                  kernel_size=self.conv1.conv.kernel_size,
                                  stride=self.conv1.conv.stride,
                                  padding=self.conv1.conv.padding,
                                  dilation=self.conv1.conv.dilation,
                                  groups=self.conv1.conv.groups,
                                  bias=True).requires_grad_(False)
            self.conv.weight.data = kernel
            self.conv.bias.data = bias
            for para in self.parameters():
                para.detach_()
            self.__delattr__('conv1')
            self.__delattr__('conv2')
            if hasattr(self, 'nm'):
                self.__delattr__('nm')
            if hasattr(self, 'bn'):
                self.__delattr__('bn')
            if hasattr(self, 'id_tensor'):
                self.__delattr__('id_tensor')
    class SP(nn.Module):
        def __init__(self, k=3, s=1):
            super(SP, self).__init__()
            self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=k // 2)

        def forward(self, x):
            return self.m(x)
    class MP(nn.Module):
        # Max pooling
        def __init__(self, k=2):
            super(MP, self).__init__()
            self.m = nn.MaxPool2d(kernel_size=k, stride=k)

        def forward(self, x):
            return self.m(x)
    class ConvTranspose(nn.Module):
        # Convolution transpose 2d layer
        default_act = nn.SiLU()  # default activation

        def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
            super().__init__()
            self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
            self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
            self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        def forward(self, x):
            return self.act(self.bn(self.conv_transpose(x)))
    class DFL(nn.Module):
        # DFL module
        def __init__(self, c1=17):
            super().__init__()
            self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
            self.conv.weight.data[:] = nn.Parameter(torch.arange(c1, dtype=torch.float).view(1, c1, 1, 1))  # / 120.0
            self.c1 = c1
            # self.bn = nn.BatchNorm2d(4)

        def forward(self, x):
            b, c, a = x.shape  # batch, channels, anchors
            return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
            # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)
    class BottleneckBase(nn.Module):
        # Standard bottleneck
        def __init__(self, c1, c2, shortcut=True, g=1, k=(1, 3),
                     e=0.5):  # ch_in, ch_out, shortcut, kernels, groups, expand
            super().__init__()
            c_ = int(c2 * e)  # hidden channels
            self.cv1 = Conv(c1, c_, k[0], 1)
            self.cv2 = Conv(c_, c2, k[1], 1, g=g)
            self.add = shortcut and c1 == c2

        def forward(self, x):
            return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    class RBottleneckBase(nn.Module):
        # Standard bottleneck
        def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 1),
                     e=0.5):  # ch_in, ch_out, shortcut, kernels, groups, expand
            super().__init__()
            c_ = int(c2 * e)  # hidden channels
            self.cv1 = Conv(c1, c_, k[0], 1)
            self.cv2 = Conv(c_, c2, k[1], 1, g=g)
            self.add = shortcut and c1 == c2

        def forward(self, x):
            return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    class RepNRBottleneckBase(nn.Module):
        # Standard bottleneck
        def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 1),
                     e=0.5):  # ch_in, ch_out, shortcut, kernels, groups, expand
            super().__init__()
            c_ = int(c2 * e)  # hidden channels
            self.cv1 = RepConvN(c1, c_, k[0], 1)
            self.cv2 = Conv(c_, c2, k[1], 1, g=g)
            self.add = shortcut and c1 == c2

        def forward(self, x):
            return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    class RepNBottleneck(nn.Module):
        # Standard bottleneck
        def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3),
                     e=0.5):  # ch_in, ch_out, shortcut, kernels, groups, expand
            super().__init__()
            c_ = int(c2 * e)  # hidden channels
            self.cv1 = RepConvN(c1, c_, k[0], 1)
            self.cv2 = Conv(c_, c2, k[1], 1, g=g)
            self.add = shortcut and c1 == c2

        def forward(self, x):
            return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    class Res(nn.Module):
        # ResNet bottleneck
        def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
            super(Res, self).__init__()
            c_ = int(c2 * e)  # hidden channels
            self.cv1 = Conv(c1, c_, 1, 1)
            self.cv2 = Conv(c_, c_, 3, 1, g=g)
            self.cv3 = Conv(c_, c2, 1, 1)
            self.add = shortcut and c1 == c2

        def forward(self, x):
            return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))
    class RepNRes(nn.Module):
        # ResNet bottleneck
        def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
            super(RepNRes, self).__init__()
            c_ = int(c2 * e)  # hidden channels
            self.cv1 = Conv(c1, c_, 1, 1)
            self.cv2 = RepConvN(c_, c_, 3, 1, g=g)
            self.cv3 = Conv(c_, c2, 1, 1)
            self.add = shortcut and c1 == c2

        def forward(self, x):
            return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))

    class CSP(nn.Module):
        # CSP Bottleneck with 3 convolutions
        def __init__(self, c1, c2, n=1, shortcut=True, g=1,
                     e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
            super().__init__()
            c_ = int(c2 * e)  # hidden channels
            self.cv1 = Conv(c1, c_, 1, 1)
            self.cv2 = Conv(c1, c_, 1, 1)
            self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
            self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

        def forward(self, x):
            return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
    class RepNCSP(nn.Module):
        # CSP Bottleneck with 3 convolutions
        def __init__(self, c1, c2, n=1, shortcut=True, g=1,
                     e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
            super().__init__()
            c_ = int(c2 * e)  # hidden channels
            self.cv1 = Conv(c1, c_, 1, 1)
            self.cv2 = Conv(c1, c_, 1, 1)
            self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
            self.m = nn.Sequential(*(RepNBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

        def forward(self, x):
            return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
    class CSPBase(nn.Module):
        # CSP Bottleneck with 3 convolutions
        def __init__(self, c1, c2, n=1, shortcut=True, g=1,
                     e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
            super().__init__()
            c_ = int(c2 * e)  # hidden channels
            self.cv1 = Conv(c1, c_, 1, 1)
            self.cv2 = Conv(c1, c_, 1, 1)
            self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
            self.m = nn.Sequential(*(BottleneckBase(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

        def forward(self, x):
            return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
    class ASPP(torch.nn.Module):

        def __init__(self, in_channels, out_channels):
            super().__init__()
            kernel_sizes = [1, 3, 3, 1]
            dilations = [1, 3, 6, 1]
            paddings = [0, 3, 6, 0]
            self.aspp = torch.nn.ModuleList()
            for aspp_idx in range(len(kernel_sizes)):
                conv = torch.nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_sizes[aspp_idx],
                    stride=1,
                    dilation=dilations[aspp_idx],
                    padding=paddings[aspp_idx],
                    bias=True)
                self.aspp.append(conv)
            self.gap = torch.nn.AdaptiveAvgPool2d(1)
            self.aspp_num = len(kernel_sizes)
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    m.bias.data.fill_(0)

        def forward(self, x):
            avg_x = self.gap(x)
            out = []
            for aspp_idx in range(self.aspp_num):
                inp = avg_x if (aspp_idx == self.aspp_num - 1) else x
                out.append(F.relu_(self.aspp[aspp_idx](inp)))
            out[-1] = out[-1].expand_as(out[-2])
            out = torch.cat(out, dim=1)
            return out
    class SPPCSPC(nn.Module):
        # CSP SPP https://github.com/WongKinYiu/CrossStagePartialNetworks
        def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
            super(SPPCSPC, self).__init__()
            c_ = int(2 * c2 * e)  # hidden channels
            self.cv1 = Conv(c1, c_, 1, 1)
            self.cv2 = Conv(c1, c_, 1, 1)
            self.cv3 = Conv(c_, c_, 3, 1)
            self.cv4 = Conv(c_, c_, 1, 1)
            self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
            self.cv5 = Conv(4 * c_, c_, 1, 1)
            self.cv6 = Conv(c_, c_, 3, 1)
            self.cv7 = Conv(2 * c_, c2, 1, 1)

        def forward(self, x):
            x1 = self.cv4(self.cv3(self.cv1(x)))
            y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
            y2 = self.cv2(x)
            return self.cv7(torch.cat((y1, y2), dim=1))
    class ReOrg(nn.Module):
        # yolo
        def __init__(self):
            super(ReOrg, self).__init__()

        def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
            return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
    class Shortcut(nn.Module):
        def __init__(self, dimension=0):
            super(Shortcut, self).__init__()
            self.d = dimension

        def forward(self, x):
            return x[0] + x[1]
    class Silence(nn.Module):
        def __init__(self):
            super(Silence, self).__init__()

        def forward(self, x):
            return x
    class SPPELAN(nn.Module):
        # spp-elan
        def __init__(self, c1, c2, c3):  # ch_in, ch_out, number, shortcut, groups, expansion
            super().__init__()
            self.c = c3
            self.cv1 = Conv(c1, c3, 1, 1)
            self.cv2 = SP(5)
            self.cv3 = SP(5)
            self.cv4 = SP(5)
            self.cv5 = Conv(4 * c3, c2, 1, 1)

        def forward(self, x):
            y = [self.cv1(x)]
            y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
            return self.cv5(torch.cat(y, 1))
    class RepNCSPELAN4(nn.Module):
        # csp-elan
        def __init__(self, c1, c2, c3, c4, c5=1):  # ch_in, ch_out, number, shortcut, groups, expansion
            super().__init__()
            self.c = c3 // 2
            self.cv1 = Conv(c1, c3, 1, 1)
            self.cv2 = nn.Sequential(RepNCSP(c3 // 2, c4, c5), Conv(c4, c4, 3, 1))
            self.cv3 = nn.Sequential(RepNCSP(c4, c4, c5), Conv(c4, c4, 3, 1))
            self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

        def forward(self, x):
            y = list(self.cv1(x).chunk(2, 1))
            y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
            return self.cv4(torch.cat(y, 1))

        def forward_split(self, x):
            y = list(self.cv1(x).split((self.c, self.c), 1))
            y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
            return self.cv4(torch.cat(y, 1))
    class CBLinear(nn.Module):
        def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):  # ch_in, ch_outs, kernel, stride, padding, groups
            super(CBLinear, self).__init__()
            self.c2s = c2s
            self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

        def forward(self, x):
            outs = self.conv(x).split(self.c2s, dim=1)
            return outs
    class CBFuse(nn.Module):
        def __init__(self, idx):
            super(CBFuse, self).__init__()
            self.idx = idx

        def forward(self, xs):
            target_size = xs[-1].shape[2:]
            res = [F.interpolate(x[self.idx[i]], size=target_size, mode='nearest') for i, x in enumerate(xs[:-1])]
            out = torch.sum(torch.stack(res + xs[-1:]), dim=0)
            return out
    class DetectMultiBackend_YOLOv9(nn.Module):
        # YOLO MultiBackend class for python inference on various backends
        def __init__(self, weights='yolo.pt', device=torch.device('cpu'), dnn=False, data=None, fp16=False, fuse=True):
            # Usage:
            #   PyTorch:              weights = *.pt
            #   TorchScript:                    *.torchscript
            #   ONNX Runtime:                   *.onnx
            #   ONNX OpenCV DNN:                *.onnx --dnn
            #   OpenVINO:                       *_openvino_model
            #   CoreML:                         *.mlmodel
            #   TensorRT:                       *.engine
            #   TensorFlow SavedModel:          *_saved_model
            #   TensorFlow GraphDef:            *.pb
            #   TensorFlow Lite:                *.tflite
            #   TensorFlow Edge TPU:            *_edgetpu.tflite
            #   PaddlePaddle:                   *_paddle_model
            from models.experimental import attempt_download_YOLOV9, \
                attempt_load_YOLOV9  # scoped to avoid circular import

            super().__init__()
            w = str(weights[0] if isinstance(weights, list) else weights)
            pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, triton = self._model_type(
                w)
            fp16 &= pt or jit or onnx or engine  # FP16
            nhwc = coreml or saved_model or pb or tflite or edgetpu  # BHWC formats (vs torch BCWH)
            stride = 32  # default stride
            cuda = torch.cuda.is_available() and device.type != 'cpu'  # use CUDA
            if not (pt or triton):
                w = attempt_download_YOLOV9(w)  # download if not local

            if pt:  # PyTorch
                model = attempt_load_YOLOV9(weights if isinstance(weights, list) else w, device=device, inplace=True,
                                            fuse=fuse)
                stride = max(int(model.stride.max()), 32)  # model stride
                names = model.module.names if hasattr(model, 'module') else model.names  # get class names
                model.half() if fp16 else model.float()
                self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
            elif jit:  # TorchScript
                LOGGER.info(f'Loading {w} for TorchScript inference...')
                extra_files = {'config.txt': ''}  # model metadata
                model = torch.jit.load(w, _extra_files=extra_files, map_location=device)
                model.half() if fp16 else model.float()
                if extra_files['config.txt']:  # load metadata dict
                    d = json.loads(extra_files['config.txt'],
                                   object_hook=lambda d: {int(k) if k.isdigit() else k: v
                                                          for k, v in d.items()})
                    stride, names = int(d['stride']), d['names']
            elif dnn:  # ONNX OpenCV DNN
                LOGGER.info(f'Loading {w} for ONNX OpenCV DNN inference...')
                check_requirements('opencv-python>=4.5.4')
                net = cv2.dnn.readNetFromONNX(w)
            elif onnx:  # ONNX Runtime
                LOGGER.info(f'Loading {w} for ONNX Runtime inference...')
                check_requirements(('onnx', 'onnxruntime-gpu' if cuda else 'onnxruntime'))
                import onnxruntime
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
                session = onnxruntime.InferenceSession(w, providers=providers)
                output_names = [x.name for x in session.get_outputs()]
                meta = session.get_modelmeta().custom_metadata_map  # metadata
                if 'stride' in meta:
                    stride, names = int(meta['stride']), eval(meta['names'])
            elif xml:  # OpenVINO
                LOGGER.info(f'Loading {w} for OpenVINO inference...')
                check_requirements('openvino')  # requires openvino-dev: https://pypi.org/project/openvino-dev/
                from openvino.runtime import Core, Layout, get_batch
                ie = Core()
                if not Path(w).is_file():  # if not *.xml
                    w = next(Path(w).glob('*.xml'))  # get *.xml file from *_openvino_model dir
                network = ie.read_model(model=w, weights=Path(w).with_suffix('.bin'))
                if network.get_parameters()[0].get_layout().empty:
                    network.get_parameters()[0].set_layout(Layout("NCHW"))
                batch_dim = get_batch(network)
                if batch_dim.is_static:
                    batch_size = batch_dim.get_length()
                executable_network = ie.compile_model(network, device_name="CPU")  # device_name="MYRIAD" for Intel NCS2
                stride, names = self._load_metadata(Path(w).with_suffix('.yaml'))  # load metadata
            elif engine:  # TensorRT
                LOGGER.info(f'Loading {w} for TensorRT inference...')
                import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
                check_version(trt.__version__, '7.0.0', hard=True)  # require tensorrt>=7.0.0
                if device.type == 'cpu':
                    device = torch.device('cuda:0')
                Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
                logger = trt.Logger(trt.Logger.INFO)
                with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
                    model = runtime.deserialize_cuda_engine(f.read())
                context = model.create_execution_context()
                bindings = OrderedDict()
                output_names = []
                fp16 = False  # default updated below
                dynamic = False
                for i in range(model.num_bindings):
                    name = model.get_binding_name(i)
                    dtype = trt.nptype(model.get_binding_dtype(i))
                    if model.binding_is_input(i):
                        if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                            dynamic = True
                            context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
                        if dtype == np.float16:
                            fp16 = True
                    else:  # output
                        output_names.append(name)
                    shape = tuple(context.get_binding_shape(i))
                    im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                    bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
                binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
                batch_size = bindings['images'].shape[0]  # if dynamic, this is instead max batch size
            elif coreml:  # CoreML
                LOGGER.info(f'Loading {w} for CoreML inference...')
                import coremltools as ct
                model = ct.models.MLModel(w)
            elif saved_model:  # TF SavedModel
                LOGGER.info(f'Loading {w} for TensorFlow SavedModel inference...')
                import tensorflow as tf
                keras = False  # assume TF1 saved_model
                model = tf.keras.models.load_model(w) if keras else tf.saved_model.load(w)
            elif pb:  # GraphDef https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
                LOGGER.info(f'Loading {w} for TensorFlow GraphDef inference...')
                import tensorflow as tf

                def wrap_frozen_graph(gd, inputs, outputs):
                    x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped
                    ge = x.graph.as_graph_element
                    return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.map_structure(ge, outputs))

                def gd_outputs(gd):
                    name_list, input_list = [], []
                    for node in gd.node:  # tensorflow.core.framework.node_def_pb2.NodeDef
                        name_list.append(node.name)
                        input_list.extend(node.input)
                    return sorted(f'{x}:0' for x in list(set(name_list) - set(input_list)) if not x.startswith('NoOp'))

                gd = tf.Graph().as_graph_def()  # TF GraphDef
                with open(w, 'rb') as f:
                    gd.ParseFromString(f.read())
                frozen_func = wrap_frozen_graph(gd, inputs="x:0", outputs=gd_outputs(gd))
            elif tflite or edgetpu:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
                try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
                    from tflite_runtime.interpreter import Interpreter, load_delegate
                except ImportError:
                    import tensorflow as tf
                    Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate,
                if edgetpu:  # TF Edge TPU https://coral.ai/software/#edgetpu-runtime
                    LOGGER.info(f'Loading {w} for TensorFlow Lite Edge TPU inference...')
                    delegate = {
                        'Linux': 'libedgetpu.so.1',
                        'Darwin': 'libedgetpu.1.dylib',
                        'Windows': 'edgetpu.dll'}[platform.system()]
                    interpreter = Interpreter(model_path=w, experimental_delegates=[load_delegate(delegate)])
                else:  # TFLite
                    LOGGER.info(f'Loading {w} for TensorFlow Lite inference...')
                    interpreter = Interpreter(model_path=w)  # load TFLite model
                interpreter.allocate_tensors()  # allocate
                input_details = interpreter.get_input_details()  # inputs
                output_details = interpreter.get_output_details()  # outputs
                # load metadata
                with contextlib.suppress(zipfile.BadZipFile):
                    with zipfile.ZipFile(w, "r") as model:
                        meta_file = model.namelist()[0]
                        meta = ast.literal_eval(model.read(meta_file).decode("utf-8"))
                        stride, names = int(meta['stride']), meta['names']
            elif tfjs:  # TF.js
                raise NotImplementedError('ERROR: YOLO TF.js inference is not supported')
            elif paddle:  # PaddlePaddle
                LOGGER.info(f'Loading {w} for PaddlePaddle inference...')
                check_requirements('paddlepaddle-gpu' if cuda else 'paddlepaddle')
                import paddle.inference as pdi
                if not Path(w).is_file():  # if not *.pdmodel
                    w = next(Path(w).rglob('*.pdmodel'))  # get *.pdmodel file from *_paddle_model dir
                weights = Path(w).with_suffix('.pdiparams')
                config = pdi.Config(str(w), str(weights))
                if cuda:
                    config.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=0)
                predictor = pdi.create_predictor(config)
                input_handle = predictor.get_input_handle(predictor.get_input_names()[0])
                output_names = predictor.get_output_names()
            elif triton:  # NVIDIA Triton Inference Server
                LOGGER.info(f'Using {w} as Triton Inference Server...')
                check_requirements('tritonclient[all]')
                from utils.triton import TritonRemoteModel
                model = TritonRemoteModel(url=w)
                nhwc = model.runtime.startswith("tensorflow")
            else:
                raise NotImplementedError(f'ERROR: {w} is not a supported format')

            # class names
            if 'names' not in locals():
                names = yaml_load(data)['names'] if data else {i: f'class{i}' for i in range(999)}
            if names[0] == 'n01440764' and len(names) == 1000:  # ImageNet
                names = yaml_load(ROOT / 'data/ImageNet.yaml')['names']  # human-readable names

            self.__dict__.update(locals())  # assign all variables to self

        def forward(self, im, augment=False, visualize=False):
            # YOLO MultiBackend inference
            b, ch, h, w = im.shape  # batch, channel, height, width
            if self.fp16 and im.dtype != torch.float16:
                im = im.half()  # to FP16
            if self.nhwc:
                im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)

            if self.pt:  # PyTorch
                y = self.model(im, augment=augment, visualize=visualize) if augment or visualize else self.model(im)
            elif self.jit:  # TorchScript
                y = self.model(im)
            elif self.dnn:  # ONNX OpenCV DNN
                im = im.cpu().numpy()  # torch to numpy
                self.net.setInput(im)
                y = self.net.forward()
            elif self.onnx:  # ONNX Runtime
                im = im.cpu().numpy()  # torch to numpy
                y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
            elif self.xml:  # OpenVINO
                im = im.cpu().numpy()  # FP32
                y = list(self.executable_network([im]).values())
            elif self.engine:  # TensorRT
                if self.dynamic and im.shape != self.bindings['images'].shape:
                    i = self.model.get_binding_index('images')
                    self.context.set_binding_shape(i, im.shape)  # reshape if dynamic
                    self.bindings['images'] = self.bindings['images']._replace(shape=im.shape)
                    for name in self.output_names:
                        i = self.model.get_binding_index(name)
                        self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))
                s = self.bindings['images'].shape
                assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
                self.binding_addrs['images'] = int(im.data_ptr())
                self.context.execute_v2(list(self.binding_addrs.values()))
                y = [self.bindings[x].data for x in sorted(self.output_names)]
            elif self.coreml:  # CoreML
                im = im.cpu().numpy()
                im = Image.fromarray((im[0] * 255).astype('uint8'))
                # im = im.resize((192, 320), Image.ANTIALIAS)
                y = self.model.predict({'image': im})  # coordinates are xywh normalized
                if 'confidence' in y:
                    box = xywh2xyxy(y['coordinates'] * [[w, h, w, h]])  # xyxy pixels
                    conf, cls = y['confidence'].max(1), y['confidence'].argmax(1).astype(np.float)
                    y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
                else:
                    y = list(reversed(y.values()))  # reversed for segmentation models (pred, proto)
            elif self.paddle:  # PaddlePaddle
                im = im.cpu().numpy().astype(np.float32)
                self.input_handle.copy_from_cpu(im)
                self.predictor.run()
                y = [self.predictor.get_output_handle(x).copy_to_cpu() for x in self.output_names]
            elif self.triton:  # NVIDIA Triton Inference Server
                y = self.model(im)
            else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
                im = im.cpu().numpy()
                if self.saved_model:  # SavedModel
                    y = self.model(im, training=False) if self.keras else self.model(im)
                elif self.pb:  # GraphDef
                    y = self.frozen_func(x=self.tf.constant(im))
                else:  # Lite or Edge TPU
                    input = self.input_details[0]
                    int8 = input['dtype'] == np.uint8  # is TFLite quantized uint8 model
                    if int8:
                        scale, zero_point = input['quantization']
                        im = (im / scale + zero_point).astype(np.uint8)  # de-scale
                    self.interpreter.set_tensor(input['index'], im)
                    self.interpreter.invoke()
                    y = []
                    for output in self.output_details:
                        x = self.interpreter.get_tensor(output['index'])
                        if int8:
                            scale, zero_point = output['quantization']
                            x = (x.astype(np.float32) - zero_point) * scale  # re-scale
                        y.append(x)
                y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
                y[0][..., :4] *= [w, h, w, h]  # xywh normalized to pixels

            if isinstance(y, (list, tuple)):
                return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
            else:
                return self.from_numpy(y)

        def from_numpy(self, x):
            return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

        def warmup(self, imgsz=(1, 3, 640, 640)):
            # Warmup model by running inference once
            warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton
            if any(warmup_types) and (self.device.type != 'cpu' or self.triton):
                im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
                for _ in range(2 if self.jit else 1):  #
                    self.forward(im)  # warmup

        @staticmethod
        def _model_type(p='path/to/model.pt'):
            # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
            # types = [pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle]
            from yolocode.yolov9.export import export_formats
            from yolocode.yolov9.utils.downloads import is_url
            sf = list(export_formats().Suffix)  # export suffixes
            if not is_url(p, check=False):
                check_suffix(p, sf)  # checks
            url = urlparse(p)  # if url may be Triton inference server
            types = [s in Path(p).name for s in sf]
            types[8] &= not types[9]  # tflite &= not edgetpu
            triton = not any(types) and all([any(s in url.scheme for s in ["http", "grpc"]), url.netloc])
            return types + [triton]

        @staticmethod
        def _load_metadata(f=Path('path/to/meta.yaml')):
            # Load metadata from meta.yaml if it exists
            if f.exists():
                d = yaml_load(f)
                return d['stride'], d['names']  # assign stride, names
            return None, None
    class AutoShape(nn.Module):
        # YOLO input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
        conf = 0.25  # NMS confidence threshold
        iou = 0.45  # NMS IoU threshold
        agnostic = False  # NMS class-agnostic
        multi_label = False  # NMS multiple labels per box
        classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
        max_det = 1000  # maximum number of detections per image
        amp = False  # Automatic Mixed Precision (AMP) inference

        def __init__(self, model, verbose=True):
            super().__init__()
            if verbose:
                LOGGER.info('Adding AutoShape... ')
            copy_attr(self, model, include=('yaml', 'nc', 'hyp', 'names', 'stride', 'abc'),
                      exclude=())  # copy attributes
            self.dmb = isinstance(model, DetectMultiBackend_YOLOv9)  # DetectMultiBackend() instance
            self.pt = not self.dmb or model.pt  # PyTorch model
            self.model = model.eval()
            if self.pt:
                m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
                m.inplace = False  # Detect.inplace=False for safe multithread inference
                m.export = True  # do not output loss values

        def _apply(self, fn):
            # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
            self = super()._apply(fn)
            from models.yolo import Detect_YOLOv9, Segment_YOLOv9
            if self.pt:
                m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
                if isinstance(m, (Detect_YOLOv9, Segment_YOLOv9)):
                    for k in 'stride', 'anchor_grid', 'stride_grid', 'grid':
                        x = getattr(m, k)
                        setattr(m, k, list(map(fn, x))) if isinstance(x, (list, tuple)) else setattr(m, k, fn(x))
            return self

        @smart_inference_mode()
        def forward(self, ims, size=640, augment=False, profile=False):
            # Inference from various sources. For size(height=640, width=1280), RGB images example inputs are:
            #   file:        ims = 'data/images/zidane.jpg'  # str or PosixPath
            #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
            #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
            #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
            #   numpy:           = np.zeros((640,1280,3))  # HWC
            #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
            #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

            dt = (Profile(), Profile(), Profile())
            with dt[0]:
                if isinstance(size, int):  # expand
                    size = (size, size)
                p = next(self.model.parameters()) if self.pt else torch.empty(1, device=self.model.device)  # param
                autocast = self.amp and (p.device.type != 'cpu')  # Automatic Mixed Precision (AMP) inference
                if isinstance(ims, torch.Tensor):  # torch
                    with amp.autocast(autocast):
                        return self.model(ims.to(p.device).type_as(p), augment=augment)  # inference

                # Pre-process
                n, ims = (len(ims), list(ims)) if isinstance(ims, (list, tuple)) else (
                    1, [ims])  # number, list of images
                shape0, shape1, files = [], [], []  # image and inference shapes, filenames
                for i, im in enumerate(ims):
                    f = f'image{i}'  # filename
                    if isinstance(im, (str, Path)):  # filename or uri
                        im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im), im
                        im = np.asarray(exif_transpose(im))
                    elif isinstance(im, Image.Image):  # PIL Image
                        im, f = np.asarray(exif_transpose(im)), getattr(im, 'filename', f) or f
                    files.append(Path(f).with_suffix('.jpg').name)
                    if im.shape[0] < 5:  # image in CHW
                        im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
                    im = im[..., :3] if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)  # enforce 3ch input
                    s = im.shape[:2]  # HWC
                    shape0.append(s)  # image shape
                    g = max(size) / max(s)  # gain
                    shape1.append([int(y * g) for y in s])
                    ims[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
                shape1 = [make_divisible(x, self.stride) for x in np.array(shape1).max(0)]  # inf shape
                x = [letterbox(im, shape1, auto=False)[0] for im in ims]  # pad
                x = np.ascontiguousarray(np.array(x).transpose((0, 3, 1, 2)))  # stack and BHWC to BCHW
                x = torch.from_numpy(x).to(p.device).type_as(p) / 255  # uint8 to fp16/32

            with amp.autocast(autocast):
                # Inference
                with dt[1]:
                    y = self.model(x, augment=augment)  # forward

                # Post-process
                with dt[2]:
                    y = non_max_suppression(y if self.dmb else y[0],
                                            self.conf,
                                            self.iou,
                                            self.classes,
                                            self.agnostic,
                                            self.multi_label,
                                            max_det=self.max_det)  # NMS
                    for i in range(n):
                        scale_boxes(shape1, y[i][:, :4], shape0[i])

                return Detections(ims, y, files, dt, self.names, x.shape)
    class Detections:
        # YOLO detections class for inference results
        def __init__(self, ims, pred, files, times=(0, 0, 0), names=None, shape=None):
            super().__init__()
            d = pred[0].device  # device
            gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in ims]  # normalizations
            self.ims = ims  # list of images as numpy arrays
            self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
            self.names = names  # class names
            self.files = files  # image filenames
            self.times = times  # profiling times
            self.xyxy = pred  # xyxy pixels
            self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
            self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
            self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
            self.n = len(self.pred)  # number of images (batch size)
            self.t = tuple(x.t / self.n * 1E3 for x in times)  # timestamps (ms)
            self.s = tuple(shape)  # inference BCHW shape

        def _run(self, pprint=False, show=False, save=False, crop=False, render=False, labels=True, save_dir=Path('')):
            s, crops = '', []
            for i, (im, pred) in enumerate(zip(self.ims, self.pred)):
                s += f'\nimage {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '  # string
                if pred.shape[0]:
                    for c in pred[:, -1].unique():
                        n = (pred[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    s = s.rstrip(', ')
                    if show or save or render or crop:
                        annotator = Annotator(im, example=str(self.names))
                        for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
                            label = f'{self.names[int(cls)]} {conf:.2f}'
                            if crop:
                                file = save_dir / 'crops' / self.names[int(cls)] / self.files[i] if save else None
                                crops.append({
                                    'box': box,
                                    'conf': conf,
                                    'cls': cls,
                                    'label': label,
                                    'im': save_one_box(box, im, file=file, save=save)})
                            else:  # all others
                                annotator.box_label(box, label if labels else '', color=colors(cls))
                        im = annotator.im
                else:
                    s += '(no detections)'

                im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
                if show:
                    display(im) if is_notebook() else im.show(self.files[i])
                if save:
                    f = self.files[i]
                    im.save(save_dir / f)  # save
                    if i == self.n - 1:
                        LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
                if render:
                    self.ims[i] = np.asarray(im)
            if pprint:
                s = s.lstrip('\n')
                return f'{s}\nSpeed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {self.s}' % self.t
            if crop:
                if save:
                    LOGGER.info(f'Saved results to {save_dir}\n')
                return crops

        @TryExcept('Showing images is not supported in this environment')
        def show(self, labels=True):
            self._run(show=True, labels=labels)  # show results

        def save(self, labels=True, save_dir='runs/detect/exp', exist_ok=False):
            save_dir = increment_path(save_dir, exist_ok, mkdir=True)  # increment save_dir
            self._run(save=True, labels=labels, save_dir=save_dir)  # save results

        def crop(self, save=True, save_dir='runs/detect/exp', exist_ok=False):
            save_dir = increment_path(save_dir, exist_ok, mkdir=True) if save else None
            return self._run(crop=True, save=save, save_dir=save_dir)  # crop results

        def render(self, labels=True):
            self._run(render=True, labels=labels)  # render results
            return self.ims

        def pandas(self):
            # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
            new = copy(self)  # return copy
            ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
            cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
            for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
                a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in
                     getattr(self, k)]  # update
                setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
            return new

        def tolist(self):
            # return a list of Detections objects, i.e. 'for result in results.tolist():'
            r = range(self.n)  # iterable
            x = [Detections([self.ims[i]], [self.pred[i]], [self.files[i]], self.times, self.names, self.s) for i in r]
            # for d in x:
            #    for k in ['ims', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
            #        setattr(d, k, getattr(d, k)[0])  # pop out of list
            return x

        def print(self):
            LOGGER.info(self.__str__())

        def __len__(self):  # override len(results)
            return self.n

        def __str__(self):  # override print(results)
            return self._run(pprint=True)  # print results

        def __repr__(self):
            return f'YOLO {self.__class__} instance\n' + self.__str__()
    class Classify(nn.Module):
        # YOLO classification head, i.e. x(b,c1,20,20) to x(b,c2)
        def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
            super().__init__()
            c_ = 1280  # efficientnet_b0 size
            self.conv = Conv(c1, c_, k, s, autopad(k, p), g)
            self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
            self.drop = nn.Dropout(p=0.0, inplace=True)
            self.linear = nn.Linear(c_, c2)  # to x(b,c2)

        def forward(self, x):
            if isinstance(x, list):
                x = torch.cat(x, 1)
            return self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
### --- YOLOv9 Code --- ###


