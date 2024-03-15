import argparse
import logging
import sys
import argparse
import contextlib
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path
import math
import torch
import torch.nn as nn
from utils import glo
try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)



yoloname = glo.get_value('yoloname')
yoloname1 = glo.get_value('yoloname1')
yoloname2 = glo.get_value('yoloname2')

yolo_name = ((str(yoloname1) if yoloname1 else '') + (str(yoloname2) if str(
    yoloname2) else '')) if yoloname1 or yoloname2 else yoloname

if "yolov5" in yolo_name:
    # YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[1]  # YOLOv5 root directory
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add ROOT to PATH
    if platform.system() != "Windows":
        ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
    from yolocode.yolov5.models.common import (
        C3,
        C3SPP,
        C3TR,
        SPP,
        SPPF,
        Bottleneck,
        BottleneckCSP,
        C3Ghost,
        C3x,
        Classify,
        Concat,
        Contract,
        Conv,
        CrossConv,
        DetectMultiBackend,
        DWConv,
        DWConvTranspose2d,
        Expand,
        Focus,
        GhostBottleneck,
        GhostConv,
        Proto,
    )
    from yolocode.yolov5.models.experimental import MixConv2d
    from yolocode.yolov5.utils.autoanchor import check_anchor_order
    from yolocode.yolov5.utils.general import LOGGER, check_version, check_yaml, colorstr, make_divisible, print_args
    from yolocode.yolov5.utils.plots import feature_visualization
    from yolocode.yolov5.utils.torch_utils import (
        fuse_conv_and_bn,
        initialize_weights,
        model_info,
        profile,
        scale_img,
        select_device,
        time_sync,
    )

    class Detect_YOLOV5(nn.Module):
        # YOLOv5 Detect head for detection models
        stride = None  # strides computed during build
        dynamic = False  # force grid reconstruction
        export = False  # export mode

        def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
            super().__init__()
            self.nc = nc  # number of classes
            self.no = nc + 5  # number of outputs per anchor
            self.nl = len(anchors)  # number of detection layers
            self.na = len(anchors[0]) // 2  # number of anchors
            self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
            self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
            self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
            self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
            self.inplace = inplace  # use inplace ops (e.g. slice assignment)

        def forward(self, x):
            z = []  # inference output
            for i in range(self.nl):
                x[i] = self.m[i](x[i])  # conv
                bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
                x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

                if not self.training:  # inference
                    if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                        self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                    if isinstance(self, Segment_YOLOV5):  # (boxes + masks)
                        xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                        xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                        wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                        y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                    else:  # Detect (boxes only)
                        xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                        xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                        wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                        y = torch.cat((xy, wh, conf), 4)
                    z.append(y.view(bs, self.na * nx * ny, self.no))

            return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

        def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, "1.10.0")):
            d = self.anchors[i].device
            t = self.anchors[i].dtype
            shape = 1, self.na, ny, nx, 2  # grid shape
            y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
            yv, xv = torch.meshgrid(y, x, indexing="ij") if torch_1_10 else torch.meshgrid(y,
                                                                                           x)  # torch>=0.7 compatibility
            grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
            anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
            return grid, anchor_grid
    class Segment_YOLOV5(Detect_YOLOV5):
        # YOLOv5 Segment head for segmentation models
        def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
            super().__init__(nc, anchors, ch, inplace)
            self.nm = nm  # number of masks
            self.npr = npr  # number of protos
            self.no = 5 + nc + self.nm  # number of outputs per anchor
            self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
            self.proto = Proto(ch[0], self.npr, self.nm)  # protos
            self.detect = Detect_YOLOV5.forward

        def forward(self, x):
            p = self.proto(x[0])
            x = self.detect(self, x)
            return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])
    class Detect(nn.Module):
        # YOLOv5 Detect head for detection models
        stride = None  # strides computed during build
        dynamic = False  # force grid reconstruction
        export = False  # export mode

        def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
            super().__init__()
            self.nc = nc  # number of classes
            self.no = nc + 5  # number of outputs per anchor
            self.nl = len(anchors)  # number of detection layers
            self.na = len(anchors[0]) // 2  # number of anchors
            self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
            self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
            self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
            self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
            self.inplace = inplace  # use inplace ops (e.g. slice assignment)

        def forward(self, x):
            z = []  # inference output
            for i in range(self.nl):
                x[i] = self.m[i](x[i])  # conv
                bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
                x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

                if not self.training:  # inference
                    if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                        self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                    if isinstance(self, Segment):  # (boxes + masks)
                        xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                        xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                        wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                        y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                    else:  # Detect (boxes only)
                        xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                        xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                        wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                        y = torch.cat((xy, wh, conf), 4)
                    z.append(y.view(bs, self.na * nx * ny, self.no))

            return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

        def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, "1.10.0")):
            d = self.anchors[i].device
            t = self.anchors[i].dtype
            shape = 1, self.na, ny, nx, 2  # grid shape
            y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
            yv, xv = torch.meshgrid(y, x, indexing="ij") if torch_1_10 else torch.meshgrid(y,
                                                                                           x)  # torch>=0.7 compatibility
            grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
            anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
            return grid, anchor_grid
    class Segment(Detect):
        # YOLOv5 Segment head for segmentation models
        def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
            super().__init__(nc, anchors, ch, inplace)
            self.nm = nm  # number of masks
            self.npr = npr  # number of protos
            self.no = 5 + nc + self.nm  # number of outputs per anchor
            self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
            self.proto = Proto(ch[0], self.npr, self.nm)  # protos
            self.detect = Detect.forward

        def forward(self, x):
            p = self.proto(x[0])
            x = self.detect(self, x)
            return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])
    class BaseModel(nn.Module):
        # YOLOv5 base model
        def forward(self, x, profile=False, visualize=False):
            return self._forward_once(x, profile, visualize)  # single-scale inference, train

        def _forward_once(self, x, profile=False, visualize=False):
            y, dt = [], []  # outputs
            for m in self.model:
                if m.f != -1:  # if not from previous layer
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                if profile:
                    self._profile_one_layer(m, x, dt)
                x = m(x)  # run
                y.append(x if m.i in self.save else None)  # save output
                if visualize:
                    feature_visualization(x, m.type, m.i, save_dir=visualize)
            return x

        def _profile_one_layer(self, m, x, dt):
            c = m == self.model[-1]  # is final layer, copy input as inplace fix
            o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1e9 * 2 if thop else 0  # FLOPs
            t = time_sync()
            for _ in range(10):
                m(x.copy() if c else x)
            dt.append((time_sync() - t) * 100)
            if m == self.model[0]:
                LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
            LOGGER.info(f"{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}")
            if c:
                LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

        def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
            LOGGER.info("Fusing layers... ")
            for m in self.model.modules():
                if isinstance(m, (Conv, DWConv)) and hasattr(m, "bn"):
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
            self.info()
            return self

        def info(self, verbose=False, img_size=640):  # print model information
            model_info(self, verbose, img_size)

        def _apply(self, fn):
            # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
            self = super()._apply(fn)
            m = self.model[-1]  # Detect()
            if isinstance(m, (Detect_YOLOV5, Segment_YOLOV5)):
                m.stride = fn(m.stride)
                m.grid = list(map(fn, m.grid))
                if isinstance(m.anchor_grid, list):
                    m.anchor_grid = list(map(fn, m.anchor_grid))
            return self
    class DetectionModel(BaseModel):
        # YOLOv5 detection model
        def __init__(self, cfg="yolov5s.yaml", ch=3, nc=None, anchors=None):  # model, input channels, number of classes
            super().__init__()
            if isinstance(cfg, dict):
                self.yaml = cfg  # model dict
            else:  # is *.yaml
                import yaml  # for torch hub

                self.yaml_file = Path(cfg).name
                with open(cfg, encoding="ascii", errors="ignore") as f:
                    self.yaml = yaml.safe_load(f)  # model dict

            # Define model
            ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
            if nc and nc != self.yaml["nc"]:
                LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
                self.yaml["nc"] = nc  # override yaml value
            if anchors:
                LOGGER.info(f"Overriding model.yaml anchors with anchors={anchors}")
                self.yaml["anchors"] = round(anchors)  # override yaml value
            self.model, self.save = parse_model_YOLOv5(deepcopy(self.yaml), ch=[ch])  # model, savelist
            self.names = [str(i) for i in range(self.yaml["nc"])]  # default names
            self.inplace = self.yaml.get("inplace", True)

            # Build strides, anchors
            m = self.model[-1]  # Detect()
            if isinstance(m, (Detect_YOLOV5, Segment_YOLOV5)):
                s = 256  # 2x min stride
                m.inplace = self.inplace
                forward = lambda x: self.forward(x)[0] if isinstance(m, Segment_YOLOV5) else self.forward(x)
                m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
                check_anchor_order(m)
                m.anchors /= m.stride.view(-1, 1, 1)
                self.stride = m.stride
                self._initialize_biases()  # only run once

            # Init weights, biases
            initialize_weights(self)
            self.info()
            LOGGER.info("")

        def forward(self, x, augment=False, profile=False, visualize=False):
            if augment:
                return self._forward_augment(x)  # augmented inference, None
            return self._forward_once(x, profile, visualize)  # single-scale inference, train

        def _forward_augment(self, x):
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self._forward_once(xi)[0]  # forward
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi = self._descale_pred(yi, fi, si, img_size)
                y.append(yi)
            y = self._clip_augmented(y)  # clip augmented tails
            return torch.cat(y, 1), None  # augmented inference, train

        def _descale_pred(self, p, flips, scale, img_size):
            # de-scale predictions following augmented inference (inverse operation)
            if self.inplace:
                p[..., :4] /= scale  # de-scale
                if flips == 2:
                    p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
                elif flips == 3:
                    p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
            else:
                x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
                if flips == 2:
                    y = img_size[0] - y  # de-flip ud
                elif flips == 3:
                    x = img_size[1] - x  # de-flip lr
                p = torch.cat((x, y, wh, p[..., 4:]), -1)
            return p

        def _clip_augmented(self, y):
            # Clip YOLOv5 augmented inference tails
            nl = self.model[-1].nl  # number of detection layers (P3-P5)
            g = sum(4 ** x for x in range(nl))  # grid points
            e = 1  # exclude layer count
            i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
            y[0] = y[0][:, :-i]  # large
            i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
            y[-1] = y[-1][:, i:]  # small
            return y

        def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
            # https://arxiv.org/abs/1708.02002 section 3.3
            # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
            m = self.model[-1]  # Detect() module
            for mi, s in zip(m.m, m.stride):  # from
                b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
                b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
                b.data[:, 5: 5 + m.nc] += (
                    math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())
                )  # cls
                mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    Model_YOLOV5 = DetectionModel  # retain YOLOv5 'Model' class for backwards compatibility
    # Model = DetectionModel  # retain YOLOv5 'Model' class for backwards compatibility
    class SegmentationModel(DetectionModel):
        # YOLOv5 segmentation model
        def __init__(self, cfg="yolov5s-seg.yaml", ch=3, nc=None, anchors=None):
            super().__init__(cfg, ch, nc, anchors)
    class ClassificationModel(BaseModel):
        # YOLOv5 classification model
        def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):  # yaml, model, number of classes, cutoff index
            super().__init__()
            self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

        def _from_detection_model(self, model, nc=1000, cutoff=10):
            # Create a YOLOv5 classification model from a YOLOv5 detection model
            if isinstance(model, DetectMultiBackend):
                model = model.model  # unwrap DetectMultiBackend
            model.model = model.model[:cutoff]  # backbone
            m = model.model[-1]  # last layer
            ch = m.conv.in_channels if hasattr(m, "conv") else m.cv1.conv.in_channels  # ch into module
            c = Classify(ch, nc)  # Classify()
            c.i, c.f, c.type = m.i, m.f, "models.common.Classify"  # index, from, type
            model.model[-1] = c  # replace
            self.model = model.model
            self.stride = model.stride
            self.save = []
            self.nc = nc

        def _from_yaml(self, cfg):
            # Create a YOLOv5 classification model from a *.yaml file
            self.model = None
    def parse_model_YOLOv5(d, ch):  # model_dict, input_channels(3)
        # Parse a YOLOv5 model.yaml dictionary
        LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
        anchors, nc, gd, gw, act, ch_mul = (
            d["anchors"],
            d["nc"],
            d["depth_multiple"],
            d["width_multiple"],
            d.get("activation"),
            d.get("channel_multiple"),
        )
        if act:
            Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print
        if not ch_mul:
            ch_mul = 8
        na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
        no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

        layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
        for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
            m = eval(m) if isinstance(m, str) else m  # eval strings
            for j, a in enumerate(args):
                with contextlib.suppress(NameError):
                    args[j] = eval(a) if isinstance(a, str) else a  # eval strings

            n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
            if m in {
                Conv,
                GhostConv,
                Bottleneck,
                GhostBottleneck,
                SPP,
                SPPF,
                DWConv,
                MixConv2d,
                Focus,
                CrossConv,
                BottleneckCSP,
                C3,
                C3TR,
                C3SPP,
                C3Ghost,
                nn.ConvTranspose2d,
                DWConvTranspose2d,
                C3x,
            }:
                c1, c2 = ch[f], args[0]
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, ch_mul)

                args = [c1, c2, *args[1:]]
                if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}:
                    args.insert(2, n)  # number of repeats
                    n = 1
            elif m is nn.BatchNorm2d:
                args = [ch[f]]
            elif m is Concat:
                c2 = sum(ch[x] for x in f)
            # TODO: channel, gw, gd
            elif m in {Detect, Segment}:
                args.append([ch[x] for x in f])
                if isinstance(args[1], int):  # number of anchors
                    args[1] = [list(range(args[1] * 2))] * len(f)
                if m is Segment:
                    args[3] = make_divisible(args[3] * gw, ch_mul)
            elif m is Contract:
                c2 = ch[f] * args[0] ** 2
            elif m is Expand:
                c2 = ch[f] // args[0] ** 2
            else:
                c2 = ch[f]

            m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
            t = str(m)[8:-2].replace("__main__.", "")  # module type
            np = sum(x.numel() for x in m_.parameters())  # number params
            m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
            LOGGER.info(f"{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}")  # print
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
            layers.append(m_)
            if i == 0:
                ch = []
            ch.append(c2)
        return nn.Sequential(*layers), sorted(save)
if "yolov7" in yolo_name:
    from yolocode.yolov7.models.common import *
    from yolocode.yolov7.models.experimental import *
    from yolocode.yolov7.utils.autoanchor import check_anchor_order
    from yolocode.yolov7.utils.general import make_divisible, check_file, set_logging
    from yolocode.yolov7.utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, \
        initialize_weights, \
        select_device, copy_attr
    from yolocode.yolov7.utils.loss import SigmoidBin

    class Detect(nn.Module):
        stride = None  # strides computed during build
        export = False  # onnx export
        end2end = False
        include_nms = False
        concat = False

        def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
            super(Detect, self).__init__()
            self.nc = nc  # number of classes
            self.no = nc + 5  # number of outputs per anchor
            self.nl = len(anchors)  # number of detection layers
            self.na = len(anchors[0]) // 2  # number of anchors
            self.grid = [torch.zeros(1)] * self.nl  # init grid
            a = torch.tensor(anchors).float().view(self.nl, -1, 2)
            self.register_buffer('anchors', a)  # shape(nl,na,2)
            self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
            self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

        def forward(self, x):
            # x = x.copy()  # for profiling
            z = []  # inference output
            self.training |= self.export
            for i in range(self.nl):
                x[i] = self.m[i](x[i])  # conv
                bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
                x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

                if not self.training:  # inference
                    if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                        self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                    y = x[i].sigmoid()
                    if not torch.onnx.is_in_onnx_export():
                        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    else:
                        xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                        xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))  # new xy
                        wh = wh ** 2 * (4 * self.anchor_grid[i].data)  # new wh
                        y = torch.cat((xy, wh, conf), 4)
                    z.append(y.view(bs, -1, self.no))

            if self.training:
                out = x
            elif self.end2end:
                out = torch.cat(z, 1)
            elif self.include_nms:
                z = self.convert(z)
                out = (z,)
            elif self.concat:
                out = torch.cat(z, 1)
            else:
                out = (torch.cat(z, 1), x)

            return out

        @staticmethod
        def _make_grid(nx=20, ny=20):
            yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
            return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

        def convert(self, z):
            z = torch.cat(z, 1)
            box = z[:, :, :4]
            conf = z[:, :, 4:5]
            score = z[:, :, 5:]
            score *= conf
            convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                          dtype=torch.float32,
                                          device=z.device)
            box @= convert_matrix
            return (box, score)
    class Detect_YOLOV7(nn.Module):
        stride = None  # strides computed during build
        export = False  # onnx export
        end2end = False
        include_nms = False
        concat = False

        def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
            super(Detect_YOLOV7, self).__init__()
            self.nc = nc  # number of classes
            self.no = nc + 5  # number of outputs per anchor
            self.nl = len(anchors)  # number of detection layers
            self.na = len(anchors[0]) // 2  # number of anchors
            self.grid = [torch.zeros(1)] * self.nl  # init grid
            a = torch.tensor(anchors).float().view(self.nl, -1, 2)
            self.register_buffer('anchors', a)  # shape(nl,na,2)
            self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
            self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

        def forward(self, x):
            # x = x.copy()  # for profiling
            z = []  # inference output
            self.training |= self.export
            for i in range(self.nl):
                x[i] = self.m[i](x[i])  # conv
                bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
                x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

                if not self.training:  # inference
                    if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                        self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                    y = x[i].sigmoid()
                    if not torch.onnx.is_in_onnx_export():
                        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    else:
                        xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                        xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))  # new xy
                        wh = wh ** 2 * (4 * self.anchor_grid[i].data)  # new wh
                        y = torch.cat((xy, wh, conf), 4)
                    z.append(y.view(bs, -1, self.no))

            if self.training:
                out = x
            elif self.end2end:
                out = torch.cat(z, 1)
            elif self.include_nms:
                z = self.convert(z)
                out = (z,)
            elif self.concat:
                out = torch.cat(z, 1)
            else:
                out = (torch.cat(z, 1), x)

            return out

        @staticmethod
        def _make_grid(nx=20, ny=20):
            yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
            return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

        def convert(self, z):
            z = torch.cat(z, 1)
            box = z[:, :, :4]
            conf = z[:, :, 4:5]
            score = z[:, :, 5:]
            score *= conf
            convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                          dtype=torch.float32,
                                          device=z.device)
            box @= convert_matrix
            return (box, score)
    class IDetect(nn.Module):
        stride = None  # strides computed during build
        export = False  # onnx export
        end2end = False
        include_nms = False
        concat = False

        def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
            super(IDetect, self).__init__()
            self.nc = nc  # number of classes
            self.no = nc + 5  # number of outputs per anchor
            self.nl = len(anchors)  # number of detection layers
            self.na = len(anchors[0]) // 2  # number of anchors
            self.grid = [torch.zeros(1)] * self.nl  # init grid
            a = torch.tensor(anchors).float().view(self.nl, -1, 2)
            self.register_buffer('anchors', a)  # shape(nl,na,2)
            self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
            self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

            self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
            self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch)

        def forward(self, x):
            # x = x.copy()  # for profiling
            z = []  # inference output
            self.training |= self.export
            for i in range(self.nl):
                x[i] = self.m[i](self.ia[i](x[i]))  # conv
                x[i] = self.im[i](x[i])
                bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
                x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

                if not self.training:  # inference
                    if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                        self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                    y = x[i].sigmoid()
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    z.append(y.view(bs, -1, self.no))

            return x if self.training else (torch.cat(z, 1), x)

        def fuseforward(self, x):
            # x = x.copy()  # for profiling
            z = []  # inference output
            self.training |= self.export
            for i in range(self.nl):
                x[i] = self.m[i](x[i])  # conv
                bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
                x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

                if not self.training:  # inference
                    if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                        self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                    y = x[i].sigmoid()
                    if not torch.onnx.is_in_onnx_export():
                        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    else:
                        xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                        xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))  # new xy
                        wh = wh ** 2 * (4 * self.anchor_grid[i].data)  # new wh
                        y = torch.cat((xy, wh, conf), 4)
                    z.append(y.view(bs, -1, self.no))

            if self.training:
                out = x
            elif self.end2end:
                out = torch.cat(z, 1)
            elif self.include_nms:
                z = self.convert(z)
                out = (z,)
            elif self.concat:
                out = torch.cat(z, 1)
            else:
                out = (torch.cat(z, 1), x)

            return out

        def fuse(self):
            print("IDetect.fuse")
            # fuse ImplicitA and Convolution
            for i in range(len(self.m)):
                c1, c2, _, _ = self.m[i].weight.shape
                c1_, c2_, _, _ = self.ia[i].implicit.shape
                self.m[i].bias += torch.matmul(self.m[i].weight.reshape(c1, c2),
                                               self.ia[i].implicit.reshape(c2_, c1_)).squeeze(1)

            # fuse ImplicitM and Convolution
            for i in range(len(self.m)):
                c1, c2, _, _ = self.im[i].implicit.shape
                self.m[i].bias *= self.im[i].implicit.reshape(c2)
                self.m[i].weight *= self.im[i].implicit.transpose(0, 1)

        @staticmethod
        def _make_grid(nx=20, ny=20):
            yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
            return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

        def convert(self, z):
            z = torch.cat(z, 1)
            box = z[:, :, :4]
            conf = z[:, :, 4:5]
            score = z[:, :, 5:]
            score *= conf
            convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                          dtype=torch.float32,
                                          device=z.device)
            box @= convert_matrix
            return (box, score)
    class IKeypoint(nn.Module):
        stride = None  # strides computed during build
        export = False  # onnx export

        def __init__(self, nc=80, anchors=(), nkpt=17, ch=(), inplace=True, dw_conv_kpt=False):  # detection layer
            super(IKeypoint, self).__init__()
            self.nc = nc  # number of classes
            self.nkpt = nkpt
            self.dw_conv_kpt = dw_conv_kpt
            self.no_det = (nc + 5)  # number of outputs per anchor for box and class
            self.no_kpt = 3 * self.nkpt  ## number of outputs per anchor for keypoints
            self.no = self.no_det + self.no_kpt
            self.nl = len(anchors)  # number of detection layers
            self.na = len(anchors[0]) // 2  # number of anchors
            self.grid = [torch.zeros(1)] * self.nl  # init grid
            self.flip_test = False
            a = torch.tensor(anchors).float().view(self.nl, -1, 2)
            self.register_buffer('anchors', a)  # shape(nl,na,2)
            self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
            self.m = nn.ModuleList(nn.Conv2d(x, self.no_det * self.na, 1) for x in ch)  # output conv

            self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
            self.im = nn.ModuleList(ImplicitM(self.no_det * self.na) for _ in ch)

            if self.nkpt is not None:
                if self.dw_conv_kpt:  # keypoint head is slightly more complex
                    self.m_kpt = nn.ModuleList(
                        nn.Sequential(DWConv_YOLOV7(x, x, k=3), Conv(x, x),
                                      DWConv_YOLOV7(x, x, k=3), Conv(x, x),
                                      DWConv_YOLOV7(x, x, k=3), Conv(x, x),
                                      DWConv_YOLOV7(x, x, k=3), Conv(x, x),
                                      DWConv_YOLOV7(x, x, k=3), Conv(x, x),
                                      DWConv_YOLOV7(x, x, k=3), nn.Conv2d(x, self.no_kpt * self.na, 1)) for x in ch)
                else:  # keypoint head is a single convolution
                    self.m_kpt = nn.ModuleList(nn.Conv2d(x, self.no_kpt * self.na, 1) for x in ch)

            self.inplace = inplace  # use in-place ops (e.g. slice assignment)

        def forward(self, x):
            # x = x.copy()  # for profiling
            z = []  # inference output
            self.training |= self.export
            for i in range(self.nl):
                if self.nkpt is None or self.nkpt == 0:
                    x[i] = self.im[i](self.m[i](self.ia[i](x[i])))  # conv
                else:
                    x[i] = torch.cat((self.im[i](self.m[i](self.ia[i](x[i]))), self.m_kpt[i](x[i])), axis=1)

                bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
                x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
                x_det = x[i][..., :6]
                x_kpt = x[i][..., 6:]

                if not self.training:  # inference
                    if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                        self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                    kpt_grid_x = self.grid[i][..., 0:1]
                    kpt_grid_y = self.grid[i][..., 1:2]

                    if self.nkpt == 0:
                        y = x[i].sigmoid()
                    else:
                        y = x_det.sigmoid()

                    if self.inplace:
                        xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                        wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2)  # wh
                        if self.nkpt != 0:
                            x_kpt[..., 0::3] = (x_kpt[..., ::3] * 2. - 0.5 + kpt_grid_x.repeat(1, 1, 1, 1, 17)) * \
                                               self.stride[i]  # xy
                            x_kpt[..., 1::3] = (x_kpt[..., 1::3] * 2. - 0.5 + kpt_grid_y.repeat(1, 1, 1, 1, 17)) * \
                                               self.stride[i]  # xy
                            # x_kpt[..., 0::3] = (x_kpt[..., ::3] + kpt_grid_x.repeat(1,1,1,1,17)) * self.stride[i]  # xy
                            # x_kpt[..., 1::3] = (x_kpt[..., 1::3] + kpt_grid_y.repeat(1,1,1,1,17)) * self.stride[i]  # xy
                            # print('=============')
                            # print(self.anchor_grid[i].shape)
                            # print(self.anchor_grid[i][...,0].unsqueeze(4).shape)
                            # print(x_kpt[..., 0::3].shape)
                            # x_kpt[..., 0::3] = ((x_kpt[..., 0::3].tanh() * 2.) ** 3 * self.anchor_grid[i][...,0].unsqueeze(4).repeat(1,1,1,1,self.nkpt)) + kpt_grid_x.repeat(1,1,1,1,17) * self.stride[i]  # xy
                            # x_kpt[..., 1::3] = ((x_kpt[..., 1::3].tanh() * 2.) ** 3 * self.anchor_grid[i][...,1].unsqueeze(4).repeat(1,1,1,1,self.nkpt)) + kpt_grid_y.repeat(1,1,1,1,17) * self.stride[i]  # xy
                            # x_kpt[..., 0::3] = (((x_kpt[..., 0::3].sigmoid() * 4.) ** 2 - 8.) * self.anchor_grid[i][...,0].unsqueeze(4).repeat(1,1,1,1,self.nkpt)) + kpt_grid_x.repeat(1,1,1,1,17) * self.stride[i]  # xy
                            # x_kpt[..., 1::3] = (((x_kpt[..., 1::3].sigmoid() * 4.) ** 2 - 8.) * self.anchor_grid[i][...,1].unsqueeze(4).repeat(1,1,1,1,self.nkpt)) + kpt_grid_y.repeat(1,1,1,1,17) * self.stride[i]  # xy
                            x_kpt[..., 2::3] = x_kpt[..., 2::3].sigmoid()

                        y = torch.cat((xy, wh, y[..., 4:], x_kpt), dim=-1)

                    else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                        xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                        wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                        if self.nkpt != 0:
                            y[..., 6:] = (y[..., 6:] * 2. - 0.5 + self.grid[i].repeat((1, 1, 1, 1, self.nkpt))) * \
                                         self.stride[i]  # xy
                        y = torch.cat((xy, wh, y[..., 4:]), -1)

                    z.append(y.view(bs, -1, self.no))

            return x if self.training else (torch.cat(z, 1), x)

        @staticmethod
        def _make_grid(nx=20, ny=20):
            yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
            return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
    class IAuxDetect(nn.Module):
        stride = None  # strides computed during build
        export = False  # onnx export
        end2end = False
        include_nms = False
        concat = False

        def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
            super(IAuxDetect, self).__init__()
            self.nc = nc  # number of classes
            self.no = nc + 5  # number of outputs per anchor
            self.nl = len(anchors)  # number of detection layers
            self.na = len(anchors[0]) // 2  # number of anchors
            self.grid = [torch.zeros(1)] * self.nl  # init grid
            a = torch.tensor(anchors).float().view(self.nl, -1, 2)
            self.register_buffer('anchors', a)  # shape(nl,na,2)
            self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
            self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch[:self.nl])  # output conv
            self.m2 = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch[self.nl:])  # output conv

            self.ia = nn.ModuleList(ImplicitA(x) for x in ch[:self.nl])
            self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch[:self.nl])

        def forward(self, x):
            # x = x.copy()  # for profiling
            z = []  # inference output
            self.training |= self.export
            for i in range(self.nl):
                x[i] = self.m[i](self.ia[i](x[i]))  # conv
                x[i] = self.im[i](x[i])
                bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
                x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

                x[i + self.nl] = self.m2[i](x[i + self.nl])
                x[i + self.nl] = x[i + self.nl].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

                if not self.training:  # inference
                    if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                        self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                    y = x[i].sigmoid()
                    if not torch.onnx.is_in_onnx_export():
                        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    else:
                        xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                        xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))  # new xy
                        wh = wh ** 2 * (4 * self.anchor_grid[i].data)  # new wh
                        y = torch.cat((xy, wh, conf), 4)
                    z.append(y.view(bs, -1, self.no))

            return x if self.training else (torch.cat(z, 1), x[:self.nl])

        def fuseforward(self, x):
            # x = x.copy()  # for profiling
            z = []  # inference output
            self.training |= self.export
            for i in range(self.nl):
                x[i] = self.m[i](x[i])  # conv
                bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
                x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

                if not self.training:  # inference
                    if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                        self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                    y = x[i].sigmoid()
                    if not torch.onnx.is_in_onnx_export():
                        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    else:
                        xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                        wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].data  # wh
                        y = torch.cat((xy, wh, y[..., 4:]), -1)
                    z.append(y.view(bs, -1, self.no))

            if self.training:
                out = x
            elif self.end2end:
                out = torch.cat(z, 1)
            elif self.include_nms:
                z = self.convert(z)
                out = (z,)
            elif self.concat:
                out = torch.cat(z, 1)
            else:
                out = (torch.cat(z, 1), x)

            return out

        def fuse(self):
            print("IAuxDetect.fuse")
            # fuse ImplicitA and Convolution
            for i in range(len(self.m)):
                c1, c2, _, _ = self.m[i].weight.shape
                c1_, c2_, _, _ = self.ia[i].implicit.shape
                self.m[i].bias += torch.matmul(self.m[i].weight.reshape(c1, c2),
                                               self.ia[i].implicit.reshape(c2_, c1_)).squeeze(1)

            # fuse ImplicitM and Convolution
            for i in range(len(self.m)):
                c1, c2, _, _ = self.im[i].implicit.shape
                self.m[i].bias *= self.im[i].implicit.reshape(c2)
                self.m[i].weight *= self.im[i].implicit.transpose(0, 1)

        @staticmethod
        def _make_grid(nx=20, ny=20):
            yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
            return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

        def convert(self, z):
            z = torch.cat(z, 1)
            box = z[:, :, :4]
            conf = z[:, :, 4:5]
            score = z[:, :, 5:]
            score *= conf
            convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                          dtype=torch.float32,
                                          device=z.device)
            box @= convert_matrix
            return (box, score)
    class IBin(nn.Module):
        stride = None  # strides computed during build
        export = False  # onnx export

        def __init__(self, nc=80, anchors=(), ch=(), bin_count=21):  # detection layer
            super(IBin, self).__init__()
            self.nc = nc  # number of classes
            self.bin_count = bin_count

            self.w_bin_sigmoid = SigmoidBin(bin_count=self.bin_count, min=0.0, max=4.0)
            self.h_bin_sigmoid = SigmoidBin(bin_count=self.bin_count, min=0.0, max=4.0)
            # classes, x,y,obj
            self.no = nc + 3 + \
                      self.w_bin_sigmoid.get_length() + self.h_bin_sigmoid.get_length()  # w-bce, h-bce
            # + self.x_bin_sigmoid.get_length() + self.y_bin_sigmoid.get_length()

            self.nl = len(anchors)  # number of detection layers
            self.na = len(anchors[0]) // 2  # number of anchors
            self.grid = [torch.zeros(1)] * self.nl  # init grid
            a = torch.tensor(anchors).float().view(self.nl, -1, 2)
            self.register_buffer('anchors', a)  # shape(nl,na,2)
            self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
            self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

            self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
            self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch)

        def forward(self, x):

            # self.x_bin_sigmoid.use_fw_regression = True
            # self.y_bin_sigmoid.use_fw_regression = True
            self.w_bin_sigmoid.use_fw_regression = True
            self.h_bin_sigmoid.use_fw_regression = True

            # x = x.copy()  # for profiling
            z = []  # inference output
            self.training |= self.export
            for i in range(self.nl):
                x[i] = self.m[i](self.ia[i](x[i]))  # conv
                x[i] = self.im[i](x[i])
                bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
                x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

                if not self.training:  # inference
                    if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                        self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                    y = x[i].sigmoid()
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    # y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                    # px = (self.x_bin_sigmoid.forward(y[..., 0:12]) + self.grid[i][..., 0]) * self.stride[i]
                    # py = (self.y_bin_sigmoid.forward(y[..., 12:24]) + self.grid[i][..., 1]) * self.stride[i]

                    pw = self.w_bin_sigmoid.forward(y[..., 2:24]) * self.anchor_grid[i][..., 0]
                    ph = self.h_bin_sigmoid.forward(y[..., 24:46]) * self.anchor_grid[i][..., 1]

                    # y[..., 0] = px
                    # y[..., 1] = py
                    y[..., 2] = pw
                    y[..., 3] = ph

                    y = torch.cat((y[..., 0:4], y[..., 46:]), dim=-1)

                    z.append(y.view(bs, -1, y.shape[-1]))

            return x if self.training else (torch.cat(z, 1), x)

        @staticmethod
        def _make_grid(nx=20, ny=20):
            yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
            return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
    class Model(nn.Module):
        def __init__(self, cfg='yolor-csp-c.yaml', ch=3, nc=None,
                     anchors=None):  # model, input channels, number of classes
            super(Model, self).__init__()
            self.traced = False
            if isinstance(cfg, dict):
                self.yaml = cfg  # model dict
            else:  # is *.yaml
                import yaml  # for torch hub
                self.yaml_file = Path(cfg).name
                with open(cfg) as f:
                    self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict

            # Define model
            ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
            if nc and nc != self.yaml['nc']:
                logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
                self.yaml['nc'] = nc  # override yaml value
            if anchors:
                logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
                self.yaml['anchors'] = round(anchors)  # override yaml value
            self.model, self.save = parse_model_YOLOV7(deepcopy(self.yaml), ch=[ch])  # model, savelist
            self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
            # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

            # Build strides, anchors
            m = self.model[-1]  # Detect()
            if isinstance(m, Detect_YOLOV7):
                s = 256  # 2x min stride
                m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
                check_anchor_order(m)
                m.anchors /= m.stride.view(-1, 1, 1)
                self.stride = m.stride
                self._initialize_biases()  # only run once
                # print('Strides: %s' % m.stride.tolist())
            if isinstance(m, IDetect):
                s = 256  # 2x min stride
                m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
                check_anchor_order(m)
                m.anchors /= m.stride.view(-1, 1, 1)
                self.stride = m.stride
                self._initialize_biases()  # only run once
                # print('Strides: %s' % m.stride.tolist())
            if isinstance(m, IAuxDetect):
                s = 256  # 2x min stride
                m.stride = torch.tensor(
                    [s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))[:4]])  # forward
                # print(m.stride)
                check_anchor_order(m)
                m.anchors /= m.stride.view(-1, 1, 1)
                self.stride = m.stride
                self._initialize_aux_biases()  # only run once
                # print('Strides: %s' % m.stride.tolist())
            if isinstance(m, IBin):
                s = 256  # 2x min stride
                m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
                check_anchor_order(m)
                m.anchors /= m.stride.view(-1, 1, 1)
                self.stride = m.stride
                self._initialize_biases_bin()  # only run once
                # print('Strides: %s' % m.stride.tolist())
            if isinstance(m, IKeypoint):
                s = 256  # 2x min stride
                m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
                check_anchor_order(m)
                m.anchors /= m.stride.view(-1, 1, 1)
                self.stride = m.stride
                self._initialize_biases_kpt()  # only run once
                # print('Strides: %s' % m.stride.tolist())

            # Init weights, biases
            initialize_weights(self)
            self.info()
            logger.info('')

        def forward(self, x, augment=False, profile=False):
            if augment:
                img_size = x.shape[-2:]  # height, width
                s = [1, 0.83, 0.67]  # scales
                f = [None, 3, None]  # flips (2-ud, 3-lr)
                y = []  # outputs
                for si, fi in zip(s, f):
                    xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                    yi = self.forward_once(xi)[0]  # forward
                    # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                    yi[..., :4] /= si  # de-scale
                    if fi == 2:
                        yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                    elif fi == 3:
                        yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                    y.append(yi)
                return torch.cat(y, 1), None  # augmented inference, train
            else:
                return self.forward_once(x, profile)  # single-scale inference, train

        def forward_once(self, x, profile=False):
            y, dt = [], []  # outputs
            for m in self.model:
                if m.f != -1:  # if not from previous layer
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

                if not hasattr(self, 'traced'):
                    self.traced = False

                if self.traced:
                    if isinstance(m, Detect) or isinstance(m, IDetect) or isinstance(m, IAuxDetect) or isinstance(m,
                                                                                                                  IKeypoint):
                        break

                if profile:
                    c = isinstance(m, (Detect, IDetect, IAuxDetect, IBin))
                    o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[
                            0] / 1E9 * 2 if thop else 0  # FLOPS
                    for _ in range(10):
                        m(x.copy() if c else x)
                    t = time_synchronized()
                    for _ in range(10):
                        m(x.copy() if c else x)
                    dt.append((time_synchronized() - t) * 100)
                    print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

                x = m(x)  # run

                y.append(x if m.i in self.save else None)  # save output

            if profile:
                print('%.1fms total' % sum(dt))
            return x

        def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
            # https://arxiv.org/abs/1708.02002 section 3.3
            # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
            m = self.model[-1]  # Detect() module
            for mi, s in zip(m.m, m.stride):  # from
                b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
                b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
                b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
                mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        def _initialize_aux_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
            # https://arxiv.org/abs/1708.02002 section 3.3
            # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
            m = self.model[-1]  # Detect() module
            for mi, mi2, s in zip(m.m, m.m2, m.stride):  # from
                b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
                b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
                b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
                mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
                b2 = mi2.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
                b2.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
                b2.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
                mi2.bias = torch.nn.Parameter(b2.view(-1), requires_grad=True)

        def _initialize_biases_bin(self, cf=None):  # initialize biases into Detect(), cf is class frequency
            # https://arxiv.org/abs/1708.02002 section 3.3
            # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
            m = self.model[-1]  # Bin() module
            bc = m.bin_count
            for mi, s in zip(m.m, m.stride):  # from
                b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
                old = b[:, (0, 1, 2, bc + 3)].data
                obj_idx = 2 * bc + 4
                b[:, :obj_idx].data += math.log(0.6 / (bc + 1 - 0.99))
                b[:, obj_idx].data += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
                b[:, (obj_idx + 1):].data += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(
                    cf / cf.sum())  # cls
                b[:, (0, 1, 2, bc + 3)].data = old
                mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        def _initialize_biases_kpt(self, cf=None):  # initialize biases into Detect(), cf is class frequency
            # https://arxiv.org/abs/1708.02002 section 3.3
            # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
            m = self.model[-1]  # Detect() module
            for mi, s in zip(m.m, m.stride):  # from
                b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
                b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
                b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
                mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        def _print_biases(self):
            m = self.model[-1]  # Detect() module
            for mi in m.m:  # from
                b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
                print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

        # def _print_weights(self):
        #     for m in self.model.modules():
        #         if type(m) is Bottleneck:
        #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

        def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
            print('Fusing layers... ')
            for m in self.model.modules():
                if isinstance(m, RepConv):
                    # print(f" fuse_repvgg_block")
                    m.fuse_repvgg_block()
                elif isinstance(m, RepConv_OREPA):
                    # print(f" switch_to_deploy")
                    m.switch_to_deploy()
                elif type(m) is Conv and hasattr(m, 'bn'):
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, 'bn')  # remove batchnorm
                    m.forward = m.fuseforward  # update forward
                elif isinstance(m, (IDetect, IAuxDetect)):
                    m.fuse()
                    m.forward = m.fuseforward
            self.info()
            return self

        def nms(self, mode=True):  # add or remove NMS module
            present = type(self.model[-1]) is NMS  # last layer is NMS
            if mode and not present:
                print('Adding NMS... ')
                m = NMS()  # module
                m.f = -1  # from
                m.i = self.model[-1].i + 1  # index
                self.model.add_module(name='%s' % m.i, module=m)  # add
                self.eval()
            elif not mode and present:
                print('Removing NMS... ')
                self.model = self.model[:-1]  # remove
            return self

        def autoshape(self):  # add autoShape module
            print('Adding autoShape... ')
            m = autoShape(self)  # wrap model
            copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
            return m

        def info(self, verbose=False, img_size=640):  # print model information
            model_info(self, verbose, img_size)
    def parse_model_YOLOV7(d, ch):  # model_dict, input_channels(3)
        logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
        anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
        na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
        no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

        layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
        for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
            m = eval(m) if isinstance(m, str) else m  # eval strings
            for j, a in enumerate(args):
                try:
                    args[j] = eval(a) if isinstance(a, str) else a  # eval strings
                except:
                    pass

            n = max(round(n * gd), 1) if n > 1 else n  # depth gain
            if m in [nn.Conv2d, Conv, RobustConv, RobustConv2, DWConv_YOLOV7, GhostConv, RepConv, RepConv_OREPA, DownC,
                     SPP, SPPF, SPPCSPC, GhostSPPCSPC, MixConv2d, Focus, Stem, GhostStem, CrossConv,
                     Bottleneck, BottleneckCSPA, BottleneckCSPB, BottleneckCSPC,
                     RepBottleneck, RepBottleneckCSPA, RepBottleneckCSPB, RepBottleneckCSPC,
                     Res, ResCSPA, ResCSPB, ResCSPC,
                     RepRes, RepResCSPA, RepResCSPB, RepResCSPC,
                     ResX, ResXCSPA, ResXCSPB, ResXCSPC,
                     RepResX, RepResXCSPA, RepResXCSPB, RepResXCSPC,
                     Ghost, GhostCSPA, GhostCSPB, GhostCSPC,
                     SwinTransformerBlock, STCSPA, STCSPB, STCSPC,
                     SwinTransformer2Block, ST2CSPA, ST2CSPB, ST2CSPC]:
                c1, c2 = ch[f], args[0]
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8)

                args = [c1, c2, *args[1:]]
                if m in [DownC, SPPCSPC, GhostSPPCSPC,
                         BottleneckCSPA, BottleneckCSPB, BottleneckCSPC,
                         RepBottleneckCSPA, RepBottleneckCSPB, RepBottleneckCSPC,
                         ResCSPA, ResCSPB, ResCSPC,
                         RepResCSPA, RepResCSPB, RepResCSPC,
                         ResXCSPA, ResXCSPB, ResXCSPC,
                         RepResXCSPA, RepResXCSPB, RepResXCSPC,
                         GhostCSPA, GhostCSPB, GhostCSPC,
                         STCSPA, STCSPB, STCSPC,
                         ST2CSPA, ST2CSPB, ST2CSPC]:
                    args.insert(2, n)  # number of repeats
                    n = 1
            elif m is nn.BatchNorm2d:
                args = [ch[f]]
            elif m is Concat:
                c2 = sum([ch[x] for x in f])
            elif m is Chuncat:
                c2 = sum([ch[x] for x in f])
            elif m is Shortcut:
                c2 = ch[f[0]]
            elif m is Foldcut:
                c2 = ch[f] // 2
            elif m in [Detect, IDetect, IAuxDetect, IBin, IKeypoint]:
                args.append([ch[x] for x in f])
                if isinstance(args[1], int):  # number of anchors
                    args[1] = [list(range(args[1] * 2))] * len(f)
            elif m is ReOrg:
                c2 = ch[f] * 4
            elif m is Contract:
                c2 = ch[f] * args[0] ** 2
            elif m is Expand:
                c2 = ch[f] // args[0] ** 2
            else:
                c2 = ch[f]

            m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
            t = str(m)[8:-2].replace('__main__.', '')  # module type
            np = sum([x.numel() for x in m_.parameters()])  # number params
            m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
            logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
            layers.append(m_)
            if i == 0:
                ch = []
            ch.append(c2)
        return nn.Sequential(*layers), sorted(save)
if "yolov9" in yolo_name:
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[1]  # YOLO root directory
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add ROOT to PATH
    if platform.system() != 'Windows':
        ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

    from yolocode.yolov9.models.common import *
    from yolocode.yolov9.models.experimental import *
    from yolocode.yolov9.utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
    from yolocode.yolov9.utils.plots import feature_visualization
    from yolocode.yolov9.utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img,
                                                   select_device,
                                                   time_sync)
    from yolocode.yolov9.utils.tal.anchor_generator import make_anchors, dist2bbox

    class Detect_YOLOV9(nn.Module):
        # YOLO Detect head for detection models
        dynamic = False  # force grid reconstruction
        export = False  # export mode
        shape = None
        anchors = torch.empty(0)  # init
        strides = torch.empty(0)  # init

        def __init__(self, nc=80, ch=(), inplace=True):  # detection layer
            super().__init__()
            self.nc = nc  # number of classes
            self.nl = len(ch)  # number of detection layers
            self.reg_max = 16
            self.no = nc + self.reg_max * 4  # number of outputs per anchor
            self.inplace = inplace  # use inplace ops (e.g. slice assignment)
            self.stride = torch.zeros(self.nl)  # strides computed during build

            c2, c3 = max((ch[0] // 4, self.reg_max * 4, 16)), max((ch[0], min((self.nc * 2, 128))))  # channels
            self.cv2 = nn.ModuleList(
                nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
            self.cv3 = nn.ModuleList(
                nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        def forward(self, x):
            shape = x[0].shape  # BCHW
            for i in range(self.nl):
                x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
            if self.training:
                return x
            elif self.dynamic or self.shape != shape:
                self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
                self.shape = shape

            box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.nc), 1)
            dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
            y = torch.cat((dbox, cls.sigmoid()), 1)
            return y if self.export else (y, x)

        def bias_init(self):
            # Initialize Detect() biases, WARNING: requires stride availability
            m = self  # self.model[-1]  # Detect() module
            # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
            # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
            for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[:m.nc] = math.log(
                    5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)
    class DDetect(nn.Module):
        # YOLO Detect head for detection models
        dynamic = False  # force grid reconstruction
        export = False  # export mode
        shape = None
        anchors = torch.empty(0)  # init
        strides = torch.empty(0)  # init

        def __init__(self, nc=80, ch=(), inplace=True):  # detection layer
            super().__init__()
            self.nc = nc  # number of classes
            self.nl = len(ch)  # number of detection layers
            self.reg_max = 16
            self.no = nc + self.reg_max * 4  # number of outputs per anchor
            self.inplace = inplace  # use inplace ops (e.g. slice assignment)
            self.stride = torch.zeros(self.nl)  # strides computed during build

            c2, c3 = make_divisible(max((ch[0] // 4, self.reg_max * 4, 16)), 4), max(
                (ch[0], min((self.nc * 2, 128))))  # channels
            self.cv2 = nn.ModuleList(
                nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3, g=4), nn.Conv2d(c2, 4 * self.reg_max, 1, groups=4)) for x
                in
                ch)
            self.cv3 = nn.ModuleList(
                nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        def forward(self, x):
            shape = x[0].shape  # BCHW
            for i in range(self.nl):
                x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
            if self.training:
                return x
            elif self.dynamic or self.shape != shape:
                self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
                self.shape = shape

            box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.nc), 1)
            dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
            y = torch.cat((dbox, cls.sigmoid()), 1)
            return y if self.export else (y, x)

        def bias_init(self):
            # Initialize Detect() biases, WARNING: requires stride availability
            m = self  # self.model[-1]  # Detect() module
            # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
            # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
            for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[:m.nc] = math.log(
                    5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)
    class DualDetect(nn.Module):
        # YOLO Detect head for detection models
        dynamic = False  # force grid reconstruction
        export = False  # export mode
        shape = None
        anchors = torch.empty(0)  # init
        strides = torch.empty(0)  # init

        def __init__(self, nc=80, ch=(), inplace=True):  # detection layer
            super().__init__()
            self.nc = nc  # number of classes
            self.nl = len(ch) // 2  # number of detection layers
            self.reg_max = 16
            self.no = nc + self.reg_max * 4  # number of outputs per anchor
            self.inplace = inplace  # use inplace ops (e.g. slice assignment)
            self.stride = torch.zeros(self.nl)  # strides computed during build

            c2, c3 = max((ch[0] // 4, self.reg_max * 4, 16)), max((ch[0], min((self.nc * 2, 128))))  # channels
            c4, c5 = max((ch[self.nl] // 4, self.reg_max * 4, 16)), max(
                (ch[self.nl], min((self.nc * 2, 128))))  # channels
            self.cv2 = nn.ModuleList(
                nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in
                ch[:self.nl])
            self.cv3 = nn.ModuleList(
                nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch[:self.nl])
            self.cv4 = nn.ModuleList(
                nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, 4 * self.reg_max, 1)) for x in
                ch[self.nl:])
            self.cv5 = nn.ModuleList(
                nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nc, 1)) for x in ch[self.nl:])
            self.dfl = DFL(self.reg_max)
            self.dfl2 = DFL(self.reg_max)

        def forward(self, x):
            shape = x[0].shape  # BCHW
            d1 = []
            d2 = []
            for i in range(self.nl):
                d1.append(torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1))
                d2.append(torch.cat((self.cv4[i](x[self.nl + i]), self.cv5[i](x[self.nl + i])), 1))
            if self.training:
                return [d1, d2]
            elif self.dynamic or self.shape != shape:
                self.anchors, self.strides = (d1.transpose(0, 1) for d1 in make_anchors(d1, self.stride, 0.5))
                self.shape = shape

            box, cls = torch.cat([di.view(shape[0], self.no, -1) for di in d1], 2).split((self.reg_max * 4, self.nc), 1)
            dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
            box2, cls2 = torch.cat([di.view(shape[0], self.no, -1) for di in d2], 2).split((self.reg_max * 4, self.nc),
                                                                                           1)
            dbox2 = dist2bbox(self.dfl2(box2), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
            y = [torch.cat((dbox, cls.sigmoid()), 1), torch.cat((dbox2, cls2.sigmoid()), 1)]
            return y if self.export else (y, [d1, d2])

        def bias_init(self):
            # Initialize Detect() biases, WARNING: requires stride availability
            m = self  # self.model[-1]  # Detect() module
            # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
            # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
            for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[:m.nc] = math.log(
                    5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)
            for a, b, s in zip(m.cv4, m.cv5, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[:m.nc] = math.log(
                    5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)
    class DualDDetect(nn.Module):
        # YOLO Detect head for detection models
        dynamic = False  # force grid reconstruction
        export = False  # export mode
        shape = None
        anchors = torch.empty(0)  # init
        strides = torch.empty(0)  # init

        def __init__(self, nc=80, ch=(), inplace=True):  # detection layer
            super().__init__()
            self.nc = nc  # number of classes
            self.nl = len(ch) // 2  # number of detection layers
            self.reg_max = 16
            self.no = nc + self.reg_max * 4  # number of outputs per anchor
            self.inplace = inplace  # use inplace ops (e.g. slice assignment)
            self.stride = torch.zeros(self.nl)  # strides computed during build

            c2, c3 = make_divisible(max((ch[0] // 4, self.reg_max * 4, 16)), 4), max(
                (ch[0], min((self.nc * 2, 128))))  # channels
            c4, c5 = make_divisible(max((ch[self.nl] // 4, self.reg_max * 4, 16)), 4), max(
                (ch[self.nl], min((self.nc * 2, 128))))  # channels
            self.cv2 = nn.ModuleList(
                nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3, g=4), nn.Conv2d(c2, 4 * self.reg_max, 1, groups=4)) for x
                in
                ch[:self.nl])
            self.cv3 = nn.ModuleList(
                nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch[:self.nl])
            self.cv4 = nn.ModuleList(
                nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3, g=4), nn.Conv2d(c4, 4 * self.reg_max, 1, groups=4)) for x
                in
                ch[self.nl:])
            self.cv5 = nn.ModuleList(
                nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nc, 1)) for x in ch[self.nl:])
            self.dfl = DFL(self.reg_max)
            self.dfl2 = DFL(self.reg_max)

        def forward(self, x):
            shape = x[0].shape  # BCHW
            d1 = []
            d2 = []
            for i in range(self.nl):
                d1.append(torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1))
                d2.append(torch.cat((self.cv4[i](x[self.nl + i]), self.cv5[i](x[self.nl + i])), 1))
            if self.training:
                return [d1, d2]
            elif self.dynamic or self.shape != shape:
                self.anchors, self.strides = (d1.transpose(0, 1) for d1 in make_anchors(d1, self.stride, 0.5))
                self.shape = shape

            box, cls = torch.cat([di.view(shape[0], self.no, -1) for di in d1], 2).split((self.reg_max * 4, self.nc), 1)
            dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
            box2, cls2 = torch.cat([di.view(shape[0], self.no, -1) for di in d2], 2).split((self.reg_max * 4, self.nc),
                                                                                           1)
            dbox2 = dist2bbox(self.dfl2(box2), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
            y = [torch.cat((dbox, cls.sigmoid()), 1), torch.cat((dbox2, cls2.sigmoid()), 1)]
            return y if self.export else (y, [d1, d2])
            # y = torch.cat((dbox2, cls2.sigmoid()), 1)
            # return y if self.export else (y, d2)
            # y1 = torch.cat((dbox, cls.sigmoid()), 1)
            # y2 = torch.cat((dbox2, cls2.sigmoid()), 1)
            # return [y1, y2] if self.export else [(y1, d1), (y2, d2)]
            # return [y1, y2] if self.export else [(y1, y2), (d1, d2)]

        def bias_init(self):
            # Initialize Detect() biases, WARNING: requires stride availability
            m = self  # self.model[-1]  # Detect() module
            # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
            # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
            for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[:m.nc] = math.log(
                    5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)
            for a, b, s in zip(m.cv4, m.cv5, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[:m.nc] = math.log(
                    5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)
    class TripleDetect(nn.Module):
        # YOLO Detect head for detection models
        dynamic = False  # force grid reconstruction
        export = False  # export mode
        shape = None
        anchors = torch.empty(0)  # init
        strides = torch.empty(0)  # init

        def __init__(self, nc=80, ch=(), inplace=True):  # detection layer
            super().__init__()
            self.nc = nc  # number of classes
            self.nl = len(ch) // 3  # number of detection layers
            self.reg_max = 16
            self.no = nc + self.reg_max * 4  # number of outputs per anchor
            self.inplace = inplace  # use inplace ops (e.g. slice assignment)
            self.stride = torch.zeros(self.nl)  # strides computed during build

            c2, c3 = max((ch[0] // 4, self.reg_max * 4, 16)), max((ch[0], min((self.nc * 2, 128))))  # channels
            c4, c5 = max((ch[self.nl] // 4, self.reg_max * 4, 16)), max(
                (ch[self.nl], min((self.nc * 2, 128))))  # channels
            c6, c7 = max((ch[self.nl * 2] // 4, self.reg_max * 4, 16)), max(
                (ch[self.nl * 2], min((self.nc * 2, 128))))  # channels
            self.cv2 = nn.ModuleList(
                nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in
                ch[:self.nl])
            self.cv3 = nn.ModuleList(
                nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch[:self.nl])
            self.cv4 = nn.ModuleList(
                nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, 4 * self.reg_max, 1)) for x in
                ch[self.nl:self.nl * 2])
            self.cv5 = nn.ModuleList(
                nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nc, 1)) for x in
                ch[self.nl:self.nl * 2])
            self.cv6 = nn.ModuleList(
                nn.Sequential(Conv(x, c6, 3), Conv(c6, c6, 3), nn.Conv2d(c6, 4 * self.reg_max, 1)) for x in
                ch[self.nl * 2:self.nl * 3])
            self.cv7 = nn.ModuleList(
                nn.Sequential(Conv(x, c7, 3), Conv(c7, c7, 3), nn.Conv2d(c7, self.nc, 1)) for x in
                ch[self.nl * 2:self.nl * 3])
            self.dfl = DFL(self.reg_max)
            self.dfl2 = DFL(self.reg_max)
            self.dfl3 = DFL(self.reg_max)

        def forward(self, x):
            shape = x[0].shape  # BCHW
            d1 = []
            d2 = []
            d3 = []
            for i in range(self.nl):
                d1.append(torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1))
                d2.append(torch.cat((self.cv4[i](x[self.nl + i]), self.cv5[i](x[self.nl + i])), 1))
                d3.append(torch.cat((self.cv6[i](x[self.nl * 2 + i]), self.cv7[i](x[self.nl * 2 + i])), 1))
            if self.training:
                return [d1, d2, d3]
            elif self.dynamic or self.shape != shape:
                self.anchors, self.strides = (d1.transpose(0, 1) for d1 in make_anchors(d1, self.stride, 0.5))
                self.shape = shape

            box, cls = torch.cat([di.view(shape[0], self.no, -1) for di in d1], 2).split((self.reg_max * 4, self.nc), 1)
            dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
            box2, cls2 = torch.cat([di.view(shape[0], self.no, -1) for di in d2], 2).split((self.reg_max * 4, self.nc),
                                                                                           1)
            dbox2 = dist2bbox(self.dfl2(box2), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
            box3, cls3 = torch.cat([di.view(shape[0], self.no, -1) for di in d3], 2).split((self.reg_max * 4, self.nc),
                                                                                           1)
            dbox3 = dist2bbox(self.dfl3(box3), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
            y = [torch.cat((dbox, cls.sigmoid()), 1), torch.cat((dbox2, cls2.sigmoid()), 1),
                 torch.cat((dbox3, cls3.sigmoid()), 1)]
            return y if self.export else (y, [d1, d2, d3])

        def bias_init(self):
            # Initialize Detect() biases, WARNING: requires stride availability
            m = self  # self.model[-1]  # Detect() module
            # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
            # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
            for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[:m.nc] = math.log(
                    5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)
            for a, b, s in zip(m.cv4, m.cv5, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[:m.nc] = math.log(
                    5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)
            for a, b, s in zip(m.cv6, m.cv7, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[:m.nc] = math.log(
                    5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)
    class TripleDDetect(nn.Module):
        # YOLO Detect head for detection models
        dynamic = False  # force grid reconstruction
        export = False  # export mode
        shape = None
        anchors = torch.empty(0)  # init
        strides = torch.empty(0)  # init

        def __init__(self, nc=80, ch=(), inplace=True):  # detection layer
            super().__init__()
            self.nc = nc  # number of classes
            self.nl = len(ch) // 3  # number of detection layers
            self.reg_max = 16
            self.no = nc + self.reg_max * 4  # number of outputs per anchor
            self.inplace = inplace  # use inplace ops (e.g. slice assignment)
            self.stride = torch.zeros(self.nl)  # strides computed during build

            c2, c3 = make_divisible(max((ch[0] // 4, self.reg_max * 4, 16)), 4), \
                max((ch[0], min((self.nc * 2, 128))))  # channels
            c4, c5 = make_divisible(max((ch[self.nl] // 4, self.reg_max * 4, 16)), 4), \
                max((ch[self.nl], min((self.nc * 2, 128))))  # channels
            c6, c7 = make_divisible(max((ch[self.nl * 2] // 4, self.reg_max * 4, 16)), 4), \
                max((ch[self.nl * 2], min((self.nc * 2, 128))))  # channels
            self.cv2 = nn.ModuleList(
                nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3, g=4),
                              nn.Conv2d(c2, 4 * self.reg_max, 1, groups=4)) for x in ch[:self.nl])
            self.cv3 = nn.ModuleList(
                nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch[:self.nl])
            self.cv4 = nn.ModuleList(
                nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3, g=4),
                              nn.Conv2d(c4, 4 * self.reg_max, 1, groups=4)) for x in ch[self.nl:self.nl * 2])
            self.cv5 = nn.ModuleList(
                nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nc, 1)) for x in
                ch[self.nl:self.nl * 2])
            self.cv6 = nn.ModuleList(
                nn.Sequential(Conv(x, c6, 3), Conv(c6, c6, 3, g=4),
                              nn.Conv2d(c6, 4 * self.reg_max, 1, groups=4)) for x in ch[self.nl * 2:self.nl * 3])
            self.cv7 = nn.ModuleList(
                nn.Sequential(Conv(x, c7, 3), Conv(c7, c7, 3), nn.Conv2d(c7, self.nc, 1)) for x in
                ch[self.nl * 2:self.nl * 3])
            self.dfl = DFL(self.reg_max)
            self.dfl2 = DFL(self.reg_max)
            self.dfl3 = DFL(self.reg_max)

        def forward(self, x):
            shape = x[0].shape  # BCHW
            d1 = []
            d2 = []
            d3 = []
            for i in range(self.nl):
                d1.append(torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1))
                d2.append(torch.cat((self.cv4[i](x[self.nl + i]), self.cv5[i](x[self.nl + i])), 1))
                d3.append(torch.cat((self.cv6[i](x[self.nl * 2 + i]), self.cv7[i](x[self.nl * 2 + i])), 1))
            if self.training:
                return [d1, d2, d3]
            elif self.dynamic or self.shape != shape:
                self.anchors, self.strides = (d1.transpose(0, 1) for d1 in make_anchors(d1, self.stride, 0.5))
                self.shape = shape

            box, cls = torch.cat([di.view(shape[0], self.no, -1) for di in d1], 2).split((self.reg_max * 4, self.nc), 1)
            dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
            box2, cls2 = torch.cat([di.view(shape[0], self.no, -1) for di in d2], 2).split((self.reg_max * 4, self.nc),
                                                                                           1)
            dbox2 = dist2bbox(self.dfl2(box2), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
            box3, cls3 = torch.cat([di.view(shape[0], self.no, -1) for di in d3], 2).split((self.reg_max * 4, self.nc),
                                                                                           1)
            dbox3 = dist2bbox(self.dfl3(box3), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
            # y = [torch.cat((dbox, cls.sigmoid()), 1), torch.cat((dbox2, cls2.sigmoid()), 1), torch.cat((dbox3, cls3.sigmoid()), 1)]
            # return y if self.export else (y, [d1, d2, d3])
            y = torch.cat((dbox3, cls3.sigmoid()), 1)
            return y if self.export else (y, d3)

        def bias_init(self):
            # Initialize Detect() biases, WARNING: requires stride availability
            m = self  # self.model[-1]  # Detect() module
            # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
            # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
            for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[:m.nc] = math.log(
                    5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)
            for a, b, s in zip(m.cv4, m.cv5, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[:m.nc] = math.log(
                    5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)
            for a, b, s in zip(m.cv6, m.cv7, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[:m.nc] = math.log(
                    5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)
    class Segment_YOLOv9(Detect_YOLOV9):
        # YOLO Segment head for segmentation models
        def __init__(self, nc=80, nm=32, npr=256, ch=(), inplace=True):
            super().__init__(nc, ch, inplace)
            self.nm = nm  # number of masks
            self.npr = npr  # number of protos
            self.proto = Proto(ch[0], self.npr, self.nm)  # protos
            self.detect = Detect_YOLOV9.forward

            c4 = max(ch[0] // 4, self.nm)
            self.cv4 = nn.ModuleList(
                nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

        def forward(self, x):
            p = self.proto(x[0])
            bs = p.shape[0]

            mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
            x = self.detect(self, x)
            if self.training:
                return x, mc, p
            return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))
    class Panoptic(Detect_YOLOV9):
        # YOLO Panoptic head for panoptic segmentation models
        def __init__(self, nc=80, sem_nc=93, nm=32, npr=256, ch=(), inplace=True):
            super().__init__(nc, ch, inplace)
            self.sem_nc = sem_nc
            self.nm = nm  # number of masks
            self.npr = npr  # number of protos
            self.proto = Proto(ch[0], self.npr, self.nm)  # protos
            self.uconv = UConv(ch[0], ch[0] // 4, self.sem_nc + self.nc)
            self.detect = Detect.forward

            c4 = max(ch[0] // 4, self.nm)
            self.cv4 = nn.ModuleList(
                nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

        def forward(self, x):
            p = self.proto(x[0])
            s = self.uconv(x[0])
            bs = p.shape[0]

            mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
            x = self.detect(self, x)
            if self.training:
                return x, mc, p, s
            return (torch.cat([x, mc], 1), p, s) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p, s))
    class Detect(nn.Module):
        # YOLO Detect head for detection models
        dynamic = False  # force grid reconstruction
        export = False  # export mode
        shape = None
        anchors = torch.empty(0)  # init
        strides = torch.empty(0)  # init

        def __init__(self, nc=80, ch=(), inplace=True):  # detection layer
            super().__init__()
            self.nc = nc  # number of classes
            self.nl = len(ch)  # number of detection layers
            self.reg_max = 16
            self.no = nc + self.reg_max * 4  # number of outputs per anchor
            self.inplace = inplace  # use inplace ops (e.g. slice assignment)
            self.stride = torch.zeros(self.nl)  # strides computed during build

            c2, c3 = max((ch[0] // 4, self.reg_max * 4, 16)), max((ch[0], min((self.nc * 2, 128))))  # channels
            self.cv2 = nn.ModuleList(
                nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
            self.cv3 = nn.ModuleList(
                nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        def forward(self, x):
            shape = x[0].shape  # BCHW
            for i in range(self.nl):
                x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
            if self.training:
                return x
            elif self.dynamic or self.shape != shape:
                self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
                self.shape = shape

            box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.nc), 1)
            dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
            y = torch.cat((dbox, cls.sigmoid()), 1)
            return y if self.export else (y, x)

        def bias_init(self):
            # Initialize Detect() biases, WARNING: requires stride availability
            m = self  # self.model[-1]  # Detect() module
            # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
            # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
            for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[:m.nc] = math.log(
                    5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)
    class BaseModel(nn.Module):
        # YOLO base model
        def forward(self, x, profile=False, visualize=False):
            return self._forward_once(x, profile, visualize)  # single-scale inference, train

        def _forward_once(self, x, profile=False, visualize=False):
            y, dt = [], []  # outputs
            for m in self.model:
                if m.f != -1:  # if not from previous layer
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                if profile:
                    self._profile_one_layer(m, x, dt)
                x = m(x)  # run
                y.append(x if m.i in self.save else None)  # save output
                if visualize:
                    feature_visualization(x, m.type, m.i, save_dir=visualize)
            return x

        def _profile_one_layer(self, m, x, dt):
            c = m == self.model[-1]  # is final layer, copy input as inplace fix
            o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
            t = time_sync()
            for _ in range(10):
                m(x.copy() if c else x)
            dt.append((time_sync() - t) * 100)
            if m == self.model[0]:
                LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
            LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
            if c:
                LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

        def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
            LOGGER.info('Fusing layers... ')
            for m in self.model.modules():
                if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, 'bn')  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
            self.info()
            return self

        def info(self, verbose=False, img_size=640):  # print model information
            model_info(self, verbose, img_size)

        def _apply(self, fn):
            # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
            self = super()._apply(fn)
            m = self.model[-1]  # Detect()
            if isinstance(m, (Detect, DualDetect, TripleDetect, DDetect, DualDDetect, TripleDDetect, Segment)):
                m.stride = fn(m.stride)
                m.anchors = fn(m.anchors)
                m.strides = fn(m.strides)
                # m.grid = list(map(fn, m.grid))
            return self
    class DetectionModel(BaseModel):
        # YOLO detection model
        def __init__(self, cfg='yolo.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
            super().__init__()
            if isinstance(cfg, dict):
                self.yaml = cfg  # model dict
            else:  # is *.yaml
                import yaml  # for torch hub
                self.yaml_file = Path(cfg).name
                with open(cfg, encoding='ascii', errors='ignore') as f:
                    self.yaml = yaml.safe_load(f)  # model dict

            # Define model
            ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
            if nc and nc != self.yaml['nc']:
                LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
                self.yaml['nc'] = nc  # override yaml value
            if anchors:
                LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
                self.yaml['anchors'] = round(anchors)  # override yaml value
            self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
            self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
            self.inplace = self.yaml.get('inplace', True)

            # Build strides, anchors
            m = self.model[-1]  # Detect()
            if isinstance(m, (Detect, DDetect, Segment)):
                s = 256  # 2x min stride
                m.inplace = self.inplace
                forward = lambda x: self.forward(x)[0] if isinstance(m, (Segment)) else self.forward(x)
                m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
                # check_anchor_order(m)
                # m.anchors /= m.stride.view(-1, 1, 1)
                self.stride = m.stride
                m.bias_init()  # only run once
            if isinstance(m, (DualDetect, TripleDetect, DualDDetect, TripleDDetect)):
                s = 256  # 2x min stride
                m.inplace = self.inplace
                # forward = lambda x: self.forward(x)[0][0] if isinstance(m, (DualSegment)) else self.forward(x)[0]
                forward = lambda x: self.forward(x)[0]
                m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
                # check_anchor_order(m)
                # m.anchors /= m.stride.view(-1, 1, 1)
                self.stride = m.stride
                m.bias_init()  # only run once

            # Init weights, biases
            initialize_weights(self)
            self.info()
            LOGGER.info('')

        def forward(self, x, augment=False, profile=False, visualize=False):
            if augment:
                return self._forward_augment(x)  # augmented inference, None
            return self._forward_once(x, profile, visualize)  # single-scale inference, train

        def _forward_augment(self, x):
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self._forward_once(xi)[0]  # forward
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi = self._descale_pred(yi, fi, si, img_size)
                y.append(yi)
            y = self._clip_augmented(y)  # clip augmented tails
            return torch.cat(y, 1), None  # augmented inference, train

        def _descale_pred(self, p, flips, scale, img_size):
            # de-scale predictions following augmented inference (inverse operation)
            if self.inplace:
                p[..., :4] /= scale  # de-scale
                if flips == 2:
                    p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
                elif flips == 3:
                    p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
            else:
                x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
                if flips == 2:
                    y = img_size[0] - y  # de-flip ud
                elif flips == 3:
                    x = img_size[1] - x  # de-flip lr
                p = torch.cat((x, y, wh, p[..., 4:]), -1)
            return p

        def _clip_augmented(self, y):
            # Clip YOLO augmented inference tails
            nl = self.model[-1].nl  # number of detection layers (P3-P5)
            g = sum(4 ** x for x in range(nl))  # grid points
            e = 1  # exclude layer count
            i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
            y[0] = y[0][:, :-i]  # large
            i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
            y[-1] = y[-1][:, i:]  # small
            return y
    Model_YOLOV9 = DetectionModel  # retain YOLO 'Model' class for backwards compatibility
    class SegmentationModel(DetectionModel):
        # YOLO segmentation model
        def __init__(self, cfg='yolo-seg.yaml', ch=3, nc=None, anchors=None):
            super().__init__(cfg, ch, nc, anchors)
    class ClassificationModel(BaseModel):
        # YOLO classification model
        def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):  # yaml, model, number of classes, cutoff index
            super().__init__()
            self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

        def _from_detection_model(self, model, nc=1000, cutoff=10):
            # Create a YOLO classification model from a YOLO detection model
            if isinstance(model, DetectMultiBackend):
                model = model.model  # unwrap DetectMultiBackend
            model.model = model.model[:cutoff]  # backbone
            m = model.model[-1]  # last layer
            ch = m.conv.in_channels if hasattr(m, 'conv') else m.cv1.conv.in_channels  # ch into module
            c = Classify(ch, nc)  # Classify()
            c.i, c.f, c.type = m.i, m.f, 'models.common.Classify'  # index, from, type
            model.model[-1] = c  # replace
            self.model = model.model
            self.stride = model.stride
            self.save = []
            self.nc = nc

        def _from_yaml(self, cfg):
            # Create a YOLO classification model from a *.yaml file
            self.model = None


    def parse_model(d, ch):  # model_dict, input_channels(3)
        # Parse a YOLO model.yaml dictionary
        LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
        anchors, nc, gd, gw, act = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d.get('activation')
        if act:
            Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print
        na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
        no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

        layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
        for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
            m = eval(m) if isinstance(m, str) else m  # eval strings
            for j, a in enumerate(args):
                with contextlib.suppress(NameError):
                    args[j] = eval(a) if isinstance(a, str) else a  # eval strings

            n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
            if m in {
                Conv, AConv, ConvTranspose,
                Bottleneck, SPP, SPPF, DWConv, BottleneckCSP, nn.ConvTranspose2d, DWConvTranspose2d, SPPCSPC, ADown,
                RepNCSPELAN4, SPPELAN}:
                c1, c2 = ch[f], args[0]
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8)

                args = [c1, c2, *args[1:]]
                if m in {BottleneckCSP, SPPCSPC}:
                    args.insert(2, n)  # number of repeats
                    n = 1
            elif m is nn.BatchNorm2d:
                args = [ch[f]]
            elif m is Concat:
                c2 = sum(ch[x] for x in f)
            elif m is Shortcut:
                c2 = ch[f[0]]
            elif m is ReOrg:
                c2 = ch[f] * 4
            elif m is CBLinear:
                c2 = args[0]
                c1 = ch[f]
                args = [c1, c2, *args[1:]]
            elif m is CBFuse:
                c2 = ch[f[-1]]
            # TODO: channel, gw, gd
            elif m in {Detect, DualDetect, TripleDetect, DDetect, DualDDetect, TripleDDetect, Segment}:
                args.append([ch[x] for x in f])
                # if isinstance(args[1], int):  # number of anchors
                #     args[1] = [list(range(args[1] * 2))] * len(f)
                if m in {Segment}:
                    args[2] = make_divisible(args[2] * gw, 8)
            elif m is Contract:
                c2 = ch[f] * args[0] ** 2
            elif m is Expand:
                c2 = ch[f] // args[0] ** 2
            else:
                c2 = ch[f]

            m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
            t = str(m)[8:-2].replace('__main__.', '')  # module type
            np = sum(x.numel() for x in m_.parameters())  # number params
            m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
            LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
            layers.append(m_)
            if i == 0:
                ch = []
            ch.append(c2)
        return nn.Sequential(*layers), sorted(save)
