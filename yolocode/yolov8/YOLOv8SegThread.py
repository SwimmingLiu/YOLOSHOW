import os.path
import time

import cv2
import numpy as np
import torch
from PySide6.QtCore import QThread, Signal
from pathlib import Path

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


class YOLOv8SegThread(QThread):
    # 输入 输出 消息
    send_input = Signal(np.ndarray)
    send_output = Signal(np.ndarray)
    send_msg = Signal(str)
    # 状态栏显示数据 进度条数据
    send_fps = Signal(str)  # fps
    # send_labels = Signal(dict)  # Detected target results (number of each category)
    send_progress = Signal(int)  # Completeness
    send_class_num = Signal(int)  # Number of categories detected
    send_target_num = Signal(int)  # Targets detected
    send_result_picture = Signal(dict)  # Send the result picture
    send_result_table = Signal(list)  # Send the result table

    def __init__(self):
        super(YOLOv8SegThread, self).__init__()
        # YOLOSHOW 界面参数设置
        self.current_model_name = None  # The detection model name to use
        self.new_model_name = None  # Models that change in real time
        self.source = None  # input source
        self.stop_dtc = True  # 停止检测
        self.is_continue = True  # continue/pause
        self.save_res = False  # Save test results
        self.iou_thres = 0.45  # iou
        self.conf_thres = 0.25  # conf
        self.speed_thres = 10  # delay, ms
        self.labels_dict = {}  # return a dictionary of results
        self.progress_value = 0  # progress bar
        self.res_status = False  # result status
        self.parent_workpath = None  # parent work path

        # YOLOv8 参数设置
        self.model = None
        self.data = 'yolocode/yolov8/cfg/datasets/coco128-seg.yaml'  # data_dict
        self.imgsz = 640
        self.device = ''
        self.dataset = None
        self.task = 'segment'
        self.dnn = False
        self.half = False
        self.agnostic_nms = False
        self.stream_buffer = False
        self.crop_fraction = 1.0
        self.done_warmup = False
        self.vid_path, self.vid_writerm, self.vid_cap = None, None, None
        self.batch = None
        self.batchsize = 1
        self.project = 'runs/segment'
        self.name = 'exp'
        self.exist_ok = False
        self.vid_stride = 1  # 视频帧率
        self.max_det = 1000  # 最大检测数
        self.classes = None  # 指定检测类别  --class 0, or --class 0 2 3
        self.line_thickness = 3
        self.results_picture = dict()  # 结果图片
        self.results_table = list()  # 结果表格
        self.callbacks = defaultdict(list, callbacks.default_callbacks)  # add callbacks
        callbacks.add_integration_callbacks(self)

    def run(self):
        if not self.model:
            self.send_msg.emit("Loading model: {}".format(os.path.basename(self.new_model_name)))
            self.setup_model(self.new_model_name)
            self.used_model_name = self.new_model_name

        source = str(self.source)
        # 判断输入源类型
        self.is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        self.is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
        self.webcam = source.isnumeric() or source.endswith(".streams") or (self.is_url and not self.is_file)
        self.screenshot = source.lower().startswith("screen")
        # 判断输入源是否是文件夹，如果是列表，则是文件夹
        self.is_folder = isinstance(self.source, list)
        if self.save_res:
            self.save_path = increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok)  # increment run
            self.save_path.mkdir(parents=True, exist_ok=True)  # make dir

        if self.is_folder:
            for source in self.source:
                self.setup_source(source)
                self.detect()
        else:
            self.setup_source(source)
            self.detect()

    def detect(
        self,
    ):
        # warmup model
        if not self.done_warmup:
            self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
            self.done_warmup = True
        self.seen, self.windows, self.dt, self.batch = 0, [], (ops.Profile(), ops.Profile(), ops.Profile()), None
        datasets = iter(self.dataset)
        self.vid_cap = self.dataset.cap if self.dataset.mode == "video" else None
        count = 0
        start_time = time.time()  # used to calculate the frame rate
        while True:
            if self.stop_dtc:
                self.send_msg.emit('Stop Detection')
                # --- 发送图片和表格结果 --- #
                self.send_result_picture.emit(self.results_picture)  # 发送图片结果
                for key, value in self.results_picture.items():
                    self.results_table.append([key, str(value)])
                self.results_picture = dict()
                self.send_result_table.emit(self.results_table)  # 发送表格结果
                self.results_table = list()
                # --- 发送图片和表格结果 --- #
                # 释放资源
                self.dataset.running = False  # stop flag for Thread
                # 判断self.dataset里面是否有threads
                if hasattr(self.dataset, 'threads'):
                    for thread in self.dataset.threads:
                        if thread.is_alive():
                            thread.join(timeout=1)  # Add timeout
                if hasattr(self.dataset, 'caps'):
                    for cap in self.dataset.caps:  # Iterate through the stored VideoCapture objects
                        try:
                            cap.release()  # release video capture
                        except Exception as e:
                            LOGGER.warning(f"WARNING ⚠️ Could not release VideoCapture object: {e}")
                cv2.destroyAllWindows()
                if isinstance(self.vid_writer[-1], cv2.VideoWriter):
                    self.vid_writer[-1].release()
                break
                #  判断是否更换模型
            if self.current_model_name != self.new_model_name:
                self.send_msg.emit('Loading Model: {}'.format(os.path.basename(self.new_model_name)))
                self.setup_model(self.new_model_name)
                self.current_model_name = self.new_model_name
            if self.is_continue:
                if self.is_file:
                    self.send_msg.emit("Detecting File: {}".format(os.path.basename(self.source)))
                elif self.webcam and not self.is_url:
                    self.send_msg.emit("Detecting Webcam: Camera_{}".format(self.source))
                elif self.is_folder:
                    self.send_msg.emit("Detecting Folder: {}".format(os.path.dirname(self.source[0])))
                elif self.is_url:
                    self.send_msg.emit("Detecting URL: {}".format(self.source))
                else:
                    self.send_msg.emit("Detecting: {}".format(self.source))
                self.batch = next(datasets)
                path, im0s, s = self.batch
                self.vid_cap = self.dataset.cap if self.dataset.mode == "video" else None
                # 原始图片送入 input框
                self.send_input.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0])
                count += 1
                percent = 0  # 进度条
                # 处理processBar
                if self.vid_cap:
                    if self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0:
                        percent = int(count / self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT) * self.progress_value)
                        self.send_progress.emit(percent)
                    else:
                        percent = 100
                        self.send_progress.emit(percent)
                else:
                    percent = self.progress_value
                if count % 5 == 0 and count >= 5:  # Calculate the frame rate every 5 frames
                    self.send_fps.emit(str(int(5 / (time.time() - start_time))))
                    start_time = time.time()
                # Preprocess
                with self.dt[0]:
                    im = self.preprocess(im0s)
                # Inference
                with self.dt[1]:
                    preds = self.inference(im)
                # Postprocess
                with self.dt[2]:
                    self.results = self.postprocess(preds, im, im0s)

                n = len(im0s)
                for i in range(n):
                    self.seen += 1
                    self.results[i].speed = {
                        "preprocess": self.dt[0].dt * 1e3 / n,
                        "inference": self.dt[1].dt * 1e3 / n,
                        "postprocess": self.dt[2].dt * 1e3 / n,
                    }
                    p, im0 = path[i], None if self.source_type.tensor else im0s[i].copy()
                    p = Path(p)

                    label_str = self.write_results(i, self.results, (p, im, im0))  # labels   /// original :s +=

                    # labels and nums dict
                    class_nums = 0
                    target_nums = 0
                    self.labels_dict = {}
                    if 'no detections' in label_str:
                        pass
                    else:
                        for each_target in label_str.split(',')[:-1]:
                            num_labelname = list(each_target.split(' '))
                            nums = 0
                            label_name = ""
                            for each in range(len(num_labelname)):
                                if num_labelname[each].isdigit() and each != len(num_labelname) - 1:
                                    nums = num_labelname[each]
                                elif len(num_labelname[each]):
                                    label_name += num_labelname[each] + " "
                            target_nums += int(nums)
                            class_nums += 1
                            if label_name in self.labels_dict:
                                self.labels_dict[label_name] += int(nums)
                            else:  # 第一次出现的类别
                                self.labels_dict[label_name] = int(nums)

                    # Send test results
                    self.send_output.emit(self.plotted_img)  # after detection
                    self.send_class_num.emit(class_nums)
                    self.send_target_num.emit(target_nums)
                    self.results_picture = self.labels_dict

                    if self.save_res:
                        save_path = str(self.save_path / p.name)  # im.jpg
                        self.res_path = self.save_preds(self.vid_cap, i, save_path)

                    if self.speed_thres != 0:
                        time.sleep(self.speed_thres / 1000)  # delay , ms

                if percent == self.progress_value and not self.webcam:
                    self.send_progress.emit(0)
                    self.send_msg.emit('Finish Detection')
                    # --- 发送图片和表格结果 --- #
                    self.send_result_picture.emit(self.results_picture)  # 发送图片结果
                    for key, value in self.results_picture.items():
                        self.results_table.append([key, str(value)])
                    self.results_picture = dict()
                    self.send_result_table.emit(self.results_table)  # 发送表格结果
                    self.results_table = list()
                    # --- 发送图片和表格结果 --- #
                    self.res_status = True
                    if self.vid_cap is not None:
                        self.vid_cap.release()
                    if isinstance(self.vid_writer[-1], cv2.VideoWriter):
                        self.vid_writer[-1].release()  # release final video writer
                    break

    def setup_model(self, model, verbose=True):
        """Initialize YOLO model with given parameters and set it to evaluation mode."""
        self.model = AutoBackend(
            model or self.model,
            device=select_device(self.device, verbose=verbose),
            dnn=self.dnn,
            data=self.data,
            fp16=self.half,
            fuse=True,
            verbose=verbose,
        )

        self.device = self.model.device  # update device
        self.half = self.model.fp16  # update half
        self.model.eval()

    def setup_source(self, source):
        """Sets up source and inference mode."""
        self.imgsz = check_imgsz(self.imgsz, stride=self.model.stride, min_dim=2)  # check image size
        self.transforms = (
            getattr(
                self.model.model,
                "transforms",
                classify_transforms(self.imgsz[0], crop_fraction=self.crop_fraction),
            )
            if self.task == "classify"
            else None
        )
        self.dataset = load_inference_source(
            source=source,
            batch=self.batchsize,
            vid_stride=self.vid_stride,
            buffer=self.stream_buffer,
        )
        self.source_type = self.dataset.source_type
        if not getattr(self, "stream", True) and (
            self.source_type.stream
            or self.source_type.screenshot
            or len(self.dataset) > 1000  # many images
            or any(getattr(self.dataset, "video_flag", [False]))
        ):  # videos
            LOGGER.warning(STREAM_WARNING)
        self.vid_path = [None] * self.dataset.bs
        self.vid_writer = [None] * self.dataset.bs
        self.vid_frame = [None] * self.dataset.bs

    def postprocess(self, preds, img, orig_imgs):
        """Applies non-max suppression and processes detections for each image in an input batch."""
        p = ops.non_max_suppression(
            preds[0],
            self.conf_thres,
            self.iou_thres,
            agnostic=self.agnostic_nms,
            max_det=self.max_det,
            nc=len(self.model.names),
            classes=self.classes,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        proto = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]  # tuple if PyTorch model or array if exported
        for i, pred in enumerate(p):
            orig_img = orig_imgs[i]
            img_path = self.batch[0][i]
            if not len(pred):  # save empty boxes
                masks = None
            else:
                masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks))
        return results

    def preprocess(self, im):
        """
        Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
        """
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        im = im.to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        if not_tensor:
            im /= 255  # 0 - 255 to 0.0 - 1.0
        return im

    def inference(self, im, *args, **kwargs):
        """Runs inference on a given image using the specified model and arguments."""
        return self.model(im, augment=False, visualize=False, embed=False, *args, **kwargs)

    def pre_transform(self, im):
        """
        Pre-transform input image before inference.

        Args:
            im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

        Returns:
            (list): A list of transformed images.
        """
        same_shapes = all(x.shape == im[0].shape for x in im)
        letterbox = LetterBox(self.imgsz, auto=same_shapes and self.model.pt, stride=self.model.stride)
        return [letterbox(image=x) for x in im]

    def save_preds(self, vid_cap, idx, save_path):
        """Save video predictions as mp4 at specified path."""
        im0 = self.plotted_img
        suffix, fourcc = (".mp4", "avc1") if MACOS else (".avi", "WMV2") if WINDOWS else (".avi", "MJPG")
        # Save imgs
        if self.dataset.mode == "image":
            cv2.imwrite(save_path, im0)
            return save_path

        else:  # 'video' or 'stream'
            if self.vid_path[idx] != save_path:  # new video
                self.vid_path[idx] = save_path
                if isinstance(self.vid_writer[idx], cv2.VideoWriter):
                    self.vid_writer[idx].release()  # release previous video writer
                if vid_cap:  # video
                    fps = int(vid_cap.get(cv2.CAP_PROP_FPS))  # integer required, floats produce error in MP4 codec
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                # suffix, fourcc = (".mp4", "avc1") if MACOS else (".avi", "WMV2") if WINDOWS else (".avi", "MJPG")
                self.vid_writer[idx] = cv2.VideoWriter(
                    str(Path(save_path).with_suffix(suffix)), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h)
                )
            # Write video
            self.vid_writer[idx].write(im0)
            return str(Path(save_path).with_suffix(suffix))

    def write_results(self, idx, results, batch):
        """Write inference results to a file or directory."""
        p, im, _ = batch
        log_string = ""
        self.data_path = p
        result = results[idx]
        log_string += result.verbose()
        result = results[idx]
        # Add bbox to image
        plot_args = {
            "line_width": self.line_thickness,
            "boxes": True,
            "conf": True,
            "labels": True,
        }
        self.plotted_img = result.plot(**plot_args)
        return log_string
