import os.path
import time
import numpy as np
import torch
from PySide6.QtCore import QThread, Signal
from pathlib import Path

from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS

from models.common import DetectMultiBackend_YOLOv5
from yolocode.yolov5.utils.dataloaders import LoadImages, LoadScreenshots, LoadStreams
from ultralytics.utils.plotting import Annotator, colors
from yolocode.yolov5.utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from yolocode.yolov5.utils.segment.general import process_mask, process_mask_native
from yolocode.yolov5.utils.torch_utils import select_device, smart_inference_mode


class YOLOv5SegThread(QThread):
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
        super(YOLOv5SegThread, self).__init__()
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

        # YOLOv5 参数设置
        self.model = None
        self.data = 'yolocode/yolov5/data/coco128.yaml'  # data_dict
        self.imgsz = (640, 640)
        self.device = ''
        self.dataset = None
        self.vid_path, self.vid_writerm, self.vid_cap = None, None, None
        self.annotator = None
        self.data_path = None
        self.source_type = None
        self.batch = None
        self.project = 'runs/segment'
        self.name = 'exp'
        self.exist_ok = False
        self.vid_stride = 1  # 视频帧率
        self.max_det = 1000  # 最大检测数
        self.classes = None  # 指定检测类别  --class 0, or --class 0 2 3
        self.line_thickness = 3
        self.retina_masks = False
        self.results_picture = dict()  # 结果图片
        self.results_table = list()  # 结果表格

    def run(self):
        source = str(self.source)
        # save_img = not nosave and not source.endswith(".txt")  # save inference images
        # 判断输入源类型
        if isinstance(IMG_FORMATS, str) or isinstance(IMG_FORMATS, tuple):
            self.is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        else:
            self.is_file = Path(source).suffix[1:] in (IMG_FORMATS | VID_FORMATS)
        self.is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
        self.webcam = source.isnumeric() or source.endswith(".streams") or (self.is_url and not self.is_file)
        self.screenshot = source.lower().startswith("screen")
        # 判断输入源是否是文件夹，如果是列表，则是文件夹
        self.is_folder = isinstance(self.source, list)
        if self.is_url and self.is_file:
            source = check_file(source)  # download

        if self.save_res:
            # 保存文件的路径
            self.save_path = increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok)  # increment run
            self.save_path.mkdir(parents=True, exist_ok=True)  # make dir

        # 加载模型
        device = select_device(self.device)
        weights = self.new_model_name
        self.current_model_name = self.new_model_name
        self.send_msg.emit(f'Loading Model: {os.path.basename(weights)}')
        model = DetectMultiBackend_YOLOv5(weights, device=device, dnn=False, data=self.data, fp16=False)
        self.stride, self.names, self.pt = model.stride, model.names, model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size

        # Dataloader 加载数据
        bs = 1  # batch_size
        vid_stride = self.vid_stride
        dataset_list = []
        if self.webcam:
            dataset = LoadStreams(source, img_size=self.imgsz, stride=self.stride, auto=self.pt, vid_stride=vid_stride)
            bs = len(dataset)
        elif self.screenshot:
            dataset = LoadScreenshots(source, img_size=self.imgsz, stride=self.stride, auto=self.pt)
        elif self.is_folder:
            for source_i in self.source:
                dataset_list.append(
                    LoadImages(source_i, img_size=self.imgsz, stride=self.stride, auto=self.pt, vid_stride=vid_stride))
        else:
            dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride, auto=self.pt, vid_stride=vid_stride)
        self.vid_path, self.vid_writer = [None] * bs, [None] * bs  # 视频路径 视频写入器
        model.warmup(imgsz=(1 if self.pt or model.triton else bs, 3, *self.imgsz))  # warmup
        self.model = model
        if self.is_folder:
            for dataset in dataset_list:
                self.detect(dataset, device, bs)
        else:
            self.detect(dataset, device, bs)

    def detect(self, dataset, device, bs):
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        # seen 表示图片计数
        datasets = iter(dataset)
        count = 0  # run location frame
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
                if hasattr(dataset, 'threads'):
                    for thread in dataset.threads:
                        if thread.is_alive():
                            thread.join(timeout=1)  # Add timeout
                if hasattr(dataset, 'cap') and dataset.cap is not None:
                    dataset.cap.release()
                cv2.destroyAllWindows()
                if isinstance(self.vid_writer[-1], cv2.VideoWriter):
                    self.vid_writer[-1].release()
                break
            #  判断是否更换模型
            if self.current_model_name != self.new_model_name:
                weights = self.current_model_name
                data = self.data
                self.send_msg.emit(f'Loading Model: {os.path.basename(weights)}')
                self.model = DetectMultiBackend_YOLOv5(weights, device=device, dnn=False, data=data, fp16=False)
                stride, names, pt = self.model.stride, self.model.names, self.model.pt
                imgsz = check_img_size(self.imgsz, s=stride)  # check image size
                self.model.warmup(imgsz=(1 if pt or self.model.triton else bs, 3, *imgsz))  # warmup
                seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
                self.current_model_name = self.new_model_name
            # 开始推理
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
                path, im, im0s, self.vid_cap, s = next(datasets)
                # 原始图片送入 input框
                self.send_input.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0])
                count += 1
                percent = 0  # 进度条
                # 处理processBar
                if self.vid_cap:
                    percent = int(count / self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT) * self.progress_value)
                    self.send_progress.emit(percent)
                else:
                    percent = self.progress_value
                if count % 5 == 0 and count >= 5:  # Calculate the frame rate every 5 frames
                    self.send_fps.emit(str(int(5 / (time.time() - start_time))))
                    start_time = time.time()

                with dt[0]:
                    im = torch.from_numpy(im).to(self.model.device)
                    im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim

                # Inference
                with dt[1]:
                    pred, proto = self.model(im, augment=False, visualize=False)[:2]

                # NMS
                with dt[2]:
                    pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, False,
                                               max_det=self.max_det, nm=32)

                # Process predictions
                for i, det in enumerate(pred):
                    seen += 1
                    # per image
                    if self.webcam:  # batch_size >= 1
                        p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    else:
                        p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

                    p = Path(p)  # to Path
                    if self.save_res:
                        save_path = str(self.save_path / p.name)  # im.jpg
                        self.res_path = save_path
                    annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))

                    # 类别数量 目标数量
                    class_nums = 0
                    target_nums = 0
                    if len(det):
                        if self.retina_masks:
                            # scale bbox first the crop masks
                            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4],
                                                     im0.shape).round()  # rescale boxes to im0 size
                            masks = process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[:2])  # HWC
                        else:
                            masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4],
                                                     im0.shape).round()  # rescale boxes to im0 size

                        # Print results
                        for c in det[:, 5].unique():
                            n = (det[:, 5] == c).sum()  # detections per class
                            s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                            class_nums += 1
                            target_nums += int(n)
                            if self.names[int(c)] in self.labels_dict:
                                self.labels_dict[self.names[int(c)]] += int(n)
                            else:  # 第一次出现的类别
                                self.labels_dict[self.names[int(c)]] = int(n)
                        # Mask plotting
                        annotator.masks(
                            masks,
                            colors=[colors(x, True) for x in det[:, 5]],
                            im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(device).permute(2, 0, 1).flip(
                                0).contiguous() /
                                   255 if self.retina_masks else im[i])

                        # Write results
                        for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                            c = int(cls)  # integer class
                            label = self.names[c]
                            confidence = float(conf)
                            confidence_str = f"{confidence:.2f}"
                            c = int(cls)  # integer class
                            label = f"{self.names[c]} {conf:.2f}"
                            annotator.box_label(xyxy, label, color=colors(c, True))

                    # 发送结果
                    im0 = annotator.result()
                    self.send_output.emit(im0)  # 输出图片
                    self.send_class_num.emit(class_nums)
                    self.send_target_num.emit(target_nums)
                    self.results_picture = self.labels_dict

                    if self.save_res:
                        if dataset.mode == "image":
                            cv2.imwrite(save_path, im0)
                        else:  # 'video' or 'stream'
                            if self.vid_path[i] != save_path:  # new video
                                self.vid_path[i] = save_path
                                if isinstance(self.vid_writer[i], cv2.VideoWriter):
                                    self.vid_writer[i].release()  # release previous video writer
                                if self.vid_cap:  # video
                                    fps = self.vid_cap.get(cv2.CAP_PROP_FPS)
                                    w = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    h = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                else:  # stream
                                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path = str(
                                    Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                                self.vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                                                     (w, h))
                            self.vid_writer[i].write(im0)

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
