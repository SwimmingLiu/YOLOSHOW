import argparse
import os
import time
from pathlib import Path
from PySide6.QtCore import Signal
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PySide6.QtCore import QThread
from numpy import random
from yolocode.yolov7.models.experimental import attempt_load
from yolocode.yolov7.utils.datasets import LoadStreams, LoadImages
from yolocode.yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, strip_optimizer, set_logging, increment_path
from yolocode.yolov7.utils.plots import plot_one_box
from yolocode.yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from yolocode.yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS

class YOLOv7Thread(QThread):
    # 输入 输出 消息
    send_input = Signal(np.ndarray)
    send_output = Signal(np.ndarray)
    send_msg = Signal(str)
    # 状态栏显示数据 进度条数据
    send_fps = Signal(str)  # fps
    send_progress = Signal(int)  # Completeness
    send_class_num = Signal(int)  # Number of categories detected
    send_target_num = Signal(int)  # Targets detected
    send_result_picture = Signal(dict)  # Send the result picture
    send_result_table = Signal(list)    # Send the result table

    def __init__(self):
        super(YOLOv7Thread, self).__init__()
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

        # YOLOv7 参数设置
        self.model = None
        self.imgsz = 640
        self.device = ''
        self.dataset = None
        self.vid_path, self.vid_writerm, self.vid_cap = None, None, None
        self.annotator = None
        self.data_path = None
        self.source_type = None
        self.batch = None
        self.project = 'runs/detect'
        self.name = 'exp'
        self.exist_ok = False
        self.vid_stride = 1  # 视频帧率
        self.max_det = 1000  # 最大检测数
        self.classes = None  # 指定检测类别  --class 0, or --class 0 2 3
        self.line_thickness = 3
        self.results_picture = dict()     # 结果图片
        self.results_table = list()         # 结果表格

    @torch.no_grad()
    def run(self):
        source = str(self.source)
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

        if self.save_res:
            self.save_dir = Path(increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok))  # increment run
            self.save_dir.mkdir(parents=True, exist_ok=True)  # make dir

        # 显卡选择
        device = select_device(self.device)
        # Load model
        weights = self.new_model_name
        self.send_msg.emit(f'Loading model: {os.path.basename(weights)}')
        self.current_model_name = self.new_model_name
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(self.imgsz, s=stride)  # check img_size

        # Set Dataloader
        self.vid_path, self.vid_writer = None, None
        dataset_list = []
        if self.webcam:
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        elif self.is_folder:
            for source_i in self.source:
                dataset_list.append(LoadImages(source_i, img_size=imgsz, stride=stride))
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride)


        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

        if self.is_folder:
            for dataset in dataset_list:
                self.detect(dataset, device, imgsz,model)
        else:
            self.detect(dataset, device, imgsz,model)


    def detect(self,dataset, device, imgsz, model):

        datasets = iter(dataset)
        # 参数设置
        start_time = time.time()  # used to calculate the frame rate
        count = 0
        # 获取模型参数
        half = device.type != 'cpu'  # half precision only supported on CUDA
        if half:
            model.half()  # to FP16
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        old_img_w = old_img_h = imgsz
        old_img_b = 1
        while True:
            # 停止检测
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
                    if dataset.threads.is_alive():
                        dataset.threads.join(timeout=5)  # Add timeout
                if hasattr(dataset, 'cap'):
                    dataset.cap.release()
                cv2.destroyAllWindows()
                if isinstance(self.vid_writer, cv2.VideoWriter):
                    self.vid_writer.release()
                break
            #  判断是否更换模型
            if self.current_model_name != self.new_model_name:
                weights = self.current_model_name
                # 显卡选择
                device = select_device(self.device)
                self.send_msg.emit(f'Loading model: {os.path.basename(weights)}')
                # Load model
                model = attempt_load(weights, map_location=device)  # load FP32 model
                stride = int(model.stride.max())  # model stride
                imgsz = check_img_size(imgsz, s=stride)  # check img_size
                half = device.type != 'cpu'  # half precision only supported on CUDA
                if half:
                    model.half()  # to FP16
                # Get names and colors
                names = model.module.names if hasattr(model, 'module') else model.names
                colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
                # Run inference
                if device.type != 'cpu':
                    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
                old_img_w = old_img_h = imgsz
                old_img_b = 1
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
                path, img, im0s, self.vid_cap = next(datasets)
                im0s_copy = im0s.copy()
                # 原始图片送入 input框
                self.send_input.emit(im0s_copy if isinstance(im0s_copy, np.ndarray) else im0s_copy[0])
                # 处理processBar
                count += 1
                percent = 0
                if self.vid_cap:
                    percent = int(count / self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT) * self.progress_value)
                    self.send_progress.emit(percent)
                else:
                    percent = self.progress_value
                if count % 5 == 0 and count >= 5:  # Calculate the frame rate every 5 frames
                    self.send_fps.emit(str(int(5 / (time.time() - start_time))))
                    start_time = time.time()

                # 处理图片
                statistic_dic = {name: 0 for name in names}
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                # Warmup
                if device.type != 'cpu' and (
                        old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                    old_img_b = img.shape[0]
                    old_img_h = img.shape[2]
                    old_img_w = img.shape[3]
                    for i in range(3):
                        model(img, augment=False)[0]

                with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
                    pred = model(img, augment=False)[0]

                # Apply NMS
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes,
                                           agnostic=False)

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    class_nums = 0
                    target_nums = 0
                    if self.webcam:  # batch_size >= 1
                        p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                    else:
                        p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                    p = Path(p)  # to Path
                    if self.save_res:
                        self.save_path = str(self.save_dir / p.name)  # img.jpg
                        self.res_path = self.save_path
                    # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, 5].unique():
                            n = (det[:, 5] == c).sum()  # detections per class
                            class_nums += 1
                            target_nums += int(n)
                            if names[int(c)] in self.labels_dict:
                                self.labels_dict[names[int(c)]] += int(n)
                            else:  # 第一次出现的类别
                                self.labels_dict[names[int(c)]] = int(n)

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            # Add bbox to image
                            c = int(cls)  # integer class
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=self.line_thickness)


                    self.send_output.emit(im0)  # 输出图片
                    self.send_class_num.emit(class_nums)
                    self.send_target_num.emit(target_nums)
                    self.results_picture = self.labels_dict

                    if self.save_res:
                        # Save results (image with detections)
                        if dataset.mode == 'image':
                            cv2.imwrite(self.save_path, im0)
                        else:  # 'video' or 'stream'
                            if self.vid_path != self.save_path:  # new video
                                self.vid_path = self.save_path
                                if isinstance(self.vid_writer, cv2.VideoWriter):
                                    self.vid_writer.release()  # release previous video writer
                                if self.vid_cap:  # video
                                    fps = self.vid_cap.get(cv2.CAP_PROP_FPS)
                                    w = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    h = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                else:  # stream
                                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                                    self.save_path += '.mp4'
                                self.vid_writer = cv2.VideoWriter(self.save_path,
                                                                  cv2.VideoWriter_fourcc(*'mp4v'),
                                                                  fps,
                                                                  (w, h))
                            self.vid_writer.write(im0)

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
                    if isinstance(self.vid_writer, cv2.VideoWriter):
                        self.vid_writer.release()  # release final video writer
                    break

