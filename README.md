# YOLOSHOW -  YOLOv5 / YOLOv7 / YOLOv8 / YOLOv9 / YOLOv10 / RTDETR GUI based on Pyside6

## Introduction

***YOLOSHOW*** is a graphical user interface (GUI) application embed with`YOLOv5` `YOLOv7` `YOLOv8` `YOLOv9` `YOLOv10` `RT-DETR` algorithm. 

 <p align="center"> 
  English &nbsp; | &nbsp; <a href="https://github.com/SwimmingLiu/YOLOSHOW/blob/master/README_cn.md">简体中文</a>
 </p>

![YOLOSHOW-Screen](https://oss.swimmingliu.cn/YOLOSHOW-SCREENSHOT.png)

## Demo Video

`YOLOSHOW v1.x` : [YOLOSHOW-YOLOv9/YOLOv8/YOLOv7/YOLOv5/RTDETR GUI](https://www.bilibili.com/video/BV1BC411x7fW)

`YOLOSHOW v2.x` : [YOLOSHOWv2.0-YOLOv9/YOLOv8/YOLOv7/YOLOv5/RTDETR GUI](https://www.bilibili.com/video/BV1ZD421E7m3)

## Todo List

- [x] Add `YOLOv9` Algorithm
- [x] Adjust User Interface (Menu Bar)
- [x] Complete Rtsp Function
- [x] Support Instance Segmentation （ `YOLOv5` & `YOLOv8` ）
- [x] Add `RT-DETR` Algorithm ( `Ultralytics` repo)
- [x] Add Model Comparison Mode（VS Mode）
- [x] Support Pose Estimation （ `YOLOv8` ）
- [x] Support Http Protocol in Rtsp Function ( Single Mode )
- [x] Support Oriented Bounding Boxes ( `YOLOv8` )
- [x] Add `YOLOv10` Algorithm
- [x] Support Dragging File Input
- [ ] Tracking & Counting ( `Industrialization` )

## Functions

### 1. Support Image / Video / Webcam / Folder (Batch) / IPCam Object Detection

Choose Image / Video / Webcam / Folder (Batch) / IPCam in the menu bar on the left to detect objects.

### 2. Change Models / Hyper Parameters dynamically

When the program is running to detect targets, you can change models / hyper Parameters

1. Support changing model in `YOLOv5` / ` YOLOv7` / `YOLOv8` / `YOLOv9` / `RTDETR` / `YOLOv5-seg` / `YOLOv8-seg` / `YOLOv10` dynamically
2. Support changing `IOU` / `Confidence` / `Delay time ` / `line thickness` dynamically

### 3. Loading Model Automatically

Our program will automatically detect  `pt` files including [YOLOv5 Models](https://github.com/ultralytics/yolov5/releases) /  [YOLOv7 Models](https://github.com/WongKinYiu/yolov7/releases/)  /  [YOLOv8 Models](https://github.com/ultralytics/assets/releases/)  / [YOLOv9 Models](https://github.com/WongKinYiu/yolov9/releases/)  / [YOLOv10 Models](https://github.com/THU-MIG/yolov10/releases/)  that were previously added to the `ptfiles` folder.

If you need add the new `pt` file, please click `Import Model` button in `Settings` box to select your `pt` file. Then our program will put it into  `ptfiles` folder.

**Notice :** 

1. All `pt` files are named including `yolov5` / `yolov7` / `yolov8` / `yolov9` / `yolov10` / `rtdetr` .  (e.g. `yolov8-test.pt`)
2. If it is a `pt` file of  segmentation mode, please name it including `yolov5n-seg` / `yolov8s-seg` .  (e.g. `yolov8n-seg-test.pt`)
3. If it is a `pt` file of  pose estimation mode, please name it including `yolov8n-pose` .  (e.g. `yolov8n-pose-test.pt`)
4. If it is a `pt` file of  oriented bounding box mode, please name it including `yolov8n-obb` .  (e.g. `yolov8n-obb-test.pt`)

### 4. Loading Configures

1.  After startup, the program will automatically loading the last configure parameters.
2.  After closedown, the program will save the changed configure parameters.

### 5. Save Results

If you need Save results, please click `Save Mode` before detection. Then you can save your detection results in selected path.

### 6. Support Object Detection, Instance Segmentation and Pose Estimation 

From ***YOLOSHOW v3.0***，our work supports both Object Detection , Instance Segmentation, Pose Estimation and Oriented Bounding Box. Meanwhile, it also supports task switching between different versions，such as switching from `YOLOv5` Object Detection task to `YOLOv8` Instance Segmentation task.

### 7. Support Model Comparison among Object Detection,  Instance Segmentation, Pose Estimation and Oriented Bounding Box

From ***YOLOSHOW v3.0***，our work supports compare model performance among Object Detection, Instance Segmentation, Pose Estimation and Oriented Bounding Box.

## Preparation

### Experimental environment

```Shell
OS : Windows 11 
CPU : Intel(R) Core(TM) i7-10750H CPU @2.60GHz 2.59 GHz
GPU : NVIDIA GeForce GTX 1660Ti 6GB
```

### 1. Create virtual environment

create a virtual environment equipped with python version 3.9, then activate environment. 

```shell
conda create -n yoloshow python=3.9
conda activate yoloshow
```

### 2. Install Pytorch frame 

```shell
Windows: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
Linux: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Change other pytorch version in  [![Pytorch](https://img.shields.io/badge/PYtorch-test?style=flat&logo=pytorch&logoColor=white&color=orange)](https://pytorch.org/)

### 3. Install dependency package

Switch the path to the location of the program

```shell
cd {the location of the program}
```

Install dependency package of program 

```shell
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 4. Add Font

#### Windows User

Copy all font files `*.ttf` in `fonts` folder into `C:\Windows\Fonts`

#### Linux User

```shell
mkdir -p ~/.local/share/fonts
sudo cp fonts/Shojumaru-Regular.ttf ~/.local/share/fonts/
sudo fc-cache -fv
```

#### MacOS User

The MacBook is so expensive that I cannot afford it, please install `.ttf` by yourself. 😂

### 5. Run Program

```shell
python main.py
```

## Frames

[![Python](https://img.shields.io/badge/python-3776ab?style=for-the-badge&logo=python&logoColor=ffd343)](https://www.python.org/)[![Pytorch](https://img.shields.io/badge/PYtorch-test?style=for-the-badge&logo=pytorch&logoColor=white&color=orange)](https://pytorch.org/)[![Static Badge](https://img.shields.io/badge/Pyside6-test?style=for-the-badge&logo=qt&logoColor=white)](https://doc.qt.io/qtforpython-6/PySide6/QtWidgets/index.html)

## Reference

### YOLO Algorithm

[YOLOv5](https://github.com/ultralytics/yolov5)   [YOLOv7](https://github.com/WongKinYiu/yolov7)  [YOLOv8](https://github.com/ultralytics/ultralytics)  [YOLOv9](https://github.com/WongKinYiu/yolov9)  [YOLOv10](https://github.com/THU-MIG/yolov10)

### YOLO Graphical User Interface

[YOLOSIDE](https://github.com/Jai-wei/YOLOv8-PySide6-GUI)	[PyQt-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)
