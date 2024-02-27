# YOLOSHOW -  YOLOv5 / YOLOv7 / YOLOv8 / YOLOv9 GUI based on Pyside6

## Introduction

***YOLOSHOW*** is a graphical user interface (GUI) application embed with`YOLOv5` `YOLOv7` `YOLOv8` `YOLOv9` algorithm. 

English | [简体中文](https://github.com/SwimmingLiu/YOLOSHOW/blob/master/README_cn.md) 

![YOLOSHOW-Screen](https://oss.swimmingliu.cn/YOLOSHOW-SCREEN.png)

## Demo Video

Bilibili Demo Video : [YOLOSHOW-YOLOv9/YOLOv8/YOLOv7/YOLOv5 GUI](https://www.bilibili.com/video/BV1BC411x7fW)

## Todo List

- [x] Add YOLOv9 Algorithm

- [x] Adjust User Interface (Menu Bar)

- [ ] Complete Rtsp Function

- [ ] Support Instance Segmentation

## Functions

### 1. Support Image / Video / Webcam / Folder (Batch ) Object Detection

Choose Image / Video / Webcam / Folder (Batch ) in the menu bar on the left to detect objects.

### 2. Change Models / Hyper Parameters dynamically

When the program is running to detect targets, you can change models / hyper Parameters

1. Support changing model in `YOLOv5` / ` YOLOv7` / `YOLOv8` / `YOLOv9` dynamically
2. Support changing `IOU` / `Confidence` / `Delay time ` / `line thickness` dynamically

### 3. Loading Model Automatically

Our program will automatically detect  `pt` files including [YOLOv5 Models](https://github.com/ultralytics/yolov5/releases) /  [YOLOv7 Models](https://github.com/WongKinYiu/yolov7/releases/)  /  [YOLOv8 Models](https://github.com/ultralytics/assets/releases/)  / [YOLOv9 Models](https://github.com/WongKinYiu/yolov9/releases/)  that were previously added to the `ptfiles` folder.

If you need add the new `pt` file, please click `Import Model` button in `Settings` box to select your `pt` file. Then our program will put it into  `ptfiles` folder.

**Notice :**  All `pt` files are named including `yolov5` / `yolov7` / `yolov8` / `yolov9` .  (e.g. `yolov8-test.pt`)

### 4. Loading Configures

1.  After startup, the program will automatically loading the last configure parameters.
2.  After closedown, the program will save the changed configure parameters.

### 5. Save Results

If you need Save results, please click `Save MP4/JPG` before detection. Then you can save your detection results in selected path.

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
pip install "PySide6-Fluent-Widgets[full]" -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -U Pyside6 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 4. Add Font

Copy all font files `*.ttf` in `fonts` folder into `C:\Windows\Fonts`

## Frames

[![Python](https://img.shields.io/badge/python-3776ab?style=for-the-badge&logo=python&logoColor=ffd343)](https://www.python.org/)[![Pytorch](https://img.shields.io/badge/PYtorch-test?style=for-the-badge&logo=pytorch&logoColor=white&color=orange)](https://pytorch.org/)[![Static Badge](https://img.shields.io/badge/Pyside6-test?style=for-the-badge&logo=qt&logoColor=white)](https://doc.qt.io/qtforpython-6/PySide6/QtWidgets/index.html)

## Reference

### YOLO Algorithm

[YOLOv5](https://github.com/ultralytics/yolov5)   [YOLOv7](https://github.com/WongKinYiu/yolov7)  [YOLOv8](https://github.com/ultralytics/ultralytics)  [YOLOv9](https://github.com/WongKinYiu/yolov9) 

### YOLO Graphical User Interface

[YOLOSIDE](https://github.com/Jai-wei/YOLOv8-PySide6-GUI)
