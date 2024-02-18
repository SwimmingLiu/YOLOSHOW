from ui.YOLOSHOWUI import Ui_mainWindow
from utils import glo

glo._init()
glo.set_value('yoloname', "yolov5 yolov7 yolov8")

import json
import os
import shutil
import sys
import cv2
import numpy as np
import torch
from PySide6.QtGui import QPixmap, QImage, QMouseEvent, QGuiApplication, QColor
from PySide6.QtWidgets import QMessageBox, QFileDialog, QMainWindow, QWidget, QApplication, QGraphicsBlurEffect, \
    QGraphicsDropShadowEffect, QMenu
from PySide6.QtUiTools import QUiLoader, loadUiType
from PySide6.QtCore import QFile, QTimer, Qt, QEventLoop, QThread, QPropertyAnimation, QEasingCurve, \
    QParallelAnimationGroup, QPoint
from PySide6 import QtCore, QtGui
from PIL import Image
from qfluentwidgets import RoundMenu, MenuAnimationType, Action

from models import common, yolo
from ui.webCamera import Camera, WebcamThread
from utils.custom_grips import CustomGrip
from yolocode.yolov5.YOLOv5Thread import YOLOv5Thread
from yolocode.yolov7.YOLOv7Thread import YOLOv7Thread
from yolocode.yolov8.YOLOv8Thread import YOLOv8Thread
# from YoloClass import YoloThread

GLOBAL_WINDOW_STATE = True
formType, baseType = loadUiType(r"ui\YOLOSHOWUI.ui")

# YOLOSHOW窗口类 动态加载UI文件 和 Ui_mainWindow
class YOLOSHOW(formType, baseType, Ui_mainWindow):
    def __init__(self):
        super().__init__()
        self.current_workpath = os.getcwd()
        self.inputPath = None
        # --- 加载UI --- #
        self.setupUi(self)
        self.setAttribute(Qt.WA_TranslucentBackground, True)  # 透明背景
        self.setWindowFlags(Qt.FramelessWindowHint)  # 无头窗口
        # --- 加载UI --- #

        # --- 最大化 最小化 关闭 --- #
        self.maximizeButton.clicked.connect(self.maxorRestore)
        self.minimizeButton.clicked.connect(self.showMinimized)
        self.closeButton.clicked.connect(self.close)
        self.topbox.doubleClickFrame.connect(self.maxorRestore)
        # --- 最大化 最小化 关闭 --- #

        # --- 播放 暂停 停止 --- #
        self.playIcon = QtGui.QIcon()
        self.playIcon.addPixmap(QtGui.QPixmap(f"{self.current_workpath}/images/newsize/play.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.playIcon.addPixmap(QtGui.QPixmap(f"{self.current_workpath}/images/newsize/pause.png"), QtGui.QIcon.Active, QtGui.QIcon.On)
        self.playIcon.addPixmap(QtGui.QPixmap(f"{self.current_workpath}/images/newsize/pause.png"), QtGui.QIcon.Selected, QtGui.QIcon.On)
        self.run_button.setCheckable(True)
        self.run_button.setIcon(self.playIcon)
        # --- 播放 暂停 停止 --- #

        # --- 侧边栏缩放 --- #
        self.src_menu.clicked.connect(self.scaleMenu)  # hide menu button
        self.src_setting.clicked.connect(self.scalSetting)  # setting button
        # --- 侧边栏缩放 --- #

        # --- 自动加载/动态改变 PT 模型 --- #
        self.pt_Path = f"{self.current_workpath}/ptfiles/"
        self.pt_list = os.listdir(f'{self.current_workpath}/ptfiles/')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.pt_list.sort(key=lambda x: os.path.getsize(f'{self.current_workpath}/ptfiles/' + x))
        self.model_box.clear()
        self.model_box.addItems(self.pt_list)
        self.qtimer_search = QTimer(self)
        self.qtimer_search.timeout.connect(lambda: self.loadModels())
        self.qtimer_search.start(2000)
        self.model_box.currentTextChanged.connect(self.changeModel)
        # --- 自动加载/动态改变 PT 模型 --- #

        # --- 超参数调整 --- #
        self.iou_spinbox.valueChanged.connect(lambda x: self.changeValue(x, 'iou_spinbox'))  # iou box
        self.iou_slider.valueChanged.connect(lambda x: self.changeValue(x, 'iou_slider'))  # iou scroll bar
        self.conf_spinbox.valueChanged.connect(lambda x: self.changeValue(x, 'conf_spinbox'))  # conf box
        self.conf_slider.valueChanged.connect(lambda x: self.changeValue(x, 'conf_slider'))  # conf scroll bar
        self.speed_spinbox.valueChanged.connect(lambda x: self.changeValue(x, 'speed_spinbox'))  # speed box
        self.speed_slider.valueChanged.connect(lambda x: self.changeValue(x, 'speed_slider'))  # speed scroll bar
        self.line_spinbox.valueChanged.connect(lambda x: self.changeValue(x, 'line_spinbox'))  # line box
        self.line_slider.valueChanged.connect(lambda x: self.changeValue(x, 'line_slider'))  # line slider
        # --- 超参数调整 --- #

        # --- 导入 图片/视频、调用摄像头、导入文件夹（批量处理）、调用网络摄像头 --- #
        self.src_img.clicked.connect(self.selectFile)
        self.src_webcam.clicked.connect(self.selectWebcam)
        self.src_folder.clicked.connect(self.selectFolder)
        self.src_camera.clicked.connect(lambda: self.showStatus('Coming soon...'))
        # --- 导入 图片/视频、调用摄像头、导入文件夹（批量处理）、调用网络摄像头 --- #

        # --- 导入模型、 导出结果 --- #
        self.import_button.clicked.connect(self.importModel)
        self.save_status_button.clicked.connect(self.saveStatus)
        self.save_button.clicked.connect(self.saveResult)
        self.save_button.setEnabled(False)
        # --- 导入模型、 导出结果 --- #

        # --- 视频、图片 预览 --- #
        self.main_leftbox.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        self.main_rightbox.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        # --- 视频、图片 预览 --- #

        # --- 状态栏 初始化 --- #
        # 状态栏阴影效果
        self.shadowStyle(self.mainBody, QColor(0, 0, 0, 38), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.Class_QF, QColor(142, 197, 252), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.classesLabel, QColor(142, 197, 252), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.Target_QF, QColor(159, 172, 230), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.targetLabel, QColor(159, 172, 230), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.Fps_QF, QColor(170, 128, 213), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.fpsLabel, QColor(170, 128, 213), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.Model_QF, QColor(162, 129, 247), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.modelLabel, QColor(162, 129, 247), top_bottom=['top', 'bottom'])
        # 状态栏默认显示
        self.model_name = self.model_box.currentText()      # 获取默认 model
        self.Class_num.setText('--')
        self.Target_num.setText('--')
        self.fps_label.setText('--')
        self.Model_label.setText(self.model_name)
        # --- 状态栏 初始化 --- #

        # --- YOLOv5 QThread --- #
        self.yolov5_thread = YOLOv5Thread()
        self.initModel("yolov5")
        # --- YOLOv5 QThread --- #

        # --- YOLOv7 QThread --- #
        self.yolov7_thread = YOLOv7Thread()
        self.initModel("yolov7")
        # --- YOLOv7 QThread --- #

        # --- YOLOv8 QThread --- #
        self.yolov8_thread = YOLOv8Thread()
        self.initModel("yolov8")
        # --- YOLOv8 QThread --- #

        # --- Setting栏 初始化 --- #
        self.loadConfig()
        # --- Setting栏 初始化 --- #

        # --- MessageBar Init --- #
        self.showStatus("Welcome to YOLOSHOW")
        # --- MessageBar Init --- #

    def initModel(self,yoloname=None):
        # --- YOLOv5 QThread --- #
        if yoloname == "yolov5":
            self.yolov5_thread.parent_workpath = self.current_workpath + '\yolocode\yolov5'
            self.yolov5_thread.new_model_name = f'{self.current_workpath}/ptfiles/' + self.model_box.currentText()
            self.yolov5_thread.progress_value = self.progress_bar.maximum()
            self.yolov5_thread.send_input.connect(lambda x: self.showImg(x, self.main_leftbox, 'img'))
            self.yolov5_thread.send_output.connect(lambda x: self.showImg(x, self.main_rightbox, 'img'))
            self.yolov5_thread.send_msg.connect(lambda x: self.showStatus(x))
            self.yolov5_thread.send_progress.connect(lambda x: self.progress_bar.setValue(x))
            self.yolov5_thread.send_fps.connect(lambda x: self.fps_label.setText(str(x)))
            self.yolov5_thread.send_class_num.connect(lambda x: self.Class_num.setText(str(x)))
            self.yolov5_thread.send_target_num.connect(lambda x: self.Target_num.setText(str(x)))
            self.run_button.clicked.connect(self.runorContinue)
            self.stop_button.clicked.connect(self.stopDetect)
        # --- YOLOv5 QThread --- #

        # --- YOLOv7 QThread --- #
        elif yoloname == "yolov7":
            self.yolov7_thread.parent_workpath = self.current_workpath + '\yolocode\yolov7'
            self.yolov7_thread.new_model_name = f'{self.current_workpath}/ptfiles/' + self.model_box.currentText()
            self.yolov7_thread.progress_value = self.progress_bar.maximum()
            self.yolov7_thread.send_input.connect(lambda x: self.showImg(x, self.main_leftbox, 'img'))
            self.yolov7_thread.send_output.connect(lambda x: self.showImg(x, self.main_rightbox, 'img'))
            self.yolov7_thread.send_msg.connect(lambda x: self.showStatus(x))
            self.yolov7_thread.send_progress.connect(lambda x: self.progress_bar.setValue(x))
            self.yolov7_thread.send_fps.connect(lambda x: self.fps_label.setText(str(x)))
            self.yolov7_thread.send_class_num.connect(lambda x: self.Class_num.setText(str(x)))
            self.yolov7_thread.send_target_num.connect(lambda x: self.Target_num.setText(str(x)))
            self.run_button.clicked.connect(self.runorContinue)
            self.stop_button.clicked.connect(self.stopDetect)
        # --- YOLOv7 QThread --- #

        # --- YOLOv8 QThread --- #
        elif yoloname == "yolov8":
            self.yolov8_thread.parent_workpath = self.current_workpath + '\yolocode\yolov8'
            self.yolov8_thread.new_model_name = f'{self.current_workpath}/ptfiles/' + self.model_box.currentText()
            self.yolov8_thread.progress_value = self.progress_bar.maximum()
            self.yolov8_thread.send_input.connect(lambda x: self.showImg(x, self.main_leftbox, 'img'))
            self.yolov8_thread.send_output.connect(lambda x: self.showImg(x, self.main_rightbox, 'img'))
            self.yolov8_thread.send_msg.connect(lambda x: self.showStatus(x))
            self.yolov8_thread.send_progress.connect(lambda x: self.progress_bar.setValue(x))
            self.yolov8_thread.send_fps.connect(lambda x: self.fps_label.setText(str(x)))
            self.yolov8_thread.send_class_num.connect(lambda x: self.Class_num.setText(str(x)))
            self.yolov8_thread.send_target_num.connect(lambda x: self.Target_num.setText(str(x)))
            self.run_button.clicked.connect(self.runorContinue)
            self.stop_button.clicked.connect(self.stopDetect)
        # --- YOLOv8 QThread --- #

    # 阴影效果
    def shadowStyle(self, widget, Color, top_bottom=None):
        shadow = QGraphicsDropShadowEffect(self)
        if 'top' in top_bottom and 'bottom' not in top_bottom:
            shadow.setOffset(0, -5)
        elif 'bottom' in top_bottom and 'top' not in top_bottom:
            shadow.setOffset(0, 5)
        else:
            shadow.setOffset(5, 5)
        shadow.setBlurRadius(10)  # 阴影半径
        shadow.setColor(Color)  # 阴影颜色
        widget.setGraphicsEffect(shadow)

    # 侧边栏缩放
    def scaleMenu(self):
        standard = 80
        maxExtend = 180
        width = self.leftBox.width()

        if width == 80:
            widthExtended = maxExtend
        else:
            widthExtended = standard

        # animation
        self.animation = QPropertyAnimation(self.leftBox, b"minimumWidth")
        self.animation.setDuration(500)  # ms
        self.animation.setStartValue(width)
        self.animation.setEndValue(widthExtended)
        self.animation.setEasingCurve(QEasingCurve.InOutQuint)
        self.animation.start()

    # 设置栏缩放
    def scalSetting(self):
        # GET WIDTH
        widthSettingBox = self.settingBox.width()  # right set column width
        widthLeftBox = self.leftBox.width()  # left column length
        maxExtend = 220
        standard = 0

        # SET MAX WIDTH
        if widthSettingBox == 0:
            widthExtended = maxExtend
            self.mainbox.setStyleSheet("""
                                  QFrame#mainbox{
                                    border: 1px solid rgba(0, 0, 0, 15%);
                                    border-bottom-left-radius: 0;
                                    border-bottom-right-radius: 0;
                                    border-radius:30%;
                                    background-color: qlineargradient(x1:0, y1:0, x2:1 , y2:0, stop:0 white, stop:0.9 #8EC5FC, stop:1 #E0C3FC);
                                }
                              """)
        else:
            widthExtended = standard
            self.mainbox.setStyleSheet("""
                                  QFrame#mainbox{
                                    border: 1px solid rgba(0, 0, 0, 15%);
                                    border-bottom-left-radius: 0;
                                    border-bottom-right-radius: 0;
                                    border-radius:30%;
                                }
                              """)

        # ANIMATION LEFT BOX
        self.left_box = QPropertyAnimation(self.leftBox, b"minimumWidth")
        self.left_box.setDuration(500)
        self.left_box.setStartValue(widthLeftBox)
        self.left_box.setEndValue(68)
        self.left_box.setEasingCurve(QEasingCurve.InOutQuart)

        # ANIMATION SETTING BOX
        self.setting_box = QPropertyAnimation(self.settingBox, b"minimumWidth")
        self.setting_box.setDuration(500)
        self.setting_box.setStartValue(widthSettingBox)
        self.setting_box.setEndValue(widthExtended)
        self.setting_box.setEasingCurve(QEasingCurve.InOutQuart)

        # SET QSS Change
        self.qss_animation = QPropertyAnimation(self.mainbox, b"styleSheet")
        self.qss_animation.setDuration(300)
        self.qss_animation.setStartValue("""
            QFrame#mainbox {
                border: 1px solid rgba(0, 0, 0, 15%);
                border-bottom-left-radius: 0;
                border-bottom-right-radius: 0;
                border-radius:30%;
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 white, stop:0.9 #8EC5FC, stop:1 #E0C3FC);
            }
        """)
        self.qss_animation.setEndValue("""
             QFrame#mainbox {
                border: 1px solid rgba(0, 0, 0, 15%);
                border-bottom-left-radius: 0;
                border-bottom-right-radius: 0;
                border-radius:30%;
            }
        """)
        self.qss_animation.setEasingCurve(QEasingCurve.InOutQuart)

        # GROUP ANIMATION
        self.group = QParallelAnimationGroup()
        self.group.addAnimation(self.left_box)
        self.group.addAnimation(self.setting_box)
        self.group.start()

    # 最大化最小化窗口
    def maxorRestore(self):
        global GLOBAL_WINDOW_STATE
        status = GLOBAL_WINDOW_STATE
        if status:
            self.showMaximized()
            self.maximizeButton.setStyleSheet("""
                          QPushButton:hover{
                               background-color:rgb(139, 29, 31);
                               border-image: url(:/leftbox/images/newsize/scalling.png);
                           }
                      """)
            GLOBAL_WINDOW_STATE = False
        else:
            self.showNormal()
            self.maximizeButton.setStyleSheet("""
                                      QPushButton:hover{
                                           background-color:rgb(139, 29, 31);
                                           border-image: url(:/leftbox/images/newsize/max.png);
                                       }
                                  """)
            GLOBAL_WINDOW_STATE = True

    # 选择照片/视频 并展示
    def selectFile(self):
        # 获取上次选择文件的路径
        config_file = f'{self.current_workpath}/config/file.json'
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        file_path = config['file_path']
        if not os.path.exists(file_path):
            file_path = os.getcwd()
        file, _ = QFileDialog.getOpenFileName(
            self,  # 父窗口对象
            "Select your Image / Video",  # 标题
            file_path,  # 默认打开路径为当前路径
            "Image / Video type (*.jpg *.jpeg *.png *.bmp *.dib *.jpe *.jp2 *.mp4)"  # 选择类型过滤项，过滤内容在括号中
        )
        if file:
            self.inputPath = file
            glo.set_value('inputPath', self.inputPath)
            # 如果是视频， 显示第一帧
            if ".avi" in self.inputPath or ".mp4" in self.inputPath:
                # 显示第一帧
                self.cap = cv2.VideoCapture(self.inputPath)
                ret, frame = self.cap.read()
                if ret:
                    # rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.showImg(frame, self.main_leftbox, 'img')
            # 如果是图片 正常显示
            else:
                self.showImg(self.inputPath, self.main_leftbox, 'path')
            self.showStatus('Loaded File：{}'.format(os.path.basename(self.inputPath)))
            config['file_path'] = os.path.dirname(self.inputPath)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)

    # 选择摄像头
    def selectWebcam(self):
        try:
            # get the number of local cameras
            cam_num, cams = Camera().get_cam_num()
            if cam_num > 0:
                popMenu = RoundMenu(parent=self)
                popMenu.setFixedWidth(self.leftbox_bottom.width())
                actions = []

                for cam in cams:
                    cam_name = f'Camera_{cam}'
                    actions.append(Action(cam_name))
                    popMenu.addAction(actions[-1])
                    actions[-1].triggered.connect(lambda: self.actionWebcam(cam))

                x = self.webcamBox.mapToGlobal(self.webcamBox.pos()).x()
                y = self.webcamBox.mapToGlobal(self.webcamBox.pos()).y()
                y = y - self.webcamBox.frameGeometry().height() * 2
                pos = QPoint(x, y)
                popMenu.exec(pos, aniType=MenuAnimationType.DROP_DOWN)
            else:
                self.showStatus('No camera found !!!')
        except Exception as e:
            self.showStatus('%s' % e)

    # 调用网络摄像头
    def actionWebcam(self, cam):
        self.showStatus(f'Loading camera：Camera_{cam}')
        self.thread = WebcamThread(cam)
        self.thread.changePixmap.connect(lambda x: self.showImg(x, self.main_leftbox, 'img'))
        self.thread.start()
        self.inputPath = int(cam)

    # 选择文件夹
    def selectFolder(self):
        config_file = f'{self.current_workpath}/config/folder.json'
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        folder_path = config['folder_path']
        if not os.path.exists(folder_path):
            folder_path = os.getcwd()
        FolderPath = QFileDialog.getExistingDirectory(
            self,
            "Select your Folder",
            folder_path  # 起始目录
        )
        if FolderPath:
            FileFormat = [".mp4", ".mkv", ".avi", ".flv", ".jpg", ".png", ".jpeg",".bmp",".dib",".jpe",".jp2"]
            Foldername = [(FolderPath + "/" + filename) for filename in os.listdir(FolderPath) for jpgname in FileFormat
                          if jpgname in filename]
            # self.yolov5_thread.source = Foldername
            self.inputPath = Foldername
            self.showStatus('Loaded Folder：{}'.format(os.path.basename(FolderPath)))
            config['folder_path'] = FolderPath
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)

    # 显示Label图片
    @staticmethod
    def showImg(img, label, flag):
        try:
            if flag == "path":
                img_src = cv2.imdecode(np.fromfile(img, dtype=np.uint8), -1)
            else:
                img_src = img
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # keep original aspect ratio
            if iw / w > ih / h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))
            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))
        except Exception as e:
            print(repr(e))

    # resize 窗口大小
    def resizeGrip(self):
        self.left_grip.setGeometry(0, 10, 10, self.height())
        self.right_grip.setGeometry(self.width() - 10, 10, 10, self.height())
        self.top_grip.setGeometry(0, 0, self.width(), 10)
        self.bottom_grip.setGeometry(0, self.height() - 10, self.width(), 10)

    # 在MessageBar显示消息
    def showStatus(self, msg):
        self.message_bar.setText(msg)
        if msg == 'Finish Detection':
            self.run_button.setChecked(False)
            self.progress_bar.setValue(0)
            if self.yolov5_thread.isRunning():
                self.yolov5_thread.quit()  # end process
            elif self.yolov7_thread.isRunning():
                self.yolov7_thread.quit()
            self.save_status_button.setEnabled(True)
        elif msg == 'Stop Detection':
            if self.yolov5_thread.isRunning():
                self.yolov5_thread.quit()  # end process
            elif self.yolov7_thread.isRunning():
                self.yolov7_thread.quit()
            self.run_button.setChecked(False)
            self.save_status_button.setEnabled(True)
            self.progress_bar.setValue(0)
            self.main_leftbox.clear()  # clear image display
            self.main_rightbox.clear()
            self.Class_num.setText('--')
            self.Target_num.setText('--')
            self.fps_label.setText('--')

    # 导入模块
    def importModel(self):
        # 获取上次选择文件的路径
        config_file = f'{self.current_workpath}/config/model.json'
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        self.model_path = config['model_path']
        if not os.path.exists(self.model_path):
            self.model_path = os.getcwd()
        file, _ = QFileDialog.getOpenFileName(
            self,  # 父窗口对象
            "Select your YOLO Model",  # 标题
            self.model_path,  # 默认打开路径为当前路径
            "Model File (*.pt)"  # 选择类型过滤项，过滤内容在括号中
        )
        if file:
            fileptPath = os.path.join(self.pt_Path,os.path.basename(file))
            if not os.path.exists(fileptPath):
                shutil.copy(file, self.pt_Path)
                self.showStatus('Loaded Model：{}'.format(os.path.basename(file)))
                config['model_path'] = os.path.dirname(file)
                config_json = json.dumps(config, ensure_ascii=False, indent=2)
                with open(config_file, 'w', encoding='utf-8') as f:
                    f.write(config_json)
            else:
                self.showStatus('Model already exists')

    # 导出结果状态判断
    def saveStatus(self):
        if self.save_status_button.checkState() == Qt.CheckState.Unchecked:
            self.showStatus('NOTE: Run image results are not saved.')
            self.yolov5_thread.save_res = False
            self.yolov7_thread.save_res = False
            self.yolov8_thread.save_res = False
            self.save_button.setEnabled(False)
        elif self.save_status_button.checkState() == Qt.CheckState.Checked:
            self.showStatus('NOTE: Run image results will be saved.')
            self.yolov5_thread.save_res = True
            self.yolov7_thread.save_res = True
            self.yolov8_thread.save_res = True
            self.save_button.setEnabled(True)

    # 导出结果
    def saveResult(self):
        if not self.yolov5_thread.res_status and not self.yolov7_thread.res_status and not self.yolov8_thread.res_status:
            self.showStatus("Please select the Image/Video before starting detection...")
            return
        config_file = f'{self.current_workpath}/config/save.json'
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        save_path = config['save_path']
        if not os.path.exists(save_path):
            save_path = os.getcwd()
        is_folder = isinstance(self.inputPath,list)
        if is_folder:
            self.OutputDir = QFileDialog.getExistingDirectory(
                self,  # 父窗口对象
                "Save Results in new Folder",  # 标题
                save_path,  # 起始目录
            )
            if "yolov5" in self.model_name:
                try:
                    output_dir = os.path.dirname(self.yolov5_thread.res_path)
                    if os.path.exists(output_dir):
                        for filename in os.listdir(output_dir):
                            source_path = os.path.join(output_dir, filename)
                            destination_path = os.path.join(self.OutputDir, filename)
                            if os.path.isfile(source_path):
                                shutil.copy(source_path, destination_path)
                        self.showStatus('Saved Successfully in {}'.format(self.OutputDir))
                    else:
                        self.showStatus('Please wait for the result to be generated')
                except Exception as err:
                    self.showStatus(f"Error occurred while saving the result: {err}")
            elif "yolov7" in self.model_name:
                try:
                    output_dir = os.path.dirname(self.yolov7_thread.res_path)
                    if os.path.exists(output_dir):
                        for filename in os.listdir(output_dir):
                            source_path = os.path.join(output_dir, filename)
                            destination_path = os.path.join( self.OutputDir, filename)
                            if os.path.isfile(source_path):
                                shutil.copy(source_path, destination_path)
                        self.showStatus('Saved Successfully in {}'.format(self.OutputDir))
                    else:
                        self.showStatus('Please wait for the result to be generated')
                except Exception as err:
                    self.showStatus(f"Error occurred while saving the result: {err}")
            elif "yolov8" in self.model_name:
                try:
                    output_dir = os.path.dirname(self.yolov8_thread.res_path)
                    if os.path.exists(output_dir):
                        for filename in os.listdir(output_dir):
                            source_path = os.path.join(output_dir, filename)
                            destination_path = os.path.join( self.OutputDir, filename)
                            if os.path.isfile(source_path):
                                shutil.copy(source_path, destination_path)
                        self.showStatus('Saved Successfully in {}'.format(self.OutputDir))
                    else:
                        self.showStatus('Please wait for the result to be generated')
                except Exception as err:
                    self.showStatus(f"Error occurred while saving the result: {err}")
        else:
            self.OutputDir, _ = QFileDialog.getSaveFileName(
                self,  # 父窗口对象
                "Save Image/Video",  # 标题
                save_path,  # 起始目录
                "Image/Vide Type (*.jpg *.jpeg *.png *.bmp *.dib  *.jpe  *.jp2 *.mp4)"  # 选择类型过滤项，过滤内容在括号中
            )
            if "yolov5" in self.model_name:
                try:
                    if os.path.exists(self.yolov5_thread.res_path):
                        shutil.copy(self.yolov5_thread.res_path, self.OutputDir)
                        self.showStatus('Saved Successfully in {}'.format(self.OutputDir))
                    else:
                        self.showStatus('Please wait for the result to be generated')
                except Exception as err:
                    self.showStatus(f"Error occurred while saving the result: {err}")
            elif "yolov7" in self.model_name:
                try:
                    if os.path.exists(self.yolov7_thread.res_path):
                        shutil.copy(self.yolov7_thread.res_path, self.OutputDir)
                        self.showStatus('Saved Successfully in {}'.format(self.OutputDir))
                    else:
                        self.showStatus('Please wait for the result to be generated')
                except Exception as err:
                    self.showStatus(f"Error occurred while saving the result: {err}")
            elif "yolov8" in self.model_name:
                try:
                    if os.path.exists(self.yolov8_thread.res_path):
                        shutil.copy(self.yolov8_thread.res_path, self.OutputDir)
                        self.showStatus('Saved Successfully in {}'.format(self.OutputDir))
                    else:
                        self.showStatus('Please wait for the result to be generated')
                except Exception as err:
                    self.showStatus(f"Error occurred while saving the result: {err}")

        config['save_path'] = self.OutputDir
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)

    # 调整超参数
    def changeValue(self, x, flag):
        if flag == 'iou_spinbox':
            self.iou_slider.setValue(int(x * 100))  # The box value changes, changing the slider
        elif flag == 'iou_slider':
            self.iou_spinbox.setValue(x / 100)  # The slider value changes, changing the box
            self.showStatus('IOU Threshold: %s' % str(x / 100))
            self.yolov5_thread.iou_thres = x / 100
            self.yolov7_thread.iou_thres = x / 100
            self.yolov8_thread.iou_thres = x / 100
        elif flag == 'conf_spinbox':
            self.conf_slider.setValue(int(x * 100))
        elif flag == 'conf_slider':
            self.conf_spinbox.setValue(x / 100)
            self.showStatus('Conf Threshold: %s' % str(x / 100))
            self.yolov5_thread.conf_thres = x / 100
            self.yolov7_thread.conf_thres = x / 100
            self.yolov8_thread.conf_thres = x / 100
        elif flag == 'speed_spinbox':
            self.speed_slider.setValue(x)
        elif flag == 'speed_slider':
            self.speed_spinbox.setValue(x)
            self.showStatus('Delay: %s ms' % str(x))
            self.yolov5_thread.speed_thres = x  # ms
            self.yolov7_thread.speed_thres = x  # ms
            self.yolov8_thread.speed_thres = x  # ms
        elif flag == 'line_spinbox':
            self.line_slider.setValue(x)
        elif flag == 'line_slider':
            self.line_spinbox.setValue(x)
            self.showStatus('Line Width: %s' % str(x))
            self.yolov5_thread.line_thickness = x
            self.yolov7_thread.line_thickness = x
            self.yolov8_thread.line_thickness = x

    # 加载 Setting 栏
    def loadConfig(self):
        config_file = 'config/setting.json'
        iou = 0.45
        conf = 0.25
        delay = 10
        line_thickness = 3
        if not os.path.exists(config_file):
            iou = 0.45
            conf = 0.25
            delay = 10
            line_thickness =  3
            new_config = {"iou": iou,
                          "conf": conf,
                          "delay": delay,
                          "line_thickness": line_thickness,
                          }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            if len(config) != 4:
                iou = 0.45
                conf = 0.25
                delay = 10
                line_thickness = 3
            else:
                iou = config['iou']
                conf = config['conf']
                delay = config['delay']
                line_thickness = config['line_thickness']
        self.iou_spinbox.setValue(iou)
        self.iou_slider.setValue(int(iou * 100))
        self.conf_spinbox.setValue(conf)
        self.conf_slider.setValue(int(conf * 100))
        self.speed_spinbox.setValue(delay)
        self.speed_slider.setValue(delay)
        self.line_spinbox.setValue(line_thickness)
        self.line_slider.setValue(line_thickness)

    # 加载 pt 模型到 model_box
    def loadModels(self):
        pt_list = os.listdir(f'{self.current_workpath}/ptfiles/')
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize(f'{self.current_workpath}/ptfiles/' + x))

        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.model_box.clear()
            self.model_box.addItems(self.pt_list)

    # 重新加载模型
    def resignModel(self,yoloname):
        if yoloname == "yolov5":
            del self.yolov5_thread
            self.yolov5_thread = YOLOv5Thread()
            self.initModel("yolov5")
            self.runModel(True)
        elif yoloname == "yolov7":
            del self.yolov7_thread
            self.yolov7_thread = YOLOv7Thread()
            self.initModel("yolov7")
            self.runModel(True)
        elif yoloname == "yolov8":
            del self.yolov8_thread
            self.yolov8_thread = YOLOv8Thread()
            self.initModel("yolov8")
            self.runModel(True)

    # 停止其他模型
    def stopOtherModel(self,yoloname):
        if self.yolov5_thread.isRunning() and yoloname == "yolov5":
            self.yolov5_thread.quit()
            self.yolov5_thread.stop_dtc = True
            self.yolov5_thread.finished.connect(lambda :self.resignModel(yoloname))
        elif self.yolov7_thread.isRunning() and yoloname == "yolov7":
            self.yolov7_thread.quit()
            self.yolov7_thread.stop_dtc = True
            self.yolov7_thread.finished.connect(lambda: self.resignModel(yoloname))
        elif self.yolov8_thread.isRunning() and yoloname == "yolov8":
            self.yolov8_thread.quit()
            self.yolov8_thread.stop_dtc = True
            self.yolov8_thread.finished.connect(lambda: self.resignModel(yoloname))

    # Model 变化
    def changeModel(self, x):
        self.model_name = self.model_box.currentText()
        self.Model_label.setText(self.model_name)   # 修改状态栏显示
        if "yolov5" in self.model_name:
            self.yolov5_thread.new_model_name = f'{self.current_workpath}/ptfiles/' + self.model_box.currentText()
            # 重载 common 和 yolo 模块
            glo.set_value('yoloname', "yolov5")
            import importlib
            importlib.reload(common)
            importlib.reload(yolo)
            # 停止其他模型
            self.stopOtherModel("yolov7")
            self.stopOtherModel("yolov8")
        elif "yolov7" in self.model_name:
            self.yolov7_thread.new_model_name = f'{self.current_workpath}/ptfiles/' + self.model_box.currentText()
            # 重载 common 和 yolo 模块
            glo.set_value('yoloname', "yolov7")
            import importlib
            importlib.reload(common)
            importlib.reload(yolo)
            # 停止其他模型
            self.stopOtherModel("yolov5")
            self.stopOtherModel("yolov8")
        elif "yolov8" in self.model_name:
            self.yolov8_thread.new_model_name = f'{self.current_workpath}/ptfiles/' + self.model_box.currentText()
            # 重载 common 和 yolo 模块
            glo.set_value('yoloname', "yolov8")
            import importlib
            importlib.reload(common)
            importlib.reload(yolo)
            # 停止其他模型
            self.stopOtherModel("yolov5")
            self.stopOtherModel("yolov7")
        else:
            self.stopOtherModel("yolov5")
            self.stopOtherModel("yolov7")
            self.stopOtherModel("yolov8")

    def runModel(self,runbuttonStatus=None):
        self.save_status_button.setEnabled(False)
        if runbuttonStatus:
            self.run_button.setChecked(True)
        if "yolov5" in self.model_name:
            self.yolov5_thread.source = self.inputPath
            self.yolov5_thread.stop_dtc = False
            if self.run_button.isChecked():
                self.yolov5_thread.is_continue = True
                if not self.yolov5_thread.isRunning():
                    self.yolov5_thread.start()
            else:
                self.yolov5_thread.is_continue = False
                self.showStatus('Pause Detection')
        elif "yolov7" in self.model_name:
            self.yolov7_thread.source = self.inputPath
            self.yolov7_thread.stop_dtc = False
            if self.run_button.isChecked():
                self.yolov7_thread.is_continue = True
                if not self.yolov7_thread.isRunning():
                    self.yolov7_thread.start()
            else:
                self.yolov7_thread.is_continue = False
                self.showStatus('Pause Detection')
        elif "yolov8" in self.model_name:
            self.yolov8_thread.source = self.inputPath
            self.yolov8_thread.stop_dtc = False
            if self.run_button.isChecked():
                self.yolov8_thread.is_continue = True
                if not self.yolov8_thread.isRunning():
                    self.yolov8_thread.start()
            else:
                self.yolov8_thread.is_continue = False
                self.showStatus('Pause Detection')
        else:
            self.showStatus('The current model is not supported')
            if self.run_button.isChecked():
                self.run_button.setChecked(False)

    # 开始/暂停 预测
    def runorContinue(self):
        if self.inputPath is not None:
            self.runModel()
        else:
            self.showStatus("Please select the Image/Video before starting detection...")
            self.run_button.setChecked(False)

    # 停止识别
    def stopDetect(self):
        if "yolov5" in self.model_name:
            if self.yolov5_thread.isRunning():
                self.yolov5_thread.quit()  # end process
            self.yolov5_thread.stop_dtc = True
        elif "yolov7" in self.model_name:
            if self.yolov7_thread.isRunning():
                self.yolov7_thread.quit()
            self.yolov7_thread.stop_dtc = True
        elif "yolov8" in self.model_name:
            if self.yolov8_thread.isRunning():
                self.yolov8_thread.quit()
            self.yolov8_thread.stop_dtc = True
        self.run_button.setChecked(False)
        self.save_status_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.main_leftbox.clear()  # clear image display
        self.main_rightbox.clear()
        self.Class_num.setText('--')
        self.Target_num.setText('--')
        self.fps_label.setText('--')


# 多套一个类 为了实现MouseLabel方法
class MyWindow(YOLOSHOW):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.center()
        # --- 拖动窗口 改变窗口大小 --- #
        self.left_grip = CustomGrip(self, Qt.LeftEdge, True)
        self.right_grip = CustomGrip(self, Qt.RightEdge, True)
        self.top_grip = CustomGrip(self, Qt.TopEdge, True)
        self.bottom_grip = CustomGrip(self, Qt.BottomEdge, True)
        # --- 拖动窗口 改变窗口大小 --- #

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.mouse_start_pt = event.globalPosition().toPoint()
            self.window_pos = self.frameGeometry().topLeft()
            self.drag = True

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.drag:
            distance = event.globalPosition().toPoint() - self.mouse_start_pt
            self.move(self.window_pos + distance)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.drag = False

    def center(self):
        # PyQt6获取屏幕参数
        screen = QGuiApplication.primaryScreen().size()
        size = self.geometry()
        self.move((screen.width() - size.width()) / 2,
                  (screen.height() - size.height()) / 2 - 10)

    # 拖动窗口 改变窗口大小
    def resizeEvent(self, event):
        # Update Size Grips
        self.resizeGrip()
    def closeEvent(self, event):
        config_file = 'config/setting.json'
        config = dict()
        config['iou'] = self.iou_spinbox.value()
        config['conf'] = self.conf_spinbox.value()
        config['delay'] = self.speed_spinbox.value()
        config['line_thickness'] = self.line_spinbox.value()
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)