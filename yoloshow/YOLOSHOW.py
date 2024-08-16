from utils import glo

glo._init()
glo.set_value('yoloname', "yolov5 yolov7 yolov8 yolov9 yolov10 yolov5-seg yolov8-seg rtdetr yolov8-pose yolov8-obb")

import json
import os
import shutil
from ui.YOLOSHOWUI import Ui_MainWindow
from PySide6.QtGui import QPixmap, QImage, QMouseEvent, QGuiApplication, QColor
from PySide6.QtWidgets import QMessageBox, QFileDialog, QMainWindow, QWidget, QApplication, QGraphicsBlurEffect, \
    QGraphicsDropShadowEffect, QMenu, QFrame, QPushButton
from PySide6.QtUiTools import QUiLoader, loadUiType
from PySide6.QtCore import QFile, QTimer, Qt, QEventLoop, QThread, QPropertyAnimation, QEasingCurve, \
    QParallelAnimationGroup, QPoint, Signal
from PySide6 import QtCore, QtGui
from yolocode.yolov5.YOLOv5Thread import YOLOv5Thread
from yolocode.yolov7.YOLOv7Thread import YOLOv7Thread
from yolocode.yolov8.YOLOv8Thread import YOLOv8Thread
from yolocode.yolov9.YOLOv9Thread import YOLOv9Thread
from yolocode.yolov5.YOLOv5SegThread import YOLOv5SegThread
from yolocode.yolov8.YOLOv8SegThread import YOLOv8SegThread
from yolocode.yolov8.RTDETRThread import RTDETRThread
from yolocode.yolov8.YOLOv8PoseThread import YOLOv8PoseThread
from yolocode.yolov8.YOLOv8ObbThread import YOLOv8ObbThread
from yolocode.yolov10.YOLOv10Thread import YOLOv10Thread
from yoloshow.YOLOSHOWBASE import YOLOSHOWBASE

GLOBAL_WINDOW_STATE = True
WIDTH_LEFT_BOX_STANDARD = 80
WIDTH_LEFT_BOX_EXTENDED = 200
WIDTH_LOGO = 60
UI_FILE_PATH = "ui/YOLOSHOWUI.ui"

KEYS_LEFT_BOX_MENU = ['src_menu', 'src_setting', 'src_webcam', 'src_folder', 'src_camera', 'src_vsmode', 'src_setting']
ALL_MODEL_NAMES = ["yolov5", "yolov7", "yolov8", "yolov9", "yolov10", "yolov5-seg", "yolov8-seg", "rtdetr",
                   "yolov8-pose", "yolov8-obb"]


# YOLOSHOW窗口类 动态加载UI文件 和 Ui_mainWindow
class YOLOSHOW(QMainWindow, YOLOSHOWBASE):
    def __init__(self):
        super().__init__()
        self.current_workpath = os.getcwd()
        self.inputPath = None
        self.allModelNames = ALL_MODEL_NAMES
        self.result_statistic = None
        self.detect_result = None

        # --- 加载UI --- #
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setAttribute(Qt.WA_TranslucentBackground, True)  # 透明背景
        self.setWindowFlags(Qt.FramelessWindowHint)  # 无头窗口
        self.initSiderWidget()
        # --- 加载UI --- #

        # --- 最大化 最小化 关闭 --- #
        self.ui.maximizeButton.clicked.connect(self.maxorRestore)
        self.ui.minimizeButton.clicked.connect(self.showMinimized)
        self.ui.closeButton.clicked.connect(self.close)
        self.ui.topbox.doubleClickFrame.connect(self.maxorRestore)
        # --- 最大化 最小化 关闭 --- #

        # --- 播放 暂停 停止 --- #
        self.playIcon = QtGui.QIcon()
        self.playIcon.addPixmap(QtGui.QPixmap(f"{self.current_workpath}/images/newsize/play.png"), QtGui.QIcon.Normal,
                                QtGui.QIcon.Off)
        self.playIcon.addPixmap(QtGui.QPixmap(f"{self.current_workpath}/images/newsize/pause.png"), QtGui.QIcon.Active,
                                QtGui.QIcon.On)
        self.playIcon.addPixmap(QtGui.QPixmap(f"{self.current_workpath}/images/newsize/pause.png"),
                                QtGui.QIcon.Selected, QtGui.QIcon.On)
        self.ui.run_button.setCheckable(True)
        self.ui.run_button.setIcon(self.playIcon)
        # --- 播放 暂停 停止 --- #

        # --- 侧边栏缩放 --- #
        self.ui.src_menu.clicked.connect(self.scaleMenu)  # hide menu button
        self.ui.src_setting.clicked.connect(self.scalSetting)  # setting button
        # --- 侧边栏缩放 --- #

        # --- 自动加载/动态改变 PT 模型 --- #
        self.pt_Path = f"{self.current_workpath}/ptfiles/"
        self.pt_list = os.listdir(f'{self.current_workpath}/ptfiles/')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.pt_list.sort(key=lambda x: os.path.getsize(f'{self.current_workpath}/ptfiles/' + x))
        self.ui.model_box.clear()
        self.ui.model_box.addItems(self.pt_list)
        self.qtimer_search = QTimer(self)
        self.qtimer_search.timeout.connect(lambda: self.loadModels())
        self.qtimer_search.start(2000)
        self.ui.model_box.currentTextChanged.connect(self.changeModel)
        # --- 自动加载/动态改变 PT 模型 --- #

        # --- 导入 图片/视频、调用摄像头、导入文件夹（批量处理）、调用网络摄像头、结果统计图片、结果统计表格 --- #
        self.ui.src_img.clicked.connect(self.selectFile)
        self.ui.src_webcam.clicked.connect(self.selectWebcam)
        self.ui.src_folder.clicked.connect(self.selectFolder)
        self.ui.src_camera.clicked.connect(self.selectRtsp)
        self.ui.src_result.clicked.connect(self.showResultStatics)
        self.ui.src_table.clicked.connect(self.showTableResult)
        # --- 导入 图片/视频、调用摄像头、导入文件夹（批量处理）、调用网络摄像头 --- #

        # --- 导入模型、 导出结果 --- #
        self.ui.import_button.clicked.connect(self.importModel)
        self.ui.save_status_button.clicked.connect(self.saveStatus)
        self.ui.save_button.clicked.connect(self.saveResult)
        self.ui.save_button.setEnabled(False)
        # --- 导入模型、 导出结果 --- #

        # --- 视频、图片 预览 --- #
        self.ui.main_leftbox.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        self.ui.main_rightbox.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        # --- 视频、图片 预览 --- #

        # --- 状态栏 初始化 --- #
        # 状态栏阴影效果
        self.shadowStyle(self.ui.mainBody, QColor(0, 0, 0, 38), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.Class_QF, QColor(142, 197, 252), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.classesLabel, QColor(142, 197, 252), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.Target_QF, QColor(159, 172, 230), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.targetLabel, QColor(159, 172, 230), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.Fps_QF, QColor(170, 128, 213), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.fpsLabel, QColor(170, 128, 213), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.Model_QF, QColor(162, 129, 247), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.modelLabel, QColor(162, 129, 247), top_bottom=['top', 'bottom'])
        # 状态栏默认显示
        self.model_name = self.ui.model_box.currentText()  # 获取默认 model
        self.ui.Class_num.setText('--')
        self.ui.Target_num.setText('--')
        self.ui.fps_label.setText('--')
        self.ui.Model_label.setText(str(self.model_name).replace(".pt", ""))
        # --- 状态栏 初始化 --- #

        # --- YOLOv5 QThread --- #
        self.yolov5_thread = YOLOv5Thread()
        self.initModel(self.yolov5_thread, "yolov5")
        # --- YOLOv5 QThread --- #

        # --- YOLOv7 QThread --- #
        self.yolov7_thread = YOLOv7Thread()
        self.initModel(self.yolov7_thread, "yolov7")
        # --- YOLOv7 QThread --- #

        # --- YOLOv8 QThread --- #
        self.yolov8_thread = YOLOv8Thread()
        self.initModel(self.yolov8_thread, "yolov8")
        # --- YOLOv8 QThread --- #

        # --- YOLOv9 QThread --- #
        self.yolov9_thread = YOLOv9Thread()
        self.initModel(self.yolov9_thread, "yolov9")
        # --- YOLOv9 QThread --- #

        # --- YOLOv9 QThread --- #
        self.yolov10_thread = YOLOv10Thread()
        self.initModel(self.yolov10_thread, "yolov10")
        # --- YOLOv9 QThread --- #

        # --- YOLOv5-Seg QThread --- #
        self.yolov5seg_thread = YOLOv5SegThread()
        self.initModel(self.yolov5seg_thread, "yolov5-seg")
        # --- YOLOv5-Seg QThread --- #

        # --- YOLOv8-Seg QThread --- #
        self.yolov8seg_thread = YOLOv8SegThread()
        self.initModel(self.yolov8seg_thread, "yolov8-seg")
        # --- YOLOv8-Seg QThread --- #

        # --- RT-DETR QThread --- #
        self.rtdetr_thread = RTDETRThread()
        self.initModel(self.rtdetr_thread, "rtdetr")
        # --- RT-DETR QThread --- #

        # --- YOLOv8-Pose QThread --- #
        self.yolov8pose_thread = YOLOv8PoseThread()
        self.initModel(self.yolov8pose_thread, "yolov8-pose")
        # --- YOLOv8-Pose QThread --- #

        # --- YOLOv8-Obb QThread --- #
        self.yolov8obb_thread = YOLOv8ObbThread()
        self.initModel(self.yolov8obb_thread, "yolov8-pose")
        # --- YOLOv8-Obb QThread --- #

        self.initThreads()

        # --- 超参数调整 --- #
        self.ui.iou_spinbox.valueChanged.connect(
            lambda x: self.changeValue(x, 'iou_spinbox'))  # iou box
        self.ui.iou_slider.valueChanged.connect(lambda x: self.changeValue(x, 'iou_slider'))  # iou scroll bar
        self.ui.conf_spinbox.valueChanged.connect(lambda x: self.changeValue(x, 'conf_spinbox'))  # conf box
        self.ui.conf_slider.valueChanged.connect(lambda x: self.changeValue(x, 'conf_slider'))  # conf scroll bar
        self.ui.speed_spinbox.valueChanged.connect(lambda x: self.changeValue(x, 'speed_spinbox'))  # speed box
        self.ui.speed_slider.valueChanged.connect(lambda x: self.changeValue(x, 'speed_slider'))  # speed scroll bar
        self.ui.line_spinbox.valueChanged.connect(lambda x: self.changeValue(x, 'line_spinbox'))  # line box
        self.ui.line_slider.valueChanged.connect(lambda x: self.changeValue(x, 'line_slider'))  # line slider
        # --- 超参数调整 --- #

        # --- 开始 / 停止 --- #
        self.ui.run_button.clicked.connect(self.runorContinue)
        self.ui.stop_button.clicked.connect(self.stopDetect)
        # --- 开始 / 停止 --- #

        # --- Setting栏 初始化 --- #
        self.loadConfig()
        # --- Setting栏 初始化 --- #

        # --- MessageBar Init --- #
        self.showStatus("Welcome to YOLOSHOW")
        # --- MessageBar Init --- #

    def initThreads(self):
        self.yolo_threads = [self.yolov5_thread, self.yolov7_thread, self.yolov8_thread, self.yolov9_thread,
                             self.yolov10_thread, self.rtdetr_thread,
                             self.yolov5seg_thread, self.yolov8seg_thread, self.yolov8pose_thread,
                             self.yolov8obb_thread]

    # 导出结果
    def saveResult(self):
        if (not self.yolov5_thread.res_status and not self.yolov7_thread.res_status
                and not self.yolov8_thread.res_status and not self.yolov9_thread.res_status and not self.yolov10_thread.res_status
                and not self.yolov5seg_thread.res_status and not self.yolov8seg_thread.res_status
                and not self.rtdetr_thread.res_status and not self.yolov8pose_thread.res_status
                and not self.yolov8obb_thread.res_status):
            self.showStatus("Please select the Image/Video before starting detection...")
            return
        config_file = f'{self.current_workpath}/config/save.json'
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        save_path = config['save_path']
        if not os.path.exists(save_path):
            save_path = os.getcwd()
        is_folder = isinstance(self.inputPath, list)
        if is_folder:
            self.OutputDir = QFileDialog.getExistingDirectory(
                self,  # 父窗口对象
                "Save Results in new Folder",  # 标题
                save_path,  # 起始目录
            )
            if "yolov5" in self.model_name and not self.checkSegName(self.model_name):
                self.saveResultProcess(self.OutputDir, self.yolov5_thread, folder=True)
            elif "yolov7" in self.model_name:
                self.saveResultProcess(self.OutputDir, self.yolov7_thread, folder=True)
            elif "yolov8" in self.model_name and not self.checkSegName(self.model_name) and not self.checkPoseName(
                    self.model_name) and not self.checkObbName(self.model_name):
                self.saveResultProcess(self.OutputDir, self.yolov8_thread, folder=True)
            elif "yolov9" in self.model_name:
                self.saveResultProcess(self.OutputDir, self.yolov9_thread, folder=True)
            elif "yolov10" in self.model_name:
                self.saveResultProcess(self.OutputDir, self.yolov10_thread, folder=True)
            elif "yolov5" in self.model_name and self.checkSegName(self.model_name):
                self.saveResultProcess(self.OutputDir, self.yolov5seg_thread, folder=True)
            elif "yolov8" in self.model_name and self.checkSegName(self.model_name):
                self.saveResultProcess(self.OutputDir, self.yolov8seg_thread, folder=True)
            elif "rtdetr" in self.model_name:
                self.saveResultProcess(self.OutputDir, self.rtdetr_thread, folder=True)
            elif "yolov8" in self.model_name and self.checkPoseName(self.model_name):
                self.saveResultProcess(self.OutputDir, self.yolov8pose_thread, folder=True)
            elif "yolov8" in self.model_name and self.checkObbName(self.model_name):
                self.saveResultProcess(self.OutputDir, self.yolov8obb_thread, folder=True)
        else:
            self.OutputDir, _ = QFileDialog.getSaveFileName(
                self,  # 父窗口对象
                "Save Image/Video",  # 标题
                save_path,  # 起始目录
                "Image/Vide Type (*.jpg *.jpeg *.png *.bmp *.dib  *.jpe  *.jp2 *.mp4)"  # 选择类型过滤项，过滤内容在括号中
            )
            if "yolov5" in self.model_name and not self.checkSegName(self.model_name):
                self.saveResultProcess(self.OutputDir, self.yolov5_thread, folder=False)
            elif "yolov7" in self.model_name:
                self.saveResultProcess(self.OutputDir, self.yolov7_thread, folder=False)
            elif "yolov8" in self.model_name and not self.checkSegName(self.model_name) and not self.checkPoseName(
                    self.model_name) and not self.checkObbName(self.model_name):
                self.saveResultProcess(self.OutputDir, self.yolov8_thread, folder=False)
            elif "yolov9" in self.model_name:
                self.saveResultProcess(self.OutputDir, self.yolov9_thread, folder=False)
            elif "yolov10" in self.model_name:
                self.saveResultProcess(self.OutputDir, self.yolov10_thread, folder=False)
            elif "yolov5" in self.model_name and self.checkSegName(self.model_name):
                self.saveResultProcess(self.OutputDir, self.yolov5seg_thread, folder=False)
            elif "yolov8" in self.model_name and self.checkSegName(self.model_name):
                self.saveResultProcess(self.OutputDir, self.yolov8seg_thread, folder=False)
            elif "rtdetr" in self.model_name:
                self.saveResultProcess(self.OutputDir, self.rtdetr_thread, folder=False)
            elif "yolov8" in self.model_name and self.checkPoseName(self.model_name):
                self.saveResultProcess(self.OutputDir, self.yolov8pose_thread, folder=False)
            elif "yolov8" in self.model_name and self.checkObbName(self.model_name):
                self.saveResultProcess(self.OutputDir, self.yolov8obb_thread, folder=False)

        config['save_path'] = self.OutputDir
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)

    # 导出检测结果 --- 过程代码
    def saveResultProcess(self, outdir, yolo_thread, folder):
        if folder:
            try:
                output_dir = os.path.dirname(yolo_thread.res_path)
                if os.path.exists(output_dir):
                    for filename in os.listdir(output_dir):
                        source_path = os.path.join(output_dir, filename)
                        destination_path = os.path.join(outdir, filename)
                        if os.path.isfile(source_path):
                            shutil.copy(source_path, destination_path)
                    self.showStatus('Saved Successfully in {}'.format(outdir))
                else:
                    self.showStatus('Please wait for the result to be generated')
            except Exception as err:
                self.showStatus(f"Error occurred while saving the result: {err}")
        else:
            try:
                if os.path.exists(yolo_thread.res_path):
                    shutil.copy(yolo_thread.res_path, outdir)
                    self.showStatus('Saved Successfully in {}'.format(outdir))
                else:
                    self.showStatus('Please wait for the result to be generated')
            except Exception as err:
                self.showStatus(f"Error occurred while saving the result: {err}")

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
            line_thickness = 3
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
        self.ui.iou_spinbox.setValue(iou)
        self.ui.iou_slider.setValue(int(iou * 100))
        self.ui.conf_spinbox.setValue(conf)
        self.ui.conf_slider.setValue(int(conf * 100))
        self.ui.speed_spinbox.setValue(delay)
        self.ui.speed_slider.setValue(delay)
        self.ui.line_spinbox.setValue(line_thickness)
        self.ui.line_slider.setValue(line_thickness)

    # 加载 pt 模型到 model_box
    def loadModels(self):
        pt_list = os.listdir(f'{self.current_workpath}/ptfiles/')
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize(f'{self.current_workpath}/ptfiles/' + x))

        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.ui.model_box.clear()
            self.ui.model_box.addItems(self.pt_list)

    def stopOtherModelProcess(self, yolo_thread, current_yoloname):
        yolo_thread.quit()
        yolo_thread.stop_dtc = True
        yolo_thread.finished.connect(lambda: self.resignModel(current_yoloname))

    # 停止其他模型
    def stopOtherModel(self, current_yoloname=None):
        modelname = self.allModelNames
        for yoloname in modelname:
            if yoloname != current_yoloname:
                if yoloname == "yolov5" and self.yolov5_thread.isRunning():
                    self.stopOtherModelProcess(self.yolov5_thread, current_yoloname)
                elif yoloname == "yolov7" and self.yolov7_thread.isRunning():
                    self.stopOtherModelProcess(self.yolov7_thread, current_yoloname)
                elif yoloname == "yolov8" and self.yolov8_thread.isRunning():
                    self.stopOtherModelProcess(self.yolov8_thread, current_yoloname)
                elif yoloname == "yolov9" and self.yolov9_thread.isRunning():
                    self.stopOtherModelProcess(self.yolov9_thread, current_yoloname)
                elif yoloname == "yolov10" and self.yolov10_thread.isRunning():
                    self.stopOtherModelProcess(self.yolov10_thread, current_yoloname)
                elif yoloname == "yolov5-seg" and self.yolov5seg_thread.isRunning():
                    self.stopOtherModelProcess(self.yolov5seg_thread, current_yoloname)
                elif yoloname == "yolov8-seg" and self.yolov8seg_thread.isRunning():
                    self.stopOtherModelProcess(self.yolov8seg_thread, current_yoloname)
                elif yoloname == "rtdetr" and self.rtdetr_thread.isRunning():
                    self.stopOtherModelProcess(self.rtdetr_thread, current_yoloname)
                elif yoloname == "yolov8-pose" and self.yolov8pose_thread.isRunning():
                    self.stopOtherModelProcess(self.yolov8pose_thread, current_yoloname)
                elif yoloname == "yolov8-obb" and self.yolov8obb_thread.isRunning():
                    self.stopOtherModelProcess(self.yolov8obb_thread, current_yoloname)

    def changeModelProcess(self, yolo_thread, yoloname):
        yolo_thread.new_model_name = f'{self.current_workpath}/ptfiles/' + self.ui.model_box.currentText()
        # 重载 common 和 yolo 模块
        glo.set_value('yoloname', yoloname)
        self.reloadModel()
        # 停止其他模型
        self.stopOtherModel(yoloname)

    # Model 变化
    def changeModel(self):
        self.model_name = self.ui.model_box.currentText()
        self.ui.Model_label.setText(str(self.model_name).replace(".pt", ""))  # 修改状态栏显示
        if "yolov5" in self.model_name and not self.checkSegName(self.model_name):
            self.changeModelProcess(self.yolov5_thread, "yolov5")
        elif "yolov7" in self.model_name:
            self.changeModelProcess(self.yolov7_thread, "yolov7")
        elif "yolov8" in self.model_name and not self.checkSegName(self.model_name) and not self.checkPoseName(
                self.model_name) and not self.checkObbName(self.model_name):
            self.changeModelProcess(self.yolov8_thread, "yolov8")
        elif "yolov9" in self.model_name:
            self.changeModelProcess(self.yolov9_thread, "yolov9")
        elif "yolov10" in self.model_name:
            self.changeModelProcess(self.yolov10_thread, "yolov10")
        elif "yolov5" in self.model_name and self.checkSegName(self.model_name):
            self.changeModelProcess(self.yolov5seg_thread, "yolov5-seg")
        elif "yolov8" in self.model_name and self.checkSegName(self.model_name):
            self.changeModelProcess(self.yolov8seg_thread, "yolov8-seg")
        elif "rtdetr" in self.model_name:
            self.changeModelProcess(self.rtdetr_thread, "rtdetr")
        elif "yolov8" in self.model_name and self.checkPoseName(self.model_name):
            self.changeModelProcess(self.yolov8pose_thread, "yolov8-pose")
        elif "yolov8" in self.model_name and self.checkObbName(self.model_name):
            self.changeModelProcess(self.yolov8obb_thread, "yolov8-obb")
        else:
            self.stopOtherModel()

    def runModelProcess(self, yolo_thread):
        yolo_thread.source = self.inputPath
        yolo_thread.stop_dtc = False
        if self.ui.run_button.isChecked():
            yolo_thread.is_continue = True
            if not yolo_thread.isRunning():
                yolo_thread.start()
        else:
            yolo_thread.is_continue = False
            self.showStatus('Pause Detection')

    def runModel(self, runbuttonStatus=None):
        self.ui.save_status_button.setEnabled(False)
        if runbuttonStatus:
            self.ui.run_button.setChecked(True)
        if "yolov5" in self.model_name and not self.checkSegName(self.model_name):
            self.runModelProcess(self.yolov5_thread)
        elif "yolov7" in self.model_name:
            self.runModelProcess(self.yolov7_thread)
        elif "yolov8" in self.model_name and not self.checkSegName(self.model_name) and not self.checkPoseName(
                self.model_name) and not self.checkObbName(self.model_name):
            self.runModelProcess(self.yolov8_thread)
        elif "yolov9" in self.model_name:
            self.runModelProcess(self.yolov9_thread)
        elif "yolov10" in self.model_name:
            self.runModelProcess(self.yolov10_thread)
        elif "yolov5" in self.model_name and self.checkSegName(self.model_name):
            self.runModelProcess(self.yolov5seg_thread)
        elif "yolov8" in self.model_name and self.checkSegName(self.model_name):
            self.runModelProcess(self.yolov8seg_thread)
        elif "rtdetr" in self.model_name:
            self.runModelProcess(self.rtdetr_thread)
        elif "yolov8" in self.model_name and self.checkPoseName(self.model_name):
            self.runModelProcess(self.yolov8pose_thread)
        elif "yolov8" in self.model_name and self.checkObbName(self.model_name):
            self.runModelProcess(self.yolov8obb_thread)
        else:
            self.showStatus('The current model is not supported')
            if self.ui.run_button.isChecked():
                self.ui.run_button.setChecked(False)

    # 重新加载模型
    def resignModel(self, yoloname):
        if yoloname == "yolov5":
            self.yolov5_thread = YOLOv5Thread()
            self.initModel(self.yolov5_thread, "yolov5")
            self.runModel(True)
        elif yoloname == "yolov7":
            self.yolov7_thread = YOLOv7Thread()
            self.initModel(self.yolov7_thread, "yolov7")
            self.runModel(True)
        elif yoloname == "yolov8":
            self.yolov8_thread = YOLOv8Thread()
            self.initModel(self.yolov8_thread, "yolov8")
            self.runModel(True)
        elif yoloname == "yolov9":
            self.yolov9_thread = YOLOv9Thread()
            self.initModel(self.yolov9_thread, "yolov9")
            self.runModel(True)
        elif yoloname == "yolov10":
            self.yolov10_thread = YOLOv10Thread()
            self.initModel(self.yolov10_thread, "yolov10")
            self.runModel(True)
        elif yoloname == "yolov5-seg":
            self.yolov5seg_thread = YOLOv5SegThread()
            self.initModel(self.yolov5seg_thread, "yolov5-seg")
            self.runModel(True)
        elif yoloname == "yolov8-seg":
            self.yolov8seg_thread = YOLOv8SegThread()
            self.initModel(self.yolov8seg_thread, "yolov8-seg")
            self.runModel(True)
        elif yoloname == "rtdetr":
            self.rtdetr_thread = RTDETRThread()
            self.initModel(self.rtdetr_thread, "rtdetr")
            self.runModel(True)
        elif yoloname == "yolov8-pose":
            self.yolov8pose_thread = YOLOv8PoseThread()
            self.initModel(self.yolov8pose_thread, "yolov8-pose")
            self.runModel(True)
        elif yoloname == "yolov8-obb":
            self.yolov8obb_thread = YOLOv8ObbThread()
            self.initModel(self.yolov8obb_thread, "yolov8-obb")
            self.runModel(True)

    # 开始/暂停 预测
    def runorContinue(self):
        if self.inputPath is not None:
            self.changeModel()
            self.runModel()
        else:
            self.showStatus("Please select the Image/Video before starting detection...")
            self.ui.run_button.setChecked(False)

    # 停止识别
    def stopDetect(self):
        self.quitRunningModel(stop_status=True)
        self.ui.run_button.setChecked(False)
        self.ui.save_status_button.setEnabled(True)
        self.ui.progress_bar.setValue(0)
        self.ui.main_leftbox.clear()  # clear image display
        self.ui.main_rightbox.clear()
        self.ui.Class_num.setText('--')
        self.ui.Target_num.setText('--')
        self.ui.fps_label.setText('--')
