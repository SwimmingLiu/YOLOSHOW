from utils import glo

glo._init()
glo.set_value('yoloname', "yolov5 yolov7 yolov8 yolov9 yolov10 yolov5-seg yolov8-seg rtdetr yolov8-pose yolov8-obb")
import json
import os
import shutil
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QFileDialog, QMainWindow
from PySide6.QtUiTools import loadUiType
from PySide6.QtCore import QTimer, Qt
from PySide6 import QtCore, QtGui
from ui.YOLOSHOWUIVS import Ui_MainWindow
from yoloshow.YOLOSHOWBASE import YOLOSHOWBASE
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

GLOBAL_WINDOW_STATE = True
WIDTH_LEFT_BOX_STANDARD = 80
WIDTH_LEFT_BOX_EXTENDED = 180
WIDTH_LOGO = 60
UI_FILE_PATH = "ui/YOLOSHOWUIVS.ui"
ALL_MODEL_NAMES = ["yolov5", "yolov7", "yolov8", "yolov9", "yolov10", "yolov5-seg", "yolov8-seg", "rtdetr",
                   "yolov8-pose", "yolov8-obb"]


# YOLOSHOW窗口类 动态加载UI文件 和 Ui_mainWindow
class YOLOSHOWVS(QMainWindow, YOLOSHOWBASE):
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
        # --- 加载UI --- #

        # 初始化侧边栏
        self.initSiderWidget()

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
        self.solveYoloConflict([f"{self.current_workpath}/ptfiles/" + pt_file for pt_file in self.pt_list])
        self.ui.model_box1.clear()
        self.ui.model_box1.addItems(self.pt_list)
        self.ui.model_box2.clear()
        self.ui.model_box2.addItems(self.pt_list)
        self.qtimer_search = QTimer(self)
        self.qtimer_search.timeout.connect(lambda: self.loadModels())
        self.qtimer_search.start(2000)
        self.ui.model_box1.currentTextChanged.connect(lambda: self.changeModel("left"))
        self.ui.model_box2.currentTextChanged.connect(lambda: self.changeModel("right"))
        # --- 自动加载/动态改变 PT 模型 --- #

        # --- 超参数调整 --- #
        self.ui.iou_spinbox.valueChanged.connect(lambda x: self.changeValue(x, 'iou_spinbox'))  # iou box
        self.ui.iou_slider.valueChanged.connect(lambda x: self.changeValue(x, 'iou_slider'))  # iou scroll bar
        self.ui.conf_spinbox.valueChanged.connect(lambda x: self.changeValue(x, 'conf_spinbox'))  # conf box
        self.ui.conf_slider.valueChanged.connect(lambda x: self.changeValue(x, 'conf_slider'))  # conf scroll bar
        self.ui.speed_spinbox.valueChanged.connect(lambda x: self.changeValue(x, 'speed_spinbox'))  # speed box
        self.ui.speed_slider.valueChanged.connect(lambda x: self.changeValue(x, 'speed_slider'))  # speed scroll bar
        self.ui.line_spinbox.valueChanged.connect(lambda x: self.changeValue(x, 'line_spinbox'))  # line box
        self.ui.line_slider.valueChanged.connect(lambda x: self.changeValue(x, 'line_slider'))  # line slider
        # --- 超参数调整 --- #

        # --- 导入 图片/视频、调用摄像头、导入文件夹（批量处理）、调用网络摄像头、结果统计图片、结果统计表格 --- #
        self.ui.src_img.clicked.connect(self.selectFile)
        # 对比模型模式 不支持同时读取摄像头流
        # self.src_webcam.clicked.connect(self.selectWebcam)
        self.ui.src_folder.clicked.connect(self.selectFolder)
        self.ui.src_camera.clicked.connect(self.selectRtsp)
        # self.ui.src_result.clicked.connect(self.showResultStatics)
        # self.ui.src_table.clicked.connect(self.showTableResult)
        # --- 导入 图片/视频、调用摄像头、导入文件夹（批量处理）、调用网络摄像头、结果统计图片、结果统计表格 --- #

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
        self.shadowStyle(self.ui.Class_QF1, QColor(142, 197, 252), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.Target_QF1, QColor(159, 172, 230), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.Fps_QF1, QColor(170, 128, 213), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.Model_QF1, QColor(162, 129, 247), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.Class_QF2, QColor(142, 197, 252), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.Target_QF2, QColor(159, 172, 230), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.Fps_QF2, QColor(170, 128, 213), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.Model_QF2, QColor(162, 129, 247), top_bottom=['top', 'bottom'])

        # 状态栏默认显示
        self.model_name1 = self.ui.model_box1.currentText()  # 获取默认 model
        self.ui.Class_num1.setText('--')
        self.ui.Target_num1.setText('--')
        self.ui.fps_label1.setText('--')
        self.ui.Model_label1.setText(str(self.model_name1).replace(".pt", ""))
        self.model_name2 = self.ui.model_box2.currentText()  # 获取默认 model
        self.ui.Class_num2.setText('--')
        self.ui.Target_num2.setText('--')
        self.ui.fps_label2.setText('--')
        self.ui.Model_label2.setText(str(self.model_name2).replace(".pt", ""))
        # --- 状态栏 初始化 --- #

        # --- YOLOv5 QThread --- #
        self.yolov5_thread1 = YOLOv5Thread()
        self.yolov5_thread2 = YOLOv5Thread()
        self.initModel(self.yolov5_thread1, "yolov5", "left")
        self.initModel(self.yolov5_thread2, "yolov5", "right")
        # --- YOLOv5 QThread --- #

        # --- YOLOv7 QThread --- #
        self.yolov7_thread1 = YOLOv7Thread()
        self.yolov7_thread2 = YOLOv7Thread()
        self.initModel(self.yolov7_thread1, "yolov7", "left")
        self.initModel(self.yolov7_thread2, "yolov7", "right")
        # --- YOLOv7 QThread --- #

        # --- YOLOv8 QThread --- #
        self.yolov8_thread1 = YOLOv8Thread()
        self.yolov8_thread2 = YOLOv8Thread()
        self.initModel(self.yolov8_thread1, "yolov8", "left")
        self.initModel(self.yolov8_thread2, "yolov8", "right")
        # --- YOLOv8 QThread --- #

        # --- YOLOv9 QThread --- #
        self.yolov9_thread1 = YOLOv9Thread()
        self.yolov9_thread2 = YOLOv9Thread()
        self.initModel(self.yolov9_thread1, "yolov9", "left")
        self.initModel(self.yolov9_thread2, "yolov9", "right")
        # --- YOLOv9 QThread --- #

        # --- YOLOv10 QThread --- #
        self.yolov10_thread1 = YOLOv10Thread()
        self.yolov10_thread2 = YOLOv10Thread()
        self.initModel(self.yolov10_thread1, "yolov10", "left")
        self.initModel(self.yolov10_thread2, "yolov10", "right")
        # --- YOLOv10 QThread --- #

        # --- YOLOv5-Seg QThread --- #
        self.yolov5seg_thread1 = YOLOv5SegThread()
        self.yolov5seg_thread2 = YOLOv5SegThread()
        self.initModel(self.yolov5seg_thread1, "yolov5-seg", "left")
        self.initModel(self.yolov5seg_thread2, "yolov5-seg", "right")
        # --- YOLOv5-Seg QThread --- #

        # --- YOLOv8-Seg QThread --- #
        self.yolov8seg_thread1 = YOLOv8SegThread()
        self.yolov8seg_thread2 = YOLOv8SegThread()
        self.initModel(self.yolov8seg_thread1, "yolov8-seg", "left")
        self.initModel(self.yolov8seg_thread2, "yolov8-seg", "right")
        # --- YOLOv8-Seg QThread --- #

        # --- RT-DETR QThread --- #
        self.rtdetr_thread1 = RTDETRThread()
        self.rtdetr_thread2 = RTDETRThread()
        self.initModel("rtdetr")
        self.initModel(self.rtdetr_thread1, "rtdetr", "left")
        self.initModel(self.rtdetr_thread2, "rtdetr", "right")
        # --- RT-DETR QThread --- #

        # --- 开始 / 停止 --- #

        # --- YOLOv8-Pose QThread --- #
        self.yolov8pose_thread1 = YOLOv8PoseThread()
        self.yolov8pose_thread2 = YOLOv8PoseThread()
        self.initModel(self.yolov8pose_thread1, "yolov8-pose", "left")
        self.initModel(self.yolov8pose_thread2, "yolov8-pose", "right")
        # --- YOLOv8-Pose QThread --- #

        # --- YOLOv8-Obb QThread --- #
        self.yolov8obb_thread1 = YOLOv8ObbThread()
        self.yolov8obb_thread2 = YOLOv8ObbThread()
        self.initModel(self.yolov8obb_thread1, "yolov8-obb", "left")
        self.initModel(self.yolov8obb_thread2, "yolov8-obb", "right")
        # --- YOLOv8-Obb QThread --- #

        self.initThreads()

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
        self.yolo_threads = [self.yolov5_thread1, self.yolov5_thread2, self.yolov7_thread1, self.yolov7_thread2,
                             self.yolov8_thread1, self.yolov8_thread2, self.yolov9_thread1, self.yolov9_thread2,
                             self.yolov10_thread1, self.yolov10_thread2, self.rtdetr_thread1, self.rtdetr_thread2,
                             self.yolov5seg_thread1, self.yolov5seg_thread2,
                             self.yolov8seg_thread1, self.yolov8seg_thread2, self.rtdetr_thread1, self.rtdetr_thread2,
                             self.yolov8pose_thread1, self.yolov8pose_thread2, self.yolov8obb_thread1,
                             self.yolov8obb_thread2]

    # 初始化模型
    def initModel(self, yolo_thread, yoloname=None, mode="all"):
        if mode == "left":
            # 左侧模型加载
            yolo_thread.new_model_name = f'{self.current_workpath}/ptfiles/' + self.ui.model_box1.currentText()
            yolo_thread.progress_value = self.ui.progress_bar.maximum()
            yolo_thread.send_output.connect(lambda x: self.showImg(x, self.ui.main_leftbox, 'img'))
            # 第一个模型来控制消息
            yolo_thread.send_msg.connect(lambda x: self.showStatus(x))
            yolo_thread.send_fps.connect(lambda x: self.ui.fps_label1.setText(str(x)))
            yolo_thread.send_class_num.connect(lambda x: self.ui.Class_num1.setText(str(x)))
            yolo_thread.send_target_num.connect(lambda x: self.ui.Target_num1.setText(str(x)))
        if mode == "right":
            # 右侧模型加载
            yolo_thread.new_model_name = f'{self.current_workpath}/ptfiles/' + self.ui.model_box2.currentText()
            yolo_thread.progress_value = self.ui.progress_bar.maximum()
            yolo_thread.send_output.connect(lambda x: self.showImg(x, self.ui.main_rightbox, 'img'))
            # 后一个模型来控制进度条
            yolo_thread.send_progress.connect(lambda x: self.ui.progress_bar.setValue(x))
            yolo_thread.send_fps.connect(lambda x: self.ui.fps_label2.setText(str(x)))
            yolo_thread.send_class_num.connect(lambda x: self.ui.Class_num2.setText(str(x)))
            yolo_thread.send_target_num.connect(lambda x: self.ui.Target_num2.setText(str(x)))

    # 在MessageBar显示消息
    def showStatus(self, msg):
        self.ui.message_bar.setText(msg)
        if msg == 'Finish Detection':
            self.quitRunningModel()
            self.ui.run_button.setChecked(False)
            self.ui.progress_bar.setValue(0)
            self.ui.save_status_button.setEnabled(True)
        elif msg == 'Stop Detection':
            self.quitRunningModel()
            self.ui.run_button.setChecked(False)
            self.ui.save_status_button.setEnabled(True)
            self.ui.progress_bar.setValue(0)
            self.ui.Class_num1.setText('--')
            self.ui.Target_num1.setText('--')
            self.ui.fps_label1.setText('--')
            self.ui.Class_num2.setText('--')
            self.ui.Target_num2.setText('--')
            self.ui.fps_label2.setText('--')
            self.ui.main_leftbox.clear()  # clear image display
            self.ui.main_rightbox.clear()

    # 导出结果状态判断
    def saveStatus(self):
        if self.ui.save_status_button.checkState() == Qt.CheckState.Unchecked:
            self.showStatus('NOTE: Run image results are not saved.')
            for yolo_thread in self.yolo_threads:
                yolo_thread.save_res = False

            self.ui.save_button.setEnabled(False)
        elif self.ui.save_status_button.checkState() == Qt.CheckState.Checked:
            self.showStatus('NOTE: Run image results will be saved. Only save left model result in VS Mode !!!')
            for yolo_thread in self.yolo_threads:
                yolo_thread.save_res = True
            self.ui.save_button.setEnabled(True)

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

    # 导出结果
    def saveResult(self):
        thread1_status = (not self.yolov5_thread1.res_status and not self.yolov7_thread1.res_status
                          and not self.yolov8_thread1.res_status and not self.yolov9_thread1.res_status and not self.yolov10_thread1
                          and not self.yolov5seg_thread1.res_status and not self.yolov8seg_thread1.res_status
                          and not self.rtdetr_thread1.res_status and not self.yolov8pose_thread1.res_status and not self.yolov8obb_thread1.res_status)
        thread2_status = (not self.yolov5_thread2.res_status and not self.yolov7_thread2.res_status
                          and not self.yolov8_thread2.res_status and not self.yolov9_thread2.res_status and not self.yolov10_thread2
                          and not self.yolov5seg_thread2.res_status and not self.yolov8seg_thread2.res_status
                          and not self.rtdetr_thread2.res_status and not self.yolov8pose_thread2.res_status and not self.yolov8obb_thread2.res_status)
        self.model_name = self.model_name1

        # 默认保存左侧模型的检测结果
        if thread1_status:
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
                self.saveResultProcess(self.OutputDir, self.yolov5_thread1, folder=True)
            elif "yolov7" in self.model_name:
                self.saveResultProcess(self.OutputDir, self.yolov7_thread1, folder=True)
            elif "yolov8" in self.model_name and not self.checkSegName(self.model_name) and not self.checkPoseName(
                    self.model_name) and not self.checkObbName(self.model_name):
                self.saveResultProcess(self.OutputDir, self.yolov8_thread1, folder=True)
            elif "yolov9" in self.model_name:
                self.saveResultProcess(self.OutputDir, self.yolov9_thread1, folder=True)
            elif "yolov10" in self.model_name:
                self.saveResultProcess(self.OutputDir, self.yolov10_thread1, folder=True)
            elif "yolov5" in self.model_name and self.checkSegName(self.model_name):
                self.saveResultProcess(self.OutputDir, self.yolov5seg_thread1, folder=True)
            elif "yolov8" in self.model_name and self.checkSegName(self.model_name):
                self.saveResultProcess(self.OutputDir, self.yolov8seg_thread1, folder=True)
            elif "rtdetr" in self.model_name:
                self.saveResultProcess(self.OutputDir, self.rtdetr_thread1, folder=True)
            elif "yolov8" in self.model_name and self.checkPoseName(self.model_name):
                self.saveResultProcess(self.OutputDir, self.yolov8pose_thread1, folder=True)
            elif "yolov8" in self.model_name and self.checkObbName(self.model_name):
                self.saveResultProcess(self.OutputDir, self.yolov8obb_thread1, folder=True)
        else:
            self.OutputDir, _ = QFileDialog.getSaveFileName(
                self,  # 父窗口对象
                "Save Image/Video",  # 标题
                save_path,  # 起始目录
                "Image/Vide Type (*.jpg *.jpeg *.png *.bmp *.dib  *.jpe  *.jp2 *.mp4)"  # 选择类型过滤项，过滤内容在括号中
            )
            if "yolov5" in self.model_name and not self.checkSegName(self.model_name):
                self.saveResultProcess(self.OutputDir, self.yolov5_thread1, folder=False)
            elif "yolov7" in self.model_name:
                self.saveResultProcess(self.OutputDir, self.yolov7_thread1, folder=False)
            elif "yolov8" in self.model_name and not self.checkSegName(self.model_name) and not self.checkPoseName(
                    self.model_name) and not self.checkObbName(self.model_name):
                self.saveResultProcess(self.OutputDir, self.yolov8_thread1, folder=False)
            elif "yolov9" in self.model_name:
                self.saveResultProcess(self.OutputDir, self.yolov9_thread1, folder=False)
            elif "yolov10" in self.model_name:
                self.saveResultProcess(self.OutputDir, self.yolov10_thread1, folder=False)
            elif "yolov5" in self.model_name and self.checkSegName(self.model_name):
                self.saveResultProcess(self.OutputDir, self.yolov5seg_thread1, folder=False)
            elif "yolov8" in self.model_name and self.checkSegName(self.model_name):
                self.saveResultProcess(self.OutputDir, self.yolov8seg_thread1, folder=False)
            elif "rtdetr" in self.model_name:
                self.saveResultProcess(self.OutputDir, self.rtdetr_thread1, folder=False)
            elif "yolov8" in self.model_name and self.checkPoseName(self.model_name):
                self.saveResultProcess(self.OutputDir, self.yolov8pose_thread1, folder=False)
            elif "yolov8" in self.model_name and self.checkObbName(self.model_name):
                self.saveResultProcess(self.OutputDir, self.yolov8obb_thread1, folder=False)
        config['save_path'] = self.OutputDir
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)

    # 加载 pt 模型到 model_box
    def loadModels(self):
        pt_list = os.listdir(f'{self.current_workpath}/ptfiles/')
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize(f'{self.current_workpath}/ptfiles/' + x))

        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.ui.model_box1.clear()
            self.ui.model_box1.addItems(self.pt_list)
            self.ui.model_box2.clear()
            self.ui.model_box2.addItems(self.pt_list)

    # 检查模型是否符合命名要求
    def checkModelName(self, modelname):
        for name in self.allModelNames:
            if modelname in name:
                return True
        return False

    # 重新加载模型
    def resignModel(self, yoloname, mode=None):
        self.reloadModel()
        if mode == "left":
            if yoloname == "yolov5":
                self.yolov5_thread1 = YOLOv5Thread()
                self.initModel(self.yolov5_thread1, "yolov5", "left")
            elif yoloname == "yolov7":
                self.yolov7_thread1 = YOLOv7Thread()
                self.initModel(self.yolov7_thread1, "yolov7", "left")
            elif yoloname == "yolov8":
                self.yolov8_thread1 = YOLOv8Thread()
                self.initModel(self.yolov8_thread1, "yolov8", "left")
            elif yoloname == "yolov9":
                self.yolov9_thread1 = YOLOv9Thread()
                self.initModel(self.yolov9_thread1, "yolov9", "left")
            elif yoloname == "yolov10":
                self.yolov10_thread1 = YOLOv10Thread()
                self.initModel(self.yolov10_thread1, "yolov10", "left")
            elif yoloname == "yolov5-seg":
                self.yolov5seg_thread1 = YOLOv5SegThread()
                self.initModel(self.yolov5seg_thread1, "yolov5-seg", "left")
            elif yoloname == "yolov8-seg":
                self.yolov8seg_thread1 = YOLOv8SegThread()
                self.initModel(self.yolov8seg_thread1, "yolov8-seg", "left")
            elif yoloname == "rtdetr":
                self.rtdetr_thread1 = RTDETRThread()
                self.initModel(self.rtdetr_thread1, "rtdetr", "left")
            elif yoloname == "yolov8-pose":
                self.yolov8pose_thread1 = YOLOv8PoseThread()
                self.initModel(self.yolov8pose_thread1, "yolov8-pose", "left")
            elif yoloname == "yolov8-obb":
                self.yolov8obb_thread1 = YOLOv8ObbThread()
                self.initModel(self.yolov8obb_thread1, "yolov8-obb", "left")
            self.ui.run_button.setChecked(True)
            self.ContinueAnotherModel(mode="right")
            self.runModel(True)
        else:
            if yoloname == "yolov5":
                self.yolov5_thread2 = YOLOv5Thread()
                self.initModel(self.yolov5_thread2, "yolov5", "right")
            elif yoloname == "yolov7":
                self.yolov7_thread2 = YOLOv7Thread()
                self.initModel(self.yolov7_thread2, "yolov7", "right")
            elif yoloname == "yolov8":
                self.yolov8_thread2 = YOLOv8Thread()
                self.initModel(self.yolov8_thread2, "yolov8", "right")
            elif yoloname == "yolov9":
                self.yolov9_thread2 = YOLOv9Thread()
                self.initModel(self.yolov9_thread2, "yolov9", "right")
            elif yoloname == "yolov10":
                self.yolov10_thread2 = YOLOv10Thread()
                self.initModel(self.yolov10_thread2, "yolov10", "right")
            elif yoloname == "yolov5-seg":
                self.yolov5seg_thread2 = YOLOv5SegThread()
                self.initModel(self.yolov5seg_thread2, "yolov5-seg", "right")
            elif yoloname == "yolov8-seg":
                self.yolov8seg_thread2 = YOLOv8SegThread()
                self.initModel(self.yolov8seg_thread2, "yolov8-seg", "right")
            elif yoloname == "rtdetr":
                self.rtdetr_thread2 = RTDETRThread()
                self.initModel(self.rtdetr_thread2, "rtdetr", "right")
            elif yoloname == "yolov8-pose":
                self.yolov8pose_thread2 = YOLOv8PoseThread()
                self.initModel(self.yolov8pose_thread2, "yolov8-pose", "right")
            elif yoloname == "yolov8-obb":
                self.yolov8obb_thread2 = YOLOv8ObbThread()
                self.initModel(self.yolov8obb_thread2, "yolov8-obb", "right")
            self.ui.run_button.setChecked(True)
            self.ContinueAnotherModel(mode="left")
            self.runModel(True)

    def stopOtherModelProcess(self, yolo_thread, current_yoloname):
        yolo_thread.quit()
        yolo_thread.stop_dtc = True
        yolo_thread.finished.connect((lambda: self.resignModel(current_yoloname, mode="left")))

    # 停止其他模型
    def stopOtherModel(self, current_yoloname=None, mode=None):
        modelname = self.allModelNames
        for yoloname in modelname:
            if yoloname != current_yoloname:
                if mode == "left":
                    if yoloname == "yolov5" and self.yolov5_thread1.isRunning():
                        self.PauseAnotherModel(mode="right")
                        self.stopOtherModelProcess(self.yolov5_thread1, current_yoloname)
                    elif yoloname == "yolov7" and self.yolov7_thread1.isRunning():
                        self.PauseAnotherModel(mode="right")
                        self.stopOtherModelProcess(self.yolov7_thread1, current_yoloname)
                    elif yoloname == "yolov8" and self.yolov8_thread1.isRunning():
                        self.PauseAnotherModel(mode="right")
                        self.stopOtherModelProcess(self.yolov8_thread1, current_yoloname)
                    elif yoloname == "yolov9" and self.yolov9_thread1.isRunning():
                        self.PauseAnotherModel(mode="right")
                        self.stopOtherModelProcess(self.yolov9_thread1, current_yoloname)
                    elif yoloname == "yolov10" and self.yolov10_thread1.isRunning():
                        self.PauseAnotherModel(mode="right")
                        self.stopOtherModelProcess(self.yolov10_thread1, current_yoloname)
                    elif yoloname == "yolov5-seg" and self.yolov5seg_thread1.isRunning():
                        self.PauseAnotherModel(mode="right")
                        self.stopOtherModelProcess(self.yolov5seg_thread1, current_yoloname)
                    elif yoloname == "yolov8-seg" and self.yolov8seg_thread1.isRunning():
                        self.PauseAnotherModel(mode="right")
                        self.stopOtherModelProcess(self.yolov8seg_thread1, current_yoloname)
                    elif yoloname == "rtdetr" and self.rtdetr_thread1.isRunning():
                        self.PauseAnotherModel(mode="right")
                        self.stopOtherModelProcess(self.rtdetr_thread1, current_yoloname)
                    elif yoloname == "yolov8-pose" and self.yolov8pose_thread1.isRunning():
                        self.PauseAnotherModel(mode="right")
                        self.stopOtherModelProcess(self.yolov8pose_thread1, current_yoloname)
                    elif yoloname == "yolov8-obb" and self.yolov8obb_thread1.isRunning():
                        self.PauseAnotherModel(mode="right")
                        self.stopOtherModelProcess(self.yolov8obb_thread1, current_yoloname)
                else:
                    if yoloname == "yolov5" and self.yolov5_thread2.isRunning():
                        self.PauseAnotherModel(mode="left")
                        self.stopOtherModelProcess(self.yolov5_thread2, current_yoloname)
                    elif yoloname == "yolov7" and self.yolov7_thread2.isRunning():
                        self.PauseAnotherModel(mode="left")
                        self.stopOtherModelProcess(self.yolov7_thread2, current_yoloname)
                    elif yoloname == "yolov8" and self.yolov8_thread2.isRunning():
                        self.PauseAnotherModel(mode="left")
                        self.stopOtherModelProcess(self.yolov8_thread2, current_yoloname)
                    elif yoloname == "yolov9" and self.yolov9_thread2.isRunning():
                        self.PauseAnotherModel(mode="left")
                        self.stopOtherModelProcess(self.yolov9_thread2, current_yoloname)
                    elif yoloname == "yolov10" and self.yolov10_thread2.isRunning():
                        self.PauseAnotherModel(mode="left")
                        self.stopOtherModelProcess(self.yolov10_thread2, current_yoloname)
                    elif yoloname == "yolov5-seg" and self.yolov5seg_thread2.isRunning():
                        self.PauseAnotherModel(mode="left")
                        self.stopOtherModelProcess(self.yolov5seg_thread2, current_yoloname)
                    elif yoloname == "yolov8-seg" and self.yolov8seg_thread2.isRunning():
                        self.PauseAnotherModel(mode="left")
                        self.stopOtherModelProcess(self.yolov8seg_thread2, current_yoloname)
                    elif yoloname == "rtdetr" and self.rtdetr_thread2.isRunning():
                        self.PauseAnotherModel(mode="left")
                        self.stopOtherModelProcess(self.rtdetr_thread2, current_yoloname)
                    elif yoloname == "yolov8-pose" and self.yolov8pose_thread2.isRunning():
                        self.PauseAnotherModel(mode="left")
                        self.stopOtherModelProcess(self.yolov8pose_thread2, current_yoloname)
                    elif yoloname == "yolov8-obb" and self.yolov8obb_thread2.isRunning():
                        self.PauseAnotherModel(mode="left")
                        self.stopOtherModelProcess(self.yolov8obb_thread2, current_yoloname)

    def PauseAnotherModelProcess(self, yolo_thread):
        yolo_thread.quit()
        yolo_thread.stop_dtc = True
        yolo_thread.wait()

    # 暂停另外一侧模型
    def PauseAnotherModel(self, mode=None):
        buttonStatus = self.ui.run_button.isChecked()
        if buttonStatus:
            if mode == "left":
                if "yolov5" in self.model_name1 and not self.checkSegName(
                        self.model_name1) and self.yolov5_thread1.isRunning():
                    self.PauseAnotherModelProcess(self.yolov5_thread1)
                elif "yolov7" in self.model_name1 and self.yolov7_thread1.isRunning():
                    self.PauseAnotherModelProcess(self.yolov7_thread1)
                elif ("yolov8" in self.model_name1 and not self.checkSegName(
                        self.model_name1) and not self.checkPoseName(self.model_name1)
                      and not self.checkObbName(self.model_name1)
                      and self.yolov8_thread1.isRunning()):
                    self.PauseAnotherModelProcess(self.yolov8_thread1)
                elif "yolov9" in self.model_name1 and self.yolov9_thread1.isRunning():
                    self.PauseAnotherModelProcess(self.yolov9_thread1)
                elif "yolov10" in self.model_name1 and self.yolov10_thread1.isRunning():
                    self.PauseAnotherModelProcess(self.yolov10_thread1)
                elif "yolov5" in self.model_name1 and self.checkSegName(
                        self.model_name1) and self.yolov5seg_thread1.isRunning():
                    self.PauseAnotherModelProcess(self.yolov5seg_thread1)
                elif "yolov8" in self.model_name1 and self.checkSegName(
                        self.model_name1) and self.yolov8seg_thread1.isRunning():
                    self.PauseAnotherModelProcess(self.yolov8seg_thread1)
                elif "rtdetr" in self.model_name1 and self.rtdetr_thread1.isRunning():
                    self.PauseAnotherModelProcess(self.rtdetr_thread1)
                elif "yolov8" in self.model_name1 and self.checkPoseName(
                        self.model_name1) and self.yolov8pose_thread1.isRunning():
                    self.PauseAnotherModelProcess(self.yolov8pose_thread1)
                elif "yolov8" in self.model_name1 and self.checkObbName(
                        self.model_name1) and self.yolov8obb_thread1.isRunning():
                    self.PauseAnotherModelProcess(self.yolov8obb_thread1)
            else:
                if "yolov5" in self.model_name2 and not self.checkSegName(
                        self.model_name2) and self.yolov5_thread2.isRunning():
                    self.PauseAnotherModelProcess(self.yolov5_thread2)
                elif "yolov7" in self.model_name2 and self.yolov7_thread2.isRunning():
                    self.PauseAnotherModelProcess(self.yolov7_thread2)
                elif ("yolov8" in self.model_name2 and not self.checkSegName(
                        self.model_name2) and not self.checkPoseName(self.model_name2) and not self.checkObbName(
                    self.model_name2)
                      and self.yolov8_thread2.isRunning()):
                    self.PauseAnotherModelProcess(self.yolov8_thread2)
                elif "yolov9" in self.model_name2 and self.yolov9_thread2.isRunning():
                    self.PauseAnotherModelProcess(self.yolov9_thread2)
                elif "yolov10" in self.model_name2 and self.yolov10_thread2.isRunning():
                    self.PauseAnotherModelProcess(self.yolov10_thread2)
                elif "yolov5" in self.model_name2 and self.checkSegName(
                        self.model_name2) and self.yolov5seg_thread2.isRunning():
                    self.PauseAnotherModelProcess(self.yolov5seg_thread2)
                elif "yolov8" in self.model_name2 and self.checkSegName(
                        self.model_name2) and self.yolov8seg_thread2.isRunning():
                    self.PauseAnotherModelProcess(self.yolov8seg_thread2)
                elif "rtdetr" in self.model_name2 and self.rtdetr_thread2.isRunning():
                    self.PauseAnotherModelProcess(self.rtdetr_thread2)
                elif "yolov8" in self.model_name2 and self.checkPoseName(
                        self.model_name2) and self.yolov8pose_thread2.isRunning():
                    self.PauseAnotherModelProcess(self.yolov8pose_thread2)
                elif "yolov8" in self.model_name2 and self.checkObbName(
                        self.model_name2) and self.yolov8obb_thread2.isRunning():
                    self.PauseAnotherModelProcess(self.yolov8obb_thread2)

    # 继续另外一侧模型
    def ContinueAnotherModel(self, mode=None):
        buttonStatus = self.ui.run_button.isChecked()
        if buttonStatus:
            if mode == "left":
                if "yolov5" in self.model_name1 and not self.checkSegName(
                        self.model_name1) and buttonStatus:
                    self.yolov5_thread1 = YOLOv5Thread()
                    self.initModel(self.yolov5_thread1, "yolov5", "left")
                elif "yolov7" in self.model_name1 and buttonStatus:
                    self.yolov7_thread1 = YOLOv7Thread()
                    self.initModel(self.yolov7_thread1, "yolov7", "left")
                elif ("yolov8" in self.model_name1 and not self.checkSegName(
                        self.model_name1) and not self.checkPoseName(self.model_name1) and not self.checkObbName(
                    self.model_name1)
                      and buttonStatus):
                    self.yolov8_thread1 = YOLOv8Thread()
                    self.initModel(self.yolov8_thread1, "yolov8", "left")
                elif "yolov9" in self.model_name1 and buttonStatus:
                    self.yolov9_thread1 = YOLOv9Thread()
                    self.initModel(self.yolov9_thread1, "yolov9", "left")
                elif "yolov10" in self.model_name1 and buttonStatus:
                    self.yolov10_thread1 = YOLOv10Thread()
                    self.initModel(self.yolov10_thread1, "yolov10", "left")
                elif "yolov5" in self.model_name1 and self.checkSegName(
                        self.model_name1) and buttonStatus:
                    self.yolov5seg_thread1 = YOLOv5SegThread()
                    self.initModel(self.yolov5seg_thread1, "yolov5-seg", "left")
                elif "yolov8" in self.model_name1 and self.checkSegName(
                        self.model_name1) and buttonStatus:
                    self.yolov8seg_thread1 = YOLOv8SegThread()
                    self.initModel(self.yolov8seg_thread1, "yolov8-seg", "left")
                elif "rtdetr" in self.model_name1 and buttonStatus:
                    self.rtdetr_thread1 = RTDETRThread()
                    self.initModel(self.rtdetr_thread1, "rtdetr", "left")
                elif "yolov8" in self.model_name1 and self.checkPoseName(self.model_name1) and buttonStatus:
                    self.yolov8pose_thread1 = YOLOv8PoseThread()
                    self.initModel(self.yolov8pose_thread1, "yolov8-pose", "left")
                elif "yolov8" in self.model_name1 and self.checkObbName(self.model_name1) and buttonStatus:
                    self.yolov8obb_thread1 = YOLOv8ObbThread()
                    self.initModel(self.yolov8obb_thread1, "yolov8-obb", "left")
            else:
                if "yolov5" in self.model_name2 and not self.checkSegName(
                        self.model_name2) and buttonStatus:

                    self.yolov5_thread2 = YOLOv5Thread()
                    self.initModel(self.yolov5_thread2, "yolov5", "right")
                elif "yolov7" in self.model_name2 and buttonStatus:

                    self.yolov7_thread2 = YOLOv7Thread()
                    self.initModel(self.yolov7_thread2, "yolov7", "right")
                elif ("yolov8" in self.model_name2 and not self.checkSegName(
                        self.model_name2) and not self.checkPoseName(self.model_name2) and not self.checkObbName(
                    self.model_name2)
                      and buttonStatus):
                    self.yolov8_thread2 = YOLOv8Thread()
                    self.initModel(self.yolov8_thread2, "yolov8", "right")
                elif "yolov9" in self.model_name2 and buttonStatus:
                    self.yolov9_thread2 = YOLOv9Thread()
                    self.initModel(self.yolov9_thread2, "yolov9", "right")
                elif "yolov10" in self.model_name2 and buttonStatus:
                    self.yolov10_thread2 = YOLOv10Thread()
                    self.initModel(self.yolov10_thread2, "yolov10", "right")
                elif "yolov5" in self.model_name2 and self.checkSegName(
                        self.model_name2) and buttonStatus:
                    self.yolov5seg_thread2 = YOLOv5SegThread()
                    self.initModel(self.yolov5seg_thread2, "yolov5-seg", "right")
                elif "yolov8" in self.model_name2 and self.checkSegName(
                        self.model_name2) and buttonStatus:
                    self.yolov8seg_thread2 = YOLOv8SegThread()
                    self.initModel(self.yolov8seg_thread2, "yolov8-seg", "right")
                elif "rtdetr" in self.model_name2 and buttonStatus:
                    self.rtdetr_thread2 = RTDETRThread()
                    self.initModel(self.rtdetr_thread2, "rtdetr", "right")
                elif "yolov8" in self.model_name2 and self.checkPoseName(self.model_name2) and buttonStatus:
                    self.yolov8pose_thread2 = YOLOv8PoseThread()
                    self.initModel(self.yolov8pose_thread2, "yolov8-pose", "right")
                elif "yolov8" in self.model_name2 and self.checkObbName(self.model_name2) and buttonStatus:
                    self.yolov8obb_thread2 = YOLOv8ObbThread()
                    self.initModel(self.yolov8obb_thread2, "yolov8-obb", "right")

    def changeModelProcess(self, yolo_thread, yoloname, mode=None):
        if mode == "left":
            yolo_thread.new_model_name = f'{self.current_workpath}/ptfiles/' + self.ui.model_box1.currentText()
            # 重载 common 和 yolo 模块
            glo.set_value('yoloname1', yoloname)
            # 停止其他模型
            self.stopOtherModel(yoloname, mode="left")
        else:
            yolo_thread.new_model_name = f'{self.current_workpath}/ptfiles/' + self.ui.model_box2.currentText()
            # 重载 common 和 yolo 模块
            glo.set_value('yoloname2', yoloname)
            # 停止其他模型
            self.stopOtherModel(yoloname, mode="right")

    # Model 变化
    def changeModel(self, mode=None):
        if mode == "left":
            # 左侧模型
            self.model_name1 = self.ui.model_box1.currentText()
            self.ui.Model_label1.setText(str(self.model_name1).replace(".pt", ""))  # 修改状态栏显示
            if "yolov5" in self.model_name1 and not self.checkSegName(self.model_name1):
                self.changeModelProcess(self.yolov5_thread1, "yolov5", "left")
            elif "yolov7" in self.model_name1:
                self.changeModelProcess(self.yolov7_thread1, "yolov7", "left")
            elif "yolov8" in self.model_name1 and not self.checkSegName(self.model_name1) \
                    and not self.checkPoseName(self.model_name1) and not self.checkObbName(self.model_name1):
                self.changeModelProcess(self.yolov8_thread1, "yolov8", "left")
            elif "yolov9" in self.model_name1:
                self.changeModelProcess(self.yolov9_thread1, "yolov9", "left")
            elif "yolov10" in self.model_name1:
                self.changeModelProcess(self.yolov10_thread1, "yolov10", "left")
            elif "yolov5" in self.model_name1 and self.checkSegName(self.model_name1):
                self.changeModelProcess(self.yolov5seg_thread1, "yolov5-seg", "left")
            elif "yolov8" in self.model_name1 and self.checkSegName(self.model_name1):
                self.changeModelProcess(self.yolov8seg_thread1, "yolov8-seg", "left")
            elif "rtdetr" in self.model_name1:
                self.changeModelProcess(self.rtdetr_thread1, "rtdetr", "left")
            elif "yolov8" in self.model_name1 and self.checkPoseName(self.model_name1):
                self.changeModelProcess(self.yolov8pose_thread1, "yolov8-pose", "left")
            elif "yolov8" in self.model_name1 and self.checkObbName(self.model_name1):
                self.changeModelProcess(self.yolov8obb_thread1, "yolov8-obb", "left")
            else:
                self.stopOtherModel(mode="left")
        else:
            # 右侧模型
            self.model_name2 = self.ui.model_box2.currentText()
            self.ui.Model_label2.setText(str(self.model_name2).replace(".pt", ""))
            if "yolov5" in self.model_name2 and not self.checkSegName(self.model_name2):
                self.changeModelProcess(self.yolov5_thread2, "yolov5", "right")
            elif "yolov7" in self.model_name2:
                self.changeModelProcess(self.yolov7_thread2, "yolov7", "right")
            elif "yolov8" in self.model_name2 and not self.checkSegName(self.model_name2) \
                    and not self.checkPoseName(self.model_name2) and not self.checkObbName(self.model_name2):
                self.changeModelProcess(self.yolov8_thread2, "yolov8", "right")
            elif "yolov9" in self.model_name2:
                self.changeModelProcess(self.yolov9_thread2, "yolov9", "right")
            elif "yolov10" in self.model_name2:
                self.changeModelProcess(self.yolov10_thread2, "yolov10", "right")
            elif "yolov5" in self.model_name2 and self.checkSegName(self.model_name2):
                self.changeModelProcess(self.yolov5seg_thread2, "yolov5-seg", "right")
            elif "yolov8" in self.model_name2 and self.checkSegName(self.model_name2):
                self.changeModelProcess(self.yolov8seg_thread2, "yolov8-seg", "right")
            elif "rtdetr" in self.model_name2:
                self.changeModelProcess(self.rtdetr_thread2, "rtdetr", "right")
            elif "yolov8" in self.model_name2 and self.checkPoseName(self.model_name2):
                self.changeModelProcess(self.yolov8pose_thread2, "yolov8-pose", "right")
            elif "yolov8" in self.model_name2 and self.checkObbName(self.model_name2):
                self.changeModelProcess(self.yolov8obb_thread2, "yolov8-obb", "right")
            else:
                self.stopOtherModel(mode="right")

    def runRightModelProcess(self, yolo_thread, mode="start"):
        if mode == "start":
            yolo_thread.source = self.inputPath
            yolo_thread.stop_dtc = False
            yolo_thread.is_continue = True
            if not yolo_thread.isRunning():
                yolo_thread.start()
        else:
            yolo_thread.is_continue = False

    # 运行右侧模型
    def runRightModel(self, mode=None):
        if mode == "start":
            if "yolov5" in self.model_name2 and not self.checkSegName(self.model_name2):
                self.runRightModelProcess(self.yolov5_thread2, "start")
            elif "yolov7" in self.model_name2:
                self.runRightModelProcess(self.yolov7_thread2, "start")
            elif "yolov8" in self.model_name2 and not self.checkSegName(self.model_name2) \
                    and not self.checkPoseName(self.model_name2) and not self.checkObbName(self.model_name2):
                self.runRightModelProcess(self.yolov8_thread2, "start")
            elif "yolov9" in self.model_name2:
                self.runRightModelProcess(self.yolov9_thread2, "start")
            elif "yolov10" in self.model_name2:
                self.runRightModelProcess(self.yolov10_thread2, "start")
            elif "yolov5" in self.model_name2 and self.checkSegName(self.model_name2):
                self.runRightModelProcess(self.yolov5seg_thread2, "start")
            elif "yolov8" in self.model_name2 and self.checkSegName(self.model_name2):
                self.runRightModelProcess(self.yolov8seg_thread2, "start")
            elif "rtdetr" in self.model_name2:
                self.runRightModelProcess(self.rtdetr_thread2, "start")
            elif "yolov8" in self.model_name2 and self.checkPoseName(self.model_name2):
                self.runRightModelProcess(self.yolov8pose_thread2, "start")
            elif "yolov8" in self.model_name2 and self.checkObbName(self.model_name2):
                self.runRightModelProcess(self.yolov8obb_thread2, "start")
        elif mode == "pause":
            if "yolov5" in self.model_name2 and not self.checkSegName(self.model_name2):
                self.runRightModelProcess(self.yolov5_thread2, "pause")
            elif "yolov7" in self.model_name2:
                self.runRightModelProcess(self.yolov7_thread2, "pause")
            elif "yolov8" in self.model_name2 and not self.checkSegName(self.model_name2) \
                    and not self.checkPoseName(self.model_name2) and not self.checkObbName(self.model_name2):
                self.runRightModelProcess(self.yolov8_thread2, "pause")
            elif "yolov9" in self.model_name2:
                self.runRightModelProcess(self.yolov9_thread2, "pause")
            elif "yolov10" in self.model_name2:
                self.runRightModelProcess(self.yolov10_thread2, "pause")
            elif "yolov5" in self.model_name2 and self.checkSegName(self.model_name2):
                self.runRightModelProcess(self.yolov5seg_thread2, "pause")
            elif "yolov8" in self.model_name2 and self.checkSegName(self.model_name2):
                self.runRightModelProcess(self.yolov8seg_thread2, "pause")
            elif "rtdetr" in self.model_name2:
                self.runRightModelProcess(self.rtdetr_thread2, "pause")
            elif "yolov8" in self.model_name2 and self.checkPoseName(self.model_name2):
                self.runRightModelProcess(self.yolov8pose_thread2, "pause")
            elif "yolov8" in self.model_name2 and self.checkObbName(self.model_name2):
                self.runRightModelProcess(self.yolov8obb_thread2, "pause")

    def runModelProcess(self, yolo_thread):
        yolo_thread.source = self.inputPath
        yolo_thread.stop_dtc = False
        if self.ui.run_button.isChecked():
            yolo_thread.is_continue = True
            if not yolo_thread.isRunning():
                yolo_thread.start()
            self.runRightModel(mode="start")
        else:
            yolo_thread.is_continue = False
            self.runRightModel(mode="pause")
            self.showStatus('Pause Detection')

    # 运行模型
    def runModel(self, runbuttonStatus=None):
        self.ui.save_status_button.setEnabled(False)
        if runbuttonStatus:
            self.ui.run_button.setChecked(True)
        # 首先判断是否两边的模型均为正确模型
        if self.checkModelName(self.model_name1) and self.checkModelName(self.model_name2):
            self.showStatus('The current model is not supported')
            if self.ui.run_button.isChecked():
                self.ui.run_button.setChecked(False)
            return
        # 左侧模型
        if "yolov5" in self.model_name1 and not self.checkSegName(self.model_name1):
            self.runModelProcess(self.yolov5_thread1)
        elif "yolov7" in self.model_name1:
            self.runModelProcess(self.yolov7_thread1)
        elif "yolov8" in self.model_name1 and not self.checkSegName(self.model_name1) \
                and not self.checkPoseName(self.model_name1) and not self.checkObbName(self.model_name1):
            self.runModelProcess(self.yolov8_thread1)
        elif "yolov9" in self.model_name1:
            self.runModelProcess(self.yolov9_thread1)
        elif "yolov10" in self.model_name1:
            self.runModelProcess(self.yolov10_thread1)
        elif "yolov5" in self.model_name1 and self.checkSegName(self.model_name1):
            self.runModelProcess(self.yolov5seg_thread1)
        elif "yolov8" in self.model_name1 and self.checkSegName(self.model_name1):
            self.runModelProcess(self.yolov8seg_thread1)
        elif "rtdetr" in self.model_name1:
            self.runModelProcess(self.rtdetr_thread1)
        elif "yolov8" in self.model_name1 and self.checkPoseName(self.model_name1):
            self.runModelProcess(self.yolov8pose_thread1)
        elif "yolov8" in self.model_name1 and self.checkObbName(self.model_name1):
            self.runModelProcess(self.yolov8obb_thread1)

    # 开始/暂停 预测
    def runorContinue(self):
        if self.inputPath is not None:
            glo.set_value('yoloname1', self.model_name1)
            glo.set_value('yoloname2', self.model_name2)
            self.reloadModel()
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
        self.ui.Class_num1.setText('--')
        self.ui.Target_num1.setText('--')
        self.ui.fps_label1.setText('--')
        self.ui.Class_num2.setText('--')
        self.ui.Target_num2.setText('--')
        self.ui.fps_label2.setText('--')
