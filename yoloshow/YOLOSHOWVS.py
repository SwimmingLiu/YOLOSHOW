from utils import glo
import json
import os
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QFileDialog, QMainWindow
from PySide6.QtCore import QTimer, Qt
from PySide6 import QtCore, QtGui
from ui.YOLOSHOWUIVS import Ui_MainWindow
from yoloshow.YOLOSHOWBASE import YOLOSHOWBASE, MODEL_THREAD_CLASSES
from yoloshow.YOLOThreadPool import YOLOThreadPool

GLOBAL_WINDOW_STATE = True
WIDTH_LEFT_BOX_STANDARD = 80
WIDTH_LEFT_BOX_EXTENDED = 200
WIDTH_LOGO = 60
UI_FILE_PATH = "ui/YOLOSHOWUIVS.ui"


# YOLOSHOW窗口类 动态加载UI文件 和 Ui_mainWindow
class YOLOSHOWVS(QMainWindow, YOLOSHOWBASE):
    def __init__(self):
        super().__init__()
        self.current_workpath = os.getcwd()
        self.inputPath = None
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

    # 初始化模型线程
    def initThreads(self):
        self.yolo_threads = YOLOThreadPool()
        # 获取当前Model
        model_name_left = self.checkCurrentModel(mode="left")
        model_name_right = self.checkCurrentModel(mode="right")
        if model_name_left:
            self.yolo_threads.set(model_name_left, MODEL_THREAD_CLASSES[model_name_left]())
            self.initModel(yoloname=model_name_left)
        if model_name_right:
            self.yolo_threads.set(model_name_right, MODEL_THREAD_CLASSES[model_name_right]())
            self.initModel(yoloname=model_name_right)

    # 加载模型
    def initModel(self, yoloname=None):
        yolo_thread = self.yolo_threads.get(yoloname)
        if not yolo_thread:
            raise ValueError(f"No thread found for '{yoloname}'")
        if yoloname.endswith("left"):
            # 左侧模型加载
            yolo_thread.new_model_name = f'{self.current_workpath}/ptfiles/' + self.ui.model_box1.currentText()
            yolo_thread.progress_value = self.ui.progress_bar.maximum()
            yolo_thread.send_output.connect(lambda x: self.showImg(x, self.ui.main_leftbox, 'img'))
            # 第一个模型来控制消息
            yolo_thread.send_msg.connect(lambda x: self.showStatus(x))
            yolo_thread.send_fps.connect(lambda x: self.ui.fps_label1.setText(str(x)))
            yolo_thread.send_class_num.connect(lambda x: self.ui.Class_num1.setText(str(x)))
            yolo_thread.send_target_num.connect(lambda x: self.ui.Target_num1.setText(str(x)))
        elif yoloname.endswith("right"):
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

    # 导出结果
    def saveResult(self):
        if not any(
                thread.res_status for thread_name, thread in self.yolo_threads.threads_pool.items() if
                thread_name.endswith("left")):
            self.showStatus("Please select the Image/Video before starting detection...")
            return
        config_file = f'{self.current_workpath}/config/save.json'
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        save_path = config.get('save_path', os.getcwd())
        is_folder = isinstance(self.inputPath, list)
        if is_folder:
            self.OutputDir = QFileDialog.getExistingDirectory(
                self,  # 父窗口对象
                "Save Results in new Folder",  # 标题
                save_path,  # 起始目录
            )
            current_model_name = self.checkCurrentModel(mode="left")
            self.saveResultProcess(self.OutputDir, current_model_name, folder=True)
        else:
            self.OutputDir, _ = QFileDialog.getSaveFileName(
                self,  # 父窗口对象
                "Save Image/Video",  # 标题
                save_path,  # 起始目录
                "Image/Vide Type (*.jpg *.jpeg *.png *.bmp *.dib  *.jpe  *.jp2 *.mp4)"  # 选择类型过滤项，过滤内容在括号中
            )
            current_model_name = self.checkCurrentModel(mode="left")
            self.saveResultProcess(self.OutputDir, current_model_name, folder=False)

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

    # 重新加载模型
    def resignModel(self, model_name, mode=None):
        self.reloadModel()
        self.yolo_threads.set(model_name, MODEL_THREAD_CLASSES[model_name]())
        self.initModel(yoloname=model_name)
        self.ui.run_button.setChecked(True)
        if mode == "left":
            self.ContinueAnotherModel(mode="right")
        else:
            self.ContinueAnotherModel(mode="left")
        self.runModel(True)

    def stopOtherModelProcess(self, model_name, current_yoloname, mode="left"):
        yolo_thread = self.yolo_threads.get(model_name)
        yolo_thread.finished.connect((lambda: self.resignModel(current_yoloname, mode=mode)))
        yolo_thread.stop_dtc = True
        self.yolo_threads.stop_thread(model_name)

    # 停止其他模型
    def stopOtherModel(self, current_yoloname=None, mode=None):
        for model_name, thread in self.yolo_threads.threads_pool.items():
            if not current_yoloname or model_name == current_yoloname:
                continue
            if not thread.isRunning():
                continue
            if mode == "left":
                self.PauseAnotherModel(mode="right")
                self.stopOtherModelProcess(model_name, current_yoloname, mode=mode)
            else:
                self.PauseAnotherModel(mode="left")
                self.stopOtherModelProcess(model_name, current_yoloname, mode=mode)

    def PauseAnotherModelProcess(self, model_name):
        yolo_thread = self.yolo_threads.get(model_name)
        yolo_thread.stop_dtc = True
        self.yolo_threads.stop_thread(model_name)

    # 暂停另外一侧模型
    def PauseAnotherModel(self, mode=None):
        buttonStatus = self.ui.run_button.isChecked()
        if buttonStatus:
            model_name_another = self.checkCurrentModel(mode=mode)
            self.PauseAnotherModelProcess(model_name_another)

    # 继续另外一侧模型
    def ContinueAnotherModel(self, mode=None):
        buttonStatus = self.ui.run_button.isChecked()
        if buttonStatus:
            model_name_another = self.checkCurrentModel(mode=mode)
            self.yolo_threads.set(model_name_another, MODEL_THREAD_CLASSES[model_name_another]())
            self.initModel(yoloname=model_name_another)

    def changeModelProcess(self, yoloname, mode=None):
        if mode == "left":
            self.stopOtherModel(yoloname, mode="left")
            yolo_thread = self.yolo_threads.get(yoloname)
            if yolo_thread is not None:
                yolo_thread.new_model_name = f'{self.current_workpath}/ptfiles/' + self.ui.model_box1.currentText()
            else:
                self.yolo_threads.set(yoloname, MODEL_THREAD_CLASSES[yoloname]())
                self.initModel(yoloname=yoloname)
            # 重载 common 和 yolo 模块
            glo.set_value('yoloname1', yoloname)
        else:
            self.stopOtherModel(yoloname, mode="right")
            yolo_thread = self.yolo_threads.get(yoloname)
            if yolo_thread is not None:
                yolo_thread.new_model_name = f'{self.current_workpath}/ptfiles/' + self.ui.model_box2.currentText()
            else:
                self.yolo_threads.set(yoloname, MODEL_THREAD_CLASSES[yoloname]())
                self.initModel(yoloname=yoloname)
            # 重载 common 和 yolo 模块
            glo.set_value('yoloname2', yoloname)

    # Model 变化
    def changeModel(self, mode=None):
        if mode == "left":
            # 左侧模型
            self.model_name1 = self.ui.model_box1.currentText()
            self.ui.Model_label1.setText(str(self.model_name1).replace(".pt", ""))  # 修改状态栏显示
            yolo_name = self.checkCurrentModel(mode="left")
            if yolo_name:
                self.changeModelProcess(yolo_name, "left")
            else:
                self.stopOtherModel(mode="left")
        else:
            # 右侧模型
            self.model_name2 = self.ui.model_box2.currentText()
            self.ui.Model_label2.setText(str(self.model_name2).replace(".pt", ""))
            yolo_name = self.checkCurrentModel(mode="right")
            if yolo_name:
                self.changeModelProcess(yolo_name, "right")
            else:
                self.stopOtherModel(mode="right")

    def runRightModelProcess(self, model_name, mode="start"):
        yolo_thread = self.yolo_threads.get(model_name)
        if mode == "start":
            yolo_thread.source = self.inputPath
            yolo_thread.stop_dtc = False
            yolo_thread.is_continue = True
            self.yolo_threads.start_thread(model_name)
        else:
            yolo_thread.is_continue = False

    # 运行右侧模型
    def runRightModel(self, mode=None):
        model_name_right = self.checkCurrentModel(mode="right")
        if mode == "start":
            self.runRightModelProcess(model_name_right, "start")
        elif mode == "pause":
            self.runRightModelProcess(model_name_right, "pause")

    def runModelProcess(self, yolo_name):
        yolo_thread = self.yolo_threads.get(yolo_name)
        yolo_thread.source = self.inputPath
        yolo_thread.stop_dtc = False
        if self.ui.run_button.isChecked():
            yolo_thread.is_continue = True
            self.yolo_threads.start_thread(yolo_name)
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
        model_name_left = self.checkCurrentModel(mode="left")
        self.runModelProcess(model_name_left)

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
