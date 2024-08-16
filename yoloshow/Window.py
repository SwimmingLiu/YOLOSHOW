from utils import glo
import json
import os
import cv2
from PySide6.QtGui import QMouseEvent, QGuiApplication
from PySide6.QtCore import Qt, QPropertyAnimation, Signal
from ui.utils.customGrips import CustomGrip
from yoloshow.YOLOSHOW import YOLOSHOW
from yoloshow.YOLOSHOWVS import YOLOSHOWVS


class YOLOSHOWWindow(YOLOSHOW):
    # 定义关闭信号
    closed = Signal()

    def __init__(self):
        super(YOLOSHOWWindow, self).__init__()
        self.center()
        # --- 拖动窗口 改变窗口大小 --- #
        self.left_grip = CustomGrip(self, Qt.LeftEdge, True)
        self.right_grip = CustomGrip(self, Qt.RightEdge, True)
        self.top_grip = CustomGrip(self, Qt.TopEdge, True)
        self.bottom_grip = CustomGrip(self, Qt.BottomEdge, True)
        self.setAcceptDrops(True)  # ==> 设置窗口支持拖动（必须设置）
        # --- 拖动窗口 改变窗口大小 --- #
        self.animation_window = None

    # 鼠标拖入事件
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():  # 检查是否为文件
            event.acceptProposedAction()  # 接受拖拽的数据


    def dropEvent(self, event):
        # files = [url.toLocalFile() for url in event.mimeData().urls()]  # 获取所有文件路径
        file = event.mimeData().urls()[0].toLocalFile()  # ==> 获取文件路径
        if file:
            # 判断是否是文件夹
            if os.path.isdir(file):
                FileFormat = [".mp4", ".mkv", ".avi", ".flv", ".jpg", ".png", ".jpeg", ".bmp", ".dib", ".jpe", ".jp2"]
                Foldername = [(file + "/" + filename) for filename in os.listdir(file) for jpgname in
                              FileFormat
                              if jpgname in filename]
                self.inputPath = Foldername
                self.showImg(self.inputPath[0], self.main_leftbox, 'path')  # 显示文件夹中第一张图片
                self.showStatus('Loaded Folder：{}'.format(os.path.basename(file)))
            # 图片 / 视频
            else:
                self.inputPath = file
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
        glo.set_value('inputPath', self.inputPath)

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

    def showEvent(self, event):
        super().showEvent(event)
        if not event.spontaneous():
            # 这里定义显示动画
            self.animation = QPropertyAnimation(self, b"windowOpacity")
            self.animation.setDuration(500)  # 动画时间500毫秒
            self.animation.setStartValue(0)  # 从完全透明开始
            self.animation.setEndValue(1)  # 到完全不透明结束
            self.animation.start()

    def closeEvent(self, event):
        if not self.animation_window:
            config_file = 'config/setting.json'
            config = dict()
            config['iou'] = self.ui.iou_spinbox.value()
            config['conf'] = self.ui.conf_spinbox.value()
            config['delay'] = self.ui.speed_spinbox.value()
            config['line_thickness'] = self.ui.line_spinbox.value()
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            self.animation_window = QPropertyAnimation(self, b"windowOpacity")
            self.animation_window.setStartValue(1)
            self.animation_window.setEndValue(0)
            self.animation_window.setDuration(500)
            self.animation_window.start()
            self.animation_window.finished.connect(self.close)
            event.ignore()
        else:
            self.setWindowOpacity(1.0)
            self.closed.emit()

# 多套一个类 为了实现MouseLabel方法
class YOLOSHOWVSWindow(YOLOSHOWVS):
    closed = Signal()

    def __init__(self):
        super(YOLOSHOWVSWindow, self).__init__()
        self.center()
        # --- 拖动窗口 改变窗口大小 --- #
        self.left_grip = CustomGrip(self, Qt.LeftEdge, True)
        self.right_grip = CustomGrip(self, Qt.RightEdge, True)
        self.top_grip = CustomGrip(self, Qt.TopEdge, True)
        self.bottom_grip = CustomGrip(self, Qt.BottomEdge, True)
        self.setAcceptDrops(True) # ==> 设置窗口支持拖动（必须设置）
        # --- 拖动窗口 改变窗口大小 --- #
        self.animation_window = None


    # 鼠标拖入事件
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():  # 检查是否为文件
            event.acceptProposedAction()  # 接受拖拽的数据


    def dropEvent(self, event):
        # files = [url.toLocalFile() for url in event.mimeData().urls()]  # 获取所有文件路径
        file = event.mimeData().urls()[0].toLocalFile()  # ==> 获取文件路径
        if file:
            # 判断是否是文件夹
            if os.path.isdir(file):
                FileFormat = [".mp4", ".mkv", ".avi", ".flv", ".jpg", ".png", ".jpeg", ".bmp", ".dib", ".jpe", ".jp2"]
                Foldername = [(file + "/" + filename) for filename in os.listdir(file) for jpgname in
                              FileFormat
                              if jpgname in filename]
                self.inputPath = Foldername
                self.showImg(self.inputPath[0], self.main_leftbox, 'path')  # 显示文件夹中第一张图片
                self.showStatus('Loaded Folder：{}'.format(os.path.basename(file)))
            # 图片 / 视频
            else:
                self.inputPath = file
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
        glo.set_value('inputPath', self.inputPath)


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

    def showEvent(self, event):
        super().showEvent(event)
        if not event.spontaneous():
            # 这里定义显示动画
            self.animation = QPropertyAnimation(self, b"windowOpacity")
            self.animation.setDuration(500)  # 动画时间500毫秒
            self.animation.setStartValue(0)  # 从完全透明开始
            self.animation.setEndValue(1)  # 到完全不透明结束
            self.animation.start()

    def closeEvent(self, event):
        if not self.animation_window:
            config_file = 'config/setting.json'
            config = dict()
            config['iou'] = self.ui.iou_spinbox.value()
            config['conf'] = self.ui.conf_spinbox.value()
            config['delay'] = self.ui.speed_spinbox.value()
            config['line_thickness'] = self.ui.line_spinbox.value()
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            self.animation_window = QPropertyAnimation(self, b"windowOpacity")
            self.animation_window.setStartValue(1)
            self.animation_window.setEndValue(0)
            self.animation_window.setDuration(500)
            self.animation_window.start()
            self.animation_window.finished.connect(self.close)
            event.ignore()
        else:
            self.setWindowOpacity(1.0)
            self.closed.emit()