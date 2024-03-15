import sys
import os
import logging
# 禁止标准输出
sys.stdout = open(os.devnull, 'w')
logging.disable(logging.CRITICAL)  # 禁用所有级别的日志
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication
from utils import glo
from YOLOSHOW import MyWindow as yoloshowWindow
from YOLOSHOWVS import MyWindow as yoloshowVSWindow

def yoloshowvsSHOW():
    yoloshow_glo = glo.get_value('yoloshow')
    yoloshowvs_glo = glo.get_value('yoloshowvs')
    glo.set_value('yoloname1', "yolov5 yolov7 yolov8 yolov9 yolov5-seg yolov8-seg rtdetr")
    glo.set_value('yoloname2', "yolov5 yolov7 yolov8 yolov9 yolov5-seg yolov8-seg rtdetr")
    yoloshowvs_glo.reloadModel()
    yoloshowvs_glo.show()
    yoloshow_glo.animation_window = None
    yoloshow_glo.closed.disconnect()

def yoloshowSHOW():
    yoloshow_glo = glo.get_value('yoloshow')
    yoloshowvs_glo = glo.get_value('yoloshowvs')
    glo.set_value('yoloname', "yolov5 yolov7 yolov8 yolov9 yolov5-seg yolov8-seg rtdetr")
    yoloshow_glo.reloadModel()
    yoloshow_glo.show()
    yoloshowvs_glo.animation_window = None
    yoloshowvs_glo.closed.disconnect()
def yoloshow2vs():
    yoloshow_glo = glo.get_value('yoloshow')
    yoloshow_glo.closed.connect(yoloshowvsSHOW)
    yoloshow_glo.close()


def vs2yoloshow():
    yoloshowvs_glo = glo.get_value('yoloshowvs')
    yoloshowvs_glo.closed.connect(yoloshowSHOW)
    yoloshowvs_glo.close()


if __name__ == '__main__':
    app = QApplication([])  # 创建应用程序实例
    app.setWindowIcon(QIcon('images/swimmingliu.ico'))  # 设置应用程序图标

    # 为整个应用程序设置样式表，去除所有QFrame的边框
    app.setStyleSheet("QFrame { border: none; }")

    # 创建窗口实例
    yoloshow = yoloshowWindow()
    yoloshowvs = yoloshowVSWindow()

    # 初始化全局变量管理器，并设置值
    glo._init()  # 初始化全局变量空间
    glo.set_value('yoloshow', yoloshow)  # 存储yoloshow窗口实例
    glo.set_value('yoloshowvs', yoloshowvs)  # 存储yoloshowvs窗口实例

    # 从全局变量管理器中获取窗口实例
    yoloshow_glo = glo.get_value('yoloshow')
    yoloshowvs_glo = glo.get_value('yoloshowvs')

    # 显示yoloshow窗口
    yoloshow_glo.show()

    # 连接信号和槽，以实现界面之间的切换
    yoloshow_glo.src_vsmode.clicked.connect(yoloshow2vs)  # 从单模式切换到对比模式
    yoloshowvs_glo.src_singlemode.clicked.connect(vs2yoloshow)  # 从对比模式切换回单模式

    app.exec()  # 启动应用程序的事件循环
