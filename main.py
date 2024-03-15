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
    app = QApplication([])
    app.setWindowIcon(QIcon('images/swimmingliu.ico'))
    yoloshow = yoloshowWindow()
    yoloshowvs = yoloshowVSWindow()
    glo._init()
    glo.set_value('yoloshow', yoloshow)
    glo.set_value('yoloshowvs', yoloshowvs)
    yoloshow_glo = glo.get_value('yoloshow')
    yoloshowvs_glo = glo.get_value('yoloshowvs')
    yoloshow_glo.show()
    yoloshow_glo.src_vsmode.clicked.connect(yoloshow2vs)
    yoloshowvs_glo.src_singlemode.clicked.connect(vs2yoloshow)
    app.exec()
