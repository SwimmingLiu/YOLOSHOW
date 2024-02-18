import cv2
import numpy as np
from PySide6.QtGui import QImage, Qt
from PySide6.QtCore import QThread, Signal


class Camera:
    def __init__(self, cam_preset_num=1):
        self.cam_preset_num = cam_preset_num

    def get_cam_num(self):
        cnt = 0
        devices = []
        for device in range(0, self.cam_preset_num):
            stream = cv2.VideoCapture(device)
            grabbed = stream.grab()
            stream.release()
            if not grabbed:
                continue
            else:
                cnt = cnt + 1
                devices.append(device)
        return cnt, devices


class WebcamThread(QThread):
    changePixmap = Signal(np.ndarray)

    def __init__(self, cam, parent=None):
        QThread.__init__(self, parent)
        self.cam = cam

    def run(self):
        cap = cv2.VideoCapture(self.cam)
        ret, frame = cap.read()
        if ret:
            self.changePixmap.emit(frame)
        cap.release()

if __name__ == '__main__':
    cam = Camera()
    cam_num, devices = cam.get_cam_num()
    print(cam_num, devices)
