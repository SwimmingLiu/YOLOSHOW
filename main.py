from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication
from YOLOSHOW import *

if __name__ == '__main__':
    app = QApplication([])
    app.setWindowIcon(QIcon('images/swimmingliu.ico'))
    yoloshow = MyWindow()
    glo._init()
    glo.set_value('yoloshow', yoloshow)
    Glo_yolo = glo.get_value('yoloshow')
    Glo_yolo.show()
    app.exec()
