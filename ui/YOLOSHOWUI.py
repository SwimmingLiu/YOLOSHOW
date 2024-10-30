# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'YOLOSHOWUI.ui'
##
## Created by: Qt User Interface Compiler version 6.8.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QDoubleSpinBox, QFrame,
    QHBoxLayout, QLabel, QLayout, QMainWindow,
    QProgressBar, QPushButton, QSizePolicy, QSlider,
    QSpacerItem, QSpinBox, QSplitter, QVBoxLayout,
    QWidget)

from qfluentwidgets import ComboBox
from ui.utils.UpdateFrame import DoubleClickQFrame
import YOLOSHOWUI_rc

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1550, 839)
        MainWindow.setStyleSheet(u"")
        self.mainWindow = QWidget(MainWindow)
        self.mainWindow.setObjectName(u"mainWindow")
        self.mainWindow.setStyleSheet(u"QWidget#mainWindow{\n"
"	border:none\n"
"}")
        self.verticalLayout_4 = QVBoxLayout(self.mainWindow)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(9, -1, -1, -1)
        self.mainBody = QFrame(self.mainWindow)
        self.mainBody.setObjectName(u"mainBody")
        self.mainBody.setStyleSheet(u"QFrame#mainBody{\n"
"	border: 0px solid rgba(0, 0, 0, 40%);\n"
"	border-bottom:none;\n"
"	border-bottom-left-radius: 0;\n"
"	border-bottom-right-radius: 0;\n"
"	border-radius:30%;\n"
"	background-color: white;\n"
"	/*	\n"
"	background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #B5FFFC, stop:0.2 #e0c3fc, stop:1 #FFDEE9);\n"
"	*/\n"
"}")
        self.mainBody.setFrameShape(QFrame.StyledPanel)
        self.mainBody.setFrameShadow(QFrame.Raised)
        self.verticalLayout = QVBoxLayout(self.mainBody)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.topbox = DoubleClickQFrame(self.mainBody)
        self.topbox.setObjectName(u"topbox")
        self.topbox.setStyleSheet(u"")
        self.topBox = QHBoxLayout(self.topbox)
        self.topBox.setSpacing(0)
        self.topBox.setObjectName(u"topBox")
        self.topBox.setContentsMargins(0, 8, 0, 8)
        self.left_top = QFrame(self.topbox)
        self.left_top.setObjectName(u"left_top")
        self.left_top.setMaximumSize(QSize(150, 16777215))
        self.left_top.setStyleSheet(u"/* QFrame#left_top{\n"
"	border: 1px solid red;\n"
"} */\n"
"QPushButton#closeButton {\n"
"    background-color: rgb(255, 59, 48); /* \u7ea2\u8272 */\n"
"    border: none;\n"
"    border-radius: 10px; /* \u4f7f\u6309\u94ae\u5706\u5f62 */\n"
"    min-width: 20px;\n"
"    max-width: 20px;\n"
"    min-height: 20px;\n"
"    max-height: 20px;\n"
"}\n"
"\n"
"QPushButton#maximizeButton {\n"
"    background-color: rgb(40, 205, 65); /* \u9ec4\u8272 */\n"
"    border: none;\n"
"    border-radius: 10px; /* \u4f7f\u6309\u94ae\u5706\u5f62 */\n"
"    min-width: 20px;\n"
"    max-width: 20px;\n"
"    min-height: 20px;\n"
"    max-height: 20px;\n"
"}\n"
"\n"
"QPushButton#minimizeButton {\n"
"    background-color: rgb(255, 214, 10); /* \u7eff\u8272 */\n"
"    border: none;\n"
"    border-radius: 10px; /* \u4f7f\u6309\u94ae\u5706\u5f62 */\n"
"    min-width: 20px;\n"
"    max-width: 20px;\n"
"    min-height: 20px;\n"
"    max-height: 20px;\n"
"}\n"
"")
        self.left_top.setFrameShape(QFrame.StyledPanel)
        self.left_top.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_2 = QHBoxLayout(self.left_top)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.closeButton = QPushButton(self.left_top)
        self.closeButton.setObjectName(u"closeButton")
        self.closeButton.setStyleSheet(u"QPushButton:hover{\n"
"	background-color:rgb(139, 29, 31);\n"
"	border-image: url(:/leftbox/images/newsize/close.png);\n"
"}\n"
"QPushButton:pressed {\n"
"	background-color: rgb(232, 59, 35);\n"
"}")

        self.horizontalLayout_2.addWidget(self.closeButton)

        self.minimizeButton = QPushButton(self.left_top)
        self.minimizeButton.setObjectName(u"minimizeButton")
        self.minimizeButton.setStyleSheet(u"QPushButton:hover{\n"
"	background-color:rgb(139, 29, 31);\n"
"	border-image: url(:/leftbox/images/newsize/min.png);\n"
"}\n"
"QPushButton:pressed {\n"
"	background-color:  rgb(255, 214, 5);\n"
"}\n"
"\n"
"")

        self.horizontalLayout_2.addWidget(self.minimizeButton)

        self.maximizeButton = QPushButton(self.left_top)
        self.maximizeButton.setObjectName(u"maximizeButton")
        self.maximizeButton.setStyleSheet(u"QPushButton:hover{\n"
"	background-color:rgb(139, 29, 31);\n"
"	border-image: url(:/leftbox/images/newsize/max.png);\n"
"}\n"
"QPushButton:pressed {\n"
"	background-color: rgb(40, 205, 60);\n"
"}")

        self.horizontalLayout_2.addWidget(self.maximizeButton)


        self.topBox.addWidget(self.left_top)

        self.right_top = QFrame(self.topbox)
        self.right_top.setObjectName(u"right_top")
        self.right_top.setStyleSheet(u"QLabel#title{\n"
"    background-color: none;\n"
"	font-size: 22px;\n"
"	font-family: \"Shojumaru\";\n"
"	color: black;\n"
"}\n"
"Spacer{\n"
"	border:none;\n"
"}")
        self.right_top.setFrameShape(QFrame.StyledPanel)
        self.right_top.setFrameShadow(QFrame.Raised)
        self.horizontalLayout = QHBoxLayout(self.right_top)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.title = QLabel(self.right_top)
        self.title.setObjectName(u"title")
        self.title.setStyleSheet(u"")
        self.title.setIndent(-1)

        self.horizontalLayout.addWidget(self.title, 0, Qt.AlignHCenter)


        self.topBox.addWidget(self.right_top, 0, Qt.AlignHCenter)

        self.topBox.setStretch(0, 1)
        self.topBox.setStretch(1, 9)

        self.verticalLayout.addWidget(self.topbox)

        self.mainbox = QFrame(self.mainBody)
        self.mainbox.setObjectName(u"mainbox")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mainbox.sizePolicy().hasHeightForWidth())
        self.mainbox.setSizePolicy(sizePolicy)
        self.mainbox.setStyleSheet(u"QFrame#mainbox{\n"
"	border: 1px solid rgba(0, 0, 0, 15%);\n"
"   /*\n"
"	background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #B5FFFC, stop:0.2 #e0c3fc, stop:1 #FFDEE9);\n"
"	 */\n"
"	border-bottom-left-radius: 0;\n"
"	border-bottom-right-radius: 0;\n"
"	border-radius:30%;\n"
"}\n"
"")
        self.mainBox = QHBoxLayout(self.mainbox)
        self.mainBox.setSpacing(0)
        self.mainBox.setObjectName(u"mainBox")
        self.mainBox.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.mainBox.setContentsMargins(0, 0, 0, 0)
        self.leftBox = QFrame(self.mainbox)
        self.leftBox.setObjectName(u"leftBox")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.leftBox.sizePolicy().hasHeightForWidth())
        self.leftBox.setSizePolicy(sizePolicy1)
        self.leftBox.setMinimumSize(QSize(180, 0))
        self.leftBox.setMaximumSize(QSize(180, 16777215))
        self.leftBox.setStyleSheet(u"QFrame#leftBox  {\n"
"    /*background-color: rgba(255, 255, 255, 80%);*/\n"
"    border: 0px solid rgba(0, 0, 0, 40%);\n"
"	border-top:none;\n"
"	border-bottom:none;\n"
"	border-left:none;\n"
"}\n"
"")
        self.leftBox.setFrameShape(QFrame.StyledPanel)
        self.leftBox.setFrameShadow(QFrame.Raised)
        self.verticalLayout_2 = QVBoxLayout(self.leftBox)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.leftbox_top = QFrame(self.leftBox)
        self.leftbox_top.setObjectName(u"leftbox_top")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.leftbox_top.sizePolicy().hasHeightForWidth())
        self.leftbox_top.setSizePolicy(sizePolicy2)
        self.leftbox_top.setMinimumSize(QSize(180, 0))
        self.leftbox_top.setFrameShape(QFrame.StyledPanel)
        self.leftbox_top.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_8 = QHBoxLayout(self.leftbox_top)
        self.horizontalLayout_8.setSpacing(5)
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.horizontalLayout_8.setContentsMargins(10, 10, 15, 3)
        self.logo = QFrame(self.leftbox_top)
        self.logo.setObjectName(u"logo")
        sizePolicy.setHeightForWidth(self.logo.sizePolicy().hasHeightForWidth())
        self.logo.setSizePolicy(sizePolicy)
        self.logo.setMinimumSize(QSize(60, 60))
        self.logo.setMaximumSize(QSize(60, 60))
        self.logo.setStyleSheet(u"image: url(:/leftbox/images/yoloshow.png);\n"
"border:2px solid rgba(0,0,0,15%);\n"
"border-radius: 15%;\n"
"")
        self.logo.setFrameShape(QFrame.StyledPanel)
        self.logo.setFrameShadow(QFrame.Raised)

        self.horizontalLayout_8.addWidget(self.logo)

        self.top_intro = QFrame(self.leftbox_top)
        self.top_intro.setObjectName(u"top_intro")
        self.top_intro.setStyleSheet(u"QLabel{\n"
"	color: black;\n"
"	font: 600 italic 9pt \"Segoe UI\";\n"
"	font-weight: bold;\n"
"}\n"
"QFrame{\n"
"	border:none;\n"
"}")
        self.top_intro.setFrameShape(QFrame.StyledPanel)
        self.top_intro.setFrameShadow(QFrame.Raised)
        self.verticalLayout_20 = QVBoxLayout(self.top_intro)
        self.verticalLayout_20.setSpacing(0)
        self.verticalLayout_20.setObjectName(u"verticalLayout_20")
        self.verticalLayout_20.setContentsMargins(5, 0, 0, 0)
        self.label = QLabel(self.top_intro)
        self.label.setObjectName(u"label")

        self.verticalLayout_20.addWidget(self.label)

        self.label_3 = QLabel(self.top_intro)
        self.label_3.setObjectName(u"label_3")

        self.verticalLayout_20.addWidget(self.label_3)


        self.horizontalLayout_8.addWidget(self.top_intro)

        self.horizontalLayout_8.setStretch(0, 4)
        self.horizontalLayout_8.setStretch(1, 6)

        self.verticalLayout_2.addWidget(self.leftbox_top)

        self.leftbox_bottom = QFrame(self.leftBox)
        self.leftbox_bottom.setObjectName(u"leftbox_bottom")
        sizePolicy.setHeightForWidth(self.leftbox_bottom.sizePolicy().hasHeightForWidth())
        self.leftbox_bottom.setSizePolicy(sizePolicy)
        self.leftbox_bottom.setMinimumSize(QSize(0, 0))
        self.leftbox_bottom.setMaximumSize(QSize(16777215, 16777215))
        self.leftbox_bottom.setStyleSheet(u"QPushButton#src_menu{\n"
"	background-image: url(:/leftbox/images/newsize/menu.png);\n"
"}\n"
"QPushButton#src_folder{\n"
"	background-image: url(:/leftbox/images/newsize/folder.png);\n"
"\n"
"}\n"
"QPushButton#src_camera{\n"
"	background-image: url(:/leftbox/images/newsize/security-camera.png);\n"
"}\n"
"QPushButton#src_img{\n"
"	background-image: url(:/leftbox/images/newsize/gallery.png);\n"
"}\n"
"QPushButton#src_webcam{\n"
"	background-image: url(:/leftbox/images/newsize/photo-camera.png);\n"
"}\n"
"QPushButton#src_setting{\n"
"	background-image:url(:/leftbox/images/newsize/setting.png);\n"
"}\n"
"QPushButton#src_vsmode{\n"
"	background-image:url(:/leftbox/images/newsize/vs.png);\n"
"}\n"
"QPushButton#src_result{\n"
"	background-image:url(:/leftbox/images/newsize/statistics.png);\n"
"}\n"
"QPushButton#src_table{\n"
"	background-image:url(:/leftbox/images/newsize/table.png);\n"
"}\n"
"QPushButton{\n"
"	border:none;\n"
"	text-align: center;\n"
"	background-repeat: no-repeat;\n"
"	background-position: left ce"
                        "nter;\n"
"	border-left: 23px solid transparent;\n"
"	color: rgba(0, 0, 0, 199);\n"
"	font: 12pt \"Times New Roman\";\n"
"	font-weight: bold;\n"
"	padding-left: 15px;\n"
"}\n"
"QFrame#cameraBox:hover{\n"
"	background-color: rgba(114, 129, 214, 59);\n"
"}\n"
"QFrame#folderBox:hover{\n"
"	background-color: rgba(114, 129, 214, 59);\n"
"}\n"
"QFrame#imgBox:hover{\n"
"	background-color: rgba(114, 129, 214, 59);\n"
"}\n"
"QFrame#menuBox:hover{\n"
"	background-color: rgba(114, 129, 214, 59);\n"
"}\n"
"QFrame#webcamBox:hover{\n"
"	background-color: rgba(114, 129, 214, 59);\n"
"}\n"
"QFrame#manageBox:hover{\n"
"	background-color: rgba(114, 129, 214, 59);\n"
"}\n"
"QFrame#vsBox:hover{\n"
"	background-color: rgba(114, 129, 214, 59);\n"
"}\n"
"QFrame#resultBox:hover{\n"
"	background-color: rgba(114, 129, 214, 59);\n"
"}\n"
"QFrame#tableBox:hover{\n"
"	background-color: rgba(114, 129, 214, 59);\n"
"}")
        self.leftbox_bottom.setFrameShape(QFrame.StyledPanel)
        self.leftbox_bottom.setFrameShadow(QFrame.Raised)
        self.verticalLayout_3 = QVBoxLayout(self.leftbox_bottom)
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 10, 0, 0)
        self.zSpacer5 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_3.addItem(self.zSpacer5)

        self.menuBox = QFrame(self.leftbox_bottom)
        self.menuBox.setObjectName(u"menuBox")
        sizePolicy2.setHeightForWidth(self.menuBox.sizePolicy().hasHeightForWidth())
        self.menuBox.setSizePolicy(sizePolicy2)
        self.menuBox.setMinimumSize(QSize(180, 0))
        self.menuBox.setMaximumSize(QSize(180, 16777215))
        self.menuBox.setLayoutDirection(Qt.LeftToRight)
        self.menuBox.setFrameShape(QFrame.StyledPanel)
        self.menuBox.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_3 = QHBoxLayout(self.menuBox)
        self.horizontalLayout_3.setSpacing(0)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.src_menu = QPushButton(self.menuBox)
        self.src_menu.setObjectName(u"src_menu")
        sizePolicy2.setHeightForWidth(self.src_menu.sizePolicy().hasHeightForWidth())
        self.src_menu.setSizePolicy(sizePolicy2)
        self.src_menu.setMinimumSize(QSize(180, 0))
        self.src_menu.setMaximumSize(QSize(180, 16777215))
        self.src_menu.setIconSize(QSize(30, 30))
        self.src_menu.setAutoDefault(False)

        self.horizontalLayout_3.addWidget(self.src_menu)


        self.verticalLayout_3.addWidget(self.menuBox)

        self.zSpacer1 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_3.addItem(self.zSpacer1)

        self.imgBox = QFrame(self.leftbox_bottom)
        self.imgBox.setObjectName(u"imgBox")
        self.imgBox.setMinimumSize(QSize(180, 0))
        self.imgBox.setMaximumSize(QSize(180, 16777215))
        self.imgBox.setLayoutDirection(Qt.LeftToRight)
        self.imgBox.setFrameShape(QFrame.StyledPanel)
        self.imgBox.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_4 = QHBoxLayout(self.imgBox)
        self.horizontalLayout_4.setSpacing(0)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.src_img = QPushButton(self.imgBox)
        self.src_img.setObjectName(u"src_img")
        sizePolicy2.setHeightForWidth(self.src_img.sizePolicy().hasHeightForWidth())
        self.src_img.setSizePolicy(sizePolicy2)
        self.src_img.setMinimumSize(QSize(180, 0))
        self.src_img.setMaximumSize(QSize(180, 16777215))
        self.src_img.setIconSize(QSize(30, 30))

        self.horizontalLayout_4.addWidget(self.src_img, 0, Qt.AlignLeft)


        self.verticalLayout_3.addWidget(self.imgBox)

        self.zSpacer2 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_3.addItem(self.zSpacer2)

        self.webcamBox = QFrame(self.leftbox_bottom)
        self.webcamBox.setObjectName(u"webcamBox")
        self.webcamBox.setMinimumSize(QSize(180, 0))
        self.webcamBox.setMaximumSize(QSize(180, 16777215))
        self.webcamBox.setLayoutDirection(Qt.LeftToRight)
        self.webcamBox.setFrameShape(QFrame.StyledPanel)
        self.webcamBox.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_7 = QHBoxLayout(self.webcamBox)
        self.horizontalLayout_7.setSpacing(0)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.horizontalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.src_webcam = QPushButton(self.webcamBox)
        self.src_webcam.setObjectName(u"src_webcam")
        sizePolicy2.setHeightForWidth(self.src_webcam.sizePolicy().hasHeightForWidth())
        self.src_webcam.setSizePolicy(sizePolicy2)
        self.src_webcam.setMinimumSize(QSize(180, 0))
        self.src_webcam.setMaximumSize(QSize(180, 16777215))
        self.src_webcam.setLayoutDirection(Qt.LeftToRight)
        self.src_webcam.setIconSize(QSize(30, 30))

        self.horizontalLayout_7.addWidget(self.src_webcam, 0, Qt.AlignLeft)


        self.verticalLayout_3.addWidget(self.webcamBox)

        self.zSpacer4 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_3.addItem(self.zSpacer4)

        self.folderBox = QFrame(self.leftbox_bottom)
        self.folderBox.setObjectName(u"folderBox")
        self.folderBox.setMinimumSize(QSize(180, 0))
        self.folderBox.setMaximumSize(QSize(180, 16777215))
        self.folderBox.setLayoutDirection(Qt.LeftToRight)
        self.folderBox.setFrameShape(QFrame.StyledPanel)
        self.folderBox.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_6 = QHBoxLayout(self.folderBox)
        self.horizontalLayout_6.setSpacing(0)
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.src_folder = QPushButton(self.folderBox)
        self.src_folder.setObjectName(u"src_folder")
        sizePolicy2.setHeightForWidth(self.src_folder.sizePolicy().hasHeightForWidth())
        self.src_folder.setSizePolicy(sizePolicy2)
        self.src_folder.setMinimumSize(QSize(180, 0))
        self.src_folder.setMaximumSize(QSize(180, 16777215))
        self.src_folder.setIconSize(QSize(30, 30))

        self.horizontalLayout_6.addWidget(self.src_folder, 0, Qt.AlignLeft)


        self.verticalLayout_3.addWidget(self.folderBox)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_3.addItem(self.verticalSpacer)

        self.cameraBox = QFrame(self.leftbox_bottom)
        self.cameraBox.setObjectName(u"cameraBox")
        self.cameraBox.setMinimumSize(QSize(180, 0))
        self.cameraBox.setMaximumSize(QSize(180, 16777215))
        self.cameraBox.setLayoutDirection(Qt.LeftToRight)
        self.cameraBox.setFrameShape(QFrame.StyledPanel)
        self.cameraBox.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_5 = QHBoxLayout(self.cameraBox)
        self.horizontalLayout_5.setSpacing(0)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.src_camera = QPushButton(self.cameraBox)
        self.src_camera.setObjectName(u"src_camera")
        sizePolicy2.setHeightForWidth(self.src_camera.sizePolicy().hasHeightForWidth())
        self.src_camera.setSizePolicy(sizePolicy2)
        self.src_camera.setMinimumSize(QSize(180, 0))
        self.src_camera.setMaximumSize(QSize(180, 16777215))
        self.src_camera.setIconSize(QSize(30, 30))

        self.horizontalLayout_5.addWidget(self.src_camera, 0, Qt.AlignLeft)


        self.verticalLayout_3.addWidget(self.cameraBox)

        self.zSpacer3 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_3.addItem(self.zSpacer3)

        self.resultBox = QFrame(self.leftbox_bottom)
        self.resultBox.setObjectName(u"resultBox")
        self.resultBox.setFrameShape(QFrame.StyledPanel)
        self.resultBox.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_27 = QHBoxLayout(self.resultBox)
        self.horizontalLayout_27.setSpacing(0)
        self.horizontalLayout_27.setObjectName(u"horizontalLayout_27")
        self.horizontalLayout_27.setContentsMargins(0, 0, 0, 0)
        self.src_result = QPushButton(self.resultBox)
        self.src_result.setObjectName(u"src_result")
        sizePolicy2.setHeightForWidth(self.src_result.sizePolicy().hasHeightForWidth())
        self.src_result.setSizePolicy(sizePolicy2)
        self.src_result.setMinimumSize(QSize(180, 0))
        self.src_result.setMaximumSize(QSize(180, 16777215))
        self.src_result.setIconSize(QSize(30, 30))

        self.horizontalLayout_27.addWidget(self.src_result)


        self.verticalLayout_3.addWidget(self.resultBox)

        self.verticalSpacer_5 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_3.addItem(self.verticalSpacer_5)

        self.tableBox = QFrame(self.leftbox_bottom)
        self.tableBox.setObjectName(u"tableBox")
        self.tableBox.setFrameShape(QFrame.StyledPanel)
        self.tableBox.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_28 = QHBoxLayout(self.tableBox)
        self.horizontalLayout_28.setSpacing(0)
        self.horizontalLayout_28.setObjectName(u"horizontalLayout_28")
        self.horizontalLayout_28.setContentsMargins(0, 0, 0, 0)
        self.src_table = QPushButton(self.tableBox)
        self.src_table.setObjectName(u"src_table")
        sizePolicy2.setHeightForWidth(self.src_table.sizePolicy().hasHeightForWidth())
        self.src_table.setSizePolicy(sizePolicy2)
        self.src_table.setMinimumSize(QSize(180, 0))
        self.src_table.setMaximumSize(QSize(180, 16777215))
        self.src_table.setIconSize(QSize(30, 30))

        self.horizontalLayout_28.addWidget(self.src_table)


        self.verticalLayout_3.addWidget(self.tableBox)

        self.verticalSpacer_6 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_3.addItem(self.verticalSpacer_6)

        self.vsBox = QFrame(self.leftbox_bottom)
        self.vsBox.setObjectName(u"vsBox")
        self.vsBox.setMinimumSize(QSize(180, 0))
        self.vsBox.setMaximumSize(QSize(180, 16777215))
        self.vsBox.setFrameShape(QFrame.StyledPanel)
        self.vsBox.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_26 = QHBoxLayout(self.vsBox)
        self.horizontalLayout_26.setSpacing(0)
        self.horizontalLayout_26.setObjectName(u"horizontalLayout_26")
        self.horizontalLayout_26.setContentsMargins(0, 0, 0, 0)
        self.src_vsmode = QPushButton(self.vsBox)
        self.src_vsmode.setObjectName(u"src_vsmode")
        sizePolicy2.setHeightForWidth(self.src_vsmode.sizePolicy().hasHeightForWidth())
        self.src_vsmode.setSizePolicy(sizePolicy2)
        self.src_vsmode.setMinimumSize(QSize(180, 0))
        self.src_vsmode.setMaximumSize(QSize(180, 16777215))
        self.src_vsmode.setStyleSheet(u"")
        self.src_vsmode.setIconSize(QSize(30, 30))

        self.horizontalLayout_26.addWidget(self.src_vsmode)


        self.verticalLayout_3.addWidget(self.vsBox)

        self.verticalSpacer_4 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_3.addItem(self.verticalSpacer_4)

        self.manageBox = QFrame(self.leftbox_bottom)
        self.manageBox.setObjectName(u"manageBox")
        self.manageBox.setMinimumSize(QSize(180, 0))
        self.manageBox.setMaximumSize(QSize(180, 16777215))
        self.manageBox.setFrameShape(QFrame.StyledPanel)
        self.manageBox.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_19 = QHBoxLayout(self.manageBox)
        self.horizontalLayout_19.setSpacing(0)
        self.horizontalLayout_19.setObjectName(u"horizontalLayout_19")
        self.horizontalLayout_19.setContentsMargins(0, 0, 0, 0)
        self.src_setting = QPushButton(self.manageBox)
        self.src_setting.setObjectName(u"src_setting")
        sizePolicy2.setHeightForWidth(self.src_setting.sizePolicy().hasHeightForWidth())
        self.src_setting.setSizePolicy(sizePolicy2)
        self.src_setting.setMinimumSize(QSize(180, 0))
        self.src_setting.setMaximumSize(QSize(180, 16777215))

        self.horizontalLayout_19.addWidget(self.src_setting)


        self.verticalLayout_3.addWidget(self.manageBox)

        self.verticalLayout_3.setStretch(0, 1)
        self.verticalLayout_3.setStretch(1, 4)
        self.verticalLayout_3.setStretch(2, 1)
        self.verticalLayout_3.setStretch(3, 4)
        self.verticalLayout_3.setStretch(4, 1)
        self.verticalLayout_3.setStretch(5, 4)
        self.verticalLayout_3.setStretch(6, 1)
        self.verticalLayout_3.setStretch(7, 4)
        self.verticalLayout_3.setStretch(8, 1)
        self.verticalLayout_3.setStretch(9, 4)
        self.verticalLayout_3.setStretch(10, 1)
        self.verticalLayout_3.setStretch(11, 4)
        self.verticalLayout_3.setStretch(12, 1)
        self.verticalLayout_3.setStretch(13, 4)
        self.verticalLayout_3.setStretch(14, 1)
        self.verticalLayout_3.setStretch(15, 4)
        self.verticalLayout_3.setStretch(16, 1)
        self.verticalLayout_3.setStretch(17, 4)

        self.verticalLayout_2.addWidget(self.leftbox_bottom)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_2.addItem(self.verticalSpacer_2)

        self.verticalLayout_2.setStretch(0, 10)
        self.verticalLayout_2.setStretch(1, 80)
        self.verticalLayout_2.setStretch(2, 10)

        self.mainBox.addWidget(self.leftBox)

        self.rightBox = QFrame(self.mainbox)
        self.rightBox.setObjectName(u"rightBox")
        sizePolicy.setHeightForWidth(self.rightBox.sizePolicy().hasHeightForWidth())
        self.rightBox.setSizePolicy(sizePolicy)
        self.rightBox.setMinimumSize(QSize(0, 0))
        self.rightBox.setStyleSheet(u"QFrame#rightBox{\n"
"	margin-top: -1px;\n"
"	margin-right: -1px;\n"
"	margin-bottom: -1px;\n"
"    background-color:  #ffffff;\n"
"    border: 1px solid rgba(0, 0, 0, 15%);\n"
"	border-radius: 30%;\n"
"	background-color: rgb(245, 249, 254);\n"
"}\n"
"QFrame#rightbox_top{\n"
"	border:2px solid rgb(255, 255, 255);\n"
"	border-radius:15%;\n"
"	background-color: rgb(238, 242, 255);\n"
"}\n"
"\n"
"QFrame#main_leftbox{\n"
"	border:2px solid rgb(255, 255, 255);\n"
"	border-radius:15%;\n"
"	background-color: rgb(238, 242, 255);\n"
"}\n"
"QFrame#main_rightbox{\n"
"	border:2px solid rgb(255, 255, 255);\n"
"	border-radius:15%;\n"
"	background-color: rgb(238, 242, 255);\n"
"}\n"
"/*QFrame#rightbox_bottom{\n"
"	border:2px solid rgb(255, 255, 255);\n"
"	border-radius:10%;\n"
"}*/")
        self.rightBox.setFrameShape(QFrame.StyledPanel)
        self.rightBox.setFrameShadow(QFrame.Raised)
        self.verticalLayout_5 = QVBoxLayout(self.rightBox)
        self.verticalLayout_5.setSpacing(3)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_5.setContentsMargins(8, 8, 8, 8)
        self.rightbox_top = QFrame(self.rightBox)
        self.rightbox_top.setObjectName(u"rightbox_top")
        self.rightbox_top.setStyleSheet(u"")
        self.rightbox_top.setFrameShape(QFrame.StyledPanel)
        self.rightbox_top.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_14 = QHBoxLayout(self.rightbox_top)
        self.horizontalLayout_14.setObjectName(u"horizontalLayout_14")
        self.horizontalLayout_14.setContentsMargins(0, 0, 0, 0)
        self.frame_6 = QFrame(self.rightbox_top)
        self.frame_6.setObjectName(u"frame_6")
        self.frame_6.setFrameShape(QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QFrame.Raised)
        self.frame_6.setLineWidth(1)
        self.horizontalLayout_22 = QHBoxLayout(self.frame_6)
        self.horizontalLayout_22.setSpacing(6)
        self.horizontalLayout_22.setObjectName(u"horizontalLayout_22")
        self.horizontalLayout_22.setContentsMargins(9, 9, 9, 9)
        self.classesLabel = QLabel(self.frame_6)
        self.classesLabel.setObjectName(u"classesLabel")
        self.classesLabel.setMaximumSize(QSize(80, 80))
        self.classesLabel.setPixmap(QPixmap(u":/statusbar/images/newsize/classesBar.png"))
        self.classesLabel.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_22.addWidget(self.classesLabel)

        self.Class_QF = QFrame(self.frame_6)
        self.Class_QF.setObjectName(u"Class_QF")
        self.Class_QF.setMinimumSize(QSize(170, 80))
        self.Class_QF.setMaximumSize(QSize(170, 80))
        self.Class_QF.setToolTipDuration(0)
        self.Class_QF.setStyleSheet(u"QFrame#Class_QF{\n"
"color: rgb(255, 255, 255);\n"
"border-radius: 15px;\n"
"background-color:qradialgradient(cx:0, cy:0, radius:1, fx:0.1, fy:0.1, stop:0  #97D9E1,  stop:1   #8EC5FC);\n"
"border: 1px outset #97D9E1;\n"
"}\n"
"")
        self.Class_QF.setFrameShape(QFrame.StyledPanel)
        self.Class_QF.setFrameShadow(QFrame.Raised)
        self.verticalLayout_7 = QVBoxLayout(self.Class_QF)
        self.verticalLayout_7.setSpacing(0)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.Class_top = QFrame(self.Class_QF)
        self.Class_top.setObjectName(u"Class_top")
        self.Class_top.setStyleSheet(u"border:none")
        self.Class_top.setFrameShape(QFrame.StyledPanel)
        self.Class_top.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_11 = QHBoxLayout(self.Class_top)
        self.horizontalLayout_11.setSpacing(0)
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.horizontalLayout_11.setContentsMargins(0, 3, 0, 3)
        self.label_5 = QLabel(self.Class_top)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setMaximumSize(QSize(16777215, 30))
        font = QFont()
        font.setFamilies([u"Segoe UI"])
        font.setPointSize(16)
        font.setBold(True)
        font.setItalic(True)
        self.label_5.setFont(font)
        self.label_5.setStyleSheet(u"color: rgba(255, 255,255, 210);\n"
"text-align:center;\n"
"font: 700 italic 16pt \"Segoe UI\";")
        self.label_5.setAlignment(Qt.AlignCenter)
        self.label_5.setIndent(0)

        self.horizontalLayout_11.addWidget(self.label_5)


        self.verticalLayout_7.addWidget(self.Class_top)

        self.line_2 = QFrame(self.Class_QF)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setMaximumSize(QSize(16777215, 1))
        self.line_2.setStyleSheet(u"background-color: rgba(255, 255, 255, 89);")
        self.line_2.setFrameShape(QFrame.Shape.HLine)
        self.line_2.setFrameShadow(QFrame.Shadow.Sunken)

        self.verticalLayout_7.addWidget(self.line_2)

        self.Class_bottom = QFrame(self.Class_QF)
        self.Class_bottom.setObjectName(u"Class_bottom")
        self.Class_bottom.setStyleSheet(u"border:none")
        self.Class_bottom.setFrameShape(QFrame.StyledPanel)
        self.Class_bottom.setFrameShadow(QFrame.Raised)
        self.verticalLayout_8 = QVBoxLayout(self.Class_bottom)
        self.verticalLayout_8.setSpacing(0)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.verticalLayout_8.setContentsMargins(0, 6, 0, 6)
        self.Class_num = QLabel(self.Class_bottom)
        self.Class_num.setObjectName(u"Class_num")
        self.Class_num.setMinimumSize(QSize(0, 30))
        self.Class_num.setMaximumSize(QSize(16777215, 30))
        font1 = QFont()
        font1.setFamilies([u"Microsoft YaHei UI"])
        font1.setPointSize(17)
        font1.setBold(False)
        font1.setItalic(False)
        font1.setUnderline(False)
        self.Class_num.setFont(font1)
        self.Class_num.setStyleSheet(u"color: rgb(255, 255, 255);\n"
"font: 17pt \"Microsoft YaHei UI\";")
        self.Class_num.setAlignment(Qt.AlignCenter)

        self.verticalLayout_8.addWidget(self.Class_num, 0, Qt.AlignTop)


        self.verticalLayout_7.addWidget(self.Class_bottom)

        self.verticalLayout_7.setStretch(1, 2)
        self.verticalLayout_7.setStretch(2, 1)

        self.horizontalLayout_22.addWidget(self.Class_QF)


        self.horizontalLayout_14.addWidget(self.frame_6)

        self.frame_7 = QFrame(self.rightbox_top)
        self.frame_7.setObjectName(u"frame_7")
        self.frame_7.setFrameShape(QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_23 = QHBoxLayout(self.frame_7)
        self.horizontalLayout_23.setSpacing(6)
        self.horizontalLayout_23.setObjectName(u"horizontalLayout_23")
        self.horizontalLayout_23.setContentsMargins(0, 9, 9, 9)
        self.targetLabel = QLabel(self.frame_7)
        self.targetLabel.setObjectName(u"targetLabel")
        self.targetLabel.setMaximumSize(QSize(80, 80))
        self.targetLabel.setPixmap(QPixmap(u":/statusbar/images/newsize/targetBar.png"))
        self.targetLabel.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_23.addWidget(self.targetLabel)

        self.Target_QF = QFrame(self.frame_7)
        self.Target_QF.setObjectName(u"Target_QF")
        self.Target_QF.setMinimumSize(QSize(170, 80))
        self.Target_QF.setMaximumSize(QSize(170, 80))
        self.Target_QF.setToolTipDuration(0)
        self.Target_QF.setStyleSheet(u"QFrame#Target_QF{\n"
"color: rgb(255, 255, 255);\n"
"border-radius: 15px;\n"
"background-color: qradialgradient(cx:0, cy:0, radius:1, fx:0.1, fy:0.1, stop:0 #E0C3FC,  stop:1  #9599E2);\n"
"border: 1px outset #9599E2;\n"
"}\n"
"")
        self.Target_QF.setFrameShape(QFrame.StyledPanel)
        self.Target_QF.setFrameShadow(QFrame.Raised)
        self.verticalLayout_9 = QVBoxLayout(self.Target_QF)
        self.verticalLayout_9.setSpacing(0)
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.verticalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.Target_top = QFrame(self.Target_QF)
        self.Target_top.setObjectName(u"Target_top")
        self.Target_top.setStyleSheet(u"border:none")
        self.Target_top.setFrameShape(QFrame.StyledPanel)
        self.Target_top.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_12 = QHBoxLayout(self.Target_top)
        self.horizontalLayout_12.setSpacing(0)
        self.horizontalLayout_12.setObjectName(u"horizontalLayout_12")
        self.horizontalLayout_12.setContentsMargins(0, 3, 0, 3)
        self.label_6 = QLabel(self.Target_top)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setMaximumSize(QSize(16777215, 30))
        self.label_6.setFont(font)
        self.label_6.setStyleSheet(u"color: rgba(255, 255,255, 210);\n"
"text-align:center;\n"
"font: 700 italic 16pt \"Segoe UI\";")
        self.label_6.setAlignment(Qt.AlignCenter)
        self.label_6.setIndent(0)

        self.horizontalLayout_12.addWidget(self.label_6)


        self.verticalLayout_9.addWidget(self.Target_top)

        self.line_3 = QFrame(self.Target_QF)
        self.line_3.setObjectName(u"line_3")
        self.line_3.setMaximumSize(QSize(16777215, 1))
        self.line_3.setStyleSheet(u"background-color: rgba(255, 255, 255, 89);")
        self.line_3.setFrameShape(QFrame.Shape.HLine)
        self.line_3.setFrameShadow(QFrame.Shadow.Sunken)

        self.verticalLayout_9.addWidget(self.line_3)

        self.Target_bottom = QFrame(self.Target_QF)
        self.Target_bottom.setObjectName(u"Target_bottom")
        self.Target_bottom.setStyleSheet(u"border:none")
        self.Target_bottom.setFrameShape(QFrame.StyledPanel)
        self.Target_bottom.setFrameShadow(QFrame.Raised)
        self.verticalLayout_10 = QVBoxLayout(self.Target_bottom)
        self.verticalLayout_10.setSpacing(0)
        self.verticalLayout_10.setObjectName(u"verticalLayout_10")
        self.verticalLayout_10.setContentsMargins(0, 6, 0, 6)
        self.Target_num = QLabel(self.Target_bottom)
        self.Target_num.setObjectName(u"Target_num")
        self.Target_num.setMinimumSize(QSize(0, 30))
        self.Target_num.setMaximumSize(QSize(16777215, 30))
        self.Target_num.setFont(font1)
        self.Target_num.setStyleSheet(u"color: rgb(255, 255, 255);\n"
"font: 17pt \"Microsoft YaHei UI\";")
        self.Target_num.setAlignment(Qt.AlignCenter)

        self.verticalLayout_10.addWidget(self.Target_num, 0, Qt.AlignTop)


        self.verticalLayout_9.addWidget(self.Target_bottom)

        self.verticalLayout_9.setStretch(1, 2)
        self.verticalLayout_9.setStretch(2, 1)

        self.horizontalLayout_23.addWidget(self.Target_QF)


        self.horizontalLayout_14.addWidget(self.frame_7)

        self.frame_8 = QFrame(self.rightbox_top)
        self.frame_8.setObjectName(u"frame_8")
        self.frame_8.setFrameShape(QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_24 = QHBoxLayout(self.frame_8)
        self.horizontalLayout_24.setSpacing(6)
        self.horizontalLayout_24.setObjectName(u"horizontalLayout_24")
        self.horizontalLayout_24.setContentsMargins(0, 9, 9, 9)
        self.fpsLabel = QLabel(self.frame_8)
        self.fpsLabel.setObjectName(u"fpsLabel")
        self.fpsLabel.setMaximumSize(QSize(80, 80))
        self.fpsLabel.setPixmap(QPixmap(u":/statusbar/images/newsize/fpsBar.png"))
        self.fpsLabel.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_24.addWidget(self.fpsLabel)

        self.Fps_QF = QFrame(self.frame_8)
        self.Fps_QF.setObjectName(u"Fps_QF")
        self.Fps_QF.setMinimumSize(QSize(170, 80))
        self.Fps_QF.setMaximumSize(QSize(170, 80))
        self.Fps_QF.setToolTipDuration(0)
        self.Fps_QF.setStyleSheet(u"QFrame#Fps_QF{\n"
"color: rgb(255, 255, 255);\n"
"border-radius: 15px;\n"
"background-color: qradialgradient(cx:0, cy:0, radius:1, fx:0.1, fy:0.1, stop:0 rgb(243, 175, 189),  stop:1 rgb(155, 118, 218));\n"
"border: 1px outset rgb(153, 117, 219)\n"
"}\n"
"")
        self.Fps_QF.setFrameShape(QFrame.StyledPanel)
        self.Fps_QF.setFrameShadow(QFrame.Raised)
        self.verticalLayout_11 = QVBoxLayout(self.Fps_QF)
        self.verticalLayout_11.setSpacing(0)
        self.verticalLayout_11.setObjectName(u"verticalLayout_11")
        self.verticalLayout_11.setContentsMargins(0, 0, 0, 0)
        self.Fps_top = QFrame(self.Fps_QF)
        self.Fps_top.setObjectName(u"Fps_top")
        self.Fps_top.setStyleSheet(u"border:none")
        self.Fps_top.setFrameShape(QFrame.StyledPanel)
        self.Fps_top.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_10 = QHBoxLayout(self.Fps_top)
        self.horizontalLayout_10.setSpacing(0)
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.horizontalLayout_10.setContentsMargins(0, 3, 7, 3)
        self.label_7 = QLabel(self.Fps_top)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setMaximumSize(QSize(16777215, 30))
        self.label_7.setFont(font)
        self.label_7.setStyleSheet(u"color: rgba(255, 255,255, 210);\n"
"text-align:center;\n"
"font: 700 italic 16pt \"Segoe UI\";")
        self.label_7.setMidLineWidth(-1)
        self.label_7.setAlignment(Qt.AlignCenter)
        self.label_7.setWordWrap(False)
        self.label_7.setIndent(0)

        self.horizontalLayout_10.addWidget(self.label_7)


        self.verticalLayout_11.addWidget(self.Fps_top)

        self.line_4 = QFrame(self.Fps_QF)
        self.line_4.setObjectName(u"line_4")
        self.line_4.setMaximumSize(QSize(16777215, 1))
        self.line_4.setStyleSheet(u"background-color: rgba(255, 255, 255, 89);")
        self.line_4.setFrameShape(QFrame.Shape.HLine)
        self.line_4.setFrameShadow(QFrame.Shadow.Sunken)

        self.verticalLayout_11.addWidget(self.line_4)

        self.Fps_bottom = QFrame(self.Fps_QF)
        self.Fps_bottom.setObjectName(u"Fps_bottom")
        self.Fps_bottom.setStyleSheet(u"border:none")
        self.Fps_bottom.setFrameShape(QFrame.StyledPanel)
        self.Fps_bottom.setFrameShadow(QFrame.Raised)
        self.verticalLayout_12 = QVBoxLayout(self.Fps_bottom)
        self.verticalLayout_12.setSpacing(0)
        self.verticalLayout_12.setObjectName(u"verticalLayout_12")
        self.verticalLayout_12.setContentsMargins(0, 6, 0, 6)
        self.fps_label = QLabel(self.Fps_bottom)
        self.fps_label.setObjectName(u"fps_label")
        self.fps_label.setMinimumSize(QSize(0, 30))
        self.fps_label.setMaximumSize(QSize(16777215, 30))
        self.fps_label.setFont(font1)
        self.fps_label.setStyleSheet(u"color: rgb(255, 255, 255);\n"
"font: 17pt \"Microsoft YaHei UI\";")
        self.fps_label.setAlignment(Qt.AlignCenter)

        self.verticalLayout_12.addWidget(self.fps_label, 0, Qt.AlignTop)


        self.verticalLayout_11.addWidget(self.Fps_bottom)

        self.verticalLayout_11.setStretch(1, 2)
        self.verticalLayout_11.setStretch(2, 1)

        self.horizontalLayout_24.addWidget(self.Fps_QF)


        self.horizontalLayout_14.addWidget(self.frame_8)

        self.frame_9 = QFrame(self.rightbox_top)
        self.frame_9.setObjectName(u"frame_9")
        self.frame_9.setFrameShape(QFrame.StyledPanel)
        self.frame_9.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_25 = QHBoxLayout(self.frame_9)
        self.horizontalLayout_25.setSpacing(6)
        self.horizontalLayout_25.setObjectName(u"horizontalLayout_25")
        self.horizontalLayout_25.setContentsMargins(0, 9, 9, 9)
        self.modelLabel = QLabel(self.frame_9)
        self.modelLabel.setObjectName(u"modelLabel")
        self.modelLabel.setMaximumSize(QSize(80, 80))
        self.modelLabel.setPixmap(QPixmap(u":/statusbar/images/newsize/modelBar.png"))
        self.modelLabel.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_25.addWidget(self.modelLabel)

        self.Model_QF = QFrame(self.frame_9)
        self.Model_QF.setObjectName(u"Model_QF")
        self.Model_QF.setMinimumSize(QSize(170, 80))
        self.Model_QF.setMaximumSize(QSize(170, 80))
        self.Model_QF.setToolTipDuration(0)
        self.Model_QF.setStyleSheet(u"QFrame#Model_QF{\n"
"color: rgb(255, 255, 255);\n"
"border-radius: 15px;\n"
"background-color: qradialgradient(cx:0, cy:0, radius:1, fx:0.1, fy:0.1, stop:0 rgb(162, 129, 247),  stop:1 rgb(119, 111, 252));\n"
"border: 1px outset rgb(98, 91, 213);\n"
"}\n"
"")
        self.Model_QF.setFrameShape(QFrame.StyledPanel)
        self.Model_QF.setFrameShadow(QFrame.Raised)
        self.verticalLayout_13 = QVBoxLayout(self.Model_QF)
        self.verticalLayout_13.setSpacing(0)
        self.verticalLayout_13.setObjectName(u"verticalLayout_13")
        self.verticalLayout_13.setContentsMargins(0, 0, 0, 0)
        self.Model_top = QFrame(self.Model_QF)
        self.Model_top.setObjectName(u"Model_top")
        self.Model_top.setStyleSheet(u"border:none")
        self.Model_top.setFrameShape(QFrame.StyledPanel)
        self.Model_top.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_13 = QHBoxLayout(self.Model_top)
        self.horizontalLayout_13.setSpacing(0)
        self.horizontalLayout_13.setObjectName(u"horizontalLayout_13")
        self.horizontalLayout_13.setContentsMargins(0, 3, 7, 3)
        self.label_8 = QLabel(self.Model_top)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setMaximumSize(QSize(16777215, 30))
        self.label_8.setFont(font)
        self.label_8.setStyleSheet(u"color: rgba(255, 255,255, 210);\n"
"text-align:center;\n"
"font: 700 italic 16pt \"Segoe UI\";")
        self.label_8.setMidLineWidth(-1)
        self.label_8.setAlignment(Qt.AlignCenter)
        self.label_8.setWordWrap(False)
        self.label_8.setIndent(0)

        self.horizontalLayout_13.addWidget(self.label_8)


        self.verticalLayout_13.addWidget(self.Model_top)

        self.line_5 = QFrame(self.Model_QF)
        self.line_5.setObjectName(u"line_5")
        self.line_5.setMaximumSize(QSize(16777215, 1))
        self.line_5.setStyleSheet(u"background-color: rgba(255, 255, 255, 89);")
        self.line_5.setFrameShape(QFrame.Shape.HLine)
        self.line_5.setFrameShadow(QFrame.Shadow.Sunken)

        self.verticalLayout_13.addWidget(self.line_5)

        self.Model_bottom = QFrame(self.Model_QF)
        self.Model_bottom.setObjectName(u"Model_bottom")
        self.Model_bottom.setStyleSheet(u"border:none")
        self.Model_bottom.setFrameShape(QFrame.StyledPanel)
        self.Model_bottom.setFrameShadow(QFrame.Raised)
        self.verticalLayout_14 = QVBoxLayout(self.Model_bottom)
        self.verticalLayout_14.setSpacing(0)
        self.verticalLayout_14.setObjectName(u"verticalLayout_14")
        self.verticalLayout_14.setContentsMargins(0, 6, 0, 6)
        self.Model_label = QLabel(self.Model_bottom)
        self.Model_label.setObjectName(u"Model_label")
        self.Model_label.setMinimumSize(QSize(0, 30))
        self.Model_label.setMaximumSize(QSize(16777215, 30))
        font2 = QFont()
        font2.setFamilies([u"Microsoft YaHei UI"])
        font2.setPointSize(15)
        font2.setBold(False)
        font2.setItalic(False)
        font2.setUnderline(False)
        self.Model_label.setFont(font2)
        self.Model_label.setStyleSheet(u"color: rgb(255, 255, 255);\n"
"font: 15pt \"Microsoft YaHei UI\";\n"
"")
        self.Model_label.setAlignment(Qt.AlignCenter)

        self.verticalLayout_14.addWidget(self.Model_label, 0, Qt.AlignTop)


        self.verticalLayout_13.addWidget(self.Model_bottom)

        self.verticalLayout_13.setStretch(1, 2)
        self.verticalLayout_13.setStretch(2, 1)

        self.horizontalLayout_25.addWidget(self.Model_QF)


        self.horizontalLayout_14.addWidget(self.frame_9)


        self.verticalLayout_5.addWidget(self.rightbox_top)

        self.rightbox_main = QFrame(self.rightBox)
        self.rightbox_main.setObjectName(u"rightbox_main")
        self.rightbox_main.setStyleSheet(u"")
        self.rightbox_main.setFrameShape(QFrame.StyledPanel)
        self.rightbox_main.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_9 = QHBoxLayout(self.rightbox_main)
        self.horizontalLayout_9.setSpacing(0)
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.horizontalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.splitter = QSplitter(self.rightbox_main)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Horizontal)
        self.main_leftbox = QLabel(self.splitter)
        self.main_leftbox.setObjectName(u"main_leftbox")
        self.main_leftbox.setMinimumSize(QSize(200, 100))
        self.splitter.addWidget(self.main_leftbox)
        self.main_rightbox = QLabel(self.splitter)
        self.main_rightbox.setObjectName(u"main_rightbox")
        self.main_rightbox.setMinimumSize(QSize(200, 100))
        self.splitter.addWidget(self.main_rightbox)

        self.horizontalLayout_9.addWidget(self.splitter)


        self.verticalLayout_5.addWidget(self.rightbox_main)

        self.rightbox_play = QFrame(self.rightBox)
        self.rightbox_play.setObjectName(u"rightbox_play")
        self.rightbox_play.setFrameShape(QFrame.StyledPanel)
        self.rightbox_play.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_20 = QHBoxLayout(self.rightbox_play)
        self.horizontalLayout_20.setObjectName(u"horizontalLayout_20")
        self.horizontalLayout_20.setContentsMargins(5, 5, 5, 5)
        self.run_button = QPushButton(self.rightbox_play)
        self.run_button.setObjectName(u"run_button")
        self.run_button.setMinimumSize(QSize(0, 30))
        self.run_button.setMaximumSize(QSize(16777215, 30))
        self.run_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.run_button.setMouseTracking(True)
        self.run_button.setStyleSheet(u"QPushButton{\n"
"background-repeat: no-repeat;\n"
"background-position: center;\n"
"border: none;\n"
"}\n"
"QPushButton:hover{\n"
"\n"
"}")
        icon = QIcon()
        icon.addFile(u":/rightbox/images/newsize/play.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.run_button.setIcon(icon)
        self.run_button.setIconSize(QSize(20, 20))
        self.run_button.setCheckable(True)
        self.run_button.setChecked(False)

        self.horizontalLayout_20.addWidget(self.run_button)

        self.progress_bar = QProgressBar(self.rightbox_play)
        self.progress_bar.setObjectName(u"progress_bar")
        self.progress_bar.setMinimumSize(QSize(0, 20))
        self.progress_bar.setMaximumSize(QSize(16777215, 20))
        self.progress_bar.setStyleSheet(u"QProgressBar{ \n"
"font: 700 10pt \"Nirmala UI\";\n"
"color: #8EC5FC; \n"
"text-align:center; \n"
"border:3px solid rgb(255, 255, 255);\n"
"border-radius: 10px; \n"
"background-color: rgba(215, 215, 215,100);\n"
"} \n"
"\n"
"QProgressBar:chunk{ \n"
"border-radius:0px; \n"
"background:  lightgrey;\n"
"border-radius: 7px;\n"
"}")
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)

        self.horizontalLayout_20.addWidget(self.progress_bar)

        self.stop_button = QPushButton(self.rightbox_play)
        self.stop_button.setObjectName(u"stop_button")
        self.stop_button.setMinimumSize(QSize(0, 30))
        self.stop_button.setMaximumSize(QSize(16777215, 30))
        self.stop_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.stop_button.setStyleSheet(u"QPushButton{\n"
"background-image: url(:/rightbox/images/newsize/stop.png);\n"
"background-repeat: no-repeat;\n"
"background-position: center;\n"
"border: none;\n"
"}\n"
"\n"
"QPushButton:hover{\n"
"\n"
"}")

        self.horizontalLayout_20.addWidget(self.stop_button)


        self.verticalLayout_5.addWidget(self.rightbox_play)

        self.rightbox_bottom = QFrame(self.rightBox)
        self.rightbox_bottom.setObjectName(u"rightbox_bottom")
        self.rightbox_bottom.setStyleSheet(u"")
        self.rightbox_bottom.setFrameShape(QFrame.StyledPanel)
        self.rightbox_bottom.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_15 = QHBoxLayout(self.rightbox_bottom)
        self.horizontalLayout_15.setSpacing(0)
        self.horizontalLayout_15.setObjectName(u"horizontalLayout_15")
        self.horizontalLayout_15.setContentsMargins(9, 0, 0, 0)
        self.message_bar = QLabel(self.rightbox_bottom)
        self.message_bar.setObjectName(u"message_bar")
        self.message_bar.setStyleSheet(u"font: 700 11pt \"Segoe UI\";\n"
"color: rgba(0, 0, 0, 140);")

        self.horizontalLayout_15.addWidget(self.message_bar)


        self.verticalLayout_5.addWidget(self.rightbox_bottom)

        self.verticalLayout_5.setStretch(0, 10)
        self.verticalLayout_5.setStretch(1, 86)
        self.verticalLayout_5.setStretch(3, 2)

        self.mainBox.addWidget(self.rightBox)

        self.settingBox = QFrame(self.mainbox)
        self.settingBox.setObjectName(u"settingBox")
        self.settingBox.setMinimumSize(QSize(0, 0))
        self.settingBox.setMaximumSize(QSize(0, 16777215))
        self.settingBox.setStyleSheet(u"QFrame#settingBox{\n"
"	margin-top: -1px;\n"
"	margin-right: -1px;\n"
"	margin-bottom: -1px;\n"
"    background-color:  #ffffff;\n"
"    border: 1px solid rgba(0, 0, 0, 15%);\n"
"	border-radius: 30%;\n"
"	border-top-left-radius: 0;\n"
"	border-bottom-left-radius: 0;\n"
"	margin-left: 1px;\n"
"	border-left:none;\n"
"}")
        self.settingBox.setFrameShape(QFrame.StyledPanel)
        self.settingBox.setFrameShadow(QFrame.Raised)
        self.verticalLayout_6 = QVBoxLayout(self.settingBox)
        self.verticalLayout_6.setSpacing(0)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.setting_page = QFrame(self.settingBox)
        self.setting_page.setObjectName(u"setting_page")
        self.setting_page.setMinimumSize(QSize(0, 0))
        self.setting_page.setMaximumSize(QSize(300, 16777215))
        self.setting_page.setStyleSheet(u"QFrame#setting_page{\n"
"	background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #8EC5FC, stop:1 #E0C3FC);\n"
"	border-top-left-radius:0px;\n"
"	border-top-right-radius:30%;\n"
"	border-bottom-right-radius:30%;\n"
"	border-bottom-left-radius:0px;\n"
"	border:none;\n"
"}")
        self.setting_page.setFrameShape(QFrame.StyledPanel)
        self.setting_page.setFrameShadow(QFrame.Raised)
        self.verticalLayout_22 = QVBoxLayout(self.setting_page)
        self.verticalLayout_22.setSpacing(15)
        self.verticalLayout_22.setObjectName(u"verticalLayout_22")
        self.verticalLayout_22.setContentsMargins(15, 15, 15, 15)
        self.label_2 = QLabel(self.setting_page)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setStyleSheet(u"padding-left: 0px;\n"
"padding-bottom: 2px;\n"
"color: white;\n"
"font: 700 italic 16pt \"Segoe UI\";")
        self.label_2.setAlignment(Qt.AlignCenter)

        self.verticalLayout_22.addWidget(self.label_2)

        self.Model_QF_2 = QWidget(self.setting_page)
        self.Model_QF_2.setObjectName(u"Model_QF_2")
        self.Model_QF_2.setMinimumSize(QSize(260, 90))
        self.Model_QF_2.setMaximumSize(QSize(260, 90))
        self.Model_QF_2.setStyleSheet(u"QWidget#Model_QF_2{\n"
"border:2px solid rgba(255, 255, 255, 70);\n"
"border-radius:15px;\n"
"}")
        self.verticalLayout_21 = QVBoxLayout(self.Model_QF_2)
        self.verticalLayout_21.setObjectName(u"verticalLayout_21")
        self.verticalLayout_21.setContentsMargins(9, 9, 9, 9)
        self.ToggleBotton_6 = QPushButton(self.Model_QF_2)
        self.ToggleBotton_6.setObjectName(u"ToggleBotton_6")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.ToggleBotton_6.sizePolicy().hasHeightForWidth())
        self.ToggleBotton_6.setSizePolicy(sizePolicy3)
        self.ToggleBotton_6.setMinimumSize(QSize(0, 30))
        self.ToggleBotton_6.setMaximumSize(QSize(16777215, 30))
        font3 = QFont()
        font3.setFamilies([u"Nirmala UI"])
        font3.setPointSize(13)
        font3.setBold(True)
        font3.setItalic(False)
        self.ToggleBotton_6.setFont(font3)
        self.ToggleBotton_6.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        self.ToggleBotton_6.setMouseTracking(True)
        self.ToggleBotton_6.setFocusPolicy(Qt.StrongFocus)
        self.ToggleBotton_6.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.ToggleBotton_6.setLayoutDirection(Qt.LeftToRight)
        self.ToggleBotton_6.setAutoFillBackground(False)
        self.ToggleBotton_6.setStyleSheet(u"QPushButton{\n"
"background-image: url(:/setting /images/newsize/model.png);\n"
"background-repeat: no-repeat;\n"
"background-position: left center;\n"
"border: none;\n"
"border-left: 20px solid transparent;\n"
"\n"
"text-align: left;\n"
"padding-left: 40px;\n"
"padding-bottom: 2px;\n"
"color: white;\n"
"font: 700 13pt \"Nirmala UI\";\n"
"}")
        self.ToggleBotton_6.setAutoDefault(False)
        self.ToggleBotton_6.setFlat(False)

        self.verticalLayout_21.addWidget(self.ToggleBotton_6)

        self.model_box = ComboBox(self.Model_QF_2)
        self.model_box.setObjectName(u"model_box")
        self.model_box.setMinimumSize(QSize(240, 22))
        self.model_box.setMaximumSize(QSize(240, 20))
        self.model_box.setStyleSheet(u"ComboBox {\n"
"            background-color: rgba(255,255,255,90);\n"
"			color: rgba(0, 0, 0, 200);\n"
"            border: 1px solid lightgray;\n"
"            border-radius: 10px;\n"
"			padding: 2px;\n"
"			text-align: left;\n"
"			font: 600 9pt \"Segoe UI\";\n"
"			padding-left: 15px;\n"
"}      \n"
"ComboBox:on {\n"
"            border: 1px solid #63acfb;       \n"
" }\n"
"\n"
"ComboBox::drop-down {\n"
"            width: 22px;\n"
"            border-left: 1px solid lightgray;\n"
"            border-top-right-radius: 15px;\n"
"            border-bottom-right-radius: 15px; \n"
"}\n"
"ComboBox::drop-down:on {\n"
"            border-left: 1px solid #63acfb;\n"
" }\n"
"\n"
"ComboBox::down-arrow {\n"
"            width: 16px;\n"
"            height: 16px;\n"
"            image: url(:/setting /images/newsize/box_down.png);\n"
" }\n"
"\n"
"ComboBox::down-arrow:on {\n"
"            image: url(:/setting /images/newsize/box_up.png);\n"
" }\n"
"")
        self.model_box.setProperty(u"minimumContentsLength", 0)

        self.verticalLayout_21.addWidget(self.model_box)


        self.verticalLayout_22.addWidget(self.Model_QF_2)

        self.IOU_QF = QFrame(self.setting_page)
        self.IOU_QF.setObjectName(u"IOU_QF")
        self.IOU_QF.setMinimumSize(QSize(260, 90))
        self.IOU_QF.setMaximumSize(QSize(260, 90))
        self.IOU_QF.setStyleSheet(u"QFrame#IOU_QF{\n"
"border:2px solid rgba(255, 255, 255, 70);\n"
"border-radius:15px;\n"
"}")
        self.verticalLayout_15 = QVBoxLayout(self.IOU_QF)
        self.verticalLayout_15.setObjectName(u"verticalLayout_15")
        self.ToggleBotton_2 = QPushButton(self.IOU_QF)
        self.ToggleBotton_2.setObjectName(u"ToggleBotton_2")
        sizePolicy3.setHeightForWidth(self.ToggleBotton_2.sizePolicy().hasHeightForWidth())
        self.ToggleBotton_2.setSizePolicy(sizePolicy3)
        self.ToggleBotton_2.setMinimumSize(QSize(0, 30))
        self.ToggleBotton_2.setMaximumSize(QSize(16777215, 30))
        self.ToggleBotton_2.setFont(font3)
        self.ToggleBotton_2.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        self.ToggleBotton_2.setMouseTracking(True)
        self.ToggleBotton_2.setFocusPolicy(Qt.StrongFocus)
        self.ToggleBotton_2.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.ToggleBotton_2.setLayoutDirection(Qt.LeftToRight)
        self.ToggleBotton_2.setAutoFillBackground(False)
        self.ToggleBotton_2.setStyleSheet(u"QPushButton{\n"
"background-image:url(:/setting /images/newsize/IOU.png);\n"
"background-repeat: no-repeat;\n"
"background-position: left center;\n"
"border: none;\n"
"border-left: 20px solid transparent;\n"
"\n"
"text-align: left;\n"
"padding-left: 40px;\n"
"padding-bottom: 4px;\n"
"color: white;\n"
"font: 700 13pt \"Nirmala UI\";\n"
"}")
        self.ToggleBotton_2.setAutoDefault(False)
        self.ToggleBotton_2.setFlat(False)

        self.verticalLayout_15.addWidget(self.ToggleBotton_2)

        self.frame_3 = QFrame(self.IOU_QF)
        self.frame_3.setObjectName(u"frame_3")
        sizePolicy.setHeightForWidth(self.frame_3.sizePolicy().hasHeightForWidth())
        self.frame_3.setSizePolicy(sizePolicy)
        self.frame_3.setMinimumSize(QSize(0, 20))
        self.frame_3.setMaximumSize(QSize(16777215, 20))
        self.horizontalLayout_16 = QHBoxLayout(self.frame_3)
        self.horizontalLayout_16.setSpacing(10)
        self.horizontalLayout_16.setObjectName(u"horizontalLayout_16")
        self.horizontalLayout_16.setContentsMargins(8, 0, 10, 0)
        self.iou_spinbox = QDoubleSpinBox(self.frame_3)
        self.iou_spinbox.setObjectName(u"iou_spinbox")
        self.iou_spinbox.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.iou_spinbox.setStyleSheet(u"QDoubleSpinBox {\n"
"border: 0px solid lightgray;\n"
"border-radius: 2px;\n"
"background-color: rgba(255,255,255,90);\n"
"font: 600 9pt \"Segoe UI\";\n"
"}\n"
"        \n"
"QDoubleSpinBox::up-button {\n"
"width: 10px;\n"
"height: 9px;\n"
"margin: 0px 3px 0px 0px;\n"
"border-image: url(:/setting /images/newsize/box_up.png);\n"
"}\n"
"QDoubleSpinBox::up-button:pressed {\n"
"margin-top: 1px;\n"
"}\n"
"            \n"
"QDoubleSpinBox::down-button {\n"
"width: 10px;\n"
"height: 9px;\n"
"margin: 0px 3px 0px 0px;\n"
"border-image:url(:/setting /images/newsize/box_down.png);\n"
"}\n"
"QDoubleSpinBox::down-button:pressed {\n"
"margin-bottom: 1px;\n"
"}")
        self.iou_spinbox.setMinimum(0.010000000000000)
        self.iou_spinbox.setMaximum(1.000000000000000)
        self.iou_spinbox.setSingleStep(0.050000000000000)
        self.iou_spinbox.setValue(0.450000000000000)

        self.horizontalLayout_16.addWidget(self.iou_spinbox)

        self.iou_slider = QSlider(self.frame_3)
        self.iou_slider.setObjectName(u"iou_slider")
        self.iou_slider.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.iou_slider.setStyleSheet(u"QSlider::groove:horizontal {\n"
"border: none;\n"
"height: 10px;\n"
"background-color: rgba(255,255,255,90);\n"
"border-radius: 5px;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"width: 10px;\n"
"margin: -1px 0px -1px 0px;\n"
"border-radius: 3px;\n"
"background-color: white;\n"
"}\n"
"\n"
"QSlider::sub-page:horizontal {\n"
"background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #59969b, stop:1 #04e7fa);\n"
"border-radius: 5px;\n"
"}")
        self.iou_slider.setMinimum(1)
        self.iou_slider.setMaximum(100)
        self.iou_slider.setValue(45)
        self.iou_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_16.addWidget(self.iou_slider)

        self.horizontalLayout_16.setStretch(0, 3)
        self.horizontalLayout_16.setStretch(1, 7)

        self.verticalLayout_15.addWidget(self.frame_3)


        self.verticalLayout_22.addWidget(self.IOU_QF)

        self.Conf_QF = QFrame(self.setting_page)
        self.Conf_QF.setObjectName(u"Conf_QF")
        self.Conf_QF.setMinimumSize(QSize(260, 90))
        self.Conf_QF.setMaximumSize(QSize(260, 90))
        self.Conf_QF.setStyleSheet(u"QFrame#Conf_QF{\n"
"border:2px solid rgba(255, 255, 255, 70);\n"
"border-radius:15px;\n"
"}")
        self.verticalLayout_18 = QVBoxLayout(self.Conf_QF)
        self.verticalLayout_18.setObjectName(u"verticalLayout_18")
        self.ToggleBotton_3 = QPushButton(self.Conf_QF)
        self.ToggleBotton_3.setObjectName(u"ToggleBotton_3")
        sizePolicy3.setHeightForWidth(self.ToggleBotton_3.sizePolicy().hasHeightForWidth())
        self.ToggleBotton_3.setSizePolicy(sizePolicy3)
        self.ToggleBotton_3.setMinimumSize(QSize(0, 30))
        self.ToggleBotton_3.setMaximumSize(QSize(16777215, 30))
        self.ToggleBotton_3.setFont(font3)
        self.ToggleBotton_3.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        self.ToggleBotton_3.setMouseTracking(True)
        self.ToggleBotton_3.setFocusPolicy(Qt.StrongFocus)
        self.ToggleBotton_3.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.ToggleBotton_3.setLayoutDirection(Qt.LeftToRight)
        self.ToggleBotton_3.setAutoFillBackground(False)
        self.ToggleBotton_3.setStyleSheet(u"QPushButton{\n"
"background-image: url(:/setting /images/newsize/conf.png);\n"
"background-repeat: no-repeat;\n"
"background-position: left center;\n"
"border: none;\n"
"border-left: 20px solid transparent;\n"
"\n"
"text-align: left;\n"
"padding-left: 40px;\n"
"padding-bottom: 4px;\n"
"color: white;\n"
"font: 700 13pt \"Nirmala UI\";\n"
"}")
        self.ToggleBotton_3.setAutoDefault(False)
        self.ToggleBotton_3.setFlat(False)

        self.verticalLayout_18.addWidget(self.ToggleBotton_3)

        self.frame = QFrame(self.Conf_QF)
        self.frame.setObjectName(u"frame")
        sizePolicy.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy)
        self.frame.setMinimumSize(QSize(0, 20))
        self.frame.setMaximumSize(QSize(16777215, 20))
        self.horizontalLayout_17 = QHBoxLayout(self.frame)
        self.horizontalLayout_17.setSpacing(10)
        self.horizontalLayout_17.setObjectName(u"horizontalLayout_17")
        self.horizontalLayout_17.setContentsMargins(8, 0, 10, 0)
        self.conf_spinbox = QDoubleSpinBox(self.frame)
        self.conf_spinbox.setObjectName(u"conf_spinbox")
        self.conf_spinbox.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.conf_spinbox.setStyleSheet(u"QDoubleSpinBox {\n"
"border: 0px solid lightgray;\n"
"border-radius: 2px;\n"
"background-color: rgba(255,255,255,90);\n"
"font: 600 9pt \"Segoe UI\";\n"
"}\n"
"        \n"
"QDoubleSpinBox::up-button {\n"
"width: 10px;\n"
"height: 9px;\n"
"margin: 0px 3px 0px 0px;\n"
"border-image: url(:/setting /images/newsize/box_up.png);\n"
"}\n"
"QDoubleSpinBox::up-button:pressed {\n"
"margin-top: 1px;\n"
"}\n"
"            \n"
"QDoubleSpinBox::down-button {\n"
"width: 10px;\n"
"height: 9px;\n"
"margin: 0px 3px 0px 0px;\n"
"border-image: url(:/setting /images/newsize/box_down.png);\n"
"}\n"
"QDoubleSpinBox::down-button:pressed {\n"
"margin-bottom: 1px;\n"
"}")
        self.conf_spinbox.setMinimum(0.010000000000000)
        self.conf_spinbox.setMaximum(1.000000000000000)
        self.conf_spinbox.setSingleStep(0.050000000000000)
        self.conf_spinbox.setValue(0.250000000000000)

        self.horizontalLayout_17.addWidget(self.conf_spinbox)

        self.conf_slider = QSlider(self.frame)
        self.conf_slider.setObjectName(u"conf_slider")
        self.conf_slider.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.conf_slider.setStyleSheet(u"QSlider::groove:horizontal {\n"
"border: none;\n"
"height: 10px;\n"
"background-color: rgba(255,255,255,90);\n"
"border-radius: 5px;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"width: 10px;\n"
"margin: -1px 0px -1px 0px;\n"
"border-radius: 3px;\n"
"background-color: white;\n"
"}\n"
"\n"
"QSlider::sub-page:horizontal {\n"
"background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #59969b, stop:1 #04e7fa);\n"
"border-radius: 5px;\n"
"}")
        self.conf_slider.setMinimum(1)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(25)
        self.conf_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_17.addWidget(self.conf_slider)

        self.horizontalLayout_17.setStretch(0, 3)
        self.horizontalLayout_17.setStretch(1, 7)

        self.verticalLayout_18.addWidget(self.frame)


        self.verticalLayout_22.addWidget(self.Conf_QF)

        self.Delay_QF = QFrame(self.setting_page)
        self.Delay_QF.setObjectName(u"Delay_QF")
        self.Delay_QF.setMinimumSize(QSize(260, 90))
        self.Delay_QF.setMaximumSize(QSize(260, 90))
        self.Delay_QF.setStyleSheet(u"QFrame#Delay_QF{\n"
"border:2px solid rgba(255, 255, 255, 70);\n"
"border-radius:15px;\n"
"}")
        self.verticalLayout_19 = QVBoxLayout(self.Delay_QF)
        self.verticalLayout_19.setObjectName(u"verticalLayout_19")
        self.ToggleBotton_4 = QPushButton(self.Delay_QF)
        self.ToggleBotton_4.setObjectName(u"ToggleBotton_4")
        sizePolicy3.setHeightForWidth(self.ToggleBotton_4.sizePolicy().hasHeightForWidth())
        self.ToggleBotton_4.setSizePolicy(sizePolicy3)
        self.ToggleBotton_4.setMinimumSize(QSize(0, 30))
        self.ToggleBotton_4.setMaximumSize(QSize(16777215, 30))
        self.ToggleBotton_4.setFont(font3)
        self.ToggleBotton_4.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        self.ToggleBotton_4.setMouseTracking(True)
        self.ToggleBotton_4.setFocusPolicy(Qt.StrongFocus)
        self.ToggleBotton_4.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.ToggleBotton_4.setLayoutDirection(Qt.LeftToRight)
        self.ToggleBotton_4.setAutoFillBackground(False)
        self.ToggleBotton_4.setStyleSheet(u"QPushButton{\n"
"background-image:url(:/setting /images/newsize/delay.png);\n"
"background-repeat: no-repeat;\n"
"background-position: left center;\n"
"border: none;\n"
"border-left: 20px solid transparent;\n"
"\n"
"text-align: left;\n"
"padding-left: 40px;\n"
"padding-bottom: 2px;\n"
"color: white;\n"
"font: 700 13pt \"Nirmala UI\";\n"
"}")
        self.ToggleBotton_4.setAutoDefault(False)
        self.ToggleBotton_4.setFlat(False)

        self.verticalLayout_19.addWidget(self.ToggleBotton_4)

        self.frame_2 = QFrame(self.Delay_QF)
        self.frame_2.setObjectName(u"frame_2")
        sizePolicy.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy)
        self.frame_2.setMinimumSize(QSize(0, 20))
        self.frame_2.setMaximumSize(QSize(16777215, 20))
        self.horizontalLayout_18 = QHBoxLayout(self.frame_2)
        self.horizontalLayout_18.setSpacing(10)
        self.horizontalLayout_18.setObjectName(u"horizontalLayout_18")
        self.horizontalLayout_18.setContentsMargins(8, 0, 10, 0)
        self.speed_spinbox = QSpinBox(self.frame_2)
        self.speed_spinbox.setObjectName(u"speed_spinbox")
        self.speed_spinbox.setStyleSheet(u"QSpinBox {\n"
"border: 0px solid lightgray;\n"
"border-radius: 2px;\n"
"background-color: rgba(255,255,255,90);\n"
"font: 600 9pt \"Segoe UI\";\n"
"}\n"
"        \n"
"QSpinBox::up-button {\n"
"width: 10px;\n"
"height: 9px;\n"
"margin: 0px 3px 0px 0px;\n"
"border-image: url(:/setting /images/newsize/box_up.png);\n"
"}\n"
"QSpinBox::up-button:pressed {\n"
"margin-top: 1px;\n"
"}\n"
"            \n"
"QSpinBox::down-button {\n"
"width: 10px;\n"
"height: 9px;\n"
"margin: 0px 3px 0px 0px;\n"
"border-image:url(:/setting /images/newsize/box_down.png);\n"
"}\n"
"QSpinBox::down-button:pressed {\n"
"margin-bottom: 1px;\n"
"}")
        self.speed_spinbox.setMaximum(50)
        self.speed_spinbox.setValue(10)

        self.horizontalLayout_18.addWidget(self.speed_spinbox)

        self.speed_slider = QSlider(self.frame_2)
        self.speed_slider.setObjectName(u"speed_slider")
        self.speed_slider.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.speed_slider.setStyleSheet(u"QSlider::groove:horizontal {\n"
"border: none;\n"
"height: 10px;\n"
"background-color: rgba(255,255,255,90);\n"
"border-radius: 5px;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"width: 10px;\n"
"margin: -1px 0px -1px 0px;\n"
"border-radius: 3px;\n"
"background-color: white;\n"
"}\n"
"\n"
"QSlider::sub-page:horizontal {\n"
"background-color: qradialgradient(cx:0, cy:0, radius:1, fx:0.1, fy:0.1, stop:0 rgb(253, 139, 133),  stop:1 rgb(248, 194, 152));\n"
"border-radius: 5px;\n"
"}")
        self.speed_slider.setMaximum(50)
        self.speed_slider.setValue(10)
        self.speed_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_18.addWidget(self.speed_slider)

        self.horizontalLayout_18.setStretch(0, 3)
        self.horizontalLayout_18.setStretch(1, 7)

        self.verticalLayout_19.addWidget(self.frame_2)


        self.verticalLayout_22.addWidget(self.Delay_QF)

        self.LINE_THICKNESS = QFrame(self.setting_page)
        self.LINE_THICKNESS.setObjectName(u"LINE_THICKNESS")
        self.LINE_THICKNESS.setMinimumSize(QSize(260, 90))
        self.LINE_THICKNESS.setMaximumSize(QSize(260, 90))
        self.LINE_THICKNESS.setStyleSheet(u"QFrame#LINE_THICKNESS{\n"
"border:2px solid rgba(255, 255, 255, 70);\n"
"border-radius:15px;\n"
"}")
        self.LINE_THICKNESS.setFrameShape(QFrame.StyledPanel)
        self.LINE_THICKNESS.setFrameShadow(QFrame.Raised)
        self.verticalLayout_23 = QVBoxLayout(self.LINE_THICKNESS)
        self.verticalLayout_23.setObjectName(u"verticalLayout_23")
        self.ToggleBotton_5 = QPushButton(self.LINE_THICKNESS)
        self.ToggleBotton_5.setObjectName(u"ToggleBotton_5")
        sizePolicy3.setHeightForWidth(self.ToggleBotton_5.sizePolicy().hasHeightForWidth())
        self.ToggleBotton_5.setSizePolicy(sizePolicy3)
        self.ToggleBotton_5.setMinimumSize(QSize(0, 30))
        self.ToggleBotton_5.setMaximumSize(QSize(16777215, 30))
        self.ToggleBotton_5.setFont(font3)
        self.ToggleBotton_5.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        self.ToggleBotton_5.setMouseTracking(True)
        self.ToggleBotton_5.setFocusPolicy(Qt.StrongFocus)
        self.ToggleBotton_5.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.ToggleBotton_5.setLayoutDirection(Qt.LeftToRight)
        self.ToggleBotton_5.setAutoFillBackground(False)
        self.ToggleBotton_5.setStyleSheet(u"QPushButton{\n"
"background-image:url(:/setting /images/newsize/line.png);\n"
"background-repeat: no-repeat;\n"
"background-position: left center;\n"
"border: none;\n"
"border-left: 20px solid transparent;\n"
"text-align: left;\n"
"padding-left: 40px;\n"
"padding-bottom: 4px;\n"
"color: white;\n"
"font: 700 13pt \"Nirmala UI\";\n"
"}")
        self.ToggleBotton_5.setAutoDefault(False)
        self.ToggleBotton_5.setFlat(False)

        self.verticalLayout_23.addWidget(self.ToggleBotton_5)

        self.frame_5 = QFrame(self.LINE_THICKNESS)
        self.frame_5.setObjectName(u"frame_5")
        sizePolicy.setHeightForWidth(self.frame_5.sizePolicy().hasHeightForWidth())
        self.frame_5.setSizePolicy(sizePolicy)
        self.frame_5.setMinimumSize(QSize(0, 20))
        self.frame_5.setMaximumSize(QSize(16777215, 20))
        self.horizontalLayout_21 = QHBoxLayout(self.frame_5)
        self.horizontalLayout_21.setSpacing(10)
        self.horizontalLayout_21.setObjectName(u"horizontalLayout_21")
        self.horizontalLayout_21.setContentsMargins(8, 0, 10, 0)
        self.line_spinbox = QDoubleSpinBox(self.frame_5)
        self.line_spinbox.setObjectName(u"line_spinbox")
        self.line_spinbox.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.line_spinbox.setStyleSheet(u"QDoubleSpinBox {\n"
"border: 0px solid lightgray;\n"
"border-radius: 2px;\n"
"background-color: rgba(255,255,255,90);\n"
"font: 600 9pt \"Segoe UI\";\n"
"}\n"
"        \n"
"QDoubleSpinBox::up-button {\n"
"width: 10px;\n"
"height: 9px;\n"
"margin: 0px 3px 0px 0px;\n"
"border-image: url(:/setting /images/newsize/box_up.png);\n"
"}\n"
"QDoubleSpinBox::up-button:pressed {\n"
"margin-top: 1px;\n"
"}\n"
"            \n"
"QDoubleSpinBox::down-button {\n"
"width: 10px;\n"
"height: 9px;\n"
"margin: 0px 3px 0px 0px;\n"
"border-image:url(:/setting /images/newsize/box_down.png);\n"
"}\n"
"QDoubleSpinBox::down-button:pressed {\n"
"margin-bottom: 1px;\n"
"}")
        self.line_spinbox.setDecimals(0)
        self.line_spinbox.setMinimum(0.000000000000000)
        self.line_spinbox.setMaximum(5.000000000000000)
        self.line_spinbox.setSingleStep(1.000000000000000)
        self.line_spinbox.setValue(3.000000000000000)

        self.horizontalLayout_21.addWidget(self.line_spinbox)

        self.line_slider = QSlider(self.frame_5)
        self.line_slider.setObjectName(u"line_slider")
        self.line_slider.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.line_slider.setStyleSheet(u"QSlider::groove:horizontal {\n"
"border: none;\n"
"height: 10px;\n"
"background-color: rgba(255,255,255,90);\n"
"border-radius: 5px;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"width: 10px;\n"
"margin: -1px 0px -1px 0px;\n"
"border-radius: 3px;\n"
"background-color: white;\n"
"}\n"
"\n"
"QSlider::sub-page:horizontal {\n"
"background-color:  qradialgradient(cx:0, cy:0, radius:1, fx:0.1, fy:0.1, stop:0 rgb(253, 139, 133),  stop:1 rgb(248, 194, 152));\n"
"border-radius: 5px;\n"
"}")
        self.line_slider.setMinimum(0)
        self.line_slider.setMaximum(5)
        self.line_slider.setPageStep(1)
        self.line_slider.setValue(3)
        self.line_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_21.addWidget(self.line_slider)

        self.horizontalLayout_21.setStretch(0, 3)
        self.horizontalLayout_21.setStretch(1, 7)

        self.verticalLayout_23.addWidget(self.frame_5)


        self.verticalLayout_22.addWidget(self.LINE_THICKNESS)

        self.Model_Manage = QFrame(self.setting_page)
        self.Model_Manage.setObjectName(u"Model_Manage")
        self.Model_Manage.setMinimumSize(QSize(190, 150))
        self.Model_Manage.setStyleSheet(u"QWidget#Model_Manage{\n"
"border:2px solid rgba(255, 255, 255, 70);\n"
"border-radius:15px;\n"
"}")
        self.Model_Manage.setFrameShape(QFrame.StyledPanel)
        self.Model_Manage.setFrameShadow(QFrame.Raised)
        self.verticalLayout_16 = QVBoxLayout(self.Model_Manage)
        self.verticalLayout_16.setObjectName(u"verticalLayout_16")
        self.frame_4 = QFrame(self.Model_Manage)
        self.frame_4.setObjectName(u"frame_4")
        self.frame_4.setFrameShape(QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QFrame.Raised)
        self.verticalLayout_17 = QVBoxLayout(self.frame_4)
        self.verticalLayout_17.setSpacing(9)
        self.verticalLayout_17.setObjectName(u"verticalLayout_17")
        self.verticalLayout_17.setContentsMargins(0, 0, 0, 0)
        self.import_button = QPushButton(self.frame_4)
        self.import_button.setObjectName(u"import_button")
        sizePolicy2.setHeightForWidth(self.import_button.sizePolicy().hasHeightForWidth())
        self.import_button.setSizePolicy(sizePolicy2)
        self.import_button.setStyleSheet(u"QPushButton{\n"
"	background-image:url(:/setting /images/newsize/import.png);\n"
"	background-repeat: no-repeat;\n"
"	background-position: left center;\n"
"	border: none;	\n"
"	border-left: 10px solid transparent;\n"
"	text-align: left;\n"
"	padding-left: 40px;\n"
"	padding-bottom: 4px;\n"
"	color: white;\n"
"	font: 700 13pt \"Nirmala UI\";\n"
"}\n"
"QPushButton:hover{\n"
"	background-color: rgba(114, 129, 214, 59);\n"
"}")

        self.verticalLayout_17.addWidget(self.import_button)

        self.save_status_button = QCheckBox(self.frame_4)
        self.save_status_button.setObjectName(u"save_status_button")
        self.save_status_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.save_status_button.setStyleSheet(u"QCheckBox {\n"
"color: white;\n"
"font: 700 13pt \"Nirmala UI\";\n"
"        }\n"
"QCheckBox::indicator {\n"
"           padding-top: 1px;\n"
"            width: 40px;\n"
"            height: 30px;\n"
"            border: none;\n"
" }\n"
"\n"
"QCheckBox::indicator:unchecked {\n"
"            image: url(:/setting /images/newsize/check_no.png);\n"
"        }\n"
"\n"
"QCheckBox::indicator:checked {\n"
"            image:url(:/setting /images/newsize/check_yes.png);\n"
"        }")

        self.verticalLayout_17.addWidget(self.save_status_button)

        self.save_button = QPushButton(self.frame_4)
        self.save_button.setObjectName(u"save_button")
        sizePolicy2.setHeightForWidth(self.save_button.sizePolicy().hasHeightForWidth())
        self.save_button.setSizePolicy(sizePolicy2)
        self.save_button.setStyleSheet(u"QPushButton{\n"
"	background-image:url(:/setting /images/newsize/save.png);\n"
"	background-repeat: no-repeat;\n"
"	background-position: left center;\n"
"	border: none;	\n"
"	border-left: 10px solid transparent;\n"
"	text-align: left;\n"
"	padding-left: 40px;\n"
"	padding-bottom: 4px;\n"
"	color: white;\n"
"	font: 700 13pt \"Nirmala UI\";\n"
"}\n"
"QPushButton:hover{\n"
"	background-color: rgba(114, 129, 214, 59);\n"
"}")
        self.save_button.setCheckable(False)

        self.verticalLayout_17.addWidget(self.save_button)


        self.verticalLayout_16.addWidget(self.frame_4)


        self.verticalLayout_22.addWidget(self.Model_Manage)

        self.verticalSpacer_3 = QSpacerItem(20, 13, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_22.addItem(self.verticalSpacer_3)


        self.verticalLayout_6.addWidget(self.setting_page)


        self.mainBox.addWidget(self.settingBox)


        self.verticalLayout.addWidget(self.mainbox)

        self.verticalLayout.setStretch(0, 5)
        self.verticalLayout.setStretch(1, 95)

        self.verticalLayout_4.addWidget(self.mainBody)

        MainWindow.setCentralWidget(self.mainWindow)

        self.retranslateUi(MainWindow)

        self.ToggleBotton_6.setDefault(False)
        self.ToggleBotton_2.setDefault(False)
        self.ToggleBotton_3.setDefault(False)
        self.ToggleBotton_4.setDefault(False)
        self.ToggleBotton_5.setDefault(False)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.closeButton.setText("")
        self.minimizeButton.setText("")
        self.maximizeButton.setText("")
        self.title.setText(QCoreApplication.translate("MainWindow", u"YOLO SHOW -YOLO Graphical User Interface based on Pyside6", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"YOLO SHOW", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"SwimmingLiu", None))
        self.src_menu.setText(QCoreApplication.translate("MainWindow", u" Menu   ", None))
        self.src_img.setText(QCoreApplication.translate("MainWindow", u"Media", None))
        self.src_webcam.setText(QCoreApplication.translate("MainWindow", u" Webcam", None))
        self.src_folder.setText(QCoreApplication.translate("MainWindow", u" Folder  ", None))
#if QT_CONFIG(shortcut)
        self.src_folder.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+S", None))
#endif // QT_CONFIG(shortcut)
        self.src_camera.setText(QCoreApplication.translate("MainWindow", u"IPcam  ", None))
        self.src_result.setText(QCoreApplication.translate("MainWindow", u"    Result Pic  ", None))
        self.src_table.setText(QCoreApplication.translate("MainWindow", u"    Result Tab  ", None))
        self.src_vsmode.setText(QCoreApplication.translate("MainWindow", u"VS Mode", None))
#if QT_CONFIG(shortcut)
        self.src_vsmode.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+S", None))
#endif // QT_CONFIG(shortcut)
        self.src_setting.setText(QCoreApplication.translate("MainWindow", u"Setting", None))
        self.classesLabel.setText("")
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"Classes", None))
        self.Class_num.setText("")
        self.targetLabel.setText("")
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"Targets", None))
        self.Target_num.setText("")
        self.fpsLabel.setText("")
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"Fps", None))
        self.fps_label.setText("")
        self.modelLabel.setText("")
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"Model", None))
        self.Model_label.setText("")
        self.main_leftbox.setText("")
        self.main_rightbox.setText("")
        self.run_button.setText("")
        self.stop_button.setText("")
        self.message_bar.setText(QCoreApplication.translate("MainWindow", u"Message Bar ... ", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Settings", None))
        self.ToggleBotton_6.setText(QCoreApplication.translate("MainWindow", u"Model", None))
        self.model_box.setProperty(u"placeholderText", "")
        self.ToggleBotton_2.setText(QCoreApplication.translate("MainWindow", u"IOU", None))
        self.ToggleBotton_3.setText(QCoreApplication.translate("MainWindow", u"Confidence", None))
        self.ToggleBotton_4.setText(QCoreApplication.translate("MainWindow", u"Delay(ms)", None))
        self.ToggleBotton_5.setText(QCoreApplication.translate("MainWindow", u"Line Width", None))
        self.import_button.setText(QCoreApplication.translate("MainWindow", u"Import Model", None))
        self.save_status_button.setText(QCoreApplication.translate("MainWindow", u"Save Mode", None))
        self.save_button.setText(QCoreApplication.translate("MainWindow", u"Save Result", None))
    # retranslateUi

