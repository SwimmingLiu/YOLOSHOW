# coding:utf-8
from typing import Union
from PySide6.QtCore import QPoint, Qt, QRect, QRectF, Signal, QSize, QMargins, QPropertyAnimation, QObject, \
    QParallelAnimationGroup, QEasingCurve
from PySide6.QtGui import QPixmap, QPainter, QColor, QPainterPath, QIcon, QImage, QCursor, QFont
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QApplication, QGraphicsDropShadowEffect, QFrame
from qfluentwidgets import  FluentIconBase, FlyoutAnimationType, \
    FlyoutAnimationManager, drawIcon, isDarkTheme, ImageLabel, TransparentToolButton, FluentIcon, FluentStyleSheet, \
    TextWrap
from qfluentwidgets.common.screen import getCurrentScreenGeometry
from qfluentwidgets.components.material import AcrylicWidget


class IconWidget(QWidget):

    def __init__(self, icon, parent=None):
        super().__init__(parent=parent)
        self.setFixedSize(36, 54)
        self.icon = icon

    def paintEvent(self, e):
        if not self.icon:
            return

        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing |
                               QPainter.SmoothPixmapTransform)

        rect = QRectF(8, (self.height() - 20) / 2, 20, 20)
        drawIcon(self.icon, painter, rect)

class FlyoutViewBase(QWidget):
    """ Flyout view base class """

    def __init__(self, parent=None):
        super().__init__(parent=parent)

    def addWidget(self, widget: QWidget, stretch=0, align=Qt.AlignLeft):
        raise NotImplementedError

    def backgroundColor(self):
        return QColor(40, 40, 40) if isDarkTheme() else QColor(248, 248, 248)

    def borderColor(self):
        return QColor(0, 0, 0, 45) if isDarkTheme() else QColor(0, 0, 0, 17)

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)

        painter.setBrush(self.backgroundColor())
        painter.setPen(self.borderColor())

        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.drawRoundedRect(rect, 8, 8)

class FlyoutView(FlyoutViewBase):
    """ Flyout view """

    closed = Signal()

    def __init__(self, title: str, content: str, icon: Union[FluentIconBase, QIcon, str] = None,
                 image: Union[str, QPixmap, QImage] = None, isClosable=False, parent=None):
        super().__init__(parent=parent)
        """
        Parameters
        ----------
        title: str
            the title of teaching tip

        content: str
            the content of teaching tip

        icon: InfoBarIcon | FluentIconBase | QIcon | str
            the icon of teaching tip

        image: str | QPixmap | QImage
            the image of teaching tip

        isClosable: bool
            whether to show the close button

        parent: QWidget
            parent widget
        """
        self.icon = icon
        self.title = title
        self.image = image
        self.content = content
        self.isClosable = isClosable

        self.vBoxLayout = QVBoxLayout(self)
        self.viewLayout = QHBoxLayout()
        self.widgetLayout = QVBoxLayout()

        self.titleLabel = QLabel(title, self)
        self.contentLabel = QLabel(content, self)
        self.iconWidget = IconWidget(icon, self)
        self.imageLabel = ImageLabel(self)
        self.closeButton = TransparentToolButton(FluentIcon.CLOSE, self)
        self.__initWidgets()

    def __initWidgets(self):
        self.imageLabel.setImage(self.image)

        self.closeButton.setFixedSize(32, 32)
        self.closeButton.setIconSize(QSize(12, 12))
        self.closeButton.setVisible(self.isClosable)
        self.titleLabel.setVisible(bool(self.title))
        self.contentLabel.setVisible(bool(self.content))
        self.iconWidget.setHidden(self.icon is None)

        self.closeButton.clicked.connect(self.closed)

        self.titleLabel.setObjectName('titleLabel')
        self.contentLabel.setObjectName('contentLabel')
        FluentStyleSheet.TEACHING_TIP.apply(self)

        self.__initLayout()

    def __initLayout(self):
        self.vBoxLayout.setContentsMargins(1, 1, 1, 1)
        self.widgetLayout.setContentsMargins(0, 8, 0, 8)
        self.viewLayout.setSpacing(4)
        self.widgetLayout.setSpacing(0)
        self.vBoxLayout.setSpacing(0)

        # add icon widget
        if not self.title or not self.content:
            self.iconWidget.setFixedHeight(36)

        self.vBoxLayout.addLayout(self.viewLayout)
        self.viewLayout.addWidget(self.iconWidget, 0, Qt.AlignTop)

        # add text
        self._adjustText()
        self.widgetLayout.addWidget(self.titleLabel)
        self.widgetLayout.addWidget(self.contentLabel)
        self.viewLayout.addLayout(self.widgetLayout)

        # add close button
        self.closeButton.setVisible(self.isClosable)
        self.viewLayout.addWidget(
            self.closeButton, 0, Qt.AlignRight | Qt.AlignTop)

        # adjust content margins
        margins = QMargins(6, 5, 6, 5)
        margins.setLeft(20 if not self.icon else 5)
        margins.setRight(20 if not self.isClosable else 6)
        self.viewLayout.setContentsMargins(margins)

        # add image
        self._adjustImage()
        self._addImageToLayout()

    def addWidget(self, widget: QWidget, stretch=0, align=Qt.AlignLeft):
        """ add widget to view """
        self.widgetLayout.addSpacing(8)
        self.widgetLayout.addWidget(widget, stretch, align)

    def _addImageToLayout(self):
        self.imageLabel.setBorderRadius(8, 8, 0, 0)
        self.imageLabel.setHidden(self.imageLabel.isNull())
        self.vBoxLayout.insertWidget(0, self.imageLabel)

    def _adjustText(self):
        w = min(900, QApplication.screenAt(
            QCursor.pos()).geometry().width() - 200)


        # adjust title
        chars = max(min(w / 10, 120), 30)
        self.titleLabel.setText(TextWrap.wrap(self.title, chars, False)[0])

        # adjust content
        chars = max(min(w / 9, 120), 30)
        self.contentLabel.setText(TextWrap.wrap(self.content, chars, False)[0])

    def _adjustImage(self):
        w = self.vBoxLayout.sizeHint().width() - 2
        self.imageLabel.scaledToWidth(w)

    def showEvent(self, e):
        super().showEvent(e)
        self._adjustImage()
        self.adjustSize()

class Flyout(QWidget):
    """ Flyout """

    closed = Signal()

    def __init__(self, view: FlyoutViewBase, parent=None, isDeleteOnClose=True):
        super().__init__(parent=parent)
        self.view = view
        self.hBoxLayout = QHBoxLayout(self)
        self.aniManager = None  # type: FlyoutAnimationManager
        self.isDeleteOnClose = isDeleteOnClose

        self.hBoxLayout.setContentsMargins(15, 8, 15, 20)
        self.hBoxLayout.addWidget(self.view)
        self.setShadowEffect()

        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint |
                            Qt.NoDropShadowWindowHint)

    def setShadowEffect(self, blurRadius=35, offset=(0, 8)):
        """ add shadow to dialog """
        color = QColor(0, 0, 0, 80 if isDarkTheme() else 30)
        self.shadowEffect = QGraphicsDropShadowEffect(self.view)
        self.shadowEffect.setBlurRadius(blurRadius)
        self.shadowEffect.setOffset(*offset)
        self.shadowEffect.setColor(color)
        self.view.setGraphicsEffect(None)
        self.view.setGraphicsEffect(self.shadowEffect)

    def closeEvent(self, e):
        if self.isDeleteOnClose:
            self.deleteLater()

        super().closeEvent(e)
        self.closed.emit()

    def showEvent(self, e):
        # fixes #780
        self.activateWindow()
        super().showEvent(e)

    def exec(self, pos: QPoint, aniType=FlyoutAnimationType.PULL_UP):
        """ show calendar view """
        self.aniManager = FlyoutAnimationManager.make(aniType, self)
        self.show()
        self.aniManager.exec(pos)

    @classmethod
    def make(cls, view: FlyoutViewBase, target: Union[QWidget, QPoint, QFrame] = None, parent=None,
             aniType=FlyoutAnimationType.PULL_UP, isDeleteOnClose=True):
        """ create and show a flyout

        Parameters
        ----------
        view: FlyoutViewBase
            flyout view

        target: QWidget | QPoint
            the target widget or position to show flyout

        parent: QWidget
            parent window

        aniType: FlyoutAnimationType
            flyout animation type

        isDeleteOnClose: bool
            whether delete flyout automatically when flyout is closed
        """
        w = cls(view, parent, isDeleteOnClose)

        if target is None:
            return w

        # show flyout first so that we can get the correct size
        w.show()

        # move flyout to the top of target
        if isinstance(target, QWidget):
            target = FlyoutAnimationManager.make(aniType, w).position(target)

        w.exec(target, aniType)
        return w

    @classmethod
    def create(cls, title: str, content: str, icon: Union[FluentIconBase, QIcon, str] = None,
               image: Union[str, QPixmap, QImage] = None, isClosable=False, target: Union[QWidget, QPoint] = None,
               parent=None, aniType=FlyoutAnimationType.PULL_UP, isDeleteOnClose=True):
        """ create and show a flyout using the default view

        Parameters
        ----------
        title: str
            the title of teaching tip

        content: str
            the content of teaching tip

        icon: InfoBarIcon | FluentIconBase | QIcon | str
            the icon of teaching tip

        image: str | QPixmap | QImage
            the image of teaching tip

        isClosable: bool
            whether to show the close button

        target: QWidget | QPoint
            the target widget or position to show flyout

        parent: QWidget
            parent window

        aniType: FlyoutAnimationType
            flyout animation type

        isDeleteOnClose: bool
            whether delete flyout automatically when flyout is closed
        """
        view = FlyoutView(title, content, icon, image, isClosable)
        w = cls.make(view, target, parent, aniType, isDeleteOnClose)
        view.closed.connect(w.close)
        return w

    def fadeOut(self):
        self.fadeOutAni = QPropertyAnimation(self, b'windowOpacity', self)
        self.fadeOutAni.finished.connect(self.close)
        self.fadeOutAni.setStartValue(1)
        self.fadeOutAni.setEndValue(0)
        self.fadeOutAni.setDuration(120)
        self.fadeOutAni.start()


class AcrylicFlyoutViewBase(AcrylicWidget, FlyoutViewBase):
    """ Acrylic flyout view base """

    def acrylicClipPath(self):
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect().adjusted(1, 1, -1, -1)), 8, 8)
        return path

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)
        self._drawAcrylic(painter)

        # draw border
        painter.setBrush(Qt.NoBrush)
        painter.setPen(self.borderColor())
        rect = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        painter.drawRoundedRect(rect, 8, 8)


class AcrylicFlyoutView(AcrylicWidget, FlyoutView):
    """ Acrylic flyout view """

    def acrylicClipPath(self):
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect().adjusted(1, 1, -1, -1)), 8, 8)
        return path

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)
        self._drawAcrylic(painter)

        # draw border
        painter.setBrush(Qt.NoBrush)
        painter.setPen(self.borderColor())
        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.drawRoundedRect(rect, 8, 8)


class AcrylicFlyout(Flyout):
    """ Acrylic flyout """

    @classmethod
    def create(cls, title: str, content: str, icon: Union[FluentIconBase, QIcon, str] = None,
               image: Union[str, QPixmap, QImage] = None, isClosable=False, target: Union[QWidget, QPoint] = None,
               parent=None, aniType=FlyoutAnimationType.PULL_UP, isDeleteOnClose=True):
        """ create and show a flyout using the default view

        Parameters
        ----------
        title: str
            the title of teaching tip

        content: str
            the content of teaching tip

        icon: InfoBarIcon | FluentIconBase | QIcon | str
            the icon of teaching tip

        image: str | QPixmap | QImage
            the image of teaching tip

        isClosable: bool
            whether to show the close button

        target: QWidget | QPoint
            the target widget or position to show flyout

        parent: QWidget
            parent window

        aniType: FlyoutAnimationType
            flyout animation type

        isDeleteOnClose: bool
            whether delete flyout automatically when flyout is closed
        """
        view = AcrylicFlyoutView(title, content, icon, image, isClosable)
        w = cls.make(view, target, parent, aniType, isDeleteOnClose)
        view.closed.connect(w.close)
        return w

    def exec(self, pos: QPoint, aniType=FlyoutAnimationType.PULL_UP):
        """ show calendar view """
        self.aniManager = FlyoutAnimationManager.make(aniType, self)

        if isinstance(self.view, AcrylicWidget):
            pos = self.aniManager._adjustPosition(pos)
            self.view.acrylicBrush.grabImage(QRect(pos, self.layout().sizeHint()))

        self.show()
        self.aniManager.exec(pos)


