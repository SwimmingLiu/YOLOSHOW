from PySide6.QtCore import QUrl
from PySide6.QtGui import QFont
from qfluentwidgets import MessageBoxBase, SubtitleLabel, LineEdit, PushButton, setTheme, Theme


class CustomMessageBox(MessageBoxBase):
    """ Custom message box """

    def __init__(self, parent=None, mode=None):
        super().__init__(parent)
        self.urlLineEdit = LineEdit(self)
        if mode == "single":
            self.titleLabel = SubtitleLabel('Input Rtsp/Http/Https URL', self)
            self.urlLineEdit.setPlaceholderText('rtsp:// - http:// - https://')
        else:
            self.titleLabel = SubtitleLabel('Input Rtsp URL', self)
            self.urlLineEdit.setPlaceholderText('rtsp://')


        self.urlLineEdit.setFont(QFont("Segoe UI", 14))
        self.urlLineEdit.setClearButtonEnabled(True)

        # add widget to view layout
        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self.urlLineEdit)

        # change the text of button
        self.yesButton.setText('Confirm')
        self.cancelButton.setText('Cancel')

        self.widget.setMinimumWidth(400)
        self.yesButton.setDisabled(True)
        self.urlLineEdit.textChanged.connect(self._validateUrl)

        # self.hideYesButton()

    def _validateUrl(self, text):
        self.yesButton.setEnabled(QUrl(text).isValid())
