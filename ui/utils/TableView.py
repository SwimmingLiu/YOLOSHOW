# coding: utf-8
import sys

from PySide6.QtWidgets import QHeaderView
from PySide6.QtCore import QModelIndex, Qt
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication, QStyleOptionViewItem, QTableWidget, QTableWidgetItem, QWidget, QHBoxLayout

from qfluentwidgets import TableWidget, isDarkTheme, setTheme, Theme, TableView, TableItemDelegate, setCustomStyleSheet


class TableViewDelegate(TableItemDelegate):
    """ Custom table item delegate """

    def initStyleOption(self, option: QStyleOptionViewItem, index: QModelIndex):
        super().initStyleOption(option, index)
        if index.column() != 1:
            return

        if isDarkTheme():
            option.palette.setColor(QPalette.Text, Qt.white)
            option.palette.setColor(QPalette.HighlightedText, Qt.white)
        else:
            option.palette.setColor(QPalette.Text, Qt.black)
            option.palette.setColor(QPalette.HighlightedText, Qt.black)


class TableViewQWidget(QWidget):

    def __init__(self,infoList=None):
        super().__init__()
        # setTheme(Theme.DARK)
        self.setWindowTitle("Result Statistics")
        self.hBoxLayout = QHBoxLayout(self)
        self.tableView = TableWidget(self)

        # NOTE: use custom item delegate
        self.tableView.setItemDelegate(TableViewDelegate(self.tableView))

        # select row on right-click
        self.tableView.setSelectRightClickedRow(True)

        # enable border
        self.tableView.setBorderVisible(True)
        self.tableView.setBorderRadius(8)

        self.tableView.setWordWrap(False)
        # 表格行数
        self.tableView.setRowCount(1000)
        self.tableView.setColumnCount(3)
        # 表格列数
        self.Infos = infoList if infoList else list()
        info_count = 1
        for i, info in enumerate(self.Infos):
            self.tableView.setItem(i, 0, QTableWidgetItem(str(info_count)))
            info_count += 1
            for j in range(1, len(info) + 1):
                self.tableView.setItem(i, j, QTableWidgetItem(info[j - 1]))

        self.tableView.verticalHeader().hide()
        self.tableView.setHorizontalHeaderLabels(['Index', 'Class', 'Frequency'])
        self.tableView.resizeColumnsToContents()
        self.tableView.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tableView.setSortingEnabled(True)

        self.setStyleSheet("TableViewQWidget{background: rgb(255, 255, 255)} ")
        self.hBoxLayout.setContentsMargins(20, 10, 20, 10)
        self.hBoxLayout.addWidget(self.tableView)
        self.resize(500, 500)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = TableViewQWidget()
    w.show()
    app.exec()
