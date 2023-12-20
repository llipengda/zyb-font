import gui.static.data as static

from PySide6.QtWidgets import QWidget, QHBoxLayout, QTextEdit
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QTextCursor


class License(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QHBoxLayout()

        self.setFixedSize(QSize(840, 500))
        self.license = QTextEdit(self)
        self.license.setPlainText(self.get_license())
        self.license.setReadOnly(True)
        self.license.setFixedSize(QSize(820, 500))
        self.license.setStyleSheet(static.data["license"]["style"])

        self.cursor = self.license.textCursor()
        self.cursor.movePosition(QTextCursor.MoveOperation.Start)
        while not self.cursor.atEnd():
            self.cursor.select(QTextCursor.SelectionType.LineUnderCursor)
            block = self.cursor.blockFormat()
            block.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.cursor.mergeBlockFormat(block)
            self.cursor.movePosition(QTextCursor.MoveOperation.NextBlock)

        self.layout.addWidget(self.license)

    def get_license(self):
        with open(static.data["license"]["file"], "r") as f:
            lic = f.read()
        return lic
