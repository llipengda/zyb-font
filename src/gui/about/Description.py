from PySide6.QtCore import *
from PySide6.QtWidgets import *

import static.data as static


class Description(QWidget):
    def __init__(self):
        super().__init__()

        self.setFixedSize(QSize(840, 80))
        self.layout = QHBoxLayout()

        self.label = QLabel(self)
        self.label.setText(static.data["description"]["text"])
        self.label.setFixedSize(QSize(820, 80))
        self.label.setStyleSheet(static.data["common"]["style"])
        self.layout.addWidget(self.label)
