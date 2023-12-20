from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *


class Setting(QWidget):
    def __init__(self):
        super().__init__()
        self.group = QGroupBox()
        self.layout = QVBoxLayout()

        self.group.setLayout(self.layout)


