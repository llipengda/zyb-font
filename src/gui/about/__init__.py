from PySide6.QtWidgets import QWidget, QGroupBox, QVBoxLayout

from gui.about.Description import Description
from gui.about.License import License


class About(QWidget):
    def __init__(self):
        super().__init__()

        self.group = QGroupBox()
        self.__layout = QVBoxLayout()

        self.__layout.addWidget(Description())
        self.__layout.addWidget(License())
        self.group.setLayout(self.__layout)
