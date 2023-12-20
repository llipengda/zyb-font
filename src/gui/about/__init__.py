from PySide6.QtWidgets import *

from gui.about.Description import Description
from gui.about.License import License


class About(QWidget):
    def __init__(self):
        super().__init__()

        self.group = QGroupBox()
        self.layout = QVBoxLayout()

        self.layout.addWidget(Description())
        self.layout.addWidget(License())
        self.group.setLayout(self.layout)
