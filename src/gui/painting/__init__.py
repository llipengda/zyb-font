from PySide6.QtWidgets import *
from gui.painting.PaintBoard import PaintBoard
from gui.painting.Widget import Widget


class Painting(QWidget):
    def __init__(self):
        super().__init__()

        self.group = QGroupBox()
        self.layout = QHBoxLayout()

        self.layout.addWidget(Widget())
        self.group.setLayout(self.layout)
