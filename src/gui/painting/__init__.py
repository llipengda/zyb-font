from PySide6.QtWidgets import QWidget, QGroupBox, QHBoxLayout

from gui.painting.Widget import Widget
from deeplearning.MNIST import predict


class Painting(QWidget):
    def __init__(self):
        super().__init__()

        self.group = QGroupBox()
        self.layout = QHBoxLayout()
        self.widget = Widget(predict())

        self.layout.addWidget(self.widget)
        self.group.setLayout(self.layout)
