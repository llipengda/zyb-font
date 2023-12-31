from PySide6.QtWidgets import QWidget, QGroupBox, QHBoxLayout

from gui.painting.Widget import Widget


class Painting(QWidget):
    def __init__(self):
        super().__init__()

        self.group = QGroupBox()
        self.__layout = QHBoxLayout()
        self.widget = Widget()

        self.__layout.addWidget(self.widget)
        self.group.setLayout(self.__layout)
