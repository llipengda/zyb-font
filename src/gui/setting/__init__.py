from PySide6.QtWidgets import QWidget, QGroupBox, QVBoxLayout

from gui.setting.WIdget import Widget


class Setting(QWidget):
    def __init__(self):
        super().__init__()
        self.group = QGroupBox()
        self.__layout = QVBoxLayout()

        self.widget = Widget()
        self.__layout.addWidget(self.widget)
        self.group.setLayout(self.__layout)
