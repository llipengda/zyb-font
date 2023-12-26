from PySide6.QtWidgets import QWidget, QGroupBox, QVBoxLayout

from gui.setting.WIdget import Widget


class Setting(QWidget):
    def __init__(self):
        super().__init__()
        self.group = QGroupBox()
        self.layout = QVBoxLayout()

        self.widget = Widget()
        self.layout.addWidget(self.widget)
        self.group.setLayout(self.layout)
