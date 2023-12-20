import gui.static.data as static

from PySide6.QtWidgets import QLabel, QWidget, QHBoxLayout
from PySide6.QtCore import Slot

from gui.basic.widgets import MiniLabel, FullLabel, CloseLabel, TopBarGroup


class TopBar(QWidget):
    def __init__(self):
        super().__init__()

        self.group = TopBarGroup()
        self.group.setStyleSheet(
            "background-color:{};border-radius:4px;".format(static.data["topnav"]["bg"]))
        self.group.setFixedHeight(52)
        self.layout = QHBoxLayout()

        self.current = QLabel(static.data["current"]["label"])
        self.current.setContentsMargins(40, 0, 0, 0)
        self.current.setStyleSheet(
            "color:{};background-color:rgba(48,55,100,0);font-size:16px".format(static.data["current"]["color"]))
        self.layout.addWidget(self.current)
        self.layout.addStretch()

        self.mini = MiniLabel()
        self.layout.addWidget(self.mini)

        self.full = FullLabel()
        self.layout.addWidget(self.full)

        self.close = CloseLabel()
        self.layout.addWidget(self.close)

        self.group.setLayout(self.layout)

    @Slot(dict)
    def get_label(self, msg):
        self.current.setText(msg["label"])
