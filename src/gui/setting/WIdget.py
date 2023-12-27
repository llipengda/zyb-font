import os
import time

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QComboBox
import gui.static.data as static
from gui.basic.widgets import Label, Button, on_pressed


class Widget(QWidget):
    idx = Signal(int)

    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(60, 10, 60, 10)
        self.layout.setSpacing(10)
        self.layout.setAlignment(Qt.AlignTop)

        self.sub_layout = QHBoxLayout()
        self.sub_layout.setContentsMargins(10, 10, 10, 10)
        self.sub_layout.setSpacing(10)
        self.sub_layout.setAlignment(Qt.AlignCenter)
        self.layout.addLayout(self.sub_layout)

        self.label = Label("数据集")
        self.label.setParent(self)
        self.sub_layout.addWidget(self.label)

        self.combo_box = QComboBox(self)
        self.combo_box.addItems(["MNIST(数字)", "HWDB(汉字)"])
        self.combo_box.setStyleSheet(static.data["combo"])
        self.combo_box.setCurrentIndex(0)
        self.combo_box.currentIndexChanged.connect(self.on_index_changed)
        self.sub_layout.addWidget(self.combo_box)

        self.clear_files = Button("清空图库")
        self.clear_files.setParent(self)
        self.clear_files.setMaximumWidth(120)
        self.clear_files.pressed.connect(self.on_clear_files)
        self.layout.addWidget(self.clear_files)

    def on_index_changed(self, value):
        self.idx.emit(value)

    def on_clear_files(self):
        on_pressed(self.clear_files)
        self.setCursor(Qt.WaitCursor)

        if os.path.exists("draw"):
            for file in os.listdir("draw"):
                os.remove(f"draw/{file}")

        self.setCursor(Qt.ArrowCursor)
