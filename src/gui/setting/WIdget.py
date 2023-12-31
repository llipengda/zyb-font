import os

import gui.static.data as static

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QComboBox
from gui.basic.widgets import Label, Button, on_pressed


class Widget(QWidget):
    idx = Signal(int)

    def __init__(self):
        super().__init__()

        self.__layout = QVBoxLayout(self)
        self.__layout.setContentsMargins(60, 10, 60, 10)
        self.__layout.setSpacing(10)
        self.__layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.sub_layout = QHBoxLayout()
        self.sub_layout.setContentsMargins(10, 10, 10, 10)
        self.sub_layout.setSpacing(10)
        self.sub_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.__layout.addLayout(self.sub_layout)

        self.label = Label("数据集")
        self.label.setParent(self)
        self.sub_layout.addWidget(self.label)

        self.combo_box = QComboBox(self)
        self.combo_box.addItems(["MNIST(数字)", "HWDB(汉字)"])
        self.combo_box.setStyleSheet(static.data["combo"])
        self.combo_box.setCurrentIndex(0)
        self.combo_box.currentIndexChanged.connect(self.on_index_changed)
        self.sub_layout.addWidget(self.combo_box)

        self.sub_layout = QHBoxLayout()
        self.sub_layout.setContentsMargins(10, 10, 10, 10)
        self.sub_layout.setSpacing(10)
        self.__layout.addLayout(self.sub_layout)

        self.clear_files = Button("清空图库")
        self.clear_files.setParent(self)
        self.clear_files.setMaximumWidth(120)
        self.clear_files.pressed.connect(self.on_clear_files)
        self.sub_layout.addWidget(self.clear_files)

        self.clear_log = Button("清空日志")
        self.clear_log.setParent(self)
        self.clear_log.setMaximumWidth(120)
        self.clear_log.pressed.connect(self.on_clear_log)
        self.sub_layout.addWidget(self.clear_log)

    def on_index_changed(self, value):
        self.idx.emit(value)

    def on_clear_files(self):
        on_pressed(self.clear_files)
        self.setCursor(Qt.CursorShape.WaitCursor)

        if os.path.exists("draw"):
            for file in os.listdir("draw"):
                os.remove(f"draw/{file}")

        self.setCursor(Qt.CursorShape.ArrowCursor)
        
    def on_clear_log(self):
        on_pressed(self.clear_log)
        self.setCursor(Qt.CursorShape.WaitCursor)

        if os.path.exists("logs"):
            for root, dirs, files in os.walk("logs", topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))

        self.setCursor(Qt.CursorShape.ArrowCursor)
