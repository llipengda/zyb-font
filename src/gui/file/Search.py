import gui.static.data as static

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLineEdit

from gui.basic.widgets import Button


class Search(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QHBoxLayout()
        self.layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.search = QLineEdit(self)
        self.search.setStyleSheet(static.data["common"]["style"])
        self.layout.addWidget(self.search)
        self.search.setPlaceholderText("按照图片名字搜索")

        self.search_button = Button("搜索")
        self.search_button.setParent(self)
        self.layout.addWidget(self.search_button)

        self.layout.setContentsMargins(10, 10, 10, 10)

        self.setLayout(self.layout)
