import gui.static.data as static

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLineEdit

from gui.basic.widgets import Button


class Search(QWidget):
    def __init__(self):
        super().__init__()

        self.__layout = QHBoxLayout()
        self.__layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.__layout.setSpacing(10)

        self.search = QLineEdit(self)
        self.search.setStyleSheet(static.data["common"]["style"])
        self.__layout.addWidget(self.search)
        self.search.setPlaceholderText("按照图片名字搜索")

        self.search_button = Button("搜索")
        self.search_button.setParent(self)
        self.search_button.setMinimumWidth(80)
        self.__layout.addWidget(self.search_button)

        self.__layout.addSpacing(20)

        self.all_check = Button("全选")
        self.all_check.setParent(self)
        self.all_check.setMinimumWidth(80)
        self.__layout.addWidget(self.all_check)

        self.all_uncheck = Button("取消全选")
        self.all_uncheck.setParent(self)
        self.all_uncheck.setMinimumWidth(80)
        self.__layout.addWidget(self.all_uncheck)

        self.__layout.addSpacing(15)

        self.generate_button = Button("生成")
        self.generate_button.setParent(self)
        self.generate_button.setMinimumWidth(100)
        self.__layout.addWidget(self.generate_button)

        self.__layout.setContentsMargins(10, 10, 10, 10)

        self.setLayout(self.__layout)
