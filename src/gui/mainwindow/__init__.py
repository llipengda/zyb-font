import gui.static.data as static

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QWidget, QHBoxLayout, QFrame

from gui.menus import LeftMenu
from gui.rightcontent import RightContent


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(static.data["mainwindow"]["TITLE"])
        self.setMinimumSize(748, 480)
        self.setStyleSheet(
            "background:{};border-radius:10px;".format(static.data["mainwindow"]["bg"]))
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.setWindowIcon(QIcon(static.data["mainwindow"]["ICON"]))

        self.left = LeftMenu()
        self.right = RightContent()

        self.main_bgQH = QHBoxLayout()
        self.main_bg = QFrame()

        self.main_layout = QHBoxLayout()
        self.main_layout.addWidget(self.left.menu_group)
        self.main_layout.addLayout(self.right.right_content_layout)
        self.main_layout.setContentsMargins(5, 5, 5, 5)

        self.setLayout(self.main_layout)

        self.left.msg.connect(self.right.nav_group.get_label)
        self.left.msg.connect(self.right.get_menu_index)
        self.left.refresh.connect(self.right.file.refresh)
        self.right.painting.widget.signal.connect(self.right.file.refresh)
        self.right.file.change.connect(self.left.set_menu_bg)
        self.right.generate.generated.connect(self.left.change_is_generated)
