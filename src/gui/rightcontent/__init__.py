import gui.static.data as static

from PySide6.QtCore import Slot
from PySide6.QtWidgets import QWidget, QVBoxLayout, QStackedLayout

from gui.about import About
from gui.file import File
from gui.generate import Generate
from gui.painting import Painting
from gui.setting import Setting
from gui.topnav import TopBar


class RightContent(QWidget):
    def __init__(self):
        super().__init__()

        self.right_content_layout = QVBoxLayout()

        self.nav_group = TopBar()
        self.right_content_layout.addWidget(self.nav_group.group)

        self.right_stack_layout = QStackedLayout()
        self.right_content = static.data["right"]

        self.painting = Painting()
        self.file = File()
        self.generate = Generate()
        self.setting = Setting()
        self.about = About()

        self.right_stack_layout.addWidget(self.painting.group)
        self.right_stack_layout.addWidget(self.file.group)
        self.right_stack_layout.addWidget(self.generate.group)
        self.right_stack_layout.addWidget(self.setting.group)
        self.right_stack_layout.addWidget(self.about.group)

        self.right_content_layout.addLayout(self.right_stack_layout)

        self.setting.widget.idx.connect(self.painting.widget.set_idx)
        self.file.ger.connect(self.generate.generate)

    @Slot(str)
    def get_menu_index(self, msg):
        self.right_stack_layout.setCurrentIndex(msg["index"])
        self.right_stack_layout.update()
