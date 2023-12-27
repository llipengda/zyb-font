import gui.static.data as static

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QHBoxLayout, QScrollArea, QGridLayout


class Show(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QHBoxLayout()
        self.layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.show_area = QScrollArea(self)
        self.show_area.setStyleSheet(static.data["scroll"]["style"])

        self.pics = QWidget(self.show_area)

        self.show_layout = QGridLayout(self.pics)
        self.pics.setLayout(self.show_layout)

        self.show_area.setWidget(self.pics)
        self.show_area.setWidgetResizable(True)
        self.show_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.layout.addWidget(self.show_area)
        self.setLayout(self.layout)
