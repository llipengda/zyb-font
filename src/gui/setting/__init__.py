from PySide6.QtWidgets import QWidget, QGroupBox, QVBoxLayout


class Setting(QWidget):
    def __init__(self):
        super().__init__()
        self.group = QGroupBox()
        self.layout = QVBoxLayout()

        self.group.setLayout(self.layout)
