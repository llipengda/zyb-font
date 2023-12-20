from gui.basic.widgets import *


class Search(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QHBoxLayout()
        self.layout.setAlignment(Qt.AlignCenter)

        self.search = QLineEdit(self)
        self.search.setStyleSheet(static.data["common"]["style"])
        self.layout.addWidget(self.search)
        self.search.setPlaceholderText("按照图片名字搜索")

        self.search_button = Button("搜索")
        self.search_button.setParent(self)
        self.layout.addWidget(self.search_button)

        self.layout.setContentsMargins(10, 10, 10, 10)

        self.setLayout(self.layout)


