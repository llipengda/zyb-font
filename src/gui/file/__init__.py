import os
from gui.basic.widgets import on_pressed
from PySide6.QtWidgets import QWidget, QGroupBox, QVBoxLayout, QLabel
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QPixmap, QKeySequence, QShortcut
from gui.file.Search import Search
from gui.file.Show import Show
import gui.static.data as static


class File(QWidget):
    def __init__(self):
        super().__init__()

        self.group = QGroupBox()
        self.layout = QVBoxLayout()

        self.__search = Search()

        self.layout.addWidget(self.__search)
        self.__search.search_button.pressed.connect(self.on_button_clicked)
        self.enter = QShortcut(QKeySequence(
            Qt.Key.Key_Return), self.__search, None)
        self.enter.activated.connect(self.on_button_clicked)

        self.__show = Show()
        self.layout.addWidget(self.__show)
        self.group.setLayout(self.layout)

        self.search_images(True)

    @Slot()
    def on_button_clicked(self):
        on_pressed(self.__search.search_button)
        self.search_images()

    def search_images(self, first=False):
        keyword = self.__search.search.text().strip() if not first else ""
        folder = static.data["draw"]["path"]
        if os.path.exists(folder) is False:
            os.mkdir(folder)
        image_files = []
        for file_name in os.listdir(folder):
            if file_name.lower().endswith("jpg"):
                if keyword.lower() in file_name.lower():
                    image_files.append(os.path.join(folder, file_name))

        image_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

        self.update_pics(image_files)

    def update_pics(self, image_files):
        for i in reversed(range(self.__show.show_layout.count())):
            remove = self.__show.show_layout.itemAt(i).widget()
            self.__show.show_layout.removeWidget(remove)
            remove.setParent(None)
            remove.destroy()

        if len(image_files) == 0:
            tip = QLabel("什么也没有搜到哦", self)
            tip.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.__show.show_layout.addWidget(tip, 0, 0)
            return

        row, col = 0, 0
        for image_file in image_files:
            pic = QLabel(self)
            pic.setFixedSize(100, 100)
            pixmap = QPixmap(image_file).scaledToWidth(100)
            pic.setPixmap(pixmap)

            image_name = os.path.splitext(os.path.basename(image_file))[0]

            if len(image_name) > 8:
                image_name = image_name[:8] + "..."

            pic_name = QLabel(image_name, self)
            pic_name.setAlignment(Qt.AlignmentFlag.AlignCenter)

            self.__show.show_layout.addWidget(pic, row, col)
            self.__show.show_layout.addWidget(pic_name, row + 1, col)

            col += 1
            if col == 5:
                row += 2
                col = 0

        self.__show.show_area.widget().update()
