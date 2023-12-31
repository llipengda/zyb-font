import os

import gui.static.data as static

from PySide6.QtWidgets import QWidget, QGroupBox, QVBoxLayout, QLabel, QCheckBox
from PySide6.QtCore import Qt, Slot, Signal
from PySide6.QtGui import QPixmap, QKeySequence, QShortcut

from gui.basic.widgets import on_pressed, MessageBox
from gui.file.Search import Search
from gui.file.Show import Show


class File(QWidget):
    ger = Signal(list)
    change = Signal(int)

    def __init__(self):
        super().__init__()

        self.group = QGroupBox()
        self.__layout = QVBoxLayout()

        self.__search = Search()

        self.__layout.addWidget(self.__search)
        self.__search.search_button.pressed.connect(self.on_button_clicked)
        self.enter = QShortcut(QKeySequence(
            Qt.Key.Key_Return), self.__search, None)
        self.enter.activated.connect(self.on_button_clicked)

        self.__search.generate_button.pressed.connect(self.on_generate)
        self.__search.all_check.pressed.connect(self.on_all_check)
        self.__search.all_uncheck.pressed.connect(self.on_all_uncheck)

        self.__show = Show()
        self.__layout.addWidget(self.__show)
        self.group.setLayout(self.__layout)

        self.search_images(True)

    @Slot()
    def on_button_clicked(self):
        on_pressed(self.__search.search_button)
        self.search_images()

    @Slot()
    def refresh(self):
        self.search_images()

    def search_images(self, first=False):
        keyword = self.__search.search.text().strip() if not first else ""
        folder = static.data["draw"]["path"]
        if os.path.exists(folder) is False:
            os.mkdir(folder)
        image_files = []
        for file_name in os.listdir(folder):
            # if file_name.lower().endswith("jpg"):
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
            pic = Label()
            pic.setFixedSize(100, 100)
            pic.setText(image_file)
            pic.mousePressEvent = lambda event, row=row, col=col: self.on_pic_clicked(event, pic, row, col) # type: ignore
            pixmap = QPixmap(image_file).scaledToWidth(100)
            pic.setPixmap(pixmap)

            image_name = os.path.splitext(os.path.basename(image_file))[0]

            if len(image_name) > 5:
                image_name = image_name[:6] + "..."

            pic_name = CheckBox(self, image_name)
            pic_name.filename = image_file

            self.__show.show_layout.addWidget(pic, row, col)
            self.__show.show_layout.addWidget(pic_name, row + 1, col)
            col += 1
            if col == 5:
                row += 2
                col = 0

        self.__show.show_area.widget().update()

    @Slot()
    def on_generate(self):
        on_pressed(self.__search.generate_button)

        files = []
        for i in range(self.__show.show_layout.count()):
            widget = self.__show.show_layout.itemAt(i).widget()
            if isinstance(widget, CheckBox):
                if widget.isChecked():
                    files.append(widget.filename)

        if len(files) == 0:
            msg = MessageBox(self)
            msg.setText("没有选中任何图片哦")
            msg.exec()

            return

        self.ger.emit(files)
        self.change.emit(2)

    @Slot()
    def on_all_check(self):
        on_pressed(self.__search.all_check)

        for i in range(self.__show.show_layout.count()):
            widget = self.__show.show_layout.itemAt(i).widget()
            if isinstance(widget, CheckBox):
                widget.setChecked(True)

    @Slot()
    def on_all_uncheck(self):
        on_pressed(self.__search.all_uncheck)

        for i in range(self.__show.show_layout.count()):
            widget = self.__show.show_layout.itemAt(i).widget()
            if isinstance(widget, CheckBox):
                widget.setChecked(False)

    def on_pic_clicked(self, event, widget, row, col):
        on_pressed(widget)

        self.__show.show_layout.itemAtPosition(row + 1, col).widget().setChecked( # type: ignore
            not self.__show.show_layout.itemAtPosition(row + 1, col).widget().isChecked()) # type: ignore


class Label(QLabel):
    def __init__(self):
        super().__init__()
        self.setStyleSheet(
            "padding:5px;color:#ffffff;font-size: 18px;")

    def enterEvent(self, event) -> None:
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            "background-color:rgb(81,93,128);padding:5px;color:#ffffff;font-size: 18px;border-width: "
            "1px;border-style: solid;border-color: #ffffff;")

    def leaveEvent(self, event) -> None:
        self.setStyleSheet(
            "background-color:rgba(81,93,128,0);padding:5px;color:#ffffff;font-size: 18px;border-width: "
            "1px;border-style: solid;border-color: #ffffff;")

    def mouseReleaseEvent(self, event) -> None:
        self.setStyleSheet(
            "background-color:rgb(81,93,128);padding:5px;color:#ffffff;font-size: 18px;border-width: "
            "1px;border-style: solid;border-color: #ffffff;")


class CheckBox(QCheckBox):
    filename = ""

    def __init__(self, parent, name):
        super().__init__(parent)

        self.setStyleSheet(static.data["checkbox"])
        self.setMaximumHeight(30)
        self.setText(name)
