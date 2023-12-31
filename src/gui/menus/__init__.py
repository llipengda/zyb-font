from PySide6.QtGui import QPixmap

import gui.static.data as static

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import QLabel, QWidget, QGroupBox, QHBoxLayout, QVBoxLayout, QFrame

from gui.basic.widgets import Label, MenuFrame, MessageBox


class LeftMenu(QWidget):
    msg = Signal(dict)
    refresh = Signal()
    is_generated = False

    def __init__(self):
        super().__init__()

        self.menu = static.data["menu"]

        self.flag = True
        self.menu_group = QGroupBox()
        self.menu_group.setStyleSheet(
            "background-color:{};border-radius:4px;".format(static.data["menus"]["bg"]))
        self.menu_group.setFixedWidth(static.data["menus"]["width"])
        self.menu_layout = QVBoxLayout()

        self.menus_layout = QVBoxLayout()

        logo_layout = QHBoxLayout()
        self.menu_logo = QLabel()
        self.menu_logo.setFixedSize(static.data["main_logo"]["width"],
                                    static.data["main_logo"]["height"])
        pixmap = QPixmap(static.data["main_logo"]["picpath"])
        self.menu_logo.setPixmap(pixmap)
        self.menu_logo.setScaledContents(True)
        self.menu_logo.setAlignment(Qt.AlignmentFlag.AlignCenter)

        logo_layout.addWidget(self.menu_logo)
        self.menu_layout.addLayout(logo_layout)

        self.menu_layout.addSpacing(20)

        for index, item in enumerate(self.menu):
            self.menu_bg = MenuFrame()
            self.menu_bg.mousePressEvent = lambda event, index=index: self.set_menu_bg(index)
            self.menu_bg.setFixedSize(static.data["menu_bg"]["width"],
                                      static.data["menu_bg"]["height"])
            self.menu_bg.setContentsMargins(2, 2, 2, 0)
            self.menu_QH = QHBoxLayout()
            self.menu_icon = QFrame()
            self.menu_icon.setStyleSheet(
                "background:url({}) no-repeat center center;".format(static.data["menu_icon"][item["type"]]))
            self.menu_label = Label(item["name"])
            self.menu_label.setStyleSheet(
                "color: #ffffff;font-size:16px;font-family:Microsoft YaHei;")
            self.menu_icon.setFixedSize(25, 25)

            self.menu_QH.addWidget(self.menu_icon)
            self.menu_QH.addWidget(self.menu_label)
            self.menu_bg.setLayout(self.menu_QH)

            self.menus_layout.addWidget(self.menu_bg)

        self.menus_layout.setSpacing(20)

        self.menu_layout.setContentsMargins(0, 20, 0, 20)

        menus_layout = QHBoxLayout()
        menus_layout.addLayout(self.menus_layout)
        self.menu_layout.addLayout(menus_layout)

        self.menu_layout.addStretch()

        self.menu_btn_frame = MenuFrame()
        self.menu_btn_frame.setFixedHeight(100)
        self.menu_btn_frame.mousePressEvent = self.toggle_menu
        self.menu_btn_layout = QHBoxLayout()
        self.menu_btn_icon = QFrame()
        self.menu_btn_icon.setStyleSheet(
            "background:url({}) no-repeat center center;".format(static.data["menu_btn"]["open"]))
        self.menu_btn_icon.setFixedSize(36, 36)
        self.menu_btn_layout.addWidget(self.menu_btn_icon)
        self.menu_btn_frame.setLayout(self.menu_btn_layout)
        self.menu_layout.addWidget(self.menu_btn_frame)

        self.menu_group.setLayout(self.menu_layout)

    def change_menu(self):
        if self.flag:
            self.menu_logo.setPixmap(QPixmap(static.data["main_logo"]["picpathm"]))
            self.menu_logo.setFixedSize(static.data["folded_logo"]["width"],
                                        static.data["folded_logo"]["height"])

            for i in range(self.menus_layout.count()):
                self.menus_layout.itemAt(i).widget().setFixedSize(static.data["folded_menu_icon"]["width"],
                                                                  static.data["folded_menu_icon"]["height"])
                self.menus_layout.itemAt(i).widget().findChildren(QLabel)[
                    0].setVisible(False)
        else:
            self.menu_logo.setPixmap(QPixmap(static.data["main_logo"]["picpath"]))
            self.menu_logo.setFixedSize(static.data["main_logo"]["width"],
                                        static.data["main_logo"]["height"])

            for i in range(self.menus_layout.count()):
                self.menus_layout.itemAt(i).widget().setFixedSize(static.data["menu_bg"]["width"],
                                                                  static.data["menu_bg"]["height"])
                self.menus_layout.itemAt(i).widget().findChildren(QLabel)[
                    0].setVisible(True)

    def toggle_menu(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.flag:
                self.menu_btn_icon.setStyleSheet(
                    "background:url({}) no-repeat center center;".format(static.data["menu_btn"]["close"]))
                self.change_menu()
                self.menu_group.setFixedWidth(60)
                self.flag = False
            else:
                self.menu_btn_icon.setStyleSheet(
                    "background:url({}) no-repeat center center;".format(static.data["menu_btn"]["open"]))
                self.change_menu()
                self.menu_group.setFixedWidth(200)
                self.flag = True

    @Slot(int)
    def set_menu_bg(self, index):
        for i in range(self.menus_layout.count()):
            if i == index:
                if index == 2 and not self.is_generated:
                    msg = MessageBox(self)
                    msg.setText("请先在图库中选择图像并生成")
                    msg.exec()
                    return

                self.menus_layout.itemAt(i).widget().flag = True # type: ignore
                self.menus_layout.itemAt(i).widget().setStyleSheet("background-color: {};border-radius:4px;".format(
                    static.data["menu_bg"]["press"]))
                dict_data = {"label": self.menus_layout.itemAt(
                    i).widget().layout().itemAt(1).widget().text(), "index": index} # type: ignore
                self.change_label(dict_data)

            elif i != index:
                self.menus_layout.itemAt(i).widget().flag = False # type: ignore
                self.menus_layout.itemAt(i).widget().setStyleSheet(
                    "background-color:{}".format(static.data["menu_bg"]["leave"]))

    def change_label(self, arg):
        self.msg.emit(arg)
        if arg["index"] == 1:
            self.refresh.emit()

    @Slot()
    def change_is_generated(self):
        self.is_generated = True
