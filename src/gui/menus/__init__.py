from gui.basic.widgets import *

import static.data as static


class LeftMenu(QWidget):
    msg = Signal(dict)

    def __init__(self):
        super().__init__()

        self.menu = static.data["menu"]

        # 初始化组件
        self.flag = True
        self.menu_group = QGroupBox()
        self.menu_group.setStyleSheet(
            "background-color:{};border-radius:4px;".format(static.data["menus"]["bg"]))
        self.menu_group.setFixedWidth(static.data["menus"]["width"])
        self.menu_layout = QVBoxLayout()

        self.menus_layout = QVBoxLayout()

        logo_layout = QHBoxLayout()
        self.menu_logo = QFrame(self)
        self.menu_logo.setFixedSize(static.data["main_logo"]["width"],
                                    static.data["main_logo"]["height"])
        self.menu_logo.setStyleSheet(
            "background:url({}) no-repeat center center;".format(static.data["main_logo"]["picpath"]))
        logo_layout.addWidget(self.menu_logo)
        self.menu_layout.addLayout(logo_layout)

        for index, item in enumerate(self.menu):
            self.menu_bg = MenuFrame()
            self.menu_bg.mousePressEvent = lambda event, index=index: self.set_menu_bg(event, index)
            self.menu_bg.setFixedSize(static.data["menu_bg"]["width"],
                                      static.data["menu_bg"]["height"])
            self.menu_bg.setContentsMargins(2, 2, 2, 0)
            self.menu_QH = QHBoxLayout()
            self.menu_icon = QFrame()
            self.menu_icon.setStyleSheet(
                "background:url({}) no-repeat center center;".format(static.data["menu_icon"][item["type"]]))
            self.menu_label = Label(item["name"])
            self.menu_label.setStyleSheet("color: #ffffff;font-size:16px;font-family:Microsoft YaHei;")
            self.menu_icon.setFixedSize(25, 25)

            self.menu_QH.addWidget(self.menu_icon)
            self.menu_QH.addWidget(self.menu_label)
            self.menu_bg.setLayout(self.menu_QH)

            self.menus_layout.addWidget(self.menu_bg)

        self.menus_layout.setSpacing(20)

        self.menu_layout.setContentsMargins(0, 20, 0, 20)

        # 使之水平居中的布局
        menus_QHlayout = QHBoxLayout()
        menus_QHlayout.addLayout(self.menus_layout)
        self.menu_layout.addLayout(menus_QHlayout)

        self.menu_layout.addStretch()

        self.menu_btn_frame = MenuFrame()
        self.menu_btn_frame.setFixedHeight(100)
        self.menu_btn_frame.mousePressEvent = self.toggle_menu
        self.menu_btn_layout = QHBoxLayout()
        self.menu_btn_icon = QFrame()
        self.menu_btn_icon.setStyleSheet(
            "background:url({}) no-repeat center center;".format(static.data["menu_btn"]["open"]))
        self.menu_btn_icon.setFixedSize(36,36)
        self.menu_btn_layout.addWidget(self.menu_btn_icon)
        self.menu_btn_frame.setLayout(self.menu_btn_layout)
        self.menu_layout.addWidget(self.menu_btn_frame)

        self.menu_group.setLayout(self.menu_layout)

    # 点击展开收起 重新渲染菜单栏
    def change_menu(self):
        if self.flag:
            # 改变logo
            self.menu_logo.setStyleSheet(
                "background:url({}) no-repeat center center;".format(static.data["main_logo"]["picpathm"]))
            self.menu_logo.setFixedSize(static.data["folded_logo"]["width"],
                                        static.data["folded_logo"]["height"])

            for i in range(self.menus_layout.count()):
                self.menus_layout.itemAt(i).widget().setFixedSize(static.data["folded_menu_icon"]["width"],
                                                                  static.data["folded_menu_icon"]["height"])
                self.menus_layout.itemAt(i).widget().findChildren(QLabel)[0].setVisible(False)
        else:
            # 改变logo
            self.menu_logo.setStyleSheet(
                "background:url({}) no-repeat center center;".format(static.data["main_logo"]["picpath"]))
            self.menu_logo.setFixedSize(static.data["main_logo"]["width"],
                                        static.data["main_logo"]["height"])

            for i in range(self.menus_layout.count()):
                self.menus_layout.itemAt(i).widget().setFixedSize(static.data["menu_bg"]["width"],
                                                                  static.data["menu_bg"]["height"])
                self.menus_layout.itemAt(i).widget().findChildren(QLabel)[0].setVisible(True)

    # 点击展开收起事件
    def toggle_menu(self, event):
        if event.button() == Qt.LeftButton:
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

    def set_menu_bg(self, event, index):
        for i in range(self.menus_layout.count()):
            if i == index:
                self.menus_layout.itemAt(i).widget().flag = True
                self.menus_layout.itemAt(i).widget().setStyleSheet("background-color: {};border-radius:4px;".format(
                    static.data["menu_bg"]["press"]))
                dict_data = {"label": self.menus_layout.itemAt(i).widget().layout().itemAt(1).widget().text(),"index": index}
                self.change_label(dict_data)

            elif i != index:
                self.menus_layout.itemAt(i).widget().flag = False
                self.menus_layout.itemAt(i).widget().setStyleSheet(
                    "background-color:{}".format(static.data["menu_bg"]["leave"]))

    def change_label(self, arg):
        self.msg.emit(arg)

