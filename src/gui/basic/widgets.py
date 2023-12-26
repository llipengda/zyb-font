from PySide6.QtWidgets import QWidget, QLabel, QPushButton, QCheckBox, QSlider, QFrame, QGroupBox
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

import gui.static.data as static


def on_pressed(widget: QWidget):
    widget.setStyleSheet(
        "background-color:rgba(81,93,128,0);padding:5px;color:#ffffff;font-size: 18px;border-width: "
        "1px;border-style: solid;border-color: #ffffff;")


class Label(QLabel):
    def __init__(self, name):
        super().__init__()
        self.setText(name)
        self.setStyleSheet(
            "padding:5px;color:#ffffff;font-size: 18px;")


class Button(QPushButton):
    def __init__(self, name):
        super().__init__()
        self.setStyleSheet(static.data["common"]["style"])
        self.setText(name)

    def enterEvent(self, event) -> None:
        self.setCursor(Qt.PointingHandCursor)
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
    def __init__(self, name):
        super().__init__()
        self.setText(name)
        self.setStyleSheet(static.data["common"]["style"])

    def enterEvent(self, event) -> None:
        self.setStyleSheet(
            "background-color:rgb(81,93,128);padding:5px;color:#ffffff;font-size: 18px;border-width: "
            "1px;border-style: solid;border-color: #ffffff;")

    def leaveEvent(self, event) -> None:
        self.setStyleSheet(
            "background-color:rgba(81,93,128,0);padding:5px;color:#ffffff;font-size: 18px;border-width: "
            "1px;border-style: solid;border-color: #ffffff;")


class Slider(QSlider):
    def __init__(self):
        super().__init__()
        self.setStyleSheet(static.data["slider"])
        self.setOrientation(Qt.Orientation.Horizontal)
        self.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.setTickInterval(5)
        self.setMaximum(50)
        self.setMinimum(0)
        self.setValue(25)
        self.setSingleStep(1)

    def enterEvent(self, event) -> None:
        self.setCursor(Qt.PointingHandCursor)


class MenuFrame(QFrame):
    def __init__(self):
        super().__init__()
        self.flag = False
        self.setStyleSheet("background-color:rgba(54,64,95,0)")

    def enterEvent(self, event):
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(
            "background-color: {};border-radius:4px;".format(static.data["menu_bg"]["enter"]))

    def leaveEvent(self, event):
        if not self.flag:
            self.setStyleSheet("background-color:{}".format(static.data["menu_bg"]["leave"]))

    def mouseReleaseEvent(self, event):
        self.setStyleSheet(
            "background-color: {};border-radius:4px;".format(static.data["menu_bg"]["enter"]))


class MiniLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.style = "background:url({}) no-repeat center center;".format(static.data["topnav"]["mini"])
        self.setStyleSheet(self.style)
        self.setFixedSize(static.data["to_btn"]["width"], static.data["to_btn"]["height"])
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def enterEvent(self, event):
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(self.style+"background-color: {};border-radius:4px;".format(
            static.data["to_btn"]["focusbackground"]))

    def leaveEvent(self, event):
        self.setStyleSheet(self.style+"background-color: {};border-radius:4px;".format(
            static.data["to_btn"]["nobackground"]))

    def mousePressEvent(self, event):
        self.parent().parent().showMinimized()


class FullLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.style = "background:url({}) no-repeat center center;".format(static.data["topnav"]["full"])
        self.setStyleSheet(self.style)
        self.setFixedSize(static.data["to_btn"]["width"], static.data["to_btn"]["height"])
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.flag = False

    def enterEvent(self, event):
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(self.style + "background-color: {};border-radius:4px;".format(
            static.data["to_btn"]["focusbackground"]))

    def leaveEvent(self, event):
        self.setStyleSheet(self.style + "background-color: {};border-radius:4px;".format(
            static.data["to_btn"]["nobackground"]))

    def mousePressEvent(self, event):
        if not self.flag:
            self.flag = True
            self.parent().parent().setWindowState(Qt.WindowState.WindowFullScreen)
        else:
            self.flag = False
            self.parent().parent().setWindowState(Qt.WindowState.WindowNoState)


class CloseLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.style = "background:url({}) no-repeat center center;".format(static.data["topnav"]["close"])
        self.setStyleSheet(self.style)
        self.setFixedSize(static.data["to_btn"]["width"], static.data["to_btn"]["height"])
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def enterEvent(self, event):
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(self.style + "background-color: {};border-radius:4px;".format(
            static.data["to_btn"]["focusbackground"]))

    def leaveEvent(self, event):
        self.setStyleSheet(self.style + "background-color: {};border-radius:4px;".format(
            static.data["to_btn"]["nobackground"]))

    def mousePressEvent(self, event):
        self.parent().parent().close()


class TopBarGroup(QGroupBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_start = None
        self.mouse_start = None
        self.window = self.parent()
        self.flag = False

    def mousePressEvent(self, event):
        self.flag = True
        self.mouse_start = self.mapToGlobal(event.pos())
        self.window_start = self.parent().pos()

    def mouseReleaseEvent(self, event):
        self.flag = False

    def mouseMoveEvent(self, event):
        if self.flag:
            distance = self.mapToGlobal(event.pos()) - self.mouse_start
            new_position = self.window_start + distance
            self.parent().move(new_position)
