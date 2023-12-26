import os
import time
import uuid

from PySide6.QtCore import Signal, Slot
from PySide6.QtGui import Qt
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QSplitter, QWidget, QLabel

from deeplearning import MNIST, HWDB
from gui.basic.widgets import Button, Label, Slider, on_pressed
from gui.painting.PaintBoard import PaintBoard
import gui.static.data as static


# TODO 适配

class Widget(QWidget):
    signal = Signal()
    msg = Signal()
    ger = Signal()

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.funcs = {
            0: MNIST.predict(),
            1: HWDB.predict(),
            2: HWDB.predict()
        }
        self.idx = 0
        self.__init_data()
        self.__init_view()

    def __init_data(self):
        self.__paint_board = PaintBoard()

    def __init_view(self):
        self.setFixedSize(840, 500)
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(10)

        main_layout.addWidget(self.__paint_board)
        sub_layout = QVBoxLayout()

        sub_layout.setContentsMargins(10, 10, 10, 10)

        self.__btn_clear = Button("清空界面")
        self.__btn_clear.setParent(self)
        self.__btn_clear.pressed.connect(self.on_clear)
        sub_layout.addWidget(self.__btn_clear)

        self.__eraser = Button("使用橡皮擦")
        self.__eraser.setParent(self)
        self.__eraser.pressed.connect(self.on_btn_eraser)
        sub_layout.addWidget(self.__eraser)

        self.__label_pen = Label("画笔粗细")
        self.__label_pen.setParent(self)
        self.__label_pen.setFixedHeight(30)
        sub_layout.addWidget(self.__label_pen)

        self.__slider_pen = Slider()
        self.__slider_pen.setParent(self)
        self.__slider_pen.valueChanged.connect(self.slider_drag)
        self.__slider_pen.sliderPressed.connect(
            lambda: self.__value.setVisible(True))
        self.__slider_pen.sliderReleased.connect(
            lambda: self.__value.setVisible(False))
        sub_layout.addWidget(self.__slider_pen)

        self.__value = Label("25")
        self.__value.setParent(self)
        self.__value.setGeometry(645, 169, 40, 40)
        self.__value.setVisible(False)

        splitter = QSplitter(self)
        sub_layout.addWidget(splitter)

        self.__btn_save = Button("保存")
        self.__btn_save.setParent(self)
        self.__btn_save.pressed.connect(self.on_btn_save)
        sub_layout.addWidget(self.__btn_save)

        self.__btn_predict = Button("保存并预测")
        self.__btn_predict.setParent(self)
        self.__btn_predict.pressed.connect(self.on_btn_save_and_predict)
        sub_layout.addWidget(self.__btn_predict)

        sub_layout.addSpacing(30)

        predict_layout = QHBoxLayout()
        predict_layout.setSpacing(10)
        sub_layout.addLayout(predict_layout)

        self.__prediction_label = Label("预测结果")
        self.__prediction_label.setParent(self)
        self.__prediction_label.setFixedHeight(30)
        predict_layout.addWidget(self.__prediction_label)

        self.__prediction_clear_button = Button("清空预测结果")
        self.__prediction_clear_button.setParent(self)
        self.__prediction_clear_button.pressed.connect(self.on_prediction_clear)
        predict_layout.addWidget(self.__prediction_clear_button)

        self.__prediction = QLabel(self)
        self.__prediction.setFixedHeight(100)
        self.__prediction.setAlignment(Qt.AlignCenter)
        self.__prediction.setStyleSheet(static.data["predict"]["style"])

        sub_layout.addWidget(self.__prediction)

        main_layout.addLayout(sub_layout)

        # 置于顶层
        self.__value.raise_()

    def on_clear(self):
        on_pressed(self.__btn_clear)
        self.__paint_board.clear()

    def on_btn_eraser(self):
        on_pressed(self.__eraser)
        self.__paint_board.eraser_mode = not self.__paint_board.eraser_mode
        if self.__paint_board.eraser_mode:
            self.__eraser.setText("使用画笔")
        else:
            self.__eraser.setText("使用橡皮擦")

    def on_btn_save(self):
        on_pressed(self.__btn_save)

        if self.__paint_board.is_empty():
            return
        name = uuid.uuid1()
        path = rf"./draw/{name}.jpg"
        image = self.__paint_board.get_content_as_image()
        if not os.path.exists("./draw"):
            os.mkdir("./draw")
        image.save(path)

        self.signal.emit()
        self.__paint_board.clear()

    def on_btn_save_and_predict(self):
        on_pressed(self.__btn_save)

        if self.__paint_board.is_empty():
            return
        name = uuid.uuid1()
        path = rf"./draw/{name}.jpg"
        image = self.__paint_board.get_content_as_image()
        if not os.path.exists("./draw"):
            os.mkdir("./draw")
        image.save(path)

        prediction = self.funcs[self.idx](path)

        self.__prediction.setText(str(prediction))
        os.rename(path, rf"./draw/{str(prediction)}-{name}.jpg")

        self.signal.emit()

        self.__paint_board.clear()

    def slider_drag(self, value):
        self.__paint_board.change_pen_thickness(value)

        left = self.__slider_pen.minimum()
        right = self.__slider_pen.maximum()
        maximum = right - left
        proportion = (value - left) / maximum

        x, y, width, height = self.__slider_pen.geometry().x(), self.__slider_pen.geometry().y(), \
                              self.__slider_pen.geometry().width(), self.__slider_pen.geometry().height()
        val = proportion * width

        self.__value.move(x + val - self.__value.width() // 2, y + height)
        self.__value.setText(str(value))

    def on_prediction_clear(self):
        on_pressed(self.__prediction_clear_button)
        self.__prediction.setText("")

    @Slot(int)
    def set_idx(self, idx):
        self.idx = idx
