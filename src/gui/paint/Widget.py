import os
import uuid

from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QPushButton, \
    QSplitter, QLabel, QSpinBox, QWidget, QCheckBox

from gui.paint.PaintBoard import PaintBoard
from deeplearning import predict


# noinspection PyUnresolvedReferences
class Widget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.__init_data()
        self.__init_view()

    def __init_data(self):
        self.__paint_board = PaintBoard(self)

    def __init_view(self):
        self.setFixedSize(800, 800)
        self.setWindowTitle("画板")
        main_layout = QHBoxLayout(self)
        # 设置布局内边距以及控件的间距为10px
        main_layout.setSpacing(10)

        # 在左侧放置画板

        main_layout.addWidget(self.__paint_board)

        # 新建垂直子布局，放置案件
        sub_layout = QVBoxLayout()

        # 设置此子布局和内部控件的间距为10px
        sub_layout.setContentsMargins(10, 10, 10, 10)

        self.__btn_clear = QPushButton("清空界面")
        self.__btn_clear.setParent(self)

        # 将按键事件和我们写的函数关联起来

        self.__btn_clear.clicked.connect(self.__paint_board.clear)
        sub_layout.addWidget(self.__btn_clear)

        self.__btn_quit = QPushButton("退出")
        self.__btn_quit.setParent(self)  # 设置父对象为本界面
        self.__btn_quit.clicked.connect(self.quit)
        sub_layout.addWidget(self.__btn_quit)

        self.__btn_save = QPushButton("预测并保存")
        self.__btn_save.setParent(self)
        self.__btn_save.clicked.connect(self.on_btn_save_clicked)
        sub_layout.addWidget(self.__btn_save)

        self.__checkbox_eraser = QCheckBox("  使用橡皮擦")
        self.__checkbox_eraser.setParent(self)
        self.__checkbox_eraser.clicked.connect(self.on_btn_eraser_clicked)
        sub_layout.addWidget(self.__checkbox_eraser)

        self.__label_predict = QLabel(self)
        self.__label_predict.setText("")
        sub_layout.addWidget(self.__label_predict)

        splitter = QSplitter(self)  # 占位符
        sub_layout.addWidget(splitter)

        self.__label_pen_thickness = QLabel(self)
        self.__label_pen_thickness.setText("画笔粗细")
        self.__label_pen_thickness.setFixedHeight(20)
        sub_layout.addWidget(self.__label_pen_thickness)

        self.__spin_box_pen_thickness = QSpinBox(self)
        self.__spin_box_pen_thickness.setMaximum(100)
        self.__spin_box_pen_thickness.setMinimum(0)
        self.__spin_box_pen_thickness.setValue(25)
        self.__spin_box_pen_thickness.setSingleStep(2)  # 最小变化值为2
        self.__spin_box_pen_thickness.valueChanged.connect(
            self.on_pen_thickness_change)  # 关联spinBox值变化信号和函数on_PenThicknessChange
        sub_layout.addWidget(self.__spin_box_pen_thickness)

        main_layout.addLayout(sub_layout)

    def on_pen_thickness_change(self):
        pen_thickness = self.__spin_box_pen_thickness.value()
        self.__paint_board.change_pen_thickness(pen_thickness)

    def on_btn_eraser_clicked(self):
        self.__paint_board.eraser_mode = self.__checkbox_eraser.isChecked()

    def quit(self):
        self.close()

    def on_btn_save_clicked(self):
        name = uuid.uuid1()
        path = rf"./draw/{name}.jpg"
        image = self.__paint_board.get_content_as_image()
        if not os.path.exists("./draw"):
            os.mkdir("./draw")
        image.save(path)

        res = predict(path)
        self.__label_predict.setText(f"预测：{res}")
        os.rename(path, f"./draw/predict_{res}_{name}.jpg")

        self.__paint_board.clear()
