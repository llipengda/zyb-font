import sys

from typing import Literal
from PyQt5.QtWidgets import QApplication

from gui.paint.Widget import Widget
from deeplearning.MNIST import predict as MNIST_predict
from deeplearning.HWDB import predict as HWDB_predict


def run(type: Literal['MNIST', 'HWDB']):
    app = QApplication(sys.argv)

    predict_func = MNIST_predict() if type == 'MNIST' else HWDB_predict()
    main_widget = Widget(predict_func)  # 新建一个主界面
    main_widget.show()  # 显示主界面

    exit(app.exec_())  # 进入消息循环


if __name__ == '__main__':
    run('MNIST')
