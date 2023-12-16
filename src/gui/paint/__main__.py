import sys

from typing import Literal
from PyQt5.QtWidgets import QApplication

from gui.paint.Widget import Widget
from deeplearning.MNIST import predict as mnist_predict
from deeplearning.HWDB import predict as hwdb_predict


def run(model_type: Literal['MNIST', 'HWDB', 'HWDB+MNIST']):
    app = QApplication(sys.argv)

    predict_func = mnist_predict() if model_type == 'MNIST' else hwdb_predict()
    main_widget = Widget(predict_func)  # 新建一个主界面
    main_widget.show()  # 显示主界面

    exit(app.exec_())  # 进入消息循环


if __name__ == '__main__':
    run('MNIST')
