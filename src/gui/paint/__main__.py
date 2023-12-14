import sys

from typing import Literal
from PyQt5.QtWidgets import QApplication

from gui.paint.Widget import Widget
from deeplearning.MNIST import predict


def run(type: Literal['MNIST', 'HWDB']):
    app = QApplication(sys.argv)
    
    # TODO: HWDB predict
    predict_func = predict() if type == 'MNIST' else predict()
    main_widget = Widget(predict_func)  # 新建一个主界面
    main_widget.show()  # 显示主界面

    exit(app.exec_())  # 进入消息循环


if __name__ == '__main__':
    run('MNIST')
