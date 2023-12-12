from gui.paint.Widget import Widget
from PyQt5.QtWidgets import QApplication

import sys


def run():
    app = QApplication(sys.argv)
    main_widget = Widget()  # 新建一个主界面
    main_widget.show()  # 显示主界面

    exit(app.exec_())  # 进入消息循环


if __name__ == '__main__':
    run()
