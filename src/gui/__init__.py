import sys

from PySide6.QtWidgets import QApplication

from gui.mainwindow import MainWindow


def run():
    app = QApplication(sys.argv)

    window = MainWindow()

    window.resize(1060, 680)
    window.show()

    sys.exit(app.exec())


__all__ = ['run']
