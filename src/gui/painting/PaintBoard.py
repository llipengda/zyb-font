from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPixmap, QPainter, QPaintEvent, QMouseEvent, QPen, QColor
from PySide6.QtCore import Qt, QPoint, QSize


class PaintBoard(QWidget):

    def __init__(self):

        super().__init__()

        self.__currentPos = QPoint(0, 0)
        self.__lastPos = QPoint(0, 0)
        self.__is_empty = True
        self.__thickness = 5

        self.__init_data()
        self.__init_view()

    def __init_data(self):

        self.__size = QSize(480, 460)
        self.setGeometry(0, 0, 480, 460)

        self.__board = QPixmap(self.__size)
        self.__board.fill(QColor("white"))

        self.__is_empty = True
        self.eraser_mode = False

        self.__lastPos = QPoint(0, 0)
        self.__currentPos = QPoint(0, 0)
        self.__painter = QPainter()

        self.__thickness = 5
        self.__penColor = QColor("black")

    def __init_view(self):
        self.setFixedSize(self.__size)

    def clear(self):
        self.__board.fill(Qt.GlobalColor.white)
        self.update()
        self.__is_empty = True

    def change_pen_thickness(self, thickness=5):
        self.__thickness = thickness

    def is_empty(self):
        return self.__is_empty

    def get_content_as_image(self):
        image = self.__board.toImage()
        return image

    def paintEvent(self, paint_event: QPaintEvent):
        """绘图事件 重写自QWidget

        Args:
            paint_event (QPaintEvent): UNUSED
        """
        self.__painter.begin(self)

        self.__painter.drawPixmap(0, 0, self.__board)
        self.__painter.end()

    def mousePressEvent(self, mouse_event: QMouseEvent):
        """鼠标按下事件 重写自QWidget

        Args:
            mouse_event (QMouseEvent): 鼠标事件
        """
        self.__currentPos = mouse_event.pos()
        self.__lastPos = self.__currentPos

    def mouseMoveEvent(self, mouse_event: QMouseEvent):
        """鼠标移动事件 重写自QWidget

        Args:
            mouse_event (QMouseEvent): 鼠标事件
        """
        self.__currentPos = mouse_event.pos()
        self.__painter.begin(self.__board)

        if not self.eraser_mode:
            self.__painter.setPen(QPen(self.__penColor, self.__thickness))
        else:
            self.__painter.setPen(QPen(Qt.GlobalColor.white, self.__thickness))

        self.__painter.drawLine(self.__lastPos, self.__currentPos)
        self.__painter.end()
        self.__lastPos = self.__currentPos

        self.update()

    def mouseReleaseEvent(self, mouse_event: QMouseEvent):
        """鼠标释放事件 重写自QWidget

        Args:
            mouse_event (QMouseEvent): UNUSED
        """
        self.__is_empty = False
