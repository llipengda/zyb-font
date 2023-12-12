from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPixmap, QPainter, QPaintEvent, QMouseEvent, QPen, QColor
from PyQt5.QtCore import Qt, QPoint, QSize


class PaintBoard(QWidget):

    def __init__(self, parent=None):

        super().__init__(parent)

        self.__currentPos = QPoint(0, 0)
        self.__lastPos = QPoint(0, 0)
        self.__is_empty = True
        self.__thickness = 10

        self.__init_data()  # 先初始化数据，再初始化界面
        self.__init_view()

    def __init_data(self):

        self.__size = QSize(480, 460)

        # 新建QPixmap作为画板，尺寸为__size
        self.__board = QPixmap(self.__size)
        self.__board.fill(QColor("white"))  # 用白色填充画板

        self.__is_empty = True  # 默认为空画板
        self.eraser_mode = False  # 默认为禁用橡皮擦模式

        self.__lastPos = QPoint(0, 0)  # 上一次鼠标位置
        self.__currentPos = QPoint(0, 0)  # 当前的鼠标位置

        self.__painter = QPainter()  # 新建绘图工具

        self.__thickness = 25  # 默认画笔粗细为25px
        self.__penColor = QColor("black")  # 设置默认画笔颜色为黑色

    def __init_view(self):
        # 设置界面的尺寸为__size
        self.setFixedSize(self.__size)

    def clear(self):
        # 清空画板
        self.__board.fill(Qt.GlobalColor.white)
        self.update()
        self.__is_empty = True

    def change_pen_thickness(self, thickness=25):
        # 改变画笔粗细
        self.__thickness = thickness

    def is_empty(self):
        # 返回画板是否为空
        return self.__is_empty

    def get_content_as_image(self):
        # 获取画板内容（返回QImage）
        image = self.__board.toImage()
        return image

    def paintEvent(self, paint_event: QPaintEvent):
        """绘图事件 重写自QWidget

        Args:
            paint_event (QPaintEvent): UNUSED
        """
        self.__painter.begin(self)
        # 0,0为绘图的左上角起点的坐标，__board即要绘制的图
        self.__painter.drawPixmap(0, 0, self.__board)
        self.__painter.end()

    def mousePressEvent(self, mouse_event: QMouseEvent):
        """鼠标按下事件 重写自QWidget

        Args:
            mouse_event (QMouseEvent): 鼠标事件
        """
        # 鼠标按下时，获取鼠标的当前位置保存为上一次位置
        self.__currentPos = mouse_event.pos()
        self.__lastPos = self.__currentPos

    def mouseMoveEvent(self, mouse_event: QMouseEvent):
        """鼠标移动事件 重写自QWidget

        Args:
            mouse_event (QMouseEvent): 鼠标事件
        """
        # 鼠标移动时，更新当前位置，并在上一个位置和当前位置间画线
        self.__currentPos = mouse_event.pos()
        self.__painter.begin(self.__board)

        if not self.eraser_mode:
            # 非橡皮擦模式
            self.__painter.setPen(
                QPen(self.__penColor, self.__thickness))  # 设置画笔颜色，粗细
        else:
            # 橡皮擦模式下画笔为纯白色，粗细为10
            self.__painter.setPen(QPen(Qt.GlobalColor.white, self.__thickness))

        # 画线
        self.__painter.drawLine(self.__lastPos, self.__currentPos)
        self.__painter.end()
        self.__lastPos = self.__currentPos

        self.update()  # 更新显示

    def mouseReleaseEvent(self, mouse_event: QMouseEvent): 
        """鼠标释放事件 重写自QWidget

        Args:
            mouse_event (QMouseEvent): UNUSED
        """
        self.__is_empty = False  # 画板不再为空
