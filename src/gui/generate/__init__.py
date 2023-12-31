from PySide6.QtCore import Slot
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QWidget, QGroupBox, QVBoxLayout, QLabel

from gui.generate.Show import Show
from deeplearning.CGAN_HWDB.Generate import Generate as GenerateModel


class Generate(QWidget):
    def __init__(self):
        super().__init__()
        self.__init_model()

        self.group = QGroupBox()
        self.__layout = QVBoxLayout()

        self.__show = Show()
        self.__layout.addWidget(self.__show)

        self.group.setLayout(self.__layout)

    def __init_model(self):
        self.model = GenerateModel('out/CGAN_HWDB/model.pth')

    @Slot(list)
    def generate(self, image_files):
        font_images = []

        resize_images = list.copy(image_files)
        while len(resize_images) < len(font_images):
            resize_images += image_files
        resize_images = resize_images[:len(font_images)]

        self.model(font_images, resize_images)

        new_images = []
        self.update_pics(new_images)

    def update_pics(self, image_files):
        for i in reversed(range(self.__show.show_layout.count())):
            remove = self.__show.show_layout.itemAt(i).widget()
            self.__show.show_layout.removeWidget(remove)
            remove.setParent(None)
            remove.destroy()

        row, col = 0, 0
        for image_file in image_files:
            pic = QLabel(self)
            pic.setFixedSize(100, 100)
            pic.setText(image_file)
            pixmap = QPixmap(image_file).scaledToWidth(100)
            pic.setPixmap(pixmap)

            self.__show.show_layout.addWidget(pic, row, col)

            col += 1
            if col == 5:
                row += 1
                col = 0

        self.__show.show_area.widget().update()
