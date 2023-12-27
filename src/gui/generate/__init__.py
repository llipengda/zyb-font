from PySide6.QtCore import Slot
from PySide6.QtGui import QPixmap, Qt
from PySide6.QtWidgets import QWidget, QGroupBox, QVBoxLayout, QLabel

from gui.generate.Show import Show
from deeplearning.CGAN_HWDB.Generate import Generate as GenerateModel


class Generate(QWidget):
    def __init__(self):
        super().__init__()
        self.__init_model()

        self.group = QGroupBox()
        self.layout = QVBoxLayout()

        self.__show = Show()
        self.layout.addWidget(self.__show)

        self.group.setLayout(self.layout)

    def __init_model(self):
        self.model = GenerateModel('out/CGAN_HWDB/model.pth.u')
        self.model_2 = GenerateModel('out/CGAN_HWDB/model.pth')

    @Slot(list)
    def generate(self, image_files):
        with open('out/CGAN_HWDB/chars.txt', 'r', encoding='utf-8') as f:
            chars = f.readline()

        font_images = [f'data/CGAN_HWDB/SIMHEI.TTF/{i}.png' for i in chars]

        resize_images = list.copy(image_files)
        while len(resize_images) < len(font_images):
            resize_images += image_files
        resize_images = resize_images[:len(font_images)]

        if resize_images[0].endswith('.png'):
            self.model_2(font_images, resize_images)
        else:
            self.model(font_images, resize_images)

        new_images = [f'gen/{i}.png' for i in chars]
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
