import os
import pickle
import torch
import random

from PIL import Image
from typing import Callable
from torchvision import transforms
from torch.utils.data import Dataset

from deeplearning.CGAN_HWDB.generate_data import generate_data
from deeplearning.CGAN_HWDB.utils import read_img, show_tensor

class CGAN_HWDB(Dataset):
    def __init__(self, transform: Callable[..., torch.Tensor], load_characters=-10, load_fonts=15):
        self.__transform = transform
        self.__load_characters = load_characters
        self.__load_fonts = load_fonts
        self.__load_data()

    def __load_data(self):
        self.__fonts = sorted(os.listdir('data/CGAN_HWDB'))[:self.__load_fonts]
        with open('data/HWDB/char_dict', 'rb') as f:
            char_dict: dict[str, str] = pickle.load(f)
        self.__characters = [c for c in char_dict.keys()]
        self.__characters = self.__characters[:self.__load_characters]
        os.makedirs('out/CGAN_HWDB', exist_ok=True)
        with open('out/CGAN_HWDB/chars.txt', 'w+') as file:
            file.writelines(self.__characters)
        self.__protype_font = 'SIMHEI.TTF'
        self.__ensure_data()
        self.__protype_imgs: list[Image.Image] = [read_img(
            f'data/CGAN_HWDB/{self.__protype_font}/{charater}.png') for charater in self.__characters] * len(self.__fonts)
        self.__protype_char_indices = list(
            range(len(self.__characters))) * len(self.__fonts)
        self.__style_imgs: list[Image.Image] = [read_img(
            f'data/CGAN_HWDB/{font}/{charater}.png') for font in self.__fonts for charater in self.__characters]
        self.__style_indices = [i for i in range(
            len(self.__fonts)) for _ in range(len(self.__characters))]
        self.__character_indices = [i for _ in range(
            len(self.__fonts)) for i in range(len(self.__characters))]

    def __ensure_data(self):
        # for font in self.__fonts:
        #     if not os.path.exists(f'data/CGAN_HWDB/{font}') \
        #             or len(os.listdir(f'data/CGAN_HWDB/{font}')) == 0:
        #         os.makedirs(f'data/CGAN_HWDB/{font}', exist_ok=True)
        #         for charater in self.__characters:
        #             generate_data(charater, font, 58)
        if not os.path.exists(f'data/CGAN_HWDB/{self.__protype_font}'):
            for charater in self.__characters:
                generate_data(charater, self.__protype_font, 58)

    def __getitem__(self, index: int):
        transform = self.__transform
        random_style_index = random.randint(0, len(self.__style_indices) - 1)

        # 原型图像
        protype_img = transform(self.__protype_imgs[index])

        # 原型图像的字符索引
        protype_index = self.__protype_char_indices[index]

        # 风格图像
        style_img = transform(self.__style_imgs[random_style_index])

        # 风格图像的风格索引
        style_index = self.__style_indices[random_style_index]

        # 风格图像的字符索引
        character_index = self.__character_indices[random_style_index]

        # 真实图像
        real_img = transform(read_img(
            f'data/CGAN_HWDB/{self.__fonts[style_index]}/{self.__characters[protype_index]}.png'))

        return (
            protype_img,
            protype_index,
            style_img,
            style_index,
            character_index,
            real_img
        )

    def __len__(self):
        return len(self.__fonts) * len(self.__characters)


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.Resize((64, 64)), transforms.ToTensor()]
    )
    dataset = CGAN_HWDB(transform)
    for (
        protype_img,
        protype_index,
        style_img,
        style_index,
        character_index,
        real_img
    ) in dataset:
        print('protype_img')
        show_tensor(protype_img)
        print('style_img')
        show_tensor(style_img)
        print('real_img')
        show_tensor(real_img)
        print('protype_index', protype_index)
        print('style_index', style_index)
        print('character_index', character_index)
        break
