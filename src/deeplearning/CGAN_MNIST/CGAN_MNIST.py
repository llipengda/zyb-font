import os
import cv2
import torch

from PIL import Image
from typing import Callable
from torchvision import transforms
from torch.utils.data import Dataset

from deeplearning.CGAN_MNIST.generate_data import generate_data


def read_img(path: str) -> Image.Image:
    return Image.open(path).convert('1')


class CGAN_MNIST(Dataset):
    def __init__(self, transform: Callable[..., torch.Tensor]):
        self.__transform = transform
        self.__load_data()

    def __load_data(self):
        self.__fonts = sorted(os.listdir('fonts'))
        self.__charaters = [str(i) for i in range(10)]
        self.__ensure_data()
        self.__protype_font = 'SIMHEI.TTF'
        self.__fonts.remove(self.__protype_font)
        self.__protype_imgs: list[Image.Image] = [read_img(
            f'data/CGAN_MNIST/{self.__protype_font}/{charater}.png') for charater in self.__charaters] * len(self.__fonts)
        self.__style_imgs: list[Image.Image] = [read_img(
            f'data/CGAN_MNIST/{font}/{charater}.png') for font in self.__fonts for charater in self.__charaters]
        self.__style_indices = [i for i in range(
            len(self.__fonts)) for _ in range(len(self.__charaters))]
        self.__character_indices = [i for _ in range(
            len(self.__fonts)) for i in range(len(self.__charaters))]

    def __ensure_data(self):
        for font in self.__fonts:
            if not os.path.exists(f'data/CGAN_MNIST/{font}'):
                os.makedirs(f'data/CGAN_MNIST/{font}')
                for charater in self.__charaters:
                    generate_data(charater, font, 64)

    def __getitem__(self, index: int):
        transform = self.__transform
        protype_img = transform(self.__protype_imgs[index])
        protype_index = index % len(self.__charaters)
        style_img = transform(self.__style_imgs[index])
        style_index = self.__style_indices[index]
        character_index = self.__character_indices[index]
        real_img = transform(read_img(
            f'data/CGAN_MNIST/{self.__fonts[style_index]}/{self.__charaters[character_index]}.png'))
        
        return (
            protype_img,
            protype_index,
            style_img,
            style_index,
            character_index,
            real_img
        )

    def __len__(self):
        return len(self.__fonts) * len(self.__charaters)


def show_tensor(tensor: torch.Tensor):
    import matplotlib.pyplot as plt
    plt.imshow(tensor.permute((1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.Resize((64, 64)), transforms.ToTensor()]
    )
    dataset = CGAN_MNIST(transform)
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
