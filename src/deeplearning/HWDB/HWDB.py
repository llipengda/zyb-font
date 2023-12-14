import os
import torch
import torchvision

from typing import Callable
from torchvision.datasets.vision import VisionDataset


class HWDB(VisionDataset):
    def __init__(self, root: str, train: bool, transform: Callable | None = None):
        super().__init__(root, transform=transform)

        train_folder = os.path.join(root, "train")
        test_folder = os.path.join(root, "test")

        if train:
            self.__dataset = torchvision.datasets.ImageFolder(
                train_folder, transform=transform)
        else:
            self.__dataset = torchvision.datasets.ImageFolder(
                test_folder, transform=transform)

    def __getitem__(self, index: int):
        image: torch.Tensor
        label: torch.Tensor
        image, label = self.__dataset[index]
        return image, label

    def __len__(self):
        return len(self.__dataset)
