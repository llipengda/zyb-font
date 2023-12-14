import os

from typing import Literal

import deeplearning.MNIST as MNIST
import deeplearning.HWDB as HWDB


def ensure_module(type: Literal['MNIST', 'HWDB'], epochs: int = 10):
    module_path = f'out/{type}/model.pth'
    if not os.path.exists(module_path):
        if type == 'MNIST':
            MNIST.train_and_test(epochs)
        else:
            HWDB.train_and_test(epochs)


__all__ = ['ensure_module']
