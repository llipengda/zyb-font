import os

from typing import Literal

import deeplearning.MNIST as MNIST
import deeplearning.HWDB as HWDB


def ensure_model(model_type: Literal['MNIST', 'HWDB', 'HWDB+MNIST'], force_train: bool = False, epochs: int = 10):
    model_path = f'out/{model_type}/model.pth'
    if model_type == 'HWDB+MNIST':
        model_path = f'out/HWDB/model.pth'
    if not os.path.exists(model_path) or force_train:
        if model_type == 'MNIST':
            MNIST.train_and_test(epochs)
        elif model_type == 'HWDB':
            HWDB.train_and_test(epochs)
        elif model_type == 'HWDB+MNIST':
            HWDB.train_and_test(epochs)
            HWDB.train_and_test_mnist(epochs)


__all__ = ['ensure_model']
