import os

from typing import Literal

import deeplearning.MNIST as MNIST
import deeplearning.HWDB as HWDB


def ensure_model(model_type: Literal['MNIST', 'HWDB'], force_train: bool = False, epochs: int = 10):
    model_path = f'out/{model_type}/model.pth'
    if not os.path.exists(model_path) or force_train:
        if model_type == 'MNIST':
            MNIST.train_and_test(epochs)
        elif model_type == 'HWDB':
            HWDB.train_and_test(epochs)
        else:
            raise ValueError(f'Unknown model type: {model_type}')


__all__ = ['ensure_model']
