import os

import deeplearning
import gui

from typing import Literal


src_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(src_path)
os.chdir(root_path)


def main(type: Literal['MNIST', 'HWDB'], force_train: bool = False, epochs: int = 10):
    deeplearning.ensure_module(type, force_train, epochs)
    gui.paint.run(type)


if __name__ == '__main__':
    main('HWDB')
