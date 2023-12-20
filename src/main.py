import os

import deeplearning
import gui

from typing import Literal


src_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(src_path)
os.chdir(root_path)


def main(model_type: Literal['MNIST', 'HWDB', 'HWDB+MNIST'], force_train=False, epochs=10):
    deeplearning.ensure_model(model_type, force_train, epochs)
    gui.run()


if __name__ == '__main__':
    main('MNIST')
