import os

import deeplearning
import gui

src_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(src_path)
os.chdir(root_path)


def main(force_train: tuple[bool, bool] = (False, False), epochs: tuple[int, int] = (10, 10)):
    deeplearning.ensure_model('HWDB', force_train[0], epochs[0])
    deeplearning.ensure_model('MNIST', force_train[1], epochs[1])
    gui.run()


if __name__ == '__main__':
    main()
