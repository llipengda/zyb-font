import os

from deeplearning.HWDB.Train import Train
from deeplearning.HWDB.fetch_data import fetch_data
from deeplearning.HWDB.process_data import load_img_from_gnt
from deeplearning.HWDB.process_mnist_data import load_mnist_data


def run(epochs=10):
    if not os.path.exists('data/HWDB/test') or not os.path.exists('data/HWDB/train') \
            or len(os.listdir('data/HWDB/test')) == 0 or len(os.listdir('data/HWDB/train')) == 0:

        if not os.path.exists('data/HWDB/raw/HWDB1.1trn_gnt') or not os.path.exists('data/HWDB/raw/HWDB1.1tst_gnt') \
                or len(os.listdir('data/HWDB/raw/HWDB1.1trn_gnt')) == 0 \
                or len(os.listdir('data/HWDB/raw/HWDB1.1tst_gnt')) == 0:
            fetch_data()

        load_img_from_gnt()

    train = Train(epochs)
    train(False)


def run_mnist(epochs=10):
    if not os.path.exists('data/HWDB/test') or not os.path.exists('data/HWDB/train') \
            or len(os.listdir('data/HWDB/test')) == 0 or len(os.listdir('data/HWDB/train')) == 0:

        if not os.path.exists('data/HWDB/raw/HWDB1.1trn_gnt') or not os.path.exists('data/HWDB/raw/HWDB1.1tst_gnt') \
                or len(os.listdir('data/HWDB/raw/HWDB1.1trn_gnt')) == 0 \
                or len(os.listdir('data/HWDB/raw/HWDB1.1tst_gnt')) == 0:
            fetch_data()

        load_img_from_gnt()
    
    if not os.path.exists('data/HWDB/MNIST/test') or not os.path.exists('data/HWDB/MNIST/train') \
            or len(os.listdir('data/HWDB/MNIST/test')) == 0 or len(os.listdir('data/HWDB/MNIST/train')) == 0:
        load_mnist_data()

    train = Train(epochs, True)
    train(False)


if __name__ == "__main__":
    run()
