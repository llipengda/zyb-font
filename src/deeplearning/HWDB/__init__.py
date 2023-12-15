from deeplearning.HWDB.Predict import Predict
from deeplearning.HWDB.__main__ import run, run_mnist


def train_and_test(epochs=10):
    run(epochs)


def train_and_test_mnist(epochs=10):
    run_mnist(epochs)


def predict():
    return Predict()


__all__ = ['train_and_test', 'predict', 'train_and_test_mnist']
