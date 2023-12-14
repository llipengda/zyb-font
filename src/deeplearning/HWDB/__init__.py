from deeplearning.HWDB.Predict import Predict
from deeplearning.HWDB.__main__ import run


def train_and_test(epochs=10):
    run(epochs)


def predict():
    return Predict()


__all__ = ['train_and_test', 'predict']
