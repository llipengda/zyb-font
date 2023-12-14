from deeplearning.MNIST.Predict import Predict
from deeplearning.MNIST.__main__ import run


def train_and_test(epochs=10):
    run(epochs, False)


def predict():
    return Predict()


__all__ = ['train_and_test', 'predict']
