from deeplearning.MNIST.Predict import Predict
from deeplearning.MNIST.__main__ import run


predict_obj = Predict()


def train_and_test(epochs=10):
    run(epochs, False)


def predict(pic_url: str):
    return predict_obj(pic_url)


__all__ = ['train_and_test', 'predict']
