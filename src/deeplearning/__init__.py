from deeplearning.__main__ import run
from deeplearning.Pridict import Pridict


pridict_obj = Pridict()


def train_and_test(epochs=10):
    run(epochs, False)


def pridict(pic_url: str):
    return pridict_obj(pic_url)


__all__ = ['train_and_test', 'pridict']
