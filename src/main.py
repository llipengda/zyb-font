import os

import deeplearning
import gui


if __name__ == "__main__":
    if not os.path.exists("out/model.pth"):
        deeplearning.train_and_test(10)

    gui.paint.run()
