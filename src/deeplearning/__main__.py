from deeplearning.Train import Train


def run(epochs=10, show_fig=True):
    train = Train(epochs)
    train.run(show_fig)


if __name__ == "__main__":
    run()
