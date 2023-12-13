from deeplearning.MNIST.Train import Train


def run(epochs=10, show_fig=True):
    train = Train(epochs)
    train(show_fig)


if __name__ == "__main__":
    run()
