import os
import torch
import torchvision
import torch.nn.functional as f
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from tqdm import tqdm

from deeplearning.MNIST.Module import Module


class Train:
    BATCH_SIZE_TRAIN = 64
    BATCH_SIZE_TEST = 1000
    LEARNING_RATE = 0.01
    MOMENTUM = 0.5
    LOG_INTERVAL = 10
    RANDOM_SEED = 1
    torch.manual_seed(RANDOM_SEED)

    def __init__(self, epochs: int = 10):
        self.__load_data()
        self.__epochs = epochs

        # for type hint
        assert isinstance(self.__train_loader.dataset,
                          torchvision.datasets.MNIST)
        assert isinstance(self.__test_loader.dataset,
                          torchvision.datasets.MNIST)

        self.__device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print("Train - Using device:", self.__device)

        self.__module = Module().to(self.__device)
        self.__optimizer = optim.SGD(self.__module.parameters(),
                                     lr=Train.LEARNING_RATE, momentum=Train.MOMENTUM)

        self.__train_losses: list[float] = []
        self.__train_counter: list[int] = []
        self.__test_losses: list[float] = []
        self.__test_counter = [i * len(self.__train_loader.dataset)
                               for i in range(self.__epochs + 1)]

    def __load_data(self):
        self.__train_loader = DataLoader(
            torchvision
            .datasets
            .MNIST('data/',
                   train=True,
                   download=True,
                   transform=torchvision.transforms.Compose([
                       torchvision.transforms.ToTensor(),
                       torchvision.transforms.Normalize(
                           (0.1307,), (0.3081,))
                   ])),
            batch_size=Train.BATCH_SIZE_TRAIN,
            shuffle=True
        )

        self.__test_loader = DataLoader(
            torchvision
            .datasets
            .MNIST('data/',
                   train=False,
                   download=True,
                   transform=torchvision.transforms.Compose([
                       torchvision.transforms.ToTensor(),
                       torchvision.transforms.Normalize(
                           (0.1307,), (0.3081,))
                   ])),
            batch_size=Train.BATCH_SIZE_TEST,
            shuffle=True
        )

    def train(self, epoch: int):
        # for type hint
        assert isinstance(self.__train_loader.dataset,
                          torchvision.datasets.MNIST)
        assert isinstance(self.__test_loader.dataset,
                          torchvision.datasets.MNIST)

        self.__module.train()

        bar = tqdm(total=len(self.__train_loader), desc=f'Train {epoch}')

        for batch_idx, (data, target) in enumerate(self.__train_loader):
            data: torch.Tensor
            target: torch.Tensor
            output: torch.Tensor

            data, target = data.to(self.__device), target.to(self.__device)
            output = self.__module(data)
            self.__optimizer.zero_grad()
            loss = f.nll_loss(output, target)
            loss.backward()
            self.__optimizer.step()

            if batch_idx % Train.LOG_INTERVAL == 0:

                bar.update(Train.LOG_INTERVAL)
                bar.set_postfix(loss=f'{loss.item():.6f}')

                self.__train_losses.append(loss.item())
                self.__train_counter.append((batch_idx * 64) +
                                            ((epoch - 1) * len(self.__train_loader.dataset)))
                os.makedirs('out/MNIST', exist_ok=True)
                torch.save(self.__module.state_dict(), 'out/MNIST/model.pth')
                torch.save(self.__optimizer.state_dict(),
                           'out/MNIST/optimizer.pth')

    def test(self):
        # for type hint
        assert isinstance(self.__train_loader.dataset,
                          torchvision.datasets.MNIST)
        assert isinstance(self.__test_loader.dataset,
                          torchvision.datasets.MNIST)

        self.__module.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.__test_loader:
                data: torch.Tensor
                target: torch.Tensor
                output: torch.Tensor
                data, target = data.to(
                    self.__device), target.to(self.__device)
                output = self.__module(data)
                test_loss += f.nll_loss(output,
                                        target, reduction='sum').item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(self.__test_loader.dataset)
        self.__test_losses.append(test_loss)
        print('\n[INFO] Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.__test_loader.dataset),
            100. * correct / len(self.__test_loader.dataset)))

    def __call__(self, show_fig: bool = True):
        # for type hint
        assert isinstance(self.__train_loader.dataset,
                          torchvision.datasets.MNIST)
        assert isinstance(self.__test_loader.dataset,
                          torchvision.datasets.MNIST)

        self.test()

        for epoch in range(1, self.__epochs + 1):
            self.train(epoch)
            self.test()

        if show_fig:
            plt.figure()
            plt.plot(self.__train_counter, self.__train_losses, color='blue')
            plt.scatter(self.__test_counter, self.__test_losses, color='red')
            plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
            plt.xlabel('number of training examples seen')
            plt.ylabel('negative log likelihood loss')
            plt.show()

        examples = enumerate(self.__test_loader)
        _, (example_data, example_targets) = next(examples)

        example_data: torch.Tensor
        example_targets: torch.Tensor
        output: torch.Tensor

        example_data, example_targets = example_data.to(
            self.__device), example_targets.to(self.__device)
        with torch.no_grad():
            output = self.__module(example_data)

        if show_fig:
            plt.figure()
            for i in range(6):
                plt.subplot(2, 3, i + 1)
                plt.tight_layout()
                img = example_data[i][0].cpu().numpy()
                plt.imshow(img, cmap='gray', interpolation='none')
                plt.title("Prediction: {}".format(
                    output.data.max(1, keepdim=True)[1][i].item()))
                plt.xticks([])
                plt.yticks([])
            plt.show()

        print("Train - Done\n")
