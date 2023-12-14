import os
import pickle
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as f
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from tqdm import tqdm

from deeplearning.HWDB.Module import Module
from deeplearning.HWDB.HWDB import HWDB


class Train:
    BATCH_SIZE_TRAIN = 1000
    BATCH_SIZE_TEST = 1000
    LEARNING_RATE = 0.01
    LOG_INTERVAL = 1

    def __init__(self, epochs: int = 10):
        self.__epochs = epochs
        self.__load_data()

        self.__device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print("Train - Using device:", self.__device)

        self.__module = Module(len(self.__char_dict)).to(self.__device)
        self.__optimizer = torch.optim.SGD(
            self.__module.parameters(), lr=Train.LEARNING_RATE)
        self.__criterion = nn.CrossEntropyLoss()

        assert isinstance(self.__train_loader.dataset, HWDB)

        self.__train_losses: list[float] = []
        self.__train_counter: list[int] = []
        self.__test_losses: list[float] = []
        self.__test_counter = [i * len(self.__train_loader.dataset)
                               for i in range(self.__epochs + 1)]

    def __load_data(self):
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

        self.__train_loader = DataLoader(
            HWDB('data/HWDB', train=True, transform=transform),
            batch_size=Train.BATCH_SIZE_TRAIN,
            shuffle=True
        )

        self.__test_loader = DataLoader(
            HWDB('data/HWDB', train=False, transform=transform),
            batch_size=Train.BATCH_SIZE_TEST,
            shuffle=True
        )

        with open('data/HWDB/use_char_dict', 'rb') as f:
            self.__char_dict: dict[str, str] = pickle.load(f)

    def train(self, epoch: int):
        # for type hint
        assert isinstance(self.__train_loader.dataset,
                          HWDB)
        assert isinstance(self.__test_loader.dataset,
                          HWDB)

        self.__module.train()

        total = 0
        correct = 0
        bar = tqdm(total=len(self.__train_loader), desc=f'Train {epoch}')

        for batch_idx, (data, target) in enumerate(self.__train_loader):
            data: torch.Tensor
            target: torch.Tensor
            output: torch.Tensor

            self.__optimizer.zero_grad()
            data, target = data.to(self.__device), target.to(self.__device)
            output = self.__module(data)
            loss: torch.Tensor = self.__criterion(output, target)
            total += target.size(0)
            correct += (output.argmax(1) == target).sum().item()
            loss.backward()
            self.__optimizer.step()

            if batch_idx % Train.LOG_INTERVAL == 0:

                bar.update(Train.LOG_INTERVAL)
                bar.set_postfix(loss=f'{loss.item():.6f}',
                                correct=f'{(correct / total) * 100.:.6f}%')

                self.__train_losses.append(loss.item())
                self.__train_counter.append((batch_idx * 64) +
                                            ((epoch - 1) * len(self.__train_loader.dataset)))
                os.makedirs('out/HWDB', exist_ok=True)
                torch.save(self.__module.state_dict(), 'out/HWDB/model.pth')
                torch.save(self.__optimizer.state_dict(),
                           'out/HWDB/optimizer.pth')

    def test(self):
        # for type hint
        assert isinstance(self.__train_loader.dataset,
                          HWDB)
        assert isinstance(self.__test_loader.dataset,
                          HWDB)

        bar = tqdm(total=len(self.__test_loader), desc=f'Test')

        self.__module.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.__test_loader:
                data: torch.Tensor
                target: torch.Tensor
                output: torch.Tensor
                data, target = data.to(
                    self.__device), target.to(self.__device)
                output = self.__module(data)
                output = output.to(self.__device)
                _, predicted = output.data.max(1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                bar.update(1)
                bar.set_postfix(
                    correct=f'{correct}/{total}({(correct / total) * 100.:.6f}%)')

    def __call__(self, show_fig: bool = True):
        # for type hint
        assert isinstance(self.__train_loader.dataset,
                          HWDB)
        assert isinstance(self.__test_loader.dataset,
                          HWDB)

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


if __name__ == "__main__":
    train = Train()
    train()
