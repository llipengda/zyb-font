import os
import pickle
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib.font_manager import FontProperties

from deeplearning.HWDB.Module import Module
from deeplearning.HWDB.HWDB import HWDB

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


class Train:
    BATCH_SIZE_TRAIN = 100
    BATCH_SIZE_TEST = 2000
    LEARNING_RATE = 0.01
    LOG_INTERVAL = 10

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

        if os.path.exists('out/HWDB/model.pth'):
            self.__module.load_state_dict(torch.load('out/HWDB/model.pth'))

        if os.path.exists('out/HWDB/optimizer.pth'):
            self.__optimizer.load_state_dict(
                torch.load('out/HWDB/optimizer.pth'))

        assert isinstance(self.__train_loader.dataset, HWDB)

        self.__train_losses: list[float] = []
        self.__train_counter: list[int] = []

    def __load_data(self):
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

        self.__train_loader = DataLoader(
            HWDB('data/HWDB', train=True, transform=transform),
            batch_size=Train.BATCH_SIZE_TRAIN,
            num_workers=24,
            pin_memory=True,
            shuffle=True
        )

        self.__test_loader = DataLoader(
            HWDB('data/HWDB', train=False, transform=transform),
            batch_size=Train.BATCH_SIZE_TEST,
            num_workers=24,
            pin_memory=True,
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
        bar = tqdm(total=len(self.__train_loader) *
                   Train.BATCH_SIZE_TRAIN, desc=f'Train {epoch}')

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

                bar.update(Train.LOG_INTERVAL * Train.BATCH_SIZE_TRAIN)
                bar.set_postfix(loss=f'{loss.item():.6f}',
                                correct=f'{(correct / total) * 100.:.6f}%')

                self.__train_losses.append(loss.item())
                self.__train_counter.append((batch_idx * self.BATCH_SIZE_TRAIN) +
                                            ((epoch - 1) * len(self.__train_loader.dataset)))

                os.makedirs('out/HWDB', exist_ok=True)
                torch.save(self.__module.state_dict(), 'out/HWDB/model.pth')
                torch.save(self.__optimizer.state_dict(),
                           'out/HWDB/optimizer.pth')

        avg_loss = sum(self.__train_losses[
            -len(self.__train_loader):]) / len(self.__train_loader)
        bar.set_postfix(loss=f'{avg_loss:.6f}',
                        correct=f'{(correct / total) * 100.:.6f}%')

    def test(self):
        # for type hint
        assert isinstance(self.__train_loader.dataset,
                          HWDB)
        assert isinstance(self.__test_loader.dataset,
                          HWDB)

        bar = tqdm(total=len(self.__test_loader) *
                   Train.BATCH_SIZE_TEST, desc=f'Test')

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
                bar.update(Train.BATCH_SIZE_TEST)
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
            plt.legend(['Train Loss'], loc='upper right')
            plt.xlabel('number of training examples seen')
            plt.ylabel('negative log likelihood loss')
            plt.show()
            plt.savefig('out/HWDB/loss.png')

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
            sim_hei = FontProperties(fname="font/SimHei.ttf")
            plt.figure()
            for i in range(6):
                plt.subplot(2, 3, i + 1)
                plt.tight_layout()
                img = example_data[i][0].cpu().numpy()
                plt.imshow(img, cmap='gray', interpolation='none')
                plt.title("Prediction: {}".format(
                    self.__char_dict[f'{output.data.max(1, keepdim=True)[1][i].item():05d}']),
                        fontproperties=sim_hei)
                plt.xticks([])
                plt.yticks([])
            plt.show()
            plt.savefig('out/HWDB/prediction.png')

        print("Train - Done\n")
