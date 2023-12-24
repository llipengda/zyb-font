import os
import pickle
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import PosixPath
from datetime import datetime
from matplotlib.font_manager import FontProperties
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import writer

from deeplearning.HWDB.Module import Module
from deeplearning.HWDB.HWDB import HWDB

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


class Train:
    BATCH_SIZE_TRAIN = 25
    BATCH_SIZE_TEST = 2000
    LEARNING_RATE = 0.005
    LOG_INTERVAL = 500
    NUM_WORKERS = 16
    WEIGHT_DECAY = 0.001

    def __init__(self, epochs=10, with_mnist=False):
        self.__epochs = epochs
        self.__with_mnist = with_mnist
        self.__load_data()

        self.__device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print("[INFO] Train - Using device:", self.__device)

        self.__module = Module(len(self.__char_dict)).to(self.__device)
        self.__optimizer = torch.optim.SGD(
            self.__module.parameters(), lr=Train.LEARNING_RATE, weight_decay=Train.WEIGHT_DECAY)
        self.__criterion = nn.CrossEntropyLoss()

        log_dir = f'logs/HWDB/{datetime.now()}'
        os.makedirs(log_dir, exist_ok=True)
        self.__writer = writer.SummaryWriter(log_dir)

        if os.path.exists('out/HWDB/model.pth'):
            self.__module.load_state_dict(torch.load('out/HWDB/model.pth'))

        if os.path.exists('out/HWDB/optimizer.pth'):
            self.__optimizer.load_state_dict(
                torch.load('out/HWDB/optimizer.pth'))

        assert isinstance(self.__train_loader.dataset, HWDB) \
            or isinstance(self.__train_loader.dataset, ConcatDataset)

        self.__train_losses: list[float] = []
        self.__train_counter: list[int] = []

    def __load_data(self):
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

        train_dataset = HWDB('data/HWDB', train=True, transform=transform)
        test_dataset = HWDB('data/HWDB', train=False, transform=transform)

        if self.__with_mnist:
            train_dataset = ConcatDataset([
                train_dataset,
                HWDB('data/HWDB/MNIST', train=True, transform=transform)
            ])
            test_dataset = ConcatDataset([
                test_dataset,
                HWDB('data/HWDB/MNIST', train=False, transform=transform)
            ])

        self.__train_loader = DataLoader(
            train_dataset,
            batch_size=Train.BATCH_SIZE_TRAIN,
            num_workers=Train.NUM_WORKERS,
            pin_memory=True,
            shuffle=True
        )

        self.__test_loader = DataLoader(
            test_dataset,
            batch_size=Train.BATCH_SIZE_TEST,
            num_workers=Train.NUM_WORKERS,
            pin_memory=True,
            shuffle=True
        )

        with open('data/HWDB/use_char_dict', 'rb') as f:
            self.__char_dict: dict[str, str] = pickle.load(f)

    def train(self, epoch: int):
        # for type hint
        assert isinstance(self.__train_loader.dataset,
                          HWDB) \
            or isinstance(self.__train_loader.dataset, ConcatDataset)
        assert isinstance(self.__test_loader.dataset,
                          HWDB) \
            or isinstance(self.__test_loader.dataset, ConcatDataset)

        self.__module.train()

        total = 0
        correct = 0
        bar = tqdm(total=len(self.__train_loader) * Train.BATCH_SIZE_TRAIN,
                   desc=f'Train {epoch}')

        for batch_idx, (data, target) in enumerate(self.__train_loader):
            data: torch.Tensor
            target: torch.Tensor
            output: torch.Tensor

            self.__optimizer.zero_grad()
            data, target = data.to(self.__device), target.to(self.__device)
            output, _ = self.__module(data)
            loss: torch.Tensor = self.__criterion(output, target)
            l2_regularization = 0
            for param in self.__module.parameters():
                l2_regularization += torch.norm(param, 2)
            loss += Train.WEIGHT_DECAY * l2_regularization
            total += target.size(0)
            correct += (output.argmax(1) == target).sum().item()
            loss.backward()
            self.__optimizer.step()

            if batch_idx % Train.LOG_INTERVAL == 0:
                if (minus := (batch_idx // Train.LOG_INTERVAL + 1) * Train.LOG_INTERVAL * Train.BATCH_SIZE_TRAIN - len(
                        self.__train_loader) * Train.BATCH_SIZE_TRAIN) > 0:
                    bar.update(Train.LOG_INTERVAL *
                               Train.BATCH_SIZE_TRAIN - minus)
                else:
                    bar.update(Train.LOG_INTERVAL * Train.BATCH_SIZE_TRAIN)
                bar.set_postfix(loss=f'{loss.item():.6f}',
                                avg_correct=f'{(correct / total) * 100.:.6f}%')

                self.__train_losses.append(loss.item())
                self.__train_counter.append((batch_idx * self.BATCH_SIZE_TRAIN) +
                                            ((epoch - 1) * len(self.__train_loader.dataset)))

                self.__writer.add_scalar(
                    'Train/Loss', loss.item(), len(self.__train_counter))
                self.__writer.add_scalar(
                    'Train/Accuracy', (correct / total) * 100., len(self.__train_counter))

                os.makedirs('out/HWDB', exist_ok=True)
                torch.save(self.__module.state_dict(), 'out/HWDB/model.pth')
                torch.save(self.__optimizer.state_dict(),
                           'out/HWDB/optimizer.pth')

        avg_loss = sum(self.__train_losses[
                       -len(self.__train_loader) // Train.LOG_INTERVAL:]) / (
                           len(self.__train_loader) // Train.LOG_INTERVAL)
        bar.set_postfix(loss=f'{avg_loss:.6f}',
                        correct=f'{(correct / total) * 100.:.6f}%')

    def test(self, epoch: int):
        # for type hint
        assert isinstance(self.__train_loader.dataset,
                          HWDB) \
            or isinstance(self.__train_loader.dataset, ConcatDataset)
        assert isinstance(self.__test_loader.dataset,
                          HWDB) \
            or isinstance(self.__test_loader.dataset, ConcatDataset)

        bar = tqdm(total=len(self.__test_loader) *
                   Train.BATCH_SIZE_TEST, desc=f'Test {epoch}')

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
                output, _ = self.__module(data)
                output = output.to(self.__device)
                _, predicted = output.data.max(1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                bar.update(Train.BATCH_SIZE_TEST)
                bar.set_postfix(
                    correct=f'{correct}/{total}({(correct / total) * 100.:.6f}%)')

        self.__writer.add_scalar(
            'Test/Accuracy', (correct / total) * 100., epoch)

    def __call__(self, show_fig: bool = True):
        # for type hint
        assert isinstance(self.__train_loader.dataset,
                          HWDB) \
            or isinstance(self.__train_loader.dataset, ConcatDataset)
        assert isinstance(self.__test_loader.dataset,
                          HWDB) \
            or isinstance(self.__test_loader.dataset, ConcatDataset)

        self.test(0)

        for epoch in range(1, self.__epochs + 1):
            self.train(epoch)
            self.test(epoch)

        os.makedirs('out/HWDB/png/loss', exist_ok=True)
        os.makedirs('out/HWDB/png/prediction', exist_ok=True)
        if self.__train_counter:
            plt.figure()
            plt.plot(self.__train_counter, self.__train_losses, color='blue')
            plt.legend(['Train Loss'], loc='upper right')
            plt.xlabel('number of training examples seen')
            plt.ylabel('negative log likelihood loss')
            if show_fig:
                plt.show()
            plt.savefig(f'out/HWDB/png/loss/{datetime.now()}.png')

        examples = enumerate(self.__test_loader)
        _, (example_data, example_targets) = next(examples)

        example_data: torch.Tensor
        example_targets: torch.Tensor
        output: torch.Tensor

        example_data, example_targets = example_data.to(
            self.__device), example_targets.to(self.__device)
        with torch.no_grad():
            output, _ = self.__module(example_data)

        sim_hei = FontProperties(fname=PosixPath("fonts/SimHei.ttf"))
        for r in range(5):
            plt.figure()
            for i in range(6):
                plt.subplot(2, 3, i + 1)
                plt.tight_layout()
                img = example_data[i + r * 6][0].cpu().numpy()
                plt.imshow(img, cmap='gray', interpolation='none')
                plt.title("Prediction: {}".format(
                    self.__char_dict[f'{output.data.max(1, keepdim=True)[1][i + r * 6].item():05d}']),
                    fontproperties=sim_hei)
                plt.xticks([])
                plt.yticks([])
            if show_fig:
                plt.show()
            plt.savefig(f'out/HWDB/png/prediction/{datetime.now()}.png')
            self.__writer.add_figure('Test/Prediction', plt.gcf(), r)

        self.__writer.close()

        print("[INFO] Train - Done\n")
