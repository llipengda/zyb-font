import torch
import torchvision
import torch.nn.functional as f
import torch.optim as optim
import matplotlib.pyplot as plt

from typing import Any
from torch.utils.data import DataLoader

from deeplearning.Module import Module


n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1
torch.manual_seed(random_seed)

train_loader = DataLoader(
    torchvision
    .datasets
    .MNIST('./data/',
           train=True,
           download=True,
           transform=torchvision.transforms.Compose([
               torchvision.transforms.ToTensor(),
               torchvision.transforms.Normalize(
                   (0.1307,), (0.3081,))
           ])),
    batch_size=batch_size_train,
    shuffle=True
)

test_loader = DataLoader(
    torchvision
    .datasets
    .MNIST('./data/',
           train=False,
           download=True,
           transform=torchvision.transforms.Compose([
               torchvision.transforms.ToTensor(),
               torchvision.transforms.Normalize(
                   (0.1307,), (0.3081,))
           ])),
    batch_size=batch_size_test,
    shuffle=True
)

# for type hint
assert isinstance(train_loader.dataset, torchvision.datasets.MNIST)
assert isinstance(test_loader.dataset, torchvision.datasets.MNIST)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

network = Module().to(device)
optimizer = optim.SGD(network.parameters(),
                      lr=learning_rate, momentum=momentum)

train_losses: list[float] = []
train_counter: list[int] = []
test_losses: list[float] = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]


def train(epoch):
    # for type hint
    assert isinstance(train_loader.dataset, torchvision.datasets.MNIST)
    assert isinstance(test_loader.dataset, torchvision.datasets.MNIST)

    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data: torch.Tensor
        target: torch.Tensor
        output: torch.Tensor

        data, target = data.to(device), target.to(device)
        output = network(data)
        optimizer.zero_grad()
        loss = f.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                  .format(epoch,
                          batch_idx * len(data),
                          len(train_loader.dataset),
                          100. * batch_idx /
                          len(train_loader),
                          loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx * 64) +
                                 ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), './model.pth')
            torch.save(optimizer.state_dict(), './optimizer.pth')


def test():
    # for type hint
    assert isinstance(train_loader.dataset, torchvision.datasets.MNIST)
    assert isinstance(test_loader.dataset, torchvision.datasets.MNIST)

    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data: torch.Tensor
            target: torch.Tensor
            output: torch.Tensor
            data, target = data.to(device), target.to(device)
            output = network(data)
            test_loss += f.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # for type hint
    assert isinstance(train_loader.dataset, torchvision.datasets.MNIST)
    assert isinstance(test_loader.dataset, torchvision.datasets.MNIST)

    train(1)
    test()

    for epoch in range(1, n_epochs + 1):
        train(epoch)
        test()

    plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()

    examples = enumerate(test_loader)
    _, (example_data, example_targets) = next(examples)

    example_data: torch.Tensor
    example_targets: torch.Tensor
    output: torch.Tensor

    example_data, example_targets = example_data.to(
        device), example_targets.to(device)
    with torch.no_grad():
        output = network(example_data)
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

    continued_network = Module()
    continued_optimizer = optim.SGD(
        network.parameters(), lr=learning_rate, momentum=momentum)

    network_state_dict: dict[str, Any] = torch.load('model.pth')
    continued_network.load_state_dict(network_state_dict)
    optimizer_state_dict: dict[str, Any] = torch.load('optimizer.pth')
    continued_optimizer.load_state_dict(optimizer_state_dict)

    for i in range(4, 9):
        test_counter.append(i * len(train_loader.dataset))
        train(i)
        test()

    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()


if __name__ == "__main__":
    main()
