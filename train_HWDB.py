import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from typing import Any, Dict
from torch.utils.data import DataLoader

# 训练次数
n_epochs = 3
# 每次训练的数据量
batch_size_train = 64
# 测试数据量
batch_size_test = 1000
# 学习率
learning_rate = 0.01
# 动量
momentum = 0.5
# 日志间隔
log_interval = 10
# 随机种子
random_seed = 1
torch.manual_seed(random_seed)

transform = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        (0.1307,), (0.3081,)
    )
])

train_loader = DataLoader(
    torchvision
        .datasets
        .MNIST('./data/',
               train=True,
               download=True,
               transform=transform
               ),
    batch_size=batch_size_train,
    shuffle=True
)

test_loader = DataLoader(
    torchvision
        .datasets
        .MNIST('./data/',
               train=False,
               download=True,
               transform=transform
               ),
    batch_size=batch_size_test,
    shuffle=True
)

# for type hint
assert isinstance(train_loader.dataset, torchvision.datasets.MNIST)
assert isinstance(test_loader.dataset, torchvision.datasets.MNIST)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

fig = plt.figure()
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        input: torch.Tensor = self.fc2(x)
        return F.log_softmax(input, dim=1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

network = Net().to(device)
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
        loss = F.nll_loss(output, target)
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
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


train(1)

test()

for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()

fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

example_data: torch.Tensor
example_targets: torch.Tensor
output: torch.Tensor

example_data, example_targets = example_data.to(
    device), example_targets.to(device)
with torch.no_grad():
    output = network(example_data)
fig = plt.figure()
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

# ----------------------------------------------------------- #

continued_network = Net()
continued_optimizer = optim.SGD(
    network.parameters(), lr=learning_rate, momentum=momentum)

network_state_dict: Dict[str, Any] = torch.load('model.pth')
continued_network.load_state_dict(network_state_dict)
optimizer_state_dict: Dict[str, Any] = torch.load('optimizer.pth')
continued_optimizer.load_state_dict(optimizer_state_dict)

# 注意不要注释前面的“for epoch in range(1, n_epochs + 1):”部分，
# 不然报错：x and y must be the same size
# 为什么是“4”开始呢，因为n_epochs=3，上面用了[1, n_epochs + 1)
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
