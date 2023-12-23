import torch
import torch.nn as nn
import torch.nn.functional as f


class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor):
        features = []  # type: list[torch.Tensor]
        x1 = f.relu(f.max_pool2d(self.conv1(x), 2))
        features.append(x1)
        x2 = f.relu(f.max_pool2d(self.conv2(x1), 2))
        features.append(x2)
        x3 = x2.view(-1, 64 * 16 * 16)
        x4 = f.relu(self.fc1(x3))
        x5 = self.fc2(x4)
        return f.log_softmax(x5, dim=1), features
