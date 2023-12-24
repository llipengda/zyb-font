import math
import torch
import torch.nn as nn
import torch.nn.functional as f

from typing import Literal


def conv_dw(inp: int, oup: int, stride: Literal[1, 2]):
    return nn.Sequential(
        # dw
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(inplace=True),
        # pw-linear
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
    )


def conv_bn(inp: int, oup: int, stride: Literal[1, 2]):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def layer_init(m):
    # 使用isinstance来判断m属于什么类型
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        # m中的weight，bias其实都是Variable，为了能学习参数以及后向传播
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()


class Module(nn.Module):
    def __init__(self, num_classes: int):
        super(Module, self).__init__()
        self.conv1 = conv_bn(3, 8, 1)  # 64x64x1
        self.conv2 = conv_bn(8, 16, 1)  # 64x64x16
        self.conv3 = conv_dw(16, 32, 1)  # 64x64x32
        self.conv4 = conv_dw(32, 32, 2)  # 32x32x32
        self.conv5 = conv_dw(32, 64, 1)  # 32x32x64
        self.conv6 = conv_dw(64, 64, 2)  # 16x16x64
        self.conv7 = conv_dw(64, 128, 1)  # 16x16x128
        self.conv8 = conv_dw(128, 128, 1)  # 16x16x128
        self.conv9 = conv_dw(128, 128, 1)  # 16x16x128
        self.conv10 = conv_dw(128, 128, 1)  # 16x16x128
        self.conv11 = conv_dw(128, 128, 1)  # 16x16x128
        self.conv12 = conv_dw(128, 256, 2)  # 8x8x256
        self.conv13 = conv_dw(256, 256, 1)  # 8x8x256
        self.conv14 = conv_dw(256, 256, 1)  # 8x8x256
        self.conv15 = conv_dw(256, 512, 2)  # 4x4x512
        self.conv16 = conv_dw(512, 512, 1)  # 4x4x512
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )
        self.__weight_init()

    def forward(self, x: torch.Tensor):
        features = [] # type: list[torch.Tensor]
        x1 = self.conv1(x)
        features.append(x1)
        x2 = self.conv2(x1)
        features.append(x2)
        x3 = self.conv3(x2)
        features.append(x3)
        x4 = self.conv4(x3)
        features.append(x4)
        x5 = self.conv5(x4)
        features.append(x5)
        x6 = self.conv6(x5)
        features.append(x6)
        x7 = self.conv7(x6)
        features.append(x7)
        x8 = self.conv8(x7)
        features.append(x8)
        x9 = self.conv9(x8)
        features.append(x9)
        x9 = f.relu(x8 + x9)
        x10 = self.conv10(x9)
        features.append(x10)
        x11 = self.conv11(x10)
        features.append(x11)
        x11 = f.relu(x10 + x11)
        x12 = self.conv12(x11)
        features.append(x12)
        x13 = self.conv13(x12)
        features.append(x13)
        x14 = self.conv14(x13)
        features.append(x14)
        x14 = f.relu(x13 + x14)
        x15 = self.conv15(x14)
        features.append(x15)
        x16 = self.conv16(x15)
        features.append(x16)
        x = x16.view(x16.size(0), -1)
        x = self.classifier(x)
        return x, features

    def __weight_init(self):
        for layer in self.modules():
            layer_init(layer)
