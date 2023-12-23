import torch
import torch.nn.functional as F

from torch import nn


class ConvBNRelu(nn.Module):
    def __init__(self, in_c: int, out_c: int, ks=5, pad=2, s=2, bn=False):
        super(ConvBNRelu, self).__init__()
        conv = nn.Conv2d(in_c, out_c, kernel_size=ks, padding=pad, stride=s)
        relu = nn.ReLU(inplace=False)
        self.module = nn.Sequential()
        if bn:
            batchnorm = nn.BatchNorm2d(out_c)
            self.module.add_module("conv", conv)
            self.module.add_module("bn", batchnorm)
            self.module.add_module("relu", relu)
        else:
            self.module.add_module("conv", conv)
            self.module.add_module("relu", relu)

    def forward(self, x):
        return self.module(x)


# 这里的ks到底设置多少，论文里面是5，但是5跑不通，可能跟pad有关？
class DeConvBNRelu(nn.Module):
    def __init__(self, in_c: int, out_c: int, ks=6, pad=2, s=2, bn=False):
        super(DeConvBNRelu, self).__init__()
        conv = nn.ConvTranspose2d(
            in_c, out_c, kernel_size=ks, padding=pad, stride=s
        )
        relu = nn.ReLU(inplace=False)  # 因为计算图有环，会重复计算，不能用inplace
        self.module = nn.Sequential()
        if bn:
            batchnorm = nn.BatchNorm2d(out_c)
            self.module.add_module("conv", conv)
            self.module.add_module("bn", batchnorm)
            self.module.add_module("relu", relu)
        else:
            self.module.add_module("conv", conv)
            self.module.add_module("relu", relu)

    def forward(self, x):
        return self.module(x)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        conv1 = ConvBNRelu(1, 64, bn=True)
        conv2 = ConvBNRelu(64, 128, bn=True)
        conv3 = ConvBNRelu(128, 256, bn=True)
        conv4 = ConvBNRelu(256, 512, bn=True)
        conv5 = ConvBNRelu(512, 512, bn=True)
        conv6 = ConvBNRelu(512, 512, bn=False)
        self.module = nn.ModuleList([
            conv1,
            conv2, conv3, conv4, conv5, conv6])

    def forward(self, x: torch.Tensor) -> 'list[torch.Tensor]':
        out: list[torch.Tensor] = []
        for mod in self.module:
            x = mod(x)
            out.append(x)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ResBlock, self).__init__()
        self.conv1 = ConvBNRelu(in_c, out_c, ks=3, pad=1, s=1, bn=True)
        self.conv2 = ConvBNRelu(out_c, out_c, ks=3, pad=1, s=1, bn=True)

    def forward(self, x) -> torch.Tensor:
        out = self.conv1(x) # type: torch.Tensor
        out = self.conv2(out)
        out = out + x
        return out


class Generater(nn.Module):
    def __init__(self, M=5, num_fonts=4, num_characters=10):
        super(Generater, self).__init__()
        self.left = Encoder()
        self.right = Encoder()
        self.left_1 = nn.Sequential(*[ResBlock(64, 64) for _ in range(M - 4)])
        self.left_2 = nn.Sequential(
            *[ResBlock(128, 128) for i in range(M - 2)])
        self.left_3 = nn.Sequential(*[ResBlock(256, 256) for _ in range(M)])
        self.right_3 = nn.Sequential(*[ResBlock(256, 256) for _ in range(M)])

        self.deconv1 = DeConvBNRelu(1024, 512, bn=False)
        self.deconv2 = DeConvBNRelu(512 * 3, 512, bn=True)
        self.deconv3 = DeConvBNRelu(512 * 3, 256, bn=True)
        self.deconv4 = DeConvBNRelu(256 * 3, 128, bn=True)
        self.deconv5 = DeConvBNRelu(128 * 2, 64, bn=True)
        self.deconv6 = DeConvBNRelu(64 * 2, 1, bn=True)

        # 这两层应该归入Discriminator
        self.fc1 = nn.Linear(512, num_fonts)
        self.fc2 = nn.Linear(512, num_characters)

    def forward(self, lx, rx):
        lout = self.left(lx) # type: list[torch.Tensor]
        rout = self.right(rx) # type: list[torch.Tensor]
        lout_0 = self.left_1(lout[0])
        lout_1 = self.left_2(lout[1])
        lout_2 = self.left_3(lout[2])
        lout_3 = lout[3]
        lout_4 = lout[4]
        lout_5 = lout[5]
        rout_2 = self.right_3(rout[2])
        rout_3 = rout[3]
        rout_4 = rout[4]
        rout_5 = rout[5]
        de_0 = self.deconv1(torch.cat([lout_5, rout_5], dim=1))
        de_1 = self.deconv2(torch.cat([lout_4, de_0, rout_4], dim=1))
        de_2 = self.deconv3(torch.cat([lout_3, de_1, rout_3], dim=1))
        de_3 = self.deconv4(torch.cat([lout_2, de_2, rout_2], dim=1))
        de_4 = self.deconv5(torch.cat([lout_1, de_3], dim=1))
        de_5 = self.deconv6(torch.cat([lout_0, de_4], dim=1)) # type: torch.Tensor

        # cls_left = self.fc1(lout[5])
        # cls_right = self.fc2(rout[5])
        # return de_5, cls_left, cls_right
        return de_5, lout_5, rout_5


# refer to https://github.com/kaonashi-tyc/zi2zi/blob/master/model/unet.py


# Discriminator要改，同时输入三张图片，三张图片在通道上拼接之后再计算，参考以下链接
# refer to  https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/pix2pix/models.py


# todo: 在Discriminator中添加layernorm
class Discriminator(nn.Module):
    def __init__(self, num_fonts=80, num_characters=3500 + 1):
        super(Discriminator, self).__init__()
        self.conv1 = ConvBNRelu(3, 64, ks=5, pad=2, s=2, bn=False)
        self.conv2 = ConvBNRelu(64, 64 * 2, ks=5, pad=2, s=2, bn=True)
        self.conv3 = ConvBNRelu(64 * 2, 64 * 4, ks=5, pad=2, s=2, bn=True)
        self.conv4 = ConvBNRelu(64 * 4, 64 * 8, ks=5, pad=2, s=1, bn=True)
        # self.convs = nn.Sequential(conv1, conv2, conv3, conv4)
        self.fc1 = nn.Linear(512 * 8 * 8, 1)
        self.fc2 = nn.Linear(512 * 8 * 8, num_fonts)
        self.fc3 = nn.Linear(512 * 8 * 8, num_characters)

    def forward(self, x1, x2, x3):
        x = torch.cat([x1, x2, x3], dim=1)
        features: list[torch.Tensor] = []
        x = self.conv1(x)
        features.append(x)
        x = self.conv2(x)
        features.append(x)
        x = self.conv3(x)
        features.append(x)
        x = self.conv4(x)
        features.append(x)
        x = x.view(-1, 512 * 8 * 8)
        x1 = torch.sigmoid(self.fc1(x))  # real or fake
        x2 = F.softmax(self.fc2(x), dim=1)  # font category
        x3 = F.softmax(self.fc3(x), dim=1)  # char category
        return x1, x2, x3, features


class ClSEncoderP(nn.Module):
    def __init__(self, num_characters=10):
        super(ClSEncoderP, self).__init__()
        self.fc = nn.Linear(512, num_characters)

    def forward(self, x):
        return self.fc(x)


class CLSEncoderS(nn.Module):
    def __init__(self, num_fonts=7):
        super(CLSEncoderS, self).__init__()
        self.fc = nn.Linear(512, num_fonts)

    def forward(self, x):
        return self.fc(x)
