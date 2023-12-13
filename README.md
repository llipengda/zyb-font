# zyb-font -- 生成你的手写字体

## 前提

- `Python` `>= 3.11.4 && < 3.12`
- `CUDA` 12.1

## 部署

你可以使用脚本自动部署，也可以手动部署。

### 使用脚本

- Linux

```bash
./init.sh
```

- Windows(CMD)

```bash
.\init.bat
```

- Windows(PowerShell)

```bash
.\init.ps1
```

### 手动部署

创建 `Python` 虚拟环境

- Linux

```bash
python3 -m venv venv
```

- Windows

```bash
python -m venv venv
```

激活虚拟环境

- Linux

```bash
source venv/bin/activate
```

- Windows(PowerShell)

```bash
.\venv\Scripts\activate.ps1
```

- Windows(CMD)

```bash
.\venv\Scripts\activate.bat
```

安装依赖

```bash
pip install -r requirements.txt
```

## 项目概述

项目涉及到多个步骤，包括数据集的获取和预处理、模型的训练、个性化学习以及字体文件的生成。

### 1. 数据集获取和预处理

首先，获取 HWDB 汉字数据集，可以在 [CASIA Online and Offline Chinese Handwriting Databases](http://www.nlpr.ia.ac.cn/databases/handwriting/Home.html) 上找到。然后，你需要编写代码来加载和预处理数据集。

```python
# 伪代码
class HWDBDataset(Dataset):
    def __init__(self, data_path, transform=None):
        # 从数据路径加载数据集
        # 进行必要的预处理，如图像缩放、归一化等
        pass

    def __len__(self):
        # 返回数据集的大小
        pass

    def __getitem__(self, index):
        # 返回单个样本的图像和标签
        pass
```

### 2. 模型训练

使用 PyTorch 定义并训练一个卷积神经网络（CNN）模型，用于识别汉字。选择使用预训练模型，也可以自己设计模型结构。

```python
# 伪代码
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        # 模型结构定义
        pass

    def forward(self, x):
        # 前向传播逻辑
        pass

# 训练循环
for epoch in range(num_epochs):
    for images, labels in dataloader:
        # 前向传播、计算损失、反向传播、优化
        pass
```

### 3. 个性化学习

为了实现个性化学习，需要添加一些用户写字的样本，并将其与原始数据集合并。这可以通过创建一个新的数据集类或修改现有的类来实现。

### 4. 字体文件生成

一旦模型训练好了，使用生成的模型来生成字体文件。通过调整模型的输入，使其生成汉字的图像，然后将图像转换为字体文件。

```python
# 伪代码
def generate_font(model, characters, font_path):
    # 使用模型生成字符图像
    # 将图像转换为字体文件
    pass
```

### 5. Qt 应用程序

使用 PyQt 或 PySide 创建一个图形用户界面（GUI）应用程序，将上述功能整合到应用程序中。

