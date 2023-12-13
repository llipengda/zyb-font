import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib
from PIL import Image

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from typing import Any, Dict
from torch.utils.data import DataLoader


class HWDBDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')
        label = int(img_path.split('/')[-2])  # 如果label是文件夹名字，可以使用这种方式获取

        if self.transform:
            image = self.transform(image)

        return image, label


# Normalization parameters for pre-trained PyTorch models
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Define transforms
train_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)])

test_transform = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean, std)])

# Define custom dataset
train_dataset = HWDBDataset(train_image_paths, transform=train_transform)
test_dataset = HWDBDataset(test_image_paths, transform=test_transform)

# Define data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)



# Load the pretrained model
model = models.resnet18(pretrained=True)

# Change the final layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3755)

# Choose device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Move model to the device
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
for epoch in range(5):  # num_epochs需要您根据实际情况设定
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{5} Loss: {loss.item()}")

