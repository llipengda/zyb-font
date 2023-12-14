import os
import pickle
import torch
import torch.nn.functional as f
import torchvision.transforms as transforms

from typing import Any
from PIL import Image

from deeplearning.HWDB.Module import Module
from deeplearning.MNIST import predict


class Predict:
    def __init__(self):
        if not os.path.exists("out/HWDB/model.pth"):
            self.__module = None
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Predict - Using device:", device)
        self.__device = device

        with open('data/HWDB/use_char_dict', 'rb') as f:
            self.__char_dict: dict[str, str] = pickle.load(f)

        module = Module(len(self.__char_dict)).to(device)
        module.load_state_dict(torch.load("out/HWDB/model.pth"))
        module.eval()
        self.__module = module

    def __call__(self, pic_url: str):
        if not self.__module:
            self.__init__()

        assert isinstance(self.__module, Module)

        img = Image.open(pic_url).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
        img_any: Any = transform(img)
        img_tensor: torch.Tensor = img_any
        img = img_tensor.unsqueeze(0)

        with torch.no_grad():
            output: torch.Tensor = self.__module(
                img.to(self.__device)).to(self.__device)

        probabilities = f.softmax(output[0], dim=0)
        predicted_num = torch.argmax(probabilities).item()
        predicted_class = self.__char_dict[f'{predicted_num:05d}']
        
        print(f"[INFO] 预测结果：{predicted_class}\n")

        return predicted_class
