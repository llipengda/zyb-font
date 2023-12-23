import os
import torch
import torch.nn.functional as f
import torchvision.transforms as transforms

from typing import Any
from PIL import Image

from deeplearning.MNIST.Module import Module


class Predict:
    def __init__(self):
        if not os.path.exists("out/MNIST/model.pth"):
            raise FileNotFoundError('out/MNIST/model.pth not found')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("[INFO] Predict - Using device:", device)
        self.__device = device
        module = Module().to(device)
        module.load_state_dict(torch.load("out/MNIST/model.pth"))
        module.eval()
        self.__module = module

    def __call__(self, pic_url: str):
        img = Image.open(pic_url).convert('L') # type: Image.Image
        img = img.resize((64, 64))
        img = img.point(lambda x: 255 - x)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        img_tensor: torch.Tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            output, _  = self.__module(
                img_tensor.to(self.__device))
            output = output.to(self.__device)

        probabilities = f.softmax(output[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()

        res_list = [round(x.item(), 4) * 100 for x in probabilities]
        res_list = [str(x) + '%' for x in res_list]
        res_map = dict(zip(range(10), res_list))

        print("\n[INFO] 预测概率：")
        for k, v in res_map.items():
            print(f"    {k}: {v}")
        print(f"[INFO] 预测结果：{predicted_class}\n")

        return predicted_class
