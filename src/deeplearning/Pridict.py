import torch
import torch.nn.functional as f
import torchvision.transforms as transforms

from typing import Any
from PIL import Image

from deeplearning.Module import Module


class Pridict:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Pridict - Using device:", device)
        self.device = device
        module = Module().to(device)
        module.load_state_dict(torch.load("out/model.pth"))
        module.eval()
        self.module = module

    def __call__(self, pic_url: str):
        img = Image.open(pic_url).convert('L')
        img = img.resize((28, 28))
        img = img.point(lambda x: 255 - x)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        img_any: Any = transform(img)
        img_tensor: torch.Tensor = img_any
        img = img_tensor.unsqueeze(0)

        with torch.no_grad():
            output: torch.Tensor = self.module(
                img.to(self.device)).to(self.device)

        probabilities = f.softmax(output[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()

        res_list = [round(x.item(), 4) * 100 for x in probabilities]
        res_list = [str(x) + '%' for x in res_list]
        res_map = dict(zip(range(10), res_list))

        print("\n预测概率：")
        for k, v in res_map.items():
            print(f"    {k}: {v}")
        print(f"预测结果：{predicted_class}\n")

        return predicted_class
