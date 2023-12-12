import torch
import torch.nn.functional as f
import torchvision.transforms as transforms

from PIL import Image

from deeplearning.Module import Module


class Pridict:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)
        self.device = device
        module = Module().to(device)
        module.load_state_dict(torch.load("out/model.pth"))
        module.eval()
        self.module = module

    def __call__(self, pic_url: str):
        img = Image.open(pic_url).convert('L')
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        img = transform(img)
        img = img.unsqueeze(0)  # type: ignore

        with torch.no_grad():
            output = self.module(img.to(self.device)).to(self.device)

        probabilities = f.softmax(output[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
        
        return predicted_class
