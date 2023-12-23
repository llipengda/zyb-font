import torch

from torchvision import transforms
from PIL import Image

from deeplearning.CGAN_MNIST.modules import Generater
from deeplearning.CGAN_MNIST.CGAN_MNIST import read_img, show_tensor


class Generate:
    def __init__(self, path: str):
        self.__path = path
        self.__load_model()

    def __load_model(self):
        self.__device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print("[INFO] Generate - Using device:", self.__device)
        self.__model = Generater().to(self.__device)
        self.__model.load_state_dict(torch.load(self.__path)['G'])
        self.__model.eval()

    def __call__(self, protype_path: str, style_path: str):
        protype_img = read_img(protype_path)
        style_img = read_img(style_path)
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        protype: torch.Tensor = transform(
            protype_img).unsqueeze(0).to(self.__device)
        style: torch.Tensor = transform(
            style_img).unsqueeze(0).to(self.__device)
        with torch.no_grad():
            out: torch.Tensor = self.__model(protype, style)[0]
            show_tensor(out.squeeze(0).cpu())
            
        normalized = torch.clamp(out.squeeze(0).cpu(), 0, 1)
        PIL_image: Image.Image = transforms.ToPILImage()(normalized)
        PIL_image.show()
        PIL_image.save('out.png')
        


if __name__ == '__main__':
    generate = Generate(
        r'out\CGAN_MNIST\9600-2023-12-24 03-15-15.635496.pth')
    generate(
        r'data/CGAN_MNIST/SIMHEI.TTF/7.png',
        r'draw/predict_3_d1f2194a-9963-11ee-b27e-103d1ccc0fd7.jpg',
    )
