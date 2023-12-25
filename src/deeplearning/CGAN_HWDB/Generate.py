import torch

from torchvision import transforms
from torchvision.utils import save_image
from typing import overload
from PIL import Image

from deeplearning.CGAN_HWDB.modules import Generater
from deeplearning.CGAN_HWDB.utils import read_img, remove_black_pixels, show_tensor


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
        
    @overload
    def __call__(self, protype_path: str, style_path: str) -> None: 
        ...
        
    @overload
    def __call__(self, protype_path: list[str], style_path: list[str]) -> None:
        ...

    def __call__(self, protype_path: str | list[str], style_path: str | list[str]):
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        if type(protype_path) is str and type(style_path) is str:
            protype_img = read_img(protype_path)
            style_img = read_img(style_path)
            protype: torch.Tensor = transform(
                protype_img).unsqueeze(0).to(self.__device)
            style: torch.Tensor = transform(
                style_img).unsqueeze(0).to(self.__device)
            with torch.no_grad():
                out: torch.Tensor = self.__model(protype, style)[0]
            show_tensor(out.squeeze(0).cpu())
                
            normalized = torch.clamp(out.squeeze(0).cpu(), 0, 1)
            PIL_image: Image.Image = transforms.ToPILImage()(normalized)
            PIL_image = remove_black_pixels(PIL_image)
            PIL_image.show()
            PIL_image.save('out.png')
        else:
            protype_imgs = [read_img(img) for img in protype_path]
            style_imgs = [read_img(img) for img in style_path]
            protypes = [transform(i) for i in protype_imgs] # type: list[torch.Tensor]
            styles = [transform(i) for i in style_imgs] # type: list[torch.Tensor]
            protype_tensor = torch.stack(protypes).to(self.__device)
            style_tensor = torch.stack(styles).to(self.__device)
            with torch.no_grad():
                out: torch.Tensor = self.__model(protype_tensor, style_tensor)[0]
            save_image(out, '1.png')


if __name__ == '__main__':
    generate = Generate(
        'out/CGAN_HWDB/model.pth')
    
    with open('out/CGAN_HWDB/chars.txt', 'r') as f:
        chars = f.readline()
    
    p = [f'data/CGAN_HWDB/SIMHEI.TTF/{c}.png' for c in chars]
    s = ['draw/predict_水_2723ead8-9b02-11ee-89ae-103d1ccc0fd7.jpg'] * len(p)

    generate(p, s)
