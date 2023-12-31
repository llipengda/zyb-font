import os
import torch

from torchvision import transforms
from PIL import Image

from deeplearning.CGAN_HWDB.modules import Generater
from deeplearning.CGAN_HWDB.utils import read_img, remove_black_pixels


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
        

    def __call__(self, protype_paths: list[str], style_paths: list[str]):
        os.makedirs('gen', exist_ok=True)
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        protype_imgs = [read_img(img) for img in protype_paths]
        style_imgs = [read_img(img) for img in style_paths]
        protypes = [transform(i) for i in protype_imgs] # type: list[torch.Tensor]
        styles = [transform(i) for i in style_imgs] # type: list[torch.Tensor]
        protype_tensor = torch.stack(protypes).to(self.__device)
        style_tensor = torch.stack(styles).to(self.__device)
        with torch.no_grad():
            out: torch.Tensor = self.__model(protype_tensor, style_tensor)[0]
        tensors = torch.split(out, 1)
        for i, t in enumerate(tensors):
            normalized = torch.clamp(t.squeeze(0).cpu(), 0, 1)
            PIL_image: Image.Image = transforms.ToPILImage()(normalized)
            PIL_image = remove_black_pixels(PIL_image)
            # PIL_image.show()
            PIL_image.save(f'gen/{protype_paths[i].split("/")[-1]}')


if __name__ == '__main__':
    generate = Generate(
        'out/CGAN_HWDB/model.pth.bak')
    
    with open('out/CGAN_HWDB/chars.txt', 'r') as f:
        chars = f.readline()
    
    p = [f'data/CGAN_HWDB/SIMHEI.TTF/{c}.png' for c in chars]
    s = [f'draw/{i}.jpg' for i in range(10)] * (len(p) // 10)
    
    generate(p, s)
