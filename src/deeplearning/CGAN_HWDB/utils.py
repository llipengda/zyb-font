import torch
import matplotlib.pyplot as plt

from PIL import Image


def read_img(path: str) -> Image.Image:
    try:
        return Image.open(path).convert('1')
    except:
        return Image.open(f'data/CGAN_HWDB/STKAITI.TTF/{path.split("/")[-1]}').convert('1')
    
def show_tensor(tensor: torch.Tensor, out_path: str | None = None):
    plt.imshow(tensor.permute((1, 2, 0)))
    plt.show()
    if out_path is not None:
        plt.savefig(out_path)
        
def remove_black_pixels(image: Image.Image):
    width, height = image.size
    for y in range(height):
        for x in range(width):
            pixel: int = image.getpixel((x, y))
            if pixel == 0:
                left_pixel = image.getpixel((x - 1, y)) if x > 0 else None
                right_pixel = image.getpixel((x + 1, y)) if x < width - 1 else None
                top_pixel = image.getpixel((x, y - 1)) if y > 0 else None
                bottom_pixel = image.getpixel((x, y + 1)) if y < height - 1 else None
                if all(p is None or p >= 160 or y == 1 or y == 0 for p in [left_pixel, right_pixel, top_pixel, bottom_pixel]):
                    image.putpixel((x, y), 255)
    return image