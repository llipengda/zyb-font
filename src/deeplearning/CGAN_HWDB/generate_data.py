import os

from PIL import ImageFont, ImageDraw, Image, ImageOps


def generate_data(word: str, font_name: str, font_size: int):
    os.makedirs(f'data/CGAN_HWDB/{font_name}', exist_ok=True)
    
    font = ImageFont.truetype(f'fonts/{font_name}', font_size)
    
    image = Image.new('L', (64, 64), 255)
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), word, fill=0, font=font)
    
    image = ImageOps.invert(image)
    img_gray = ImageOps.grayscale(image)
    bbox = img_gray.getbbox()
    image = image.crop(bbox)
    new_image = Image.new('L', (font_size, font_size), 255)
    width, height = image.width, image.height
    width = (64 - width) / 2
    height = (64 - height) / 2
    draw = ImageDraw.Draw(new_image)
    draw.bitmap((width, height), image, fill=0)
    image = new_image
    
    image.save(f'data/CGAN_HWDB/{font_name}/{word}.png')
    return image

