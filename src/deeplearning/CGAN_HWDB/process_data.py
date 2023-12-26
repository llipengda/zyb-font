import os
import struct

from PIL import Image, ImageOps, ImageDraw, ImageEnhance
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


from deeplearning.HWDB.process_data import read_from_gnt_dir

def process_data():
    gnt_dir = 'data/HWDB/raw/HWDB1.1trn_gnt'
    thread_pool = ThreadPoolExecutor(12)
    bar = tqdm(total=897758, desc='[INFO] Processing data')
    for image, label, writer in read_from_gnt_dir(gnt_dir):
        thread_pool.submit(save_img, image, label, writer)
        bar.update(1)
    thread_pool.shutdown()
        
def save_img(image, label, writer):
    decoded_label = struct.pack('>H', label).decode('gb2312')
    img = Image.fromarray(image)
    img = ImageOps.invert(img)
    w, h = int(img.width * 0.7), int(img.height * 0.7)
    img = img.resize((w, h))
    img = ImageEnhance.Contrast(img).enhance(5)
    img_gray = ImageOps.grayscale(img)
    bbox = img_gray.getbbox()
    img = img.crop(bbox)
    new_image = Image.new('L', (64, 64), 255)
    width, height = img.width, img.height
    width = (64 - width) // 2
    height = (64 - height) // 2
    draw = ImageDraw.Draw(new_image)
    draw.bitmap((width, height), img, fill=0)
    img = new_image
    dir_name = f'data/CGAN_HWDB/{writer}'
    os.makedirs(dir_name, exist_ok=True)
    img.convert('L').save(f'{dir_name}/{decoded_label}.png')
    
        
if __name__ == '__main__':
    process_data()