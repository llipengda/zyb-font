import os
import struct
import pickle
import threading
import numpy as np

from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from typing import Any, BinaryIO

TRAIN_IMG_CNT = 897758
TEST_IMG_CNT = 223991
CHAR_CNT = 3755


# 处理单个gnt文件获取图像与标签
def read_from_gnt_dir(gnt_dir: str):
    def one_file(file: BinaryIO):
        header_size = 10
        while True:
            header = np.fromfile(file, dtype=np.dtype(
                'uint8'), count=header_size)
            if not header.size:
                break
            sample_size: int = header[0] + \
                (header[1] << 8) + (header[2] << 16) + (header[3] << 24)
            _label: int = header[5] + (header[4] << 8)
            width: int = header[6] + (header[7] << 8)
            height: int = header[8] + (header[9] << 8)
            if header_size + width * height != sample_size:
                break
            _image = np.fromfile(file, dtype=np.dtype(
                'uint8'), count=width * height).reshape((height, width))
            yield _image, _label

    for file_name in os.listdir(gnt_dir):
        if file_name.endswith('.gnt'):
            file_path = os.path.join(gnt_dir, file_name)
            with open(file_path, 'rb') as f:
                for image, label in one_file(f):
                    yield image, label


def gnt_to_img(gnt_dir: str, img_dir: str, char_dict: dict[str, str]):
    def save_img(_label: int, _image: np.ndarray[Any, np.dtype[np.uint8]], _counter: int):
        decoded_label = struct.pack('>H', _label).decode('gb2312')
        img = Image.fromarray(_image)
        dir_name = os.path.join(img_dir, char_dict[str(decoded_label)])
        os.makedirs(dir_name, exist_ok=True)
        img.convert('RGB').save(dir_name + '/' + str(_counter) + '.png')

    mode = 'train' if 'train' in img_dir else 'test'
    total = TRAIN_IMG_CNT if mode == 'train' else TEST_IMG_CNT
    bar = tqdm(total=total, desc=f'Converting {mode} gnt to img')
    counter = 0
    thread_pool = ThreadPoolExecutor(20)
    for image, label in read_from_gnt_dir(gnt_dir):
        thread_pool.submit(save_img, label, image, counter)
        counter += 1
        bar.update(1)
    thread_pool.shutdown()
    bar.close()


def generate_char_dict(data_dir: str, test_gnt_dir: str):
    char_dict_path = os.path.join(data_dir, 'char_dict')
    use_char_dict_path = os.path.join(data_dir, 'use_char_dict')

    if not os.path.exists(char_dict_path) or not os.path.exists(use_char_dict_path):
        print('Generating char dict...')

        all_imgs = TEST_IMG_CNT
        bar = tqdm(total=all_imgs, desc='Generating char dict')

        char_set = set()
        for _, tagcode in read_from_gnt_dir(gnt_dir=test_gnt_dir):
            tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
            char_set.add(tagcode_unicode)
            bar.update(1)
        bar.close()

        char_list = list(char_set)
        char_range = ['%0.5d' % x for x in range(len(char_list))]
        char_dict = dict(zip(sorted(char_list), char_range))
        use_char_dict = {v: k for k, v in char_dict.items()}

        with open(char_dict_path, 'wb') as f:
            pickle.dump(char_dict, f)
        with open(use_char_dict_path, 'wb') as f:
            pickle.dump(use_char_dict, f)

    else:
        with open(char_dict_path, 'rb') as f:
            char_dict: dict[str, str] = pickle.load(f)

    return char_dict


def load_img_from_gnt():
    # 路径
    gnt_dir = 'data/HWDB/raw'
    data_dir = 'data/HWDB/'
    train_gnt_dir = os.path.join(gnt_dir, 'HWDB1.1trn_gnt')
    test_gnt_dir = os.path.join(gnt_dir, 'HWDB1.1tst_gnt')
    train_img_dir = os.path.join(data_dir, 'train')
    test_img_dir = os.path.join(data_dir, 'test')
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)

    char_dict = generate_char_dict(data_dir, test_gnt_dir)

    if len(os.listdir(train_img_dir)) == 0 or len(os.listdir(test_img_dir)) == 0:
        print('Converting gnt to img...')

        train_thread = threading.Thread(
            target=gnt_to_img, args=(train_gnt_dir, train_img_dir, char_dict))
        test_thread = threading.Thread(
            target=gnt_to_img, args=(test_gnt_dir, test_img_dir, char_dict))

        train_thread.start()
        test_thread.start()

        train_thread.join()
        test_thread.join()


if __name__ == "__main__":
    load_img_from_gnt()
