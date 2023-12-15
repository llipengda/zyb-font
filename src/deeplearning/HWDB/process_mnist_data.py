import os
import pickle
import concurrent.futures

from concurrent.futures import ThreadPoolExecutor
from torchvision import datasets, transforms
from tqdm import tqdm
from PIL import Image


def load_mnist_data():
    # if os.path.exists('data/HWDB/MNIST/train') and os.path.exists('data/HWDB/MNIST/test') \
    #         and len(os.listdir('data/HWDB/MNIST/train')) == 10 and len(os.listdir('data/HWDB/MNIST/test')) == 10:
    #     return

    train_dataset = datasets.MNIST(root='./data',
                                   train=True,
                                   download=True,
                                   transform=transforms.Compose([
                                       transforms.Resize((64, 64)),
                                       transforms.ToTensor()
                                   ]))
    test_dataset = datasets.MNIST(root='./data',
                                  train=False,
                                  transform=transforms.Compose([
                                      transforms.Resize((64, 64)),
                                      transforms.ToTensor()
                                  ]))
    
    with open('data/HWDB/use_char_dict', 'rb') as f:
        tmp_dict: dict[str, str] = pickle.load(f)
        print(tmp_dict)
        ignore = [str(i) for i in range(10)]
        tmp_dict = {k: v for k, v in tmp_dict.items() if v not in ignore}
        offset = len(tmp_dict)

    train_folders = [f'data/HWDB/MNIST/train/{i + offset}' for i in range(10)]
    test_folders = [f'data/HWDB/MNIST/test/{i + offset}' for i in range(10)]

    with open('data/HWDB/use_char_dict', 'rb') as f:
        use_char_dict: dict[str, str] = pickle.load(f)
        for i in range(10):
            use_char_dict[f'{i + offset:05d}'] = str(i)
            
    with open('data/HWDB/use_char_dict', 'wb') as f:
        pickle.dump(use_char_dict, f)

    with open('data/HWDB/char_dict', 'rb') as f:
        char_dict: dict[str, str] = pickle.load(f)
        for i in range(10):
            char_dict[str(i)] = f'{i + offset:05d}'
            
    with open('data/HWDB/char_dict', 'wb') as f:
        pickle.dump(char_dict, f)

    for folder in train_folders + test_folders:
        os.makedirs(folder, exist_ok=True)

    bar = tqdm(total=len(train_dataset.data) +
               len(test_dataset.data), desc='Process MNIST data')

    def process_image(index, data, target, folders):
        label = int(target.item())
        folder = folders[label]
        image = Image.fromarray(data.numpy(), mode='L')
        image = image.resize((64, 64))
        image = image.point(lambda x: 255 - x)
        image.save(f'{folder}/{index}.png')
        bar.update()

    with ThreadPoolExecutor(max_workers=8) as executor:
        # 处理训练集
        futures_train = [executor.submit(process_image, i, data, target, train_folders)
                         for i, (data, target) in enumerate(zip(train_dataset.data, train_dataset.targets))]

        # 处理测试集
        futures_test = [executor.submit(process_image, i, data, target, test_folders)
                        for i, (data, target) in enumerate(zip(test_dataset.data, test_dataset.targets))]

        concurrent.futures.wait(futures_train + futures_test)


if __name__ == "__main__":
    load_mnist_data()
