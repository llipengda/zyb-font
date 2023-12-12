import struct
from pathlib import Path
from PIL import Image
from multiprocessing import Pool

save_dir = '../data/HWDB/train'


def process_gnt_file(gnt_paths):
    label_list = []
    for gnt_path in gnt_paths:
        count = 0
        print(f'gnt路径--->{gnt_path}')

        with open(str(gnt_path), 'rb') as f:
            while f.read(1) != "":
                f.seek(-1, 1)
                count += 1
                try:
                    # 按类型提取gnt格式文件中的数据
                    length_bytes = struct.unpack('<I', f.read(4))[0]

                    tag_code = f.read(2)

                    width = struct.unpack('<H', f.read(2))[0]

                    height = struct.unpack('<H', f.read(2))[0]

                    im = Image.new('RGB', (width, height))
                    img_array = im.load()  # 返回像素值
                    for x in range(height):
                        for y in range(width):
                            # 读取像素值
                            pixel = struct.unpack('<B', f.read(1))[0]
                            # 赋值
                            img_array[y, x] = (pixel, pixel, pixel)

                    filename = str(count) + '.png'

                    # 转换为中文的格式
                    tag_code = tag_code.decode('gbk').strip('\x00')
                    save_path = f'{save_dir}/{gnt_path.stem}'
                    if not Path(save_path).exists():
                        Path(save_path).mkdir(parents=True, exist_ok=True)
                    im.save(f'{save_path}/{filename}')

                    # 保存格式为：文件路径/文件名 图片中的文字 : 1290-c/563.png	兼
                    label_list.append(f'{gnt_path.stem}/{filename}\t{tag_code}')
                except Exception as e:
                    print(f"break because of exception:{e}")
                    break

    return label_list


def write_txt(save_path: str, content: list, mode='w'):
    """
    将list内容写入txt中
    @param
    content: list格式内容
    save_path: 绝对路径str
    @return:None
    """
    with open(save_path, mode, encoding='utf-8') as f:
        for value in content:
            print(f'value--->{value}')
            f.write(value + '\n')


if __name__ == '__main__':
    # 数据读取路径
    path = "../data/raw/HWDB1.1trn_gnt"

    gnt_paths = list(Path(path).iterdir())
    print(f'gnt_paths--->{gnt_paths}')

    # 将 gnt_paths 分割成子列表
    n = 10
    gnt_sublists = [gnt_paths[i:i + n] for i in range(0, len(gnt_paths), n)]
    print(f'gnt_sublists--->{gnt_sublists}')

    # 多进程处理
    with Pool(10) as p:
        label_list = p.map(process_gnt_file, gnt_sublists)
        print(f'label_list--->{label_list}')

    # 合并子列表
    label_list = [item for sublist in label_list for item in sublist]
    print(f'label_list--->{label_list}')

    write_txt(f'{save_dir}/zf_gnt_train.txt', label_list)
