import os
import zipfile
import patoolib

from DownloadKit import DownloadKit
from tqdm import tqdm


RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

TRAIN_PATH = "data/HWDB/train/"
TEST_PATH = "data/HWDB/test/"
TRAIN_GNT_PATH = "data/HWDB/raw/HWDB1.1trn_gnt/"
TEST_GNT_PATH = "data/HWDB/raw/HWDB1.1tst_gnt/"


def download():
    print(
        YELLOW + '[WARN]  This script will download 2.23GB data from the Internet.' + RESET)
    print("Downloading data...\n")

    download_kit = DownloadKit(roads=20)

    raw_path = "data/HWDB/raw/"
    os.makedirs(raw_path, exist_ok=True)

    train_url = "http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1trn_gnt.zip"
    test_url = "http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1tst_gnt.zip"

    download_kit.add(
        train_url, raw_path, file_exists='overwrite')
    download_kit.add(
        test_url, raw_path, file_exists='overwrite')
    download_kit.wait(show=True)

    print("Download complete.")


def is_unalz_installed():
    paths = os.environ['PATH'].split(os.pathsep)
    for path in paths:
        return os.path.exists(os.path.join(path, 'unalz.exe')) or os.path.exists(os.path.join(path, 'unalz'))
    return False


def unzip_alz(file_path: str, output_path: str):
    print("Unzipping alz file... (This may take a while)")
    patoolib.extract_archive(file_path, outdir=output_path)


def unzip():
    print("Unzipping data...")

    train_zip_path = "data/HWDB/raw/HWDB1.1trn_gnt.zip"
    train_output_path = "data/HWDB/raw/"

    test_zip_path = "data/HWDB/raw/HWDB1.1tst_gnt.zip"
    test_output_path = TEST_GNT_PATH

    with zipfile.ZipFile(train_zip_path, 'r') as train_zip:
        train_files = train_zip.namelist()
        for file in tqdm(train_files, desc="Unzipping Training Data"):
            train_zip.extract(file, train_output_path)

    alz_path = "data/HWDB/raw/HWDB1.1trn_gnt.alz"
    unzip_alz(alz_path, TRAIN_GNT_PATH)

    with zipfile.ZipFile(test_zip_path, 'r') as test_zip:
        test_files = test_zip.namelist()
        for file in tqdm(test_files, desc="Unzipping Testing Data"):
            test_zip.extract(file, test_output_path)

    os.remove(train_zip_path)
    os.remove(test_zip_path)
    os.remove(alz_path)

    print("Unzip complete.\n")


def fetch_data():
    if os.path.exists(TRAIN_PATH) and os.path.exists(TEST_PATH) \
            and len(os.listdir(TRAIN_PATH)) > 0 and len(os.listdir(TEST_PATH)) > 0:
        return

    if os.path.exists(TRAIN_GNT_PATH) and os.path.exists(TEST_GNT_PATH) \
            and len(os.listdir(TRAIN_GNT_PATH)) > 0 and len(os.listdir(TEST_GNT_PATH)) > 0:
        return

    if not is_unalz_installed():
        print(
            f'{RED}[ERROR] unalz is not installed. Please install it first.{RESET}')
        exit(1)

    download()
    unzip()


if __name__ == "__main__":
    fetch_data()
