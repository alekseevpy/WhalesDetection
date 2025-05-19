import requests
import os
from urllib.parse import urlparse
from tqdm import tqdm
import yaml
import py7zr
from PIL import Image
import shutil
import random

def download_archive(url, save_path=None, chunk_size=8192, progress_bar=True):
    try:
        if save_path is None:
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            if not filename:
                raise ValueError("Не удалось определить имя файла из URL")
            save_path = filename

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with requests.get(url, stream=True) as r:
            r.raise_for_status() 
            
            total_size = int(r.headers.get('content-length', 0))

            progress = tqdm(
                total=total_size, 
                unit='B', 
                unit_scale=True, 
                desc=f"Загрузка {os.path.basename(save_path)}", 
                disable=not progress_bar
            )
            
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    progress.update(len(chunk))
            
            progress.close()

        return os.path.abspath(save_path)

    except Exception as e:
        raise type(e)(f"Ошибка при загрузке файла с {url}: {str(e)}") from e


def extract_7z_archive(archive_path, extract_dir=None, get_files_list=False):
    if not os.path.exists(archive_path):
        raise FileNotFoundError(f"Архив не найден: {archive_path}")

    if extract_dir is None:
        base_name = os.path.splitext(os.path.basename(archive_path))[0]
        extract_dir = os.path.join(os.path.dirname(archive_path), base_name)

    os.makedirs(extract_dir, exist_ok=True)

    with py7zr.SevenZipFile(archive_path, mode='r') as archive:
        file_list = archive.getnames()
        archive.extractall(path=extract_dir)

    if get_files_list:
        return file_list
    return None     


def split_whale_data(source_dir, train_ratio=0.8):
    for root, dirs, files in os.walk("../Whales"):
        for file in files:
            if file.endswith(".png"):
                os.remove(os.path.join(root, file))
    
    whales = os.listdir(source_dir)
    for whale in tqdm(whales):
        whale_path = os.path.join(source_dir, whale)
        dates = os.listdir(whale_path)
        random.shuffle(dates)

        total_photos = sum(
            len(os.listdir(os.path.join(whale_path, date)))
            for date in dates
        )
        target_test_photos = int(total_photos * (1 - train_ratio))

        train_dates, test_dates = [], []
        test_photo_count = 0

        for date in dates:
            date_photos = len(os.listdir(os.path.join(whale_path, date)))
            if test_photo_count + date_photos <= target_test_photos:
                test_dates.append(date)
                test_photo_count += date_photos
            else:
                train_dates.append(date)

        if not test_dates:
            test_dates = [random.choice(dates)]
            train_dates = [d for d in dates if d not in test_dates]

        for date in train_dates:
            src = os.path.join(whale_path, date)
            dst = os.path.join("../whales_processed/train", whale, date)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copytree(src, dst)

        for date in test_dates:
            src = os.path.join(whale_path, date)
            dst = os.path.join("../whales_processed/test", whale, date)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copytree(src, dst)


def apply_mask(img, mask_img):
  image = Image.open(img)
  mask = Image.open(mask_img)

  return Image.composite(image, Image.new("RGB", image.size, (0, 0, 0)), mask)


def split_whale_data_with_masks(source_dir, train_ratio=0.8):
    whales = os.listdir(source_dir)
    for whale in tqdm(whales):
        whale_path = os.path.join(source_dir, whale)
        dates = os.listdir(whale_path)
        random.shuffle(dates)

        total_photos = sum(
            len(os.listdir(os.path.join(whale_path, date)))
            for date in dates
        )
        target_test_photos = int(total_photos * (1 - train_ratio))

        train_dates, test_dates = [], []
        test_photo_count = 0

        for date in dates:
            date_photos = len(os.listdir(os.path.join(whale_path, date)))
            if test_photo_count + date_photos <= target_test_photos:
                test_dates.append(date)
                test_photo_count += date_photos
            else:
                train_dates.append(date)

        if not test_dates:
            test_dates = [random.choice(dates)]
            train_dates = [d for d in dates if d not in test_dates]

        for date in train_dates:
            src = os.path.join(whale_path, date)
            dst = os.path.join("../whales_processed/train", whale, date)
            os.makedirs(dst, exist_ok=True)
            src_imgs = os.listdir(src)
            for src_img in src_imgs:
                if src_img.split('.')[1] == 'jpg':
                    stem = src_img.split('.')[0]
                    original_path = os.path.join(src, src_img)
                    mask_path = os.path.join(src, stem + ".png")
                    apply_mask(original_path, mask_path).save(os.path.join(dst, stem + ".jpg"))

        for date in test_dates:
            src = os.path.join(whale_path, date)
            dst = os.path.join("../whales_processed/test", whale, date)
            os.makedirs(dst, exist_ok=True)
            src_imgs = os.listdir(src)
            for src_img in src_imgs:
                if src_img.split('.')[1] == 'jpg':
                    stem = src_img.split('.')[0]
                    original_path = os.path.join(src, src_img)
                    mask_path = os.path.join(src, stem + ".png")
                    apply_mask(original_path, mask_path).save(os.path.join(dst, stem + ".jpg"))


def check_balance(root_dir, min_train_percent=70, max_train_percent=90):
    stats = {}
    deleted_whales = []

    for split in ["train", "test"]:
        split_path = os.path.join(root_dir, split)
        if not os.path.exists(split_path):
            continue

        for whale in os.listdir(split_path):
            if whale not in stats:
                stats[whale] = {"train": {"photos": 0, "dates": 0},
                               "test": {"photos": 0, "dates": 0}}

            whale_path = os.path.join(split_path, whale)
            photo_count = 0
            date_count = 0

            for date in os.listdir(whale_path):
                date_path = os.path.join(whale_path, date)
                if os.path.isdir(date_path):
                    date_count += 1
                    photos = [f for f in os.listdir(date_path)
                             if f.lower().endswith(('.jpg'))]
                    photo_count += len(photos)

            stats[whale][split]["photos"] = photo_count
            stats[whale][split]["dates"] = date_count

    for whale, data in stats.copy().items():
        total_photos = data["train"]["photos"] + data["test"]["photos"]
        total_dates = data["train"]["dates"] + data["test"]["dates"]

        if total_photos == 0 or total_dates <= 2:
            for split in ["train", "test"]:
                whale_path = os.path.join(root_dir, split, whale)
                if os.path.exists(whale_path):
                    shutil.rmtree(whale_path)

            deleted_whales.append(whale)
            del stats[whale]
            continue

        train_percent = (data["train"]["photos"] / total_photos) * 100

        if not (min_train_percent <= train_percent <= max_train_percent):
            for split in ["train", "test"]:
                whale_path = os.path.join(root_dir, split, whale)
                if os.path.exists(whale_path):
                    shutil.rmtree(whale_path)

            deleted_whales.append(whale)
            del stats[whale]

    print("\n{:<15} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        "Кит", "Train фото", "Test фото", "Train %", "Test %", "Даты (train/test)"))
    print("-" * 70)

    for whale, data in stats.items():
        total = data["train"]["photos"] + data["test"]["photos"]
        train_p = (data["train"]["photos"] / total) * 100
        test_p = (data["test"]["photos"] / total) * 100

        print("{:<15} {:<10} {:<10} {:<10.1f}% {:<10.1f}% {:<10}".format(
            whale,
            data["train"]["photos"],
            data["test"]["photos"],
            train_p,
            test_p,
            f"{data['train']['dates']}/{data['test']['dates']}"
        ))

    if deleted_whales:
        print("\nУдаленные папки с плохим балансом или малым числом дат:")
        for whale in deleted_whales:
            print(f"- {whale}")
    else:
        print("\nВсе папки соответствуют критериям!")


def flatten_directory_structure(root_dir):
    for split in ["train", "test"]:
        split_path = os.path.join(root_dir, split)
        if not os.path.exists(split_path):
            continue

        for whale in tqdm(os.listdir(split_path), desc=f"Processing {split}"):
            whale_path = os.path.join(split_path, whale)

            for root, dirs, files in os.walk(whale_path):
                for file in files:
                    if file.lower().endswith(('.jpg')):
                        src_path = os.path.join(root, file)

                        date_folder = os.path.basename(root)
                        new_filename = f"{date_folder}_{file}"

                        dest_path = os.path.join(whale_path, new_filename)

                        shutil.move(src_path, dest_path)

            for root, dirs, files in os.walk(whale_path, topdown=False):
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    if not os.listdir(dir_path):
                        os.rmdir(dir_path)


def keep_common_folders(dataset_path, max_folders_to_keep=8):
    """Оставляет только указанное количество общих папок в train/test"""
    train_path = os.path.join(dataset_path, "train")
    test_path = os.path.join(dataset_path, "test")

    train_folders = set(os.listdir(train_path)) if os.path.exists(train_path) else set()
    test_folders = set(os.listdir(test_path)) if os.path.exists(test_path) else set()

    common_folders = list(train_folders & test_folders)
    print(f"Найдено общих папок: {len(common_folders)}")

    if len(common_folders) < max_folders_to_keep:
        print(f"Внимание: доступно только {len(common_folders)} общих папок")
        max_folders_to_keep = len(common_folders)

    folders_to_keep = random.sample(common_folders, max_folders_to_keep)
    print(f"Сохраняем папки: {', '.join(folders_to_keep)}")

    if os.path.exists(train_path):
        for folder in tqdm(os.listdir(train_path), desc="Обработка train"):
            if folder not in folders_to_keep:
                shutil.rmtree(os.path.join(train_path, folder))
    if os.path.exists(test_path):
        for folder in tqdm(os.listdir(test_path), desc="Обработка test"):
            if folder not in folders_to_keep:
                shutil.rmtree(os.path.join(test_path, folder))


def prepare_data(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    download_info_conf = config['download_info']

    if download_info_conf['archive_downloaded'] == False:
        download_archive(download_info_conf['archive_web_path'], download_info_conf['archive_save_path'])
    if download_info_conf['archive_extracted'] == False:
        extract_7z_archive(download_info_conf['archive_saved_path'], download_info_conf['folder_path'])
    
    datasets_conf = config['datasets']
    if datasets_conf['use_masks']:
      split_whale_data_with_masks(download_info_conf['folder_path'] + '/Whale ReId 2')
    else:
      split_whale_data(download_info_conf['folder_path'] + '/Whale ReId 2')
    check_balance('../whales_processed', datasets_conf['min_pers'], datasets_conf['max_pers'])
    flatten_directory_structure('../whales_processed')
    if datasets_conf['n_folders']['keep_n_folders']:
      keep_common_folders('../whales_processed', datasets_conf['n_folders']['n'])