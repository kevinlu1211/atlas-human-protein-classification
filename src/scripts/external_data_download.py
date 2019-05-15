import requests
import pandas as pd
from miniutils.progress_bar import parallel_progbar, progbar

from src.training.data import DataPaths

COLORS = ['red', 'green', 'blue', 'yellow']
SAVE_PATH = DataPaths.TRAIN_IMAGES_HPAv18
SAVE_PATH.mkdir(exist_ok=True, parents=True)
HPAv18_URL = 'http://v18.proteinatlas.org/images/'
IMG_IDS = pd.read_csv(DataPaths.TRAIN_LABELS_HPAv18)["Id"]


def get_names_and_urls_of_img():
    names = []
    urls = []
    for img_id in progbar(IMG_IDS):
        folder_name, *file_name = img_id.split('_')
        for color in COLORS:
            img_path = f"{folder_name}/{'_'.join(file_name)}_{color}.jpg"
            img_name = f"{img_id}_{color}.jpg"
            img_url = HPAv18_URL + img_path
            urls.append(img_url)
            names.append(img_name)
    return names, urls


def download_from_url(data):
    name, url = data
    r = requests.get(url, allow_redirects=True)
    open(SAVE_PATH / name, 'wb').write(r.content)


names, urls = get_names_and_urls_of_img()
names_and_urls = zip(names, urls)
parallel_progbar(download_from_url, names_and_urls)

print("Finished downloading all external images!")
print(f"These images are saved in: {SAVE_PATH}")
