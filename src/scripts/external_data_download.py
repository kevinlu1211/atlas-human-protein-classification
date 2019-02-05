import sys
from multiprocessing import Pool
from pathlib import Path

from progress.bar import Bar
import requests
import pandas as pd
from miniutils.progress_bar import parallel_progbar, progbar

from src.data import DataPaths

COLORS = ['red', 'green', 'blue', 'yellow']
SAVE_PATH = "tmp"
SAVE_PATH.mkdir(exist_ok=True, parents=True)
HPAv18_URL = 'http://v18.proteinatlas.org/images/'
IMG_IDS = pd.read_csv(DataPaths.TRAIN_LABELS_HPAv18)["Id"]

def get_url_and_names_of_img():
    urls = []
    names = []
    for img_id in progbar(IMG_IDS):
        folder_name, *file_name = img_id.split('_')
        for color in COLORS:
            img_path = f"{folder_name}/{'_'.join(file_name)}_{color}.jpg"
            img_name = f"{img_id}_{color}.jpg"
            img_url = HPAv18_URL + img_path
            urls.append(img_url)
            names.append(img_name)
    return urls, names

def download_from_url(data):
    url, name = data
    r = requests.get(url, allow_redirects=True)
    open(SAVE_PATH / name, 'wb').write(r.content)


urls, names = get_url_and_names_of_img()
iterable = zip(urls, names)
parallel_progbar(download_from_url, iterable)
