from progress.bar import Bar
from multiprocessing import Pool
from pathlib import Path
import requests
import pandas as pd
from tornado import ioloop, httpclient

colors = ['red','green','blue','yellow']
DIR = Path("../data/HPAv18_train/")
DIR.mkdir(exist_ok=True, parents=True)
v18_url = 'http://v18.proteinatlas.org/images/'
imgList = pd.read_csv("../data/HPAv18RBGY_wodpl.csv")

def job(data):
    url, img_name = data
    r = requests.get(url, allow_redirects=True)
    open(DIR / img_name, 'wb').write(r.content)

url_and_names = []
for i in imgList['Id']: # [:5] means downloard only first 5 samples, if it works, please remove it
    img = i.split('_')
    for color in colors:
        img_path = img[0] + '/' + "_".join(img[1:]) + "_" + color + ".jpg"
        img_name = i + "_" + color + ".jpg"
        img_url = v18_url + img_path
        url_and_names.append((img_url, img_name))
       
pool = Pool()
bar = Bar("Processing", max=len(url_and_names))
for i in pool.imap(job, url_and_names):
    bar.next()
bar.finish()
