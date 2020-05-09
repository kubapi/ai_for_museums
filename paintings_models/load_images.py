import numpy as np
import pandas as pd

#file managment
import os
import cv2

#progress bar
from tqdm import tqdm

import random

DATADIR = 'C:/Users/kubas/Desktop/paintings'
metadata = pd.read_csv('../paintings_scrape/paintings.csv', sep = '&', encoding ='utf-8', index_col = 0)

#retruns image title based on id that is in the directory
def get_image_title(image_url):
    return metadata[metadata['ID'] == int(image_url.split('.')[0])].values[0,0]

def get_image_path_by_label(label):
    return os.path.join(DATADIR,str(metadata[metadata['Title'] == label].values[0, 2])+".jpg")

def convert_image(image):
    ''' Changes BGR to RGB'''
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def load_data(dir = DATADIR):
    X = []
    labels = []

    # to change if more computing power
    IMG_SIZE = 224

    # to see which one are not working
    exceptions = []

    print("Loading images from directory: ", dir)
    for image in tqdm(os.listdir(dir)):
        try:
            X.append(cv2.resize(cv2.imread(os.path.join(dir, image)), (IMG_SIZE, IMG_SIZE)))
            labels.append(get_image_title(image))
        except:
            exceptions.append(image)

    # reshaping data
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    labels = np.array(labels)

    #saving data for further use
    np.save('data/X_data.npy', X)
    np.save('data/labels_data.npy', labels)
    print("Data saved!")

if __name__ == '__main__':
    load_data(DATADIR)
