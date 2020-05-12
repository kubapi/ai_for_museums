import numpy as np
import pandas as pd

from sklearn.manifold import TSNE

from PIL import Image
import os

from tqdm import tqdm
from load_images import get_image_path_by_label

def tsne_map(dim_red_features, title):
    labels = np.load("data/labels_data.npy")

    tx, ty = dim_red_features[:,0], dim_red_features[:,1]
    tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

    width = 4000
    height = 3000
    max_dim = 100

    full_image = Image.new('RGBA', (width, height))
    iter = 0
    for x, y in zip(tx, ty):
        tile = Image.open(get_image_path_by_label(labels[iter].replace("\\", "/")))
        rs = max(1, tile.width/max_dim, tile.height/max_dim)
        tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
        full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))
        iter += 1

    full_image.save("tsne_results/"+title)

if __name__ == '__main__':
    perplexities = [5, 15, 25, 30, 35, 50]
    for path_dim_red_features in tqdm(os.listdir("data/dim_red")):
        for perplexity in perplexities:
            dim_red_features = TSNE(n_components=2, perplexity=perplexity, n_iter = 5000).fit_transform(np.load("data/dim_red/"+path_dim_red_features))
            tsne_map(dim_red_features, title = str(path_dim_red_features.split(".")[0]+"_tsne.png"))
