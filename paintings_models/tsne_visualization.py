import numpy as np
import pandas as pd

from sklearn.manifold import TSNE

from PIL import Image

from tqdm import tqdm
from load_images import get_image_path_by_label

def tsne_map(tsne_features):
    labels = np.load("data/labels_data.npy")

    tx, ty = tsne_features[:,0], tsne_features[:,1]
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

    full_image.save("tSNE.png")

if __name__ == '__main__':
    #using PCA or LSA top components (t-SNE doesn't like big number of featurers)
    print("Calculating TSNE features!")
    tsne_features = TSNE(n_components=2).fit_transform(np.load("data/pca_features.npy"))
    tsne_map(tsne_features)
