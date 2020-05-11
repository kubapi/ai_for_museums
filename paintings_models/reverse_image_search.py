import os
os.environ["KMP_WARNINGS"] = "FALSE"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm
import cv2

import random
from scipy.spatial import distance

#pca and truncatedSVD for dimensionality reduction
from sklearn.decomposition import PCA, TruncatedSVD

import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input

import h5py
from keras.models import Model, load_model

from load_images import get_image_path_by_label

#local path to files
path_extracted_features = "data/extracted_features.npy"
path_lsa_features = "data/lsa_features.npy"
path_pca_features = "data/pca_features.npy"
path_rasta_model = "data/rasta_model.h5"

def load_VGG16_feature_extractor():
    model = keras.applications.VGG16(weights='imagenet', include_top=True)
    #returns last before output layer that serves as a feature extractor
    return Model(inputs=model.input, outputs=model.get_layer("fc2").output)

def load_RASTA_feature_extractor():
    model = load_model(path_rasta_model)
    return Model(inputs=model.input, output=model.get_layer("add_16").output)

def clear_data():
    '''
    Removes features extracted for vgg16, PCA and LSA data
    '''
    os.remove(path_extracted_features_vgg16)
    os.remove(path_lsa_features)
    os.remove(path_pca_features)

def pre_extract_features(X):
    '''
    Extracts features using pre-trained model, as input accepts feature_extractor (VGG16 default) and numpy array of images
    '''
    print("Starting feature extraction")
    feature_extractor = load_RASTA_feature_extractor()
    if os.path.exists(path_extracted_features):
        print("Pre-traiend features found!")
        return feature_extractor, np.load(path_extracted_features)
    else:
        pre_features = [feature_extractor.predict(np.expand_dims(x, 0))[0] for x in tqdm(X)]
        return feature_extractor, np.array(pre_features)

def get_models_and_features(pre_features, lsa_n_components = 100, pca_n_components = 50):
    '''
    Extracts features using PCA, LSA methods as a final extraction
    Returns [lsa_model, lsa_features, pca_model, pca_features]
    '''
    print("Extracting PCA and LSA features!")
    lsa = TruncatedSVD(n_components=lsa_n_components, algorithm='randomized')
    lsa_features = lsa.fit_transform(pre_features)
    np.save(path_lsa_features, lsa_features)

    pca = PCA(n_components=pca_n_components)
    pca_features = pca.fit_transform(pre_features)
    np.save(path_pca_features, pca_features)

    return lsa, lsa_features, pca, pca_features

def reverse_search(image_url, feature_extractor, model, features, N, labels):
    '''
    returns image paths for DATADIR folder
    '''
    # preprocessing on given image
    image =  cv2.resize(cv2.imread(os.path.join(image_url)), (224, 224))
    np.array(image).reshape(-1, 224, 224, 3)

    new_features = model.transform(feature_extractor.predict(np.expand_dims(image, 0)))[0]
    distances = [ distance.cosine(new_features, feature) for feature in features ]
    closests = sorted(range(len(distances)), key=lambda k: distances[k])[0:N]
    return [get_image_path_by_label(labels[idx]).replace("\\", "/") for idx in closests]

if __name__ == '__main__':
    #loading labels and X
    X = np.load('data/X_data.npy')
    labels = np.load('data/labels_data.npy')

    # loading extractor and applying extraction
    feature_extractor, pre_features = pre_extract_features(X)

    #loading and using pca and lsa
    lsa, lsa_features, pca, pca_features = get_models_and_features(pre_features)
    # assert "8745170876.jpg" in reverse_search("data/test.jpg", feature_extractor, pca, pca_features, 5, labels)[0] == True, "Problem with reverse search!"
    print(reverse_search("data/test.jpg", feature_extractor, pca, pca_features, 5, labels))
    # print(reverse_search("king.jpg", feature_extractor, pca, pca_features, 5, labels))
