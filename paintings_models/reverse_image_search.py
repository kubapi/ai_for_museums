import os
os.environ["KMP_WARNINGS"] = "FALSE"

import logging
import tensorflow as tf
logging.getLogger('tensorflow').disabled = True


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pk

from tqdm import tqdm
import cv2

import random
from scipy.spatial import distance

#pca and truncatedSVD for dimensionality reduction
from sklearn.decomposition import PCA, TruncatedSVD

import keras
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16

import h5py
from keras.models import Model, load_model

from load_images import get_image_path_by_label

#local path to files
path_extracted_features = "data/ext_feat/extracted_features_"

path_dim_red_features = "data/dim_red/features_"
path_dim_red_models = "data/dim_red/model_"

path_rasta_model = "data/models/rasta_model.h5"

def load_VGG16_feature_extractor():
    model = VGG16(weights='imagenet', include_top = True)
    model.name = "vg116"
    return model.name, Model(inputs=model.input, outputs=model.get_layer("fc2").output)

def load_RESNET50_feature_extractor():
    model = ResNet50(weights='imagenet', include_top = True)
    model.name = "resnet50"
    return model.name, Model(inputs=model.input, outputs=model.get_layer("avg_pool").output)

def load_RASTA_feature_extractor():
    model = load_model(path_rasta_model)
    model.name="rasta"
    #returns penultimate layer that serves as a feature extractor
    return model.name, Model(inputs=model.input, output=model.get_layer("global_average_pooling2d_1").output)

def feature_extraction(X, loaded_feature_extractor):
    '''
    Extracts features using pre-trained model, as input accepts feature_extractor (VGG16 default) and numpy array of images
    '''
    print("Starting feature extraction")
    name, feature_extractor = loaded_feature_extractor
    path = path_extracted_features+name+".npy"
    if os.path.exists(path):
        print("Pre-traiend features found!", path)
        return name, feature_extractor, np.load(path)
    else:
        pre_features = [feature_extractor.predict(np.expand_dims(x, 0))[0] for x in tqdm(X)]
        np.save(path_extracted_features+name+".npy", pre_features)
        return name, feature_extractor, np.array(pre_features)

def dim_red(name ,pre_features, n):
    '''
    Extracts features using PCA, LSA methods as a final extraction
    Returns [lsa_model, lsa_features, pca_model, pca_features]
    '''
    print("Extracting PCA and LSA features from ", name)

    lsa = TruncatedSVD(n_components=n, algorithm='randomized')
    lsa_features = lsa.fit_transform(pre_features)
    np.save(path_dim_red_features+"lsa_"+name+"_"+str(n)+".npy", lsa_features)
    pk.dump(lsa, open(path_dim_red_models+"lsa_"+name+"_"+str(n)+".pkl","wb"))

    pca = PCA(n_components=n)
    pca_features = pca.fit_transform(pre_features)
    np.save(path_dim_red_features+"pca_"+name+"_"+str(n)+".npy", lsa_features)
    pk.dump(pca, open(path_dim_red_models+"pca_"+name+"_"+str(n)+".pkl","wb"))


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

    # # loading extractor and applying extraction
    name1, feature_extractor1, pre_features1 = feature_extraction(X, load_RESNET50_feature_extractor())
    name2, feature_extractor2, pre_features2 = feature_extraction(X, load_RASTA_feature_extractor())
    name3, feature_extractor3, pre_features3 = feature_extraction(X, load_VGG16_feature_extractor())

    names = [name1, name2, name3]
    features = [pre_features1, pre_features2, pre_features3]
    for feature in features:
        print(feature.shape)
    values = [25, 50, 100]

    for e, name in enumerate(names):
        for value in values:
            dim_red(name, features[e], value)


    # print(reverse_search("data/test.jpg", feature_extractor, pca, pca_features, 5, labels))
