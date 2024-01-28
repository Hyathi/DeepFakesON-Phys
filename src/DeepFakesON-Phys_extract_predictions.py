
import numpy as np
import os
import cv2
from imageio import imread
from skimage.transform import resize
import tensorflow.keras as keras
from tensorflow.keras.models import Model,Sequential,load_model
import pandas as pd
import h5py
import glob
import sys
import scipy.io
import time 
import argparse

def load_test_motion(carpeta):
    X_test = []
    images_names = []
    image_path = carpeta
    print(f"Reading test images from: {image_path}")
    
    for imagen in os.listdir(image_path):
        imagenes = os.path.join(image_path, imagen)
        print(f"Image: {imagenes}")
        img = cv2.resize(cv2.imread(imagenes, cv2.IMREAD_COLOR), (36, 36))
        img = img.transpose((-1, 0, 1))
        X_test.append(img)
        images_names.append(imagenes)
        
    return X_test, images_names

def load_test_attention(carpeta):
    X_test = []
    images_names = []
    image_path = carpeta
    print(f"Reading test images from: {image_path}")
    
    for imagen in os.listdir(image_path):
        imagenes = os.path.join(image_path, imagen)
        print(f"Image: {imagenes}")
        img = cv2.resize(cv2.imread(imagenes, cv2.IMREAD_COLOR), (36, 36))
        img = img.transpose((-1, 0, 1))
        X_test.append(img)
        images_names.append(imagenes)
        
    return X_test, images_names

np.set_printoptions(threshold=np.inf)
data = []
batch_size = 128
model = load_model('../models/DeepFakesON-Phys_CelebDF_V2.h5')
print(model.summary())
input("Press Enter to continue...")

parser = argparse.ArgumentParser("Predict test videos")
parser.add_argument('input_dir', type=str, help="path to directory with videos")
args = parser.parse_args()
input_dir = args.input_dir

image_path = os.path.join(input_dir)
print(f"image_path: {image_path}")
carpeta_deep= os.path.join(image_path, "DeepFrames")
print(f"carpeta_deep: {carpeta_deep}")
carpeta_raw= os.path.join(image_path, "RawFrames")
print(f"carpeta_raw: {carpeta_raw}")

input("Press Enter to proceed with predictions...")
test_data, images_names = load_test_motion(carpeta_deep)
test_data2, images_names = load_test_attention(carpeta_raw)
#print(f"test_data with load_test_motion: {test_data}")
#print(f"test_data2 with load_test_attention: {test_data2}")

input("Press Enter to proceed with predictions...")
test_data = np.array(test_data, copy=False, dtype=np.float32)
test_data2 = np.array(test_data2, copy=False, dtype=np.float32)
#print(f"test_data with np.array: {test_data}")
#print(f"test_data2 with np.array: {test_data2}")

input("Press Enter to proceed with predictions...")
# if data is empty, show error
if len(test_data) == 0:
    print(f"test_data Error: No images found in the provided path: {image_path}")
if len(test_data2) == 0:
    print(f"test_data2 Error: No images found in the provided path: {image_path}")
predictions = model.predict([test_data, test_data2], batch_size=batch_size, verbose=1)
bufsize = 1
nombre_fichero_scores = 'deepfake_scores.txt'
fichero_scores = open(nombre_fichero_scores,'w',buffering=bufsize)
fichero_scores.write("img;score\n")
for i in range(predictions.shape[0]):
    fichero_scores.write("%s" % images_names[i]) #fichero
    # if float(predictions[i])<0:
        # predictions[i]='0'
    # elif float(predictions[i])>1:
        # predictions[i]='1'
    fichero_scores.write(";%s\n" % predictions[i]) #scores predichas