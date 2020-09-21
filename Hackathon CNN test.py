# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 14:46:36 2020

@author: Abhijeet
"""


import keras
from keras.preprocessing import image
from glob import glob
import cv2, os
import numpy as np
import matplotlib.pyplot as plt

path = r'C:\Users\Abhijeet\Desktop\Skin dataset\data\test\benign'
path1 = r'C:\Users\Abhijeet\Desktop\Skin dataset\data\test\malignant'
ROW, COL = 96, 96
benigns, malignants = [], []
y_benigns, y_malignants = [], []

def load_benigns():
    print('Loading all benign images\n')
    benign_path = os.path.join(path, '*g')
    for benign_img in glob(benign_path):
        benign = cv2.imread(benign_img)
        benign = cv2.cvtColor(benign, cv2.COLOR_BGR2GRAY)
        benign = cv2.resize(benign, (ROW, COL))
        benign = image.img_to_array(benign)
        benigns.append(benign)
    print('All benign images loaded')
    

load_benigns()
def load_malignants():
    print('Loading all malignant images\n')
    malignant_path = os.path.join(path1, '*g')
    for malignant_img in glob(malignant_path):
        malignant = cv2.imread(malignant_img)
        
        malignant = cv2.cvtColor(malignant, cv2.COLOR_BGR2GRAY)
        malignant = cv2.resize(malignant, (ROW, COL))
        malignant = image.img_to_array(malignant)
        malignants.append(malignant)
    print('All malignant images loaded')
load_malignants()

y_benigns = [1 for item in enumerate(benigns)]
y_malignants = [0 for item in enumerate(malignants)]


benigns = np.asarray(benigns).astype('float32')
malignants = np.asarray(malignants).astype('float32')
y_benigns = np.asarray(y_benigns).astype('int32')
y_malignants = np.asarray(y_malignants).astype('int32')
benigns /= 255
malignants /= 255

X = np.concatenate((benigns,malignants), axis=0)
y = np.concatenate((y_benigns, y_malignants), axis=0)

from keras.models import load_model 

model = load_model(r'C:\Users\Abhijeet\Desktop\Skin dataset\my_modelCNN.h5')
pred = model.predict_classes(X) 



from sklearn.metrics import accuracy_score 
print(accuracy_score(y, pred)) 
