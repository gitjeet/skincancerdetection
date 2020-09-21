import keras
from keras.preprocessing import image
from glob import glob
import cv2, os
import numpy as np
import matplotlib.pyplot as plt

path = r'C:\Users\Abhijeet\Desktop\Skin dataset\data\train\benign'
path1 = r'C:\Users\Abhijeet\Desktop\Skin dataset\data\train\malignant'
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

X = X.reshape(2637,96*96)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

import pickle

# save
with open('modelrandomforest.pkl','wb') as f4:
    pickle.dump(classifier,f4)

# load
with open('modelrandomforest.pkl', 'rb') as f4:
    clf4 = pickle.load(f4)

