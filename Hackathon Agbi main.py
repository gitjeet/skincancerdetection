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



classes = ['benign', 'malignant']


from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Flatten, Dropout, Activation
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.models import save_model, load_model

IMG_CHANNEL = 1
BATCH_SIZE = 128
N_EPOCH = 16
VERBOSE = 2
VALIDAION_SPLIT = .2
OPTIM = Adam()
N_CLASSES = len(classes)



y = np_utils.to_categorical(y, N_CLASSES)
plt.xlabel('epochs') 
print('One-Hot Encoding done')

model = Sequential([
    Conv2D(32, (3,3), padding='same', input_shape=(ROW, COL, IMG_CHANNEL), activation='relu'),
    Conv2D(32, (3,3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(.25),
    Conv2D(64, (3,3), padding='same', activation='relu'),
    Conv2D(64, (3,3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(.25),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(.5),
    Dense(N_CLASSES, activation='softmax')
])
print('The model was created by following config:')
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=OPTIM, metrics=['accuracy'])

checkpoint = ModelCheckpoint('model_checkpoint/benigns_vs_malignants.h5')


history=model.fit(X, y, batch_size=BATCH_SIZE, epochs=N_EPOCH, validation_split=VALIDAION_SPLIT, verbose=VERBOSE)


model.save("my_modelCNN.h5") 
plt.figure(0) 
plt.plot(history.history['accuracy'], label='training accuracy') 
plt.plot(history.history['val_accuracy'], label='val accuracy') 
plt.title('Accuracy') 
plt.xlabel('epochs') 
plt.ylabel('accuracy') 
plt.legend() 
plt.show() 
 
plt.figure(1) 
plt.plot(history.history['loss'], label='training loss') 
plt.plot(history.history['val_loss'], label='val loss') 
plt.title('Loss') 
plt.ylabel('loss') 
plt.legend() 
plt.show() 

#testing the 















