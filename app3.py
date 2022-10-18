# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 16:04:54 2022

@author: 103077
"""

#import the required libararies 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow 
import keras
from keras.utils import to_categorical
from keras.utils.np_utils import to_categoricalr
import h5py
from keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout
from PIL import ImageTk, Image
from keras.models import load_model
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

imgs_path = "E:/Kiran/DL/Trafiic_Signal/gtsrb-preprocessed/train"
data = []
labels = []
classes = 43
for i in range(classes):
    img_path = os.path.join(imgs_path, str(i)) #0-42
    for img in os.listdir(img_path):
        im = Image.open(img_path + '/'  + img)
        im = im.resize((30,30))
        im = np.array(im)
        data.append(im)
        labels.append(i)
#data = np.array(data)
data = np.expand_dims(data, axis=0)
labels = np.array(labels)
print("success") 


x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

input_shape=np.shape(x_train)

x_train = np.array(x_train)
x_test  = np.array(x_test)
#y_train = np.array(y_train)
#y_train = y_train.reshape(1,-1)
#Building the model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=x_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))
#Compilation of the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
epochs = 15
history = model.fit(x_train, y_train, epochs=epochs, batch_size=64, validation_data=(x_test, y_test))

plt.figure(0)
plt.plot(history.history['accuracy'], label="Training accuracy")
plt.plot(history.history['val_accuracy'], label="val accuracy")
plt.title("Accuracy")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.figure(1)
plt.plot(history.history['loss'], label="training loss")
plt.plot(history.history['val_loss'], label="val loss")
plt.title("Loss")
plt.xlabel("epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

model.save('traffic_classifier.h5')
#tesing 



model = load_model('traffic_classifier.h5')
#dictionary to label all traffic signs class.
classes = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)',
            3:'Speed limit (50km/h)',
            4:'Speed limit (60km/h)',
            5:'Speed limit (70km/h)',
            6:'Speed limit (80km/h)',
            7:'End of speed limit (80km/h)',
            8:'Speed limit (100km/h)',
            9:'Speed limit (120km/h)',
            10:'No passing',
            11:'No passing veh over 3.5 tons',
            12:'Right-of-way at intersection',
            13:'Priority road',
            14:'Yield',
            15:'Stop',
            16:'No vehicles',
            17:'Veh > 3.5 tons prohibited',
            18:'No entry',
            19:'General caution',
            20:'Dangerous curve left',
            21:'Dangerous curve right',
            22:'Double curve',
            23:'Bumpy road',
            24:'Slippery road',
            25:'Road narrows on the right',
            26:'Road work',
            27:'Traffic signals',
            28:'Pedestrians',
            29:'Children crossing',
            30:'Bicycles crossing',
            31:'Beware of ice/snow',
            32:'Wild animals crossing',
            33:'End speed + passing limits',
            34:'Turn right ahead',
            35:'Turn left ahead',
            36:'Ahead only',
            37:'Go straight or right',
            38:'Go straight or left',
            39:'Keep right',
            40:'Keep left',
            41:'Roundabout mandatory',
            42:'End of no passing',
            43:'End no passing veh > 3.5 tons' }







sample_data = []
sample_labels = []
#classes = 43
sample_im = Image.open("E:/Kiran/DL/Trafiic_Signal/gtsrb-preprocessed/test/00024.png")
sample_im = sample_im.resize((30,30))
sample_im = np.array(sample_im)
sample_data.append(sample_im)
#labels.append(i)
sample_data = np.array(sample_data)
#sample_data = np.expand_dims(sample_data, axis=0)
pred = model.predict(sample_data)
maxindex = pred.argmax()

sign = classes[maxindex+1]

print("Your image is => ",sign)

