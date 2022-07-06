import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

import cv2 as cv
import glob
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random
import seaborn as sns
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Nadam
from keras.utils import to_categorical
from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout, 
                          Flatten, Input, MaxPooling2D)
from sklearn.model_selection import train_test_split


SHAPE = (96,96)

daisy_dir = glob.glob(os.path.join('flowers', 'daisy', '*'))
dandelion_dir = glob.glob(os.path.join('flowers', 'dandelion', '*'))
rose_dir = glob.glob(os.path.join('flowers', 'rose', '*'))
sunflower_dir = glob.glob(os.path.join('flowers', 'sunflower', '*'))
tulip_dir = glob.glob(os.path.join('flowers', 'tulip', '*'))


# Entrada Rede Neural
X_path = daisy_dir + dandelion_dir + rose_dir + sunflower_dir + tulip_dir
X = [
    np.array(
        cv.resize( # type: ignore
            cv.imread(f), SHAPE, # type: ignore
            interpolation = cv.INTER_AREA)) for f in X_path] # type: ignore
X = np.array(X) / 255


l_daisy = np.zeros(len(daisy_dir))
l_dandelion = np.ones(len(dandelion_dir))
l_rose = 2 * np.ones(len(rose_dir))
l_sunflower = 3 * np.ones(len(sunflower_dir))
l_tulip = 4 * np.ones(len(tulip_dir))

y = np.concatenate((l_daisy, l_dandelion, l_rose, l_sunflower, l_tulip))
y = to_categorical(y, 5)

X_train, X_val, y_train, y_val = train_test_split(
                                        X, y, test_size=0.2, random_state=42)


# Fitting data
datagen = ImageDataGenerator(
        zoom_range=0.1, # Zoom aleatorio
        rotation_range=15,
        width_shift_range=0.1, # Shift horizontal
        height_shift_range=0.1, # Shift vertical
        horizontal_flip=True,
        vertical_flip=True)
datagen.fit(X_train)


# CNN from scratch
inp = Input((*SHAPE, 3))

conv1 = Conv2D(64, (5,5), padding='valid', activation='relu')(inp)
conv1 = MaxPooling2D(pool_size=(2,2))(conv1)
conv1 = BatchNormalization()(conv1)

conv2 = Conv2D(96, (4,4), padding='valid', activation='relu')(conv1)
conv2 = MaxPooling2D(pool_size=(2,2))(conv2)
conv2 = BatchNormalization()(conv2)

conv3 = Conv2D(128, (3,3), padding='valid', activation='relu')(conv2)
conv3 = MaxPooling2D(pool_size=(2,2))(conv3)
conv3 = BatchNormalization()(conv3)

conv4 = Conv2D(256, (3,3), padding='valid', activation='relu')(conv3)
conv4 = MaxPooling2D(pool_size=(2,2))(conv4)
conv4 = BatchNormalization()(conv4)

flat = Flatten()(conv4)

dense1 = Dense(512, activation='relu')(flat)
dense1 = Dropout(0.5)(dense1)

dense2 = Dense(64, activation='relu')(dense1)
dense2 = Dropout(0.1)(dense2)

out = Dense(5, activation='softmax')(dense2)
model = Model(inp, out)
model.compile(
        optimizer=Nadam(lr=0.0001), 
        loss='categorical_crossentropy',
        metrics=['accuracy'])


# train_test_split
history = model.fit(X_train, y_train, 
        batch_size=32, epochs=50, 
        initial_epoch=0, validation_data=(X_val, y_val))


# Transfer learning
vgg = keras.applications.VGG16(input_shape=(*SHAPE, 3), # type: ignore
        include_top=False,
        weights='imagenet')

x = vgg.output
x = Flatten()(x)
x = Dense(3078, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)

out = Dense(5, activation='softmax')(x)
tf_model = Model(inputs=vgg.input, outputs=out)
for layer in tf_model.layers[:20]:
    layer.trainable=False

history = tf_model.fit(X_train, y_train, 
        batch_size=1, epochs=30, 
        initial_epoch=0, validation_data=(X_val, y_val))

pred = tf_model.predict(X_val)
pred = np.argmax(pred, axis=1)
