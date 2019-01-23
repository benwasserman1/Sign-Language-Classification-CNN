#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 22:14:13 2019

@author: benjaminwasserman
"""

import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator


datagen = ImageDataGenerator(
    rotation_range=16,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=.08)


# EDA: show the first image, show the shape, split into training and test, 
# find length of eac3h

x = np.load("Sign-language-digits-dataset 2/X.npy")
y = np.load("Sign-language-digits-dataset 2/Y.npy")

# vgg16

# plot the first image in the dataset
plt.imshow(x[0], cmap = "gray")

# check the image shape
#print(x[0].shape)

# split into test and training data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.15)

# get array length
x_train_count = len(x_train)
x_test_count = len(x_test)


# get full shape
print(x_train.shape)
print(x_train.shape[1:])


# reshape data to fit model
x_train = x_train.reshape(x_train_count, 64, 64, 1)
x_test = x_test.reshape(x_test_count, 64, 64, 1)


# create model
model = Sequential()

#add model layers
model.add(Conv2D(64, kernel_size=4, activation='relu', input_shape=(64,64, 1)))
model.add(Conv2D(64, kernel_size=4, activation='relu'))
model.add(MaxPooling2D(pool_size=4))

model.add(Dropout(0.3))

model.add(Dense(units = 256, activation = 'relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

#compile model using accuracy to measure model performance
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

#train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)

model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=64, epochs=8)

score = model.evaluate(x_test, y_test, verbose=0)

print('Loss: {:.4f}  Accuaracy: {:.4}%'.format(score[0],score[1]))

model.predict(x_test[0:])
y_test[0:]


