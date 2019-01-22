#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 22:14:13 2019

@author: benjaminwasserman
"""

import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from sklearn.model_selection import train_test_split

x = np.load("Sign-language-digits-dataset 2/X.npy")
y = np.load("Sign-language-digits-dataset 2/Y.npy")

# plot the first image in the dataset
plt.imshow(x[0])

# check the image shape
print(x[0].shape)

# split into test and training data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)

# get array length
x_train_count = len(x_train)
x_test_count = len(x_test)

# reshape data to fit model
x_train = x_train.reshape(x_train_count, 64, 64, 1)
x_test = x_test.reshape(x_test_count, 64, 64, 1)


#create model
model = Sequential()

#add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(64,64, 1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3)


