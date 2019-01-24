#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 22:14:13 2019

@author: benjaminwasserman and ryanturley

Ben Wasserman and Ryan Turley
CPSC393 Machine Learning
January 24th, 2018
"""

# Import libraries: matplotlib for showing images, sklearn for splitting, numpy for arrays, keras for CNN
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator


# Generate augmented data so the results will generalize better 
aug_data = ImageDataGenerator(
    rotation_range=3,
    width_shift_range=0.04,
    height_shift_range=0.08,
    zoom_range=.12,
    shear_range = .1)

# EDA: show the first image, show the shape, split into training and test, 
# and find length of each so x_train and x_test can be reshaped properly
x = np.load("Sign-language-digits-dataset 2/X.npy")
y = np.load("Sign-language-digits-dataset 2/Y.npy")


# plot the first image in the dataset
plt.imshow(x[0], cmap = "gray")

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

# create the model
model = Sequential()

# add two convolutional layers and max pool
model.add(Conv2D(64, kernel_size=4, activation='relu', input_shape=(64,64, 1)))
model.add(Conv2D(64, kernel_size=4, activation='relu'))
model.add(MaxPooling2D(pool_size=4))

# dropout to help with overiftting
model.add(Dropout(0.35))
model.add(Flatten())

# pass through fully connected layer
model.add(Dense(units = 256, activation = 'relu'))

# use BatchNormalization to minimize how layers affect each other
model.add(BatchNormalization())

# softmax for output layer
model.add(Dense(10, activation='softmax'))

#compile model
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

# train the model with the augmented data from datagen
model.fit_generator(aug_data.flow(x_train, y_train, batch_size=64),
                    steps_per_epoch=64, epochs=7)


# show validation accuracy of model
score = model.evaluate(x_test, y_test, verbose=0)
print('Validation loss: {:.4f}  Validation accuracy: {:.4}%'.format(score[0],score[1]))


# get results array and convert it to a list
results = model.predict(x_test[0:])
results = results.tolist()

# convert actual labels to a list
y_test[0:]
encodings = y_test[0:].tolist()

predicted_labels = []
proper_labels = []

# create a list of the predicted and proper values for each image
for encoding in encodings:
    value = max(encoding)
    proper_labels.append(encoding.index(value))

for result in results:
    value = (max(result))
    predicted_labels.append(result.index(value))
    
print(proper_labels)
print(predicted_labels)


# print out what was predicted improperly 
for i in range(len(proper_labels)):
    if proper_labels[i] != predicted_labels[i]:
        print("Mistook " + str(predicted_labels[i]) + " for " + str(proper_labels[i]))
        
        


