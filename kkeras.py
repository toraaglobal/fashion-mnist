# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 20:12:33 2019

@author: teeja
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

# MODEL
# BUILD THE BASELINE

def baseline_model(num_pixels,num_classes, optimizer='adam',metrics=['accuracy']):
    # create model
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
    return model


