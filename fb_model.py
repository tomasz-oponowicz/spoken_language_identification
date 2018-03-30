import os
import time
import re

# supress all warnings (especially matplotlib warnings)
import warnings
warnings.filterwarnings("ignore")

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

from sklearn import preprocessing
from sklearn.metrics import classification_report

from keras.models import Model, load_model, Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten
from keras.layers import Dropout, Input, Activation
from keras.optimizers import Nadam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.models import load_model
from keras.layers.normalization import BatchNormalization

import tensorflow as tf

from constants import *
import common

# for reproducibility
np.random.seed(SEED)

# supress tensorflow debug logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def build_model(input_shape):
    model = Sequential()

    model.add(Conv2D(8, (3, 3), strides=(1, 1), padding='same', input_shape=input_shape))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3), strides=(1,2), padding='same'))

    model.add(Conv2D(8, (3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same'))

    model.add(Conv2D(16, (3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(5,5), strides=(2,5), padding='same'))

    model.add(Conv2D(16, (3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(5,5), strides=(1,5), padding='same'))

    model.add(Flatten())

    model.add(Dense(64))
    model.add(Activation('elu'))
    model.add(BatchNormalization())

    model.add(Dense(32))
    model.add(Activation('elu'))
    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    model.add(Dense(len(LANGUAGES)))
    model.add(Activation('softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Nadam(lr=1e-4),
        metrics=['accuracy']
    )

    return model

if __name__ == "__main__":
    input_shape = (FB_HEIGHT, WIDTH, COLOR_DEPTH)

    accuracies = []
    generator = common.train_generator(14, 'fb', input_shape, max_iterations=1)

    first = True
    for train_labels, train_features, test_labels, test_features in generator:
        model = build_model(input_shape)
        if first:
            model.summary()
            first = False

        earlystop = EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=1,
            verbose=0,
            mode='auto'
        )

        model.fit(
            train_features,
            train_labels,
            epochs=20,
            callbacks=[earlystop],
            verbose=2,
            validation_split=0.1
        )

        scores = model.evaluate(test_features, test_labels, verbose=0)
        accuracy = scores[1]

        print('Accuracy:', accuracy)
        accuracies.append(accuracy)

    accuracies = np.array(accuracies)

    print('\n## Summary')
    print("Mean: {mean}, Std {std}".format(mean=accuracies.mean(), std=accuracies.std()))