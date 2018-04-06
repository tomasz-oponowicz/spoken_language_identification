
## Avoid randomness
# https://machinelearningmastery.com/reproducible-results-neural-networks-keras/
# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
# https://machinelearningmastery.com/reproducible-results-neural-networks-keras/#comment-414394

import numpy as np
import tensorflow as tf
import random as rn

import os
os.environ['PYTHONHASHSEED'] = '42'

# supress tensorflow debug logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# disable auto tune
# https://github.com/tensorflow/tensorflow/issues/5048
# os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'

np.random.seed(42)

rn.seed(42)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

tf.set_random_seed(42)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

import time
import re

# supress all warnings (especially matplotlib warnings)
import warnings
warnings.filterwarnings("ignore")

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

from constants import *
import common

def build_model(input_shape):
    model = Sequential()

    # 12x1000

    model.add(Conv2D(16, (3, 5), strides=(1, 1), padding='same', input_shape=input_shape))
    model.add(Conv2D(16, (3, 5), strides=(1, 1), padding='same'))
    model.add(Conv2D(16, (3, 5), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(1,2), padding='same'))

    # 12x500

    model.add(Conv2D(16, (3, 5), strides=(1, 1), padding='same'))
    model.add(Conv2D(16, (3, 5), strides=(1, 1), padding='same'))
    model.add(Conv2D(16, (3, 5), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,1), padding='same'))

    # 6x500

    model.add(Conv2D(16, (3, 5), strides=(1, 1), padding='same'))
    model.add(Conv2D(16, (3, 5), strides=(1, 1), padding='same'))
    model.add(Conv2D(16, (3, 5), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(1,2), padding='same'))

    # 6x250

    model.add(Conv2D(16, (3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(1,2), padding='same'))

    # 6x125

    model.add(Conv2D(16, (3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(1,2), padding='same'))

    # 6x62

    model.add(Conv2D(16, (3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(1,2), padding='same'))

    # 6x32

    model.add(Conv2D(16, (3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(1,2), padding='same'))

    # 6x16

    model.add(Conv2D(16, (3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(1,2), padding='same'))

    # 6x8

    model.add(Flatten())

    model.add(Dense(64))
    model.add(Activation('tanh'))

    model.add(Dense(32))
    model.add(Activation('tanh'))

    model.add(Dense(len(LANGUAGES)))
    model.add(Activation('softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='sgd',
        metrics=['accuracy']
    )

    return model

if __name__ == "__main__":
    input_shape = (MFCC_HEIGHT, WIDTH, COLOR_DEPTH)

    accuracies = []
    generator = common.train_generator(14, 'mfcc', input_shape, max_iterations=1)

    first = True
    for train_labels, train_features, test_labels, test_features, test_metadata, clazzes in generator:
        model = build_model(input_shape)
        if first:
            model.summary()
            first = False

        checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=0,
            save_best_only=True, mode='min')

        earlystop = EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=3,
            verbose=0,
            mode='auto'
        )

        model.fit(
            train_features,
            train_labels,
            epochs=20,
            callbacks=[checkpoint, earlystop],
            verbose=1,
            validation_split=0.1
        )

        model = load_model('model.h5')

        scores = model.evaluate(test_features, test_labels, verbose=0)
        accuracy = scores[1]

        print('Accuracy:', accuracy)
        accuracies.append(accuracy)

        common.test(test_labels, test_features, test_metadata, model, clazzes)

    accuracies = np.array(accuracies)

    print('\n## Summary')
    print("Mean: {mean}, Std {std}".format(mean=accuracies.mean(), std=accuracies.std()))