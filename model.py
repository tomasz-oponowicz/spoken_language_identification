from constants import *

import time
import re

# supress all warnings (especially matplotlib warnings)
import warnings
warnings.filterwarnings("ignore")

# RANDOMNESS
# https://machinelearningmastery.com/reproducible-results-neural-networks-keras/
# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development

import os
os.environ['PYTHONHASHSEED'] = '0'

import random
random.seed(SEED)

import numpy as np
np.random.seed(SEED)

# supress tensorflow debug logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# disable auto tune
# https://github.com/tensorflow/tensorflow/issues/5048
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'

import tensorflow as tf
session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1)
from keras import backend as K
tf.set_random_seed(SEED)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

from sklearn import preprocessing
from sklearn.metrics import classification_report

from keras.models import Model, load_model, Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten
from keras.layers import Dropout, Input, Activation
from keras.optimizers import Nadam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from keras import regularizers

import common


def build_model(input_shape):
    model = Sequential()

    # 40x1000

    model.add(Conv2D(
        16,
        (3, 3),
        strides=(1, 1),
        padding='same',
        kernel_regularizer=regularizers.l2(0.001),
        input_shape=input_shape))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    # 20x500

    model.add(Conv2D(
        32,
        (3, 3),
        strides=(1, 1),
        padding='same',
        kernel_regularizer=regularizers.l2(0.001)))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    # 10x250

    model.add(Conv2D(
        64,
        (3, 3),
        strides=(1, 1),
        padding='same',
        kernel_regularizer=regularizers.l2(0.001)))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    # 5x125

    model.add(Conv2D(
        128,
        (3, 5),
        strides=(1, 1),
        padding='same',
        kernel_regularizer=regularizers.l2(0.001)))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(3, 5), strides=(1, 5), padding='same'))

    # 5x25

    model.add(Conv2D(
        256,
        (3, 5),
        strides=(1, 1),
        padding='same',
        kernel_regularizer=regularizers.l2(0.001)))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(3, 5), strides=(1, 5), padding='same'))
    model.add(AveragePooling2D(
        pool_size=(5, 5),
        strides=(5, 5),
        padding='valid'))

    # 1x1

    model.add(Flatten())

    model.add(Dense(
        32,
        activation='elu',
        kernel_regularizer=regularizers.l2(0.001)))

    model.add(Dropout(0.5))

    model.add(Dense(len(LANGUAGES)))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.0, nesterov=False)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=sgd,
        metrics=['accuracy'])

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument(
        '--test',
        dest='test',
        action='store_true',
        help='test the previously trained model against the test set')
    parser.set_defaults(test=False)

    args = parser.parse_args()

    input_shape = (FB_HEIGHT, WIDTH, COLOR_DEPTH)

    if args.test:
        model = load_model('model.h5')

        input_shape = (FB_HEIGHT, WIDTH, COLOR_DEPTH)
        label_binarizer, clazzes = common.build_label_binarizer()

        test_labels, test_features, test_metadata = common.load_data(
            label_binarizer, 'build/folds', 'test', [1], input_shape)

        common.test(test_labels, test_features, test_metadata, model, clazzes)
    else:
        accuracies = []
        generator = common.train_generator(
            14, 'build/folds', input_shape, max_iterations=1)

        first = True
        for (train_labels,
             train_features,
             test_labels,
             test_features,
             test_metadata,
             clazzes) in generator:

            # TODO reset tensorflow

            model = build_model(input_shape)
            if first:
                model.summary()
                first = False

            checkpoint = ModelCheckpoint(
                'model.h5',
                monitor='val_loss',
                verbose=0,
                save_best_only=True,
                mode='min')

            earlystop = EarlyStopping(
                monitor='val_loss',
                min_delta=0,
                patience=3,
                verbose=0,
                mode='auto')

            model.fit(
                train_features,
                train_labels,
                epochs=20,
                callbacks=[checkpoint, earlystop],
                verbose=1,
                validation_data=(test_features, test_labels),
                batch_size=8)

            model = load_model('model.h5')

            scores = model.evaluate(test_features, test_labels, verbose=0)
            accuracy = scores[1]

            print('Accuracy:', accuracy)
            accuracies.append(accuracy)

            common.test(
                test_labels,
                test_features,
                test_metadata,
                model,
                clazzes)

        accuracies = np.array(accuracies)

        print('\n## Summary\n')
        print("Mean: {mean}, Std {std}".format(
            mean=accuracies.mean(),
            std=accuracies.std()))
