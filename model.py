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

SEED = 42

# for reproducibility
np.random.seed(SEED)

# supress tensorflow debug logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

GENDERS = ['m', 'f']
GENDER_INDEX = 1

LANGUAGES = ['en', 'de', 'es']
LANGUAGE_INDEX = 0

TEST_GROUP = 'valid'
VALID_GROUP = 'test'
TRAIN_GROUP = 'train'

COLOR_DEPTH = 1
WIDTH = 192
HEIGHT = 192

THRESHOLD = 0.8

in_dim = (192,192,1)
out_dim = len(LANGUAGES)

def load_data(group, label_binarizer, pattern=None, skip_input_validation=False):
    metadata = np.load("{0}_metadata.npy".format(group))

    features = np.memmap("{0}_features.npy".format(group), dtype='float16', mode='r',
        shape=(len(metadata), WIDTH, HEIGHT, COLOR_DEPTH))

    if pattern:
        mask = []
        for info in metadata:
            filename = info[2]
            if pattern.match(filename):
                mask.append(True)
            else:
                mask.append(False)

        metadata = metadata[mask]
        features = features[mask]

    labels = label_binarizer.transform(metadata[:, 0])

    print("[{group}] labels: {labels}, features: {features}".format(
        group=group, labels=labels.shape, features=features.shape
    ))

    image = features[0, :, :, 0]

    # plt.imgshow doesn't support float16
    image = image.astype('float32')

    plt.figure() # reset plot
    plt.imshow(image)
    plt.savefig(group + '.png', bbox_inches='tight')

    if not skip_input_validation:
        validate(labels, features, metadata, label_binarizer.classes_)

    return (labels, features, metadata)

def flatten(binary_labels):
    return np.argmax(binary_labels, axis=1)

def validate(binary_labels, features, metadata, classes):
    assert len(binary_labels.shape) == 2
    assert binary_labels.shape[1] == len(LANGUAGES)

    assert len(features.shape) == 4
    assert features.shape[1] == 192
    assert features.shape[2] == 192
    assert features.shape[3] == 1

    # langauge, gender, info
    assert len(metadata.shape) == 2
    assert metadata.shape[1] == 3

    # everything should have the same length
    assert len(metadata) == len(binary_labels)
    assert len(metadata) == len(features)

    # values should be between 0 and 1
    assert np.max(features) == 1.0
    assert np.min(features) == 0.0

    classes = list(classes)
    labels = flatten(binary_labels)

    # test if first 3 elements have the same class index
    assert labels[0] == classes.index(metadata[0][LANGUAGE_INDEX])
    assert labels[1] == classes.index(metadata[1][LANGUAGE_INDEX])
    assert labels[2] == classes.index(metadata[2][LANGUAGE_INDEX])

    # test if samples are equally balanced per language
    en_count = len(np.where(metadata == 'en')[0])
    de_count = len(np.where(metadata == 'de')[0])
    es_count = len(np.where(metadata == 'es')[0])

    assert en_count > 0
    assert en_count == de_count
    assert en_count == es_count

    # test if samples are equally balanced per gender
    male_count = len(np.where(metadata == 'm')[0])
    female_count = len(np.where(metadata == 'f')[0])

    assert male_count > 0
    assert male_count == female_count

def test(labels, features, metadata, model, clazzes, title=""):
    probabilities = model.predict(features, verbose=0)

    expected = flatten(labels)
    actual = flatten(probabilities)

    print("\n## {title}\n".format(title=title))

    max_probabilities = np.amax(probabilities, axis=1)

    plt.figure() # reset plot
    plot = sns.distplot(max_probabilities, bins=10)
    plot.figure.savefig("{title}_probabilities.png".format(title=title))

    print("Average confidence: {average}\n".format(
        average=np.mean(max_probabilities)
    ))

    errors = pd.DataFrame(np.zeros((len(clazzes), len(GENDERS)), dtype=int),
        index=clazzes, columns=GENDERS)
    threshold_errors = pd.DataFrame(np.zeros((len(clazzes), len(GENDERS)), dtype=int),
        index=clazzes, columns=GENDERS)
    threshold_scores = pd.DataFrame(np.zeros((len(clazzes), len(GENDERS)), dtype=int),
        index=clazzes, columns=GENDERS)
    for index in range(len(actual)):
        clazz = metadata[index][LANGUAGE_INDEX]
        gender = metadata[index][GENDER_INDEX]
        if actual[index] != expected[index]:
            errors[gender][clazz] += 1
        if actual[index] >= THRESHOLD:
            if actual[index] != expected[index]:
                threshold_errors[gender][clazz] += 1
            if actual[index] == expected[index]:
                threshold_scores[gender][clazz] += 1

    print("Amount of errors by gender:")
    print(errors, "\n")
    print("Amount of errors by gender (threshold {0}):".format(THRESHOLD))
    print(threshold_errors, "\n")
    print("Amount of scores by gender (threshold {0}):".format(THRESHOLD))
    print(threshold_scores, "\n")

    print(classification_report(expected, actual, target_names=clazzes))

def train_model(train_labels, train_features, valid_labels, valid_features,
                epochs=100, enable_model_summary=True, enable_early_stop=True):

    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=in_dim))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(4,2)))

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,4)))

    model.add(Flatten())

    model.add(Dense(64))
    model.add(Activation('elu'))
    model.add(BatchNormalization())

    model.add(Dense(32))
    model.add(Activation('elu'))
    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    model.add(Dense(out_dim))
    model.add(Activation('softmax'))

    if enable_model_summary:
        model.summary()

    checkpoint = ModelCheckpoint('model.h5', monitor='val_acc', verbose=0,
        save_best_only=True, mode='max')

    # https://stackoverflow.com/questions/43906048/keras-early-stopping
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2,
        verbose=0, mode='auto')

    # https://keras.io/callbacks/#tensorboard
    # $ tensorboard --logdir=/home/myveo/projects/bert/logs
    # tensorboard = TensorBoard(
    #     log_dir='logs', histogram_freq=10, batch_size=batch_size,
    #     write_graph=True, write_grads=True, write_images=False,
    #     embeddings_freq=0, embeddings_layer_names=None,
    #     embeddings_metadata=None
    # )
    # ...then:
    # callbacks[tensorboard]

    callbacks = [checkpoint]
    if enable_early_stop:
        callbacks.append(earlystop)

    model.compile(loss='categorical_crossentropy', optimizer=Nadam(lr=1e-4),
        metrics=['accuracy'])

    history = model.fit(train_features, train_labels, epochs=epochs,
        callbacks=callbacks, verbose=1,
        validation_data=(valid_features, valid_labels))

    return (model, history.history)

def plot_metrics(history, metrics, file):
    plt.figure()
    data = pd.DataFrame(history)[metrics]
    data.plot(xticks=data.index)
    plt.savefig(file)

def compare_deformation_accuracies(label_binarizer, valid_labels, valid_features,
        skip_input_validation=False):
    epochs = 10

    all_accuracies = None

    patternsAndColumns = [
        (re.compile("^.+fragment\d+$"), 'base'),
        (re.compile("^.+pitch\d+$"), 'pitch'),
        (re.compile("^.+speed\d+$"), 'speed'),
        (re.compile("^.+noise\d+$"), 'noise'),
        (None, 'all')
    ];

    for pattern, column in patternsAndColumns:
        print("# {title}\n".format(title=column)) # separator

        train_labels, train_features, train_metadata = load_data(
            TRAIN_GROUP, label_binarizer, pattern=pattern,
            skip_input_validation=skip_input_validation)
        model, history = train_model(train_labels, train_features,
            valid_labels, valid_features, epochs=epochs,
            enable_model_summary=False, enable_early_stop=False)

        current_accuracies = pd.DataFrame(history['val_acc'], columns=[column])
        if all_accuracies is None:
            all_accuracies = current_accuracies
        else:
            # merge columns
            all_accuracies = pd.concat([all_accuracies, current_accuracies],
                axis=1, join='inner')

    plt.figure()
    all_accuracies.plot(xticks=all_accuracies.index)
    plt.savefig('deformation_acurracies.png')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate speech language recognition model.')
    parser.add_argument('--compare-deformations', dest='compare_deformations', action='store_true')
    parser.add_argument('--skip-input-validation', dest='skip_input_validation', action='store_true')
    parser.add_argument('--all', dest='all', action='store_true')
    parser.add_argument('--test', dest='test', action='store_true')
    parser.set_defaults(compare_deformations=False, skip_input_validation=False, all=False, test=False)

    args = parser.parse_args()

    label_binarizer = preprocessing.LabelBinarizer()
    label_binarizer.fit(LANGUAGES)
    clazzes = list(label_binarizer.classes_)
    print("Classes:", clazzes)

    start = time.time()
    valid_labels, valid_features, valid_metadata = load_data(VALID_GROUP, label_binarizer,
        skip_input_validation=args.skip_input_validation)

    if args.compare_deformations:
        compare_deformation_accuracies(label_binarizer, valid_labels, valid_features,
            skip_input_validation=args.skip_input_validation)
    elif args.test:
        test_labels, test_features, test_metadata = load_data(TEST_GROUP, label_binarizer,
            skip_input_validation=args.skip_input_validation)
        print("Loaded data in [s]: ", time.time() - start)

        model = load_model('model.h5')

        test(valid_labels, valid_features, valid_metadata, model, clazzes, title="valid")
        test(test_labels, test_features, test_metadata, model, clazzes, title="test")
    else:
        test_labels, test_features, test_metadata = load_data(TEST_GROUP, label_binarizer,
            skip_input_validation=args.skip_input_validation)

        # without deformations
        pattern = pattern=re.compile("^.+fragment\d+$")
        if args.all:
            pattern = None
        train_labels, train_features, train_metadata = load_data(TRAIN_GROUP, label_binarizer,
            pattern=pattern, skip_input_validation=args.skip_input_validation)
        print("Loaded data in [s]: ", time.time() - start)

        start = time.time()
        model, history = train_model(train_labels, train_features,
            valid_labels, valid_features)
        print("Generated model in [s]: ", time.time() - start)

        plot_metrics(history, ['acc', 'val_acc'], file='history_accuracy.png')
        plot_metrics(history, ['loss', 'val_loss'], file='history_loss.png')

        # delete current model
        del model
        # load the best model checkpoint instead
        model = load_model('model.h5')

        test(valid_labels, valid_features, valid_metadata, model, clazzes, title="valid")
        test(test_labels, test_features, test_metadata, model, clazzes, title="test")
