import os
import time
import re

import numpy as np

import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.metrics import classification_report

from keras.models import Model, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.layers import Dropout, Input, BatchNormalization
from keras.optimizers import Nadam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

import tensorflow as tf

# disable tensorflow debug logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

in_dim = (192,192,1)
out_dim = 3

LANGUAGES = ['en', 'de', 'es']
LANGUAGE_INDEX = 0

def load_data(file, label_binarizer, use_augmented_samples=True):
    bundle = np.load(file)

    metadata = bundle['labels']
    features = bundle['features']

    if not use_augmented_samples:

        # filename without augmentation
        pattern = re.compile("^.+fragment\d+$")

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

    features = features.reshape((len(features), 192, 192, 1))
    features = np.divide(features, 255.)

    print("[{file}] labels: {labels}, features: {features}".format(
        file=file, labels=labels.shape, features=features.shape
    ))

    image = features[0, :, :, 0]
    plt.imshow(image)
    plt.savefig(file + '.png', bbox_inches='tight')

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

def test(labels, features, model, classes, title=""):
    probabilities = model.predict(features, batch_size=32, verbose=0)

    expected = flatten(labels)
    actual = flatten(probabilities)

    print("\n## {title}\n".format(title=title))

    print("Average confidence: {average}\n".format(
        average=np.mean(np.amax(probabilities, axis=1))
    ))

    print(classification_report(expected, actual, target_names=classes))

def create_model(use_augmented_samples):
    label_binarizer = preprocessing.LabelBinarizer()
    label_binarizer.fit(LANGUAGES)
    classes = list(label_binarizer.classes_)
    print("classes:", classes)

    start = time.time()
    train_labels, train_features, train_metadata = load_data('train.npz', label_binarizer, use_augmented_samples=use_augmented_samples)
    valid_labels, valid_features, valid_metadata = load_data('valid.npz', label_binarizer)
    test_labels, test_features, test_metadata = load_data('test.npz', label_binarizer)
    print("Loaded data in [s]: ", time.time() - start)

    i = Input(shape=in_dim)
    m = Conv2D(1, (3, 3), activation='elu', padding='same')(i)
    m = MaxPooling2D()(m)
    # m = Conv2D(16, (3, 3), activation='elu', padding='same')(i)
    # m = MaxPooling2D()(m)
    # m = Conv2D(32, (3, 3), activation='elu', padding='same')(m)
    # m = MaxPooling2D()(m)
    # m = Conv2D(64, (3, 3), activation='elu', padding='same')(m)
    # m = MaxPooling2D()(m)
    # m = Conv2D(128, (3, 3), activation='elu', padding='same')(m)
    # m = MaxPooling2D()(m)
    # m = Conv2D(256, (3, 3), activation='elu', padding='same')(m)
    # m = MaxPooling2D()(m)
    m = Flatten()(m)
    m = Dense(1, activation='elu')(m)
    # m = Dense(512, activation='elu')(m)
    # m = Dropout(0.5)(m)
    o = Dense(out_dim, activation='softmax')(m)

    model = Model(inputs=i, outputs=o)
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer=Nadam(lr=1e-4), metrics=['accuracy'])
    model.fit(train_features, train_labels, epochs=3, verbose=1, validation_data=(valid_features, valid_labels))

    model.save('language.h5')

    test(valid_labels, valid_features, model, classes, title="validation")
    test(test_labels, test_features, model, classes, title="test")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate speech language recognition model.')
    parser.add_argument('--use-augmented-samples', dest='use_augmented_samples', action='store_true')
    parser.set_defaults(use_augmented_samples=False)

    args = parser.parse_args()

    start = time.time()
    create_model(args.use_augmented_samples)
    print("Generated model in [s]: ", time.time() - start)
