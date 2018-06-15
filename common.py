from constants import *

import os

import numpy as np

import pandas as pd

from sklearn import preprocessing
from sklearn.metrics import classification_report


def can_ignore(file, key):
    if key in file:
        return True
    return False


def flatten(binary_labels):
    return np.argmax(binary_labels, axis=1)


def test(labels, features, metadata, model, clazzes, title="test"):
    probabilities = model.predict(features, verbose=0)

    expected = flatten(labels)
    actual = flatten(probabilities)

    print("\n## {title}\n".format(title=title))

    max_probabilities = np.amax(probabilities, axis=1)

    print("Average confidence: {average}\n".format(
        average=np.mean(max_probabilities)))

    errors = pd.DataFrame(np.zeros((len(clazzes), len(GENDERS)), dtype=int),
                          index=clazzes, columns=GENDERS)
    threshold_errors = pd.DataFrame(
        np.zeros((len(clazzes), len(GENDERS)), dtype=int),
        index=clazzes,
        columns=GENDERS)
    threshold_scores = pd.DataFrame(
        np.zeros((len(clazzes), len(GENDERS)), dtype=int),
        index=clazzes,
        columns=GENDERS)
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


def load_data(label_binarizer, input_dir, group, fold_indexes, input_shape):
    all_metadata = []
    all_features = []

    for fold_index in fold_indexes:
        filename = "{group}_metadata.fold{index}.npy".format(
            group=group, index=fold_index)
        metadata = np.load(os.path.join(input_dir, filename))

        filename = "{group}_data.fold{index}.npy".format(
            group=group, index=fold_index)
        features = np.memmap(
            os.path.join(input_dir, filename),
            dtype=DATA_TYPE,
            mode='r',
            shape=(len(metadata),) + input_shape)

        all_metadata.append(metadata)
        all_features.append(features)

    all_metadata = np.concatenate(all_metadata)
    all_features = np.concatenate(all_features)
    all_labels = label_binarizer.transform(all_metadata[:, 0])

    print("[{group}] labels: {labels}, features: {features}".format(
        group=group, labels=all_labels.shape, features=all_features.shape))

    return all_labels, all_features, all_metadata


def build_label_binarizer():
    label_binarizer = preprocessing.LabelBinarizer()
    label_binarizer.fit(LANGUAGES)
    clazzes = list(label_binarizer.classes_)
    print("Classes:", clazzes)

    return label_binarizer, clazzes


def train_generator(fold_count, input_dir, input_shape, max_iterations=1):
    label_binarizer, clazzes = build_label_binarizer()

    fold_indexes = list(range(1, fold_count + 1))

    iteration = 0
    for fold_index in fold_indexes:
        train_fold_indexes = fold_indexes.copy()
        train_fold_indexes.remove(fold_index)
        train_labels, train_features, train_metadata = load_data(
            label_binarizer,
            input_dir,
            'train',
            train_fold_indexes,
            input_shape)

        test_fold_indexes = [fold_index]
        test_labels, test_features, test_metadata = load_data(
            label_binarizer,
            input_dir,
            'train',
            test_fold_indexes,
            input_shape)

        yield (train_labels, train_features, test_labels,
               test_features, test_metadata, clazzes)

        del train_labels
        del train_features
        del train_metadata

        del test_labels
        del test_features
        del test_metadata

        iteration += 1
        if iteration == max_iterations:
            return


def remove_extension(file):
    return os.path.splitext(file)[0]


def get_filename(file):
    return os.path.basename(remove_extension(file))


def group_uids(files):
    uids = dict()

    # intialize empty sets
    for language in LANGUAGES:
        uids[language] = dict()
        for gender in GENDERS:
            uids[language][gender] = set()

    # extract uids and append to language/gender sets
    for file in files:
        info = get_filename(file).split('_')

        language = info[0]
        gender = info[1]
        uid = info[2].split('.')[0]

        uids[language][gender].add(uid)

    # convert sets to lists
    for language in LANGUAGES:
        for gender in GENDERS:
            uids[language][gender] = sorted(list(uids[language][gender]))

    return uids


if __name__ == "__main__":
    generator = train_generator(3, 'fb', (FB_HEIGHT, WIDTH, COLOR_DEPTH))
    for train_labels, train_features, test_labels, test_features in generator:
        print(train_labels.shape)
