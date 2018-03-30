import os

import numpy as np

from sklearn import preprocessing

from constants import *


def load_data(label_binarizer, input_dir, group, fold_indexes, shape):
    all_metadata = []
    all_features = []

    for fold_index in fold_indexes:
        filename = "{group}_metadata.fold{index}.npy".format(group=group, index=fold_index)
        metadata = np.load(os.path.join(input_dir, filename))

        filename = "{group}_data.fold{index}.npy".format(group=group, index=fold_index)
        features = np.memmap(os.path.join(input_dir, filename),
            dtype=DATA_TYPE, mode='r', shape=(len(metadata),) + shape)

        all_metadata.append(metadata)
        all_features.append(features)

    all_metadata = np.concatenate(all_metadata)
    all_features = np.concatenate(all_features)
    all_labels = label_binarizer.transform(all_metadata[:, 0])

    print("[{group}] labels: {labels}, features: {features}".format(
        group=group, labels=all_labels.shape, features=all_features.shape
    ))

    return all_labels, all_features, all_metadata

def build_label_binarizer():
    label_binarizer = preprocessing.LabelBinarizer()
    label_binarizer.fit(LANGUAGES)
    clazzes = list(label_binarizer.classes_)
    print("Classes:", clazzes)

    return label_binarizer, clazzes

def train_generator(fold_count, input_dir, shape):
    label_binarizer, clazzes = build_label_binarizer()

    fold_indexes = list(range(1, fold_count + 1))

    for fold_index in fold_indexes:
        train_fold_indexes = fold_indexes.copy()
        train_fold_indexes.remove(fold_index)
        train_labels, train_features, train_metadata = load_data(label_binarizer,
            input_dir, 'train', train_fold_indexes, shape)

        test_fold_indexes = [fold_index]
        test_labels, test_features, test_metadata = load_data(label_binarizer,
            input_dir, 'train', test_fold_indexes, shape)

        yield train_labels, train_features, test_labels, test_features

        del train_labels
        del train_features
        del train_metadata

        del test_labels
        del test_features
        del test_metadata

if __name__ == "__main__":
    generator = train_generator(14, 'fb', (FB_HEIGHT, WIDTH, COLOR_DEPTH))
    for train_labels, train_features, test_labels, test_features in generator:
        print(train_labels.shape)