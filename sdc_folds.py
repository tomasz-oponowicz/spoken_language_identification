import os
from glob import glob
import time

import numpy as np
from sklearn.utils import shuffle
import speechpy

from constants import *
import common

HEIGHT = MFCC_HEIGHT * 2
DTYPE = 'float64'

def generate_folds(input_dir, input_langauge, input_ext, output_dir, output_group):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = glob(os.path.join(input_dir, input_langauge + '*' + input_ext))

    uids = common.group_uids(files)

    fold_index = 1
    while has_uids(uids, input_langauge):
        print("[{group}] Fold {index}".format(group=output_group, index=fold_index))

        generate_fold(
            uids, input_dir, input_langauge,
            input_ext, output_dir, output_group,
            fold_index
        )

        fold_index += 1

def has_uids(uids, language):
    for gender in GENDERS:
        if len(uids[language][gender]) == 0:
            return False

    return True

def generate_fold(uids, input_dir, input_langauge, input_ext, output_dir, output_group, fold_index):

    # pull uid for each a gender pair
    fold_uids = []
    for gender in GENDERS:
        fold_uids.append(uids[input_langauge][gender].pop())

    # find files for given uids
    fold_files = []
    for fold_uid in fold_uids:
        filename = '*{uid}*{extension}'.format(uid=fold_uid, extension=input_ext)
        fold_files.extend(glob(os.path.join(input_dir, filename)))

    fold_files = sorted(fold_files)
    fold_files = shuffle(fold_files, random_state=SEED)

    features = []

    processed_files = 0
    for fold_file in fold_files:
        print(fold_file)
        processed_files += 1

        entries = np.load(fold_file)[DATA_KEY]

        # source: https://github.com/astorfi/speechpy/blob/master/speechpy/processing.py

        # cepstral mean and variance normalization
        entries = speechpy.processing.cmvn(entries, variance_normalization=True)

        # deltas
        deltas = speechpy.processing.derivative_extraction(entries, DeltaWindows=2)

        entries = np.hstack((entries, deltas))

        assert entries.dtype == DTYPE

        for entry in entries:
            features.append(entry)

    # create a file array
    filename = "{language}_{group}.fold{index}".format(
        language=input_langauge,
        group=output_group,
        index=fold_index
    )

    features = np.array(features)
    print(features.shape)

    assert features.shape == (processed_files * WIDTH, HEIGHT)
    assert features.dtype == DTYPE

    np.savez_compressed(os.path.join(output_dir, filename), data=features)

    del features


if __name__ == "__main__":
    start = time.time()

    for language in LANGUAGES:
        generate_folds(
            './build/test',
            input_langauge=language, input_ext='.mfcc.npz',
            output_dir='mfcc', output_group='test'
        )
        generate_folds(
            './build/train',
            input_langauge=language, input_ext='.mfcc.npz',
            output_dir='mfcc', output_group='train'
        )

    end = time.time()
    print("It took [s]: ", end - start)
