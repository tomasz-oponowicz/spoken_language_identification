import os
from glob import glob
import time

import numpy as np
from sklearn.utils import shuffle

from constants import *
import common

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

def can_ignore(file, key, numbers):
    for number in numbers:
        if "{k}{n}".format(k=key, n=number) in file:
            return True
    return False


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

        # limit number of samples
        if can_ignore(fold_file, 'speed', [1,3,6,8]):
            continue
        if can_ignore(fold_file, 'pitch', [1,3,6,8]):
            continue
        if can_ignore(fold_file, 'noise', [2,3,5,6,8,9,11,12]):
            continue

        print(fold_file)

        processed_files += 1

        entries = np.load(fold_file)[DATA_KEY]

        assert np.min(entries) >= 0. and np.max(entries) <= 1.
        assert entries.shape == (WIDTH, SDC_HEIGHT)
        assert entries.dtype == DATA_TYPE

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

    assert features.shape == (processed_files * WIDTH, SDC_HEIGHT)
    assert features.dtype == DATA_TYPE

    np.savez_compressed(os.path.join(output_dir, filename), data=features)

    del features


if __name__ == "__main__":
    start = time.time()

    for language in LANGUAGES:
        generate_folds(
            './build/test',
            input_langauge=language, input_ext='.sdc2.npz',
            output_dir='sdc', output_group='test'
        )
        generate_folds(
            './build/train',
            input_langauge=language, input_ext='.sdc2.npz',
            output_dir='sdc', output_group='train'
        )

    end = time.time()
    print("It took [s]: ", end - start)
