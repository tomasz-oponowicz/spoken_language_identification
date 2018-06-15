import imageio
from glob import glob
import os
import numpy as np
from sklearn.utils import shuffle
import time
import speechpy

from constants import *
import common


def has_uids(uids):
    for language in LANGUAGES:
        for gender in GENDERS:
            if len(uids[language][gender]) == 0:
                return False

    return True


def generate_fold(
        uids,
        input_dir,
        input_ext,
        output_dir,
        group,
        fold_index,
        input_shape,
        normalize,
        output_shape):

    # pull uid for each a language, gender pair
    fold_uids = []
    for language in LANGUAGES:
        for gender in GENDERS:
            fold_uids.append(uids[language][gender].pop())

    # find files for given uids
    fold_files = []
    for fold_uid in fold_uids:
        filename = '*{uid}*{extension}'.format(
            uid=fold_uid,
            extension=input_ext)
        fold_files.extend(glob(os.path.join(input_dir, filename)))

    fold_files = sorted(fold_files)
    fold_files = shuffle(fold_files, random_state=SEED)

    metadata = []

    # create a file array
    filename = "{group}_data.fold{index}.npy".format(
        group=group, index=fold_index)
    features = np.memmap(
        os.path.join(output_dir, filename),
        dtype=DATA_TYPE,
        mode='w+',
        shape=(len(fold_files),) + output_shape)

    # append data to a file array
    # append metadata to an array
    for index, fold_file in enumerate(fold_files):
        print(fold_file)

        filename = common.get_filename(fold_file)
        language = filename.split('_')[0]
        gender = filename.split('_')[1]

        data = np.load(fold_file)[DATA_KEY]
        assert data.shape == input_shape
        assert data.dtype == DATA_TYPE

        features[index] = normalize(data)
        metadata.append((language, gender, filename))

    assert len(metadata) == len(fold_files)

    # store metadata in a file
    filename = "{group}_metadata.fold{index}.npy".format(
        group=group,
        index=fold_index)
    np.save(
        os.path.join(output_dir, filename),
        metadata)

    # flush changes to a disk
    features.flush()
    del features


def generate_folds(
        input_dir,
        input_ext,
        output_dir,
        group,
        input_shape,
        normalize,
        output_shape):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = glob(os.path.join(input_dir, '*' + input_ext))

    uids = common.group_uids(files)

    fold_index = 1
    while has_uids(uids):
        print("[{group}] Fold {index}".format(group=group, index=fold_index))

        generate_fold(
            uids,
            input_dir,
            input_ext,
            output_dir,
            group,
            fold_index,
            input_shape,
            normalize,
            output_shape)

        fold_index += 1


def normalize_fb(spectrogram):

    # Mean and Variance Normalization
    spectrogram = speechpy.processing.cmvn(
        spectrogram,
        variance_normalization=True)

    # MinMax Scaler, scale values between (0,1)
    normalized = (
        (spectrogram - np.min(spectrogram)) /
        (np.max(spectrogram) - np.min(spectrogram))
    )

    # Rotate 90deg
    normalized = np.swapaxes(normalized, 0, 1)

    # Reshape, tensor 3d
    (height, width) = normalized.shape
    normalized = normalized.reshape(height, width, COLOR_DEPTH)

    assert normalized.dtype == DATA_TYPE
    assert np.max(normalized) == 1.0
    assert np.min(normalized) == 0.0

    return normalized


if __name__ == "__main__":
    start = time.time()

    # fb
    generate_folds(
        os.path.join(common.DATASET_DIST, 'test'),
        '.fb.npz',
        output_dir='build/folds',
        group='test',
        input_shape=(WIDTH, FB_HEIGHT),
        normalize=normalize_fb,
        output_shape=(FB_HEIGHT, WIDTH, COLOR_DEPTH)
    )
    generate_folds(
        os.path.join(common.DATASET_DIST, 'train'),
        '.fb.npz',
        output_dir='build/folds',
        group='train',
        input_shape=(WIDTH, FB_HEIGHT),
        normalize=normalize_fb,
        output_shape=(FB_HEIGHT, WIDTH, COLOR_DEPTH)
    )

    end = time.time()
    print("It took [s]: ", end - start)
