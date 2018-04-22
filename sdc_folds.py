import os
from glob import glob
import time

import numpy
from sklearn.utils import shuffle
import speechpy

from constants import *
import common

HEIGHT = MFCC_HEIGHT * 2

PARAM_TYPE = numpy.float32
DATA_TYPE = 'float64'

def compute_delta(features,
                  win=3,
                  method='filter',
                  filt=numpy.array([.25, .5, .25, 0, -.25, -.5, -.25])):
    """features is a 2D-ndarray  each row of features is a a frame

    :param features: the feature frames to compute the delta coefficients
    :param win: parameter that set the length of the computation window.
            The size of the window is (win x 2) + 1
    :param method: method used to compute the delta coefficients
        can be diff or filter
    :param filt: definition of the filter to use in "filter" mode, default one
        is similar to SPRO4:  filt=numpy.array([.2, .1, 0, -.1, -.2])

    :return: the delta coefficients computed on the original features.
    """
    # First and last features are appended to the begining and the end of the
    # stream to avoid border effect
    x = numpy.zeros((features.shape[0] + 2 * win, features.shape[1]), dtype=PARAM_TYPE)
    x[:win, :] = features[0, :]
    x[win:-win, :] = features
    x[-win:, :] = features[-1, :]

    delta = numpy.zeros(x.shape, dtype=PARAM_TYPE)

    if method == 'diff':
        filt = numpy.zeros(2 * win + 1, dtype=PARAM_TYPE)
        filt[0] = -1
        filt[-1] = 1

    for i in range(features.shape[1]):
        delta[:, i] = numpy.convolve(features[:, i], filt)

    return delta[win:-win, :]

def shifted_delta_cepstral(cep, d=1, p=3, k=7):
    """
    Compute the Shifted-Delta-Cepstral features for language identification

    :param cep: matrix of feature, 1 vector per line
    :param d: represents the time advance and delay for the delta computation
    :param k: number of delta-cepstral blocks whose delta-cepstral
       coefficients are stacked to form the final feature vector
    :param p: time shift between consecutive blocks.

    return: cepstral coefficient concatenated with shifted deltas
    """

    y = numpy.r_[numpy.resize(cep[0, :], (d, cep.shape[1])),
                 cep,
                 numpy.resize(cep[-1, :], (k * 3 + d, cep.shape[1]))]

    delta = compute_delta(y, win=d, method='diff')
    sdc = numpy.empty((cep.shape[0], cep.shape[1] * k))

    idx = numpy.zeros(delta.shape[0], dtype='bool')
    for ii in range(k):
        idx[d + ii * p] = True
    for ff in range(len(cep)):
        sdc[ff, :] = delta[idx, :].reshape(1, -1)
        idx = numpy.roll(idx, 1)

    # return numpy.hstack((cep, sdc))
    return sdc

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

        entries = numpy.load(fold_file)[DATA_KEY]

        # source: https://github.com/astorfi/speechpy/blob/master/speechpy/processing.py

        # cepstral mean and variance normalization
        entries = speechpy.processing.cmvn(entries, variance_normalization=True)

        # # deltas
        # deltas = speechpy.processing.derivative_extraction(entries, DeltaWindows=2)
        # entries = numpy.hstack((entries, deltas))
        # assert entries.dtype == DTYPE

        # sdc
        entries = shifted_delta_cepstral(entries, d=3, p=2, k=2)

        assert entries.dtype == DATA_TYPE

        for entry in entries:
            features.append(entry)

    # create a file array
    filename = "{language}_{group}.fold{index}".format(
        language=input_langauge,
        group=output_group,
        index=fold_index
    )

    features = numpy.array(features)
    print(features.shape)

    assert features.shape == (processed_files * WIDTH, HEIGHT)
    assert entries.dtype == DATA_TYPE

    numpy.savez_compressed(os.path.join(output_dir, filename), data=features)

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
