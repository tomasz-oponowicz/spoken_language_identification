import imageio
import glob
import os
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import time
import numpy as np

from constants import *

def remove_extension(file):
    return os.path.splitext(file)[0]


def get_filename(file):
    return os.path.basename(remove_extension(file))


def images_to_bundle(input_dir, group):
    files = glob.glob(os.path.join(input_dir, '*.npz'))

    # shuffle files
    files = sorted(files)
    files = shuffle(files, random_state=SEED)

    metadata = []

    features = np.memmap("{0}_features.npy".format(group), dtype=DATA_TYPE, mode='w+',
        shape=(len(files), HEIGHT, WIDTH, COLOR_DEPTH))

    for index, file in enumerate(files):
        print(file)

        filename = get_filename(file)
        language = filename.split('_')[0]
        gender = filename.split('_')[1]

        spectogram = np.load(file)[DATA_KEY]

        assert spectogram.shape == (HEIGHT, WIDTH, COLOR_DEPTH)
        assert spectogram.dtype == DATA_TYPE

        metadata.append([language, gender, filename])

        features[index] = spectogram

    np.save("{0}_metadata.npy".format(group), metadata)

    # flush changes to a disk
    features.flush()
    del features


if __name__ == "__main__":
    start = time.time()

    images_to_bundle('./build/valid', 'valid')
    images_to_bundle('./build/test', 'test')
    images_to_bundle('./build/train', 'train')

    end = time.time()
    print("It took [s]: ", end - start)
