import imageio
import glob
import os
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import time
import numpy as np

SEED = 42

COLOR_DEPTH = 1
WIDTH = 192
HEIGHT = 192

def remove_extension(file):
    return os.path.splitext(file)[0]


def get_filename(file):
    return os.path.basename(remove_extension(file))


def images_to_bundle(input_dir, group):
    files = glob.glob(os.path.join(input_dir, '*.png'))

    # shuffle files
    files = sorted(files)
    files = shuffle(files, random_state=SEED)

    metadata = []

    features = np.memmap("{0}_features.npy".format(group), dtype='float16', mode='w+',
        shape=(len(files), WIDTH, HEIGHT, COLOR_DEPTH))

    for index, file in enumerate(files):
        print(file)

        filename = get_filename(file)
        language = filename.split('_')[0]
        gender = filename.split('_')[1]

        image = imageio.imread(file)
        image = image.reshape(WIDTH, HEIGHT, COLOR_DEPTH)
        image = np.divide(image, 255.)

        metadata.append([language, gender, filename])

        features[index] = image

    np.save("{0}_metadata.npy".format(group), metadata)

    # flush changes to disk
    features.flush()
    del features


if __name__ == "__main__":
    start = time.time()

    images_to_bundle('./build/valid', 'valid')
    images_to_bundle('./build/test', 'test')
    images_to_bundle('./build/train', 'train')

    end = time.time()
    print("It took [s]: ", end - start)
