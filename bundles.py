import imageio
import glob
import os
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import time


def remove_extension(file):
    return os.path.splitext(file)[0]


def get_filename(file):
    return os.path.basename(remove_extension(file))


def images_to_bundle(input_dir, group):
    files = glob.glob(os.path.join(input_dir, '*.png'))

    # shuffle files
    files = sorted(files)
    files = shuffle(files, random_state=47)

    labels = []
    features = []

    for index, file in enumerate(files):
        print(file)

        filename = get_filename(file)
        metadata = filename.split('_')
        language = metadata[0]
        gender = metadata[1]
        info = metadata[2]

        image = imageio.imread(file)

        labels.append([language, gender, info])
        features.append(image)

    np.savez("{0}.npz".format(group), labels=np.array(labels), features=np.array(features))


if __name__ == "__main__":
    start = time.time()

    images_to_bundle('./build/test', 'test')
    images_to_bundle('./build/valid', 'valid')
    images_to_bundle('./build/train', 'train')

    end = time.time()
    print("It took [s]: ", end - start)

    # bundle = np.load('test.npz')
    # print(bundle['labels'].shape)
    # print(bundle['features'].shape)
    # image = bundle['features'][0]
    # plt.imshow(image)
    # plt.show()
