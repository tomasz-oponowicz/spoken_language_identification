import dask.array as da


def images_to_bundle(source, target, name):
    da.image.imread(source).to_hdf5(target, name)


if __name__ == "__main__":
    images_to_bundle('./build/valid/*.png', 'valid.h5', 'data')
    # images_to_bundle('./build/test/*.png', 'test.h5', 'data')
    # images_to_bundle('./build/train/*.png', 'train.h5', 'data')
