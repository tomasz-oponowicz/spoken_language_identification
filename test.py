from keras.models import load_model

import common
from constants import *

model = load_model('model.h5')

input_shape = (FB_HEIGHT, WIDTH, COLOR_DEPTH)
label_binarizer, clazzes = common.build_label_binarizer()

test_labels, test_features, test_metadata = common.load_data(label_binarizer,
    'fb', 'test', [1], input_shape)

common.test(test_labels, test_features, test_metadata, model, clazzes)
