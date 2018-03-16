import time
import numpy as np

from sklearn import preprocessing

from keras.models import Model, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.layers import Dropout, Input, BatchNormalization
from keras.optimizers import Nadam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

in_dim = (192,192,1)
out_dim = 3
batch_size = 32

def create_model():
    lb = preprocessing.LabelBinarizer()
    lb.fit(['en', 'de', 'es'])
    print(lb.classes_)

    train = np.load('train.npz')
    valid = np.load('valid.npz')
    test = np.load('test.npz')

    train_metadata = train['labels']
    train_labels = lb.transform(train_metadata[:, 0])
    train_features = train['features']
    train_features = train_features.reshape((len(train_features), 192, 192, 1))
    train_features = np.divide(train_features, 255.)
    print(train_labels.shape, train_features.shape, np.max(train_features), np.min(train_features))

    valid_metadata = valid['labels']
    valid_labels = lb.transform(valid_metadata[:, 0])
    valid_features = valid['features']
    valid_features = valid_features.reshape((len(valid_features), 192, 192, 1))
    valid_features = np.divide(valid_features, 255.)
    print(valid_labels.shape, valid_features.shape, np.max(valid_features), np.min(valid_features))

    test_metadata = test['labels']
    test_labels = lb.transform(test_metadata[:, 0])
    test_features = test['features']
    test_features = test_features.reshape((len(test_features), 192, 192, 1))
    test_features = np.divide(test_features, 255.)
    print(test_labels.shape, test_features.shape, np.max(test_features), np.min(test_features))

    i = Input(shape=in_dim)
    m = Conv2D(16, (3, 3), activation='elu', padding='same')(i)
    m = MaxPooling2D()(m)
    m = Conv2D(32, (3, 3), activation='elu', padding='same')(m)
    m = MaxPooling2D()(m)
    m = Conv2D(64, (3, 3), activation='elu', padding='same')(m)
    m = MaxPooling2D()(m)
    m = Conv2D(128, (3, 3), activation='elu', padding='same')(m)
    m = MaxPooling2D()(m)
    m = Conv2D(256, (3, 3), activation='elu', padding='same')(m)
    m = MaxPooling2D()(m)
    m = Flatten()(m)
    m = Dense(512, activation='elu')(m)
    m = Dropout(0.5)(m)
    o = Dense(out_dim, activation='softmax')(m)

    model = Model(inputs=i, outputs=o)
    model.summary()
    
    model.compile(loss='categorical_crossentropy', optimizer=Nadam(lr=1e-4), metrics=['accuracy'])
    model.fit(train_features, train_labels, epochs=10, verbose=1, validation_data=(valid_features, valid_labels))

    print(model.evaluate(test_features, test_labels))


if __name__ == "__main__":
    start = time.time()

    create_model()

    end = time.time()
    print("It took [s]: ", end - start)
