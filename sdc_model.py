SEED = 42

import os
os.environ['PYTHONHASHSEED'] = '42'

import random
random.seed(SEED)

import numpy as np
np.random.seed(SEED)

import time

from sklearn.utils import shuffle
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.externals import joblib

from constants import *
import common

BASE_DIR = 'mfcc'

# source: http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
# covariance_type : {‘full’, ‘tied’, ‘diag’, ‘spherical’}
COVARIANCE = 'diag'
REGULARIZATION = 1e-6
TOLERANCE = 1e-3
N_INIT = 1

SAMPLE_LENGTH = 1000
STEP = 1

K = 256
P = 1
V = 3

def train(X, k, file):
    if os.path.isfile(file):
        return joblib.load(file)

    gmm = GaussianMixture(
        n_components=k,
        covariance_type=COVARIANCE,
        n_init=N_INIT,
        tol=TOLERANCE,
        reg_covar=REGULARIZATION,
        random_state=SEED
    )

    gmm.fit(X)
    joblib.dump(gmm, file)

    return gmm

def remove_silence(X):
    s = np.sum(np.abs(X), axis=1)
    t = np.mean(s) * 0.5
    return X[np.where(s > t)]

def build_filename(language):
    return "{l}_gmm_n={n}_p={p}_t={t}_v={v}.pkl".format(
        l=language,
        n=K,
        p=int(P * 100),
        t=COVARIANCE,
        v=V
    )

en_train = []
de_train = []
es_train = []
for i in [1,2,3,4,5,6,7,9,10,11]:
    en_train.append(np.load("{0}/en_train.fold{1}.npz".format(BASE_DIR, i))[DATA_KEY])
    de_train.append(np.load("{0}/de_train.fold{1}.npz".format(BASE_DIR, i))[DATA_KEY])
    es_train.append(np.load("{0}/es_train.fold{1}.npz".format(BASE_DIR, i))[DATA_KEY])

en_train = np.concatenate(en_train)
de_train = np.concatenate(de_train)
es_train = np.concatenate(es_train)

# en_train = remove_silence(en_train)
# de_train = remove_silence(de_train)
# es_train = remove_silence(es_train)

size = np.min((len(en_train), len(de_train), len(es_train)))
percent = int(P * size)

en_train = shuffle(en_train, random_state=SEED)[:percent]
de_train = shuffle(de_train, random_state=SEED)[:percent]
es_train = shuffle(es_train, random_state=SEED)[:percent]

print(en_train.shape)
print(de_train.shape)
print(es_train.shape)

assert len(en_train) == len(de_train)
assert len(de_train) == len(es_train)

print("Train...")
start = time.time()

print("==> en")
en_gmm = train(en_train, K, build_filename('en'))
print("==> de")
de_gmm = train(de_train, K, build_filename('de'))
print("==> es")
es_gmm = train(es_train, K, build_filename('es'))

end = time.time()
print("It trained in [s]: ", end - start)

print("Test...")
start = time.time()

accuracies = []
languages = ['en', 'de', 'es']

for fold in range(12, 15):
    for language_idx, language in enumerate(languages):
        file = "{0}/{1}_train.fold{2}.npz".format(BASE_DIR, language, fold)

        samples = np.load(file)[DATA_KEY]

        correct_samples = 0
        samples_count = int(len(samples) / SAMPLE_LENGTH)

        # print(file, samples.shape, samples_count, language_idx)

        for sample_idx in range(0, samples_count):
            begin = SAMPLE_LENGTH * sample_idx
            end = begin + SAMPLE_LENGTH

            vectors = samples[begin:end:STEP]
            # vectors = remove_silence(vectors)

            results = np.zeros(len(languages))
            for vector in vectors:
                vector = vector.reshape((1, 24))

                results[0] += en_gmm.score_samples(vector)
                results[1] += de_gmm.score_samples(vector)
                results[2] += es_gmm.score_samples(vector)

            if np.argmax(results) == language_idx:
                correct_samples += 1

        accuracy = correct_samples / samples_count
        accuracies.append(accuracy)

        print("{lang} acc: {acc:.2f}".format(lang=language, acc=accuracy))

print("==> acc: {acc:.2f}".format(acc=np.mean(accuracies)))

end = time.time()
print("It tested in [s]: ", end - start)
