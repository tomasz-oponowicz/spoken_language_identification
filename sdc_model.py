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

from constants import *
import common

BASE_DIR = 'mfcc'

# source: http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
K = 40

# covariance_type : {‘full’, ‘tied’, ‘diag’, ‘spherical’}
COVARIANCE = 'diag'
REGULARIZATION = 1e-6
TOLERANCE = 1e-3
N_INIT = 1

SAMPLE_LENGTH = 1000
STEP = 10

def bicGMMModelSelection(X, k):
    best_gmm = None
    best_bic = np.infty

    for n_components in np.arange(k, k + 1):
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=COVARIANCE,
            n_init=N_INIT,
            tol=TOLERANCE,
            reg_covar=REGULARIZATION,
            random_state=SEED
        )

        gmm.fit(X)
        bic = gmm.aic(X)

        print("n_components: {0}, bic: {1:,.2f}".format(n_components, bic))

        if bic < best_bic:
            best_bic = bic
            best_n_components = n_components
            best_gmm = gmm

    print('==> best_n_components:', best_n_components)

    return best_gmm

en_train = []
de_train = []
es_train = []
for i in [1,2,3,4,5,6,7,9,10,11]:
    en_train.append(np.load("{0}/en_train.fold{1}.npz".format(BASE_DIR, i))[DATA_KEY])
    de_train.append(np.load("{0}/de_train.fold{1}.npz".format(BASE_DIR, i))[DATA_KEY])
    es_train.append(np.load("{0}/es_train.fold{1}.npz".format(BASE_DIR, i))[DATA_KEY])

en_train = np.concatenate(en_train);
de_train = np.concatenate(de_train);
es_train = np.concatenate(es_train);

print(en_train.shape)
print(de_train.shape)
print(es_train.shape)

assert len(en_train) == len(de_train)
assert len(de_train) == len(es_train)

partial = int(0.3 * len(en_train))
en_train = shuffle(en_train, random_state=SEED)[:partial]
de_train = shuffle(de_train, random_state=SEED)[:partial]
es_train = shuffle(es_train, random_state=SEED)[:partial]

print("Train...")
start = time.time()

print("==> de")
de_gmm = bicGMMModelSelection(de_train, 25)
print("==> es")
es_gmm = bicGMMModelSelection(es_train, 30)

end = time.time()
print("It trained in [s]: ", end - start)

print("Test...")
start = time.time()

for fold in range(12, 15):
    accuracies = []

    for language_idx, language in enumerate(['de', 'es']):
        file = "{0}/{1}_train.fold{2}.npz".format(BASE_DIR, language, fold)

        samples = np.load(file)[DATA_KEY]

        correct = 0
        size = int(len(samples) / SAMPLE_LENGTH)

        print(file, samples.shape, size, language_idx)

        for i in range(0, size):
            begin = SAMPLE_LENGTH * i
            end = SAMPLE_LENGTH * (i + 1)

            sample = samples[begin:end:STEP]

            scores = [
                # en_gmm.score(sample),
                # np.mean(np.max(de_gmm.predict_proba(sample), axis=1)),
                # np.mean(np.max(es_gmm.predict_proba(sample), axis=1))
                np.max(de_gmm.score_samples(sample)),
                np.max(es_gmm.score_samples(sample))
            ]

            if np.argmax(scores) == language_idx:
                correct += 1

        accuracy = correct / size
        accuracies.append(accuracy)

        print("{lang} acc: {acc:.2f}".format(lang=language, acc=accuracy))

    print("==> fold{fold} acc: {acc:.2f}".format(fold=fold, acc=np.mean(accuracies)))

end = time.time()
print("It tested in [s]: ", end - start)
