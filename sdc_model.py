SEED = 42

import os
os.environ['PYTHONHASHSEED'] = '42'

import random
random.seed(SEED)

import numpy as np
np.random.seed(SEED)

import time

from sklearn.utils import shuffle
from sklearn.mixture import GaussianMixture

from constants import *
import common

BASE_DIR = 'mfcc'

# source: http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
K = 12

# covariance_type : {‘full’, ‘tied’, ‘diag’, ‘spherical’}
COVARIANCE = 'spherical'
REGULARIZATION = 1e-5

SAMPLE_LENGTH = 1000
STEP = 10

en_train = []
de_train = []
es_train = []
for i in [2,3,4,5,6,7,9,10,11,12,13,14]:
	en_train.append(np.load("{0}/en_train.fold{1}.npz".format(BASE_DIR, i))[DATA_KEY])
	de_train.append(np.load("{0}/de_train.fold{1}.npz".format(BASE_DIR, i))[DATA_KEY])
	es_train.append(np.load("{0}/es_train.fold{1}.npz".format(BASE_DIR, i))[DATA_KEY])

en_train = np.concatenate(en_train);
de_train = np.concatenate(de_train);
es_train = np.concatenate(es_train);

assert len(en_train) == len(de_train)
assert len(de_train) == len(es_train)

partial = int(0.1 * len(en_train))
en_train = shuffle(en_train, random_state=SEED)[:partial]
de_train = shuffle(de_train, random_state=SEED)[:partial]
es_train = shuffle(es_train, random_state=SEED)[:partial]

print(en_train[0])

print(en_train.shape)
print(de_train.shape)
print(es_train.shape)

en_gmm = GaussianMixture(n_components=K, covariance_type=COVARIANCE, reg_covar=REGULARIZATION, random_state=SEED)
de_gmm = GaussianMixture(n_components=K, covariance_type=COVARIANCE, reg_covar=REGULARIZATION, random_state=SEED)
es_gmm = GaussianMixture(n_components=K, covariance_type=COVARIANCE, reg_covar=REGULARIZATION, random_state=SEED)

print("Train...")
start = time.time()

en_gmm.fit(en_train)
de_gmm.fit(de_train)
es_gmm.fit(es_train)

end = time.time()
print("It trained in [s]: ", end - start)

print("Test...")
start = time.time()

for fold in range(1, 2):
	accuracies = []

	for language_idx, language in enumerate(['en', 'de', 'es']):
		file = "{0}/{1}_train.fold{2}.npz".format(BASE_DIR, language, fold)
		print(file)

		samples = np.load(file)[DATA_KEY]

		correct = 0
		size = int(len(samples) / SAMPLE_LENGTH)

		print(samples.shape, size, language_idx)

		for i in range(0, size):
			begin = SAMPLE_LENGTH * i
			end = SAMPLE_LENGTH * (i + 1)

			sample = samples[begin:end:STEP]

			scores = [
				en_gmm.score(sample),
				de_gmm.score(sample),
				es_gmm.score(sample)
			]

			if np.argmax(scores) == language_idx:
				correct += 1

		accuracy = correct / size
		accuracies.append(accuracy)

		print("{lang} acc: {acc:.2f}".format(lang=language, acc=accuracy))

	print("fold{fold} acc: {acc:.2f}".format(fold=fold, acc=np.mean(accuracies)))

end = time.time()
print("It tested in [s]: ", end - start)
