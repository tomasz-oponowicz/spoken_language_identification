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

K = 128
SAMPLE_LENGTH = 1000

en_train = []
de_train = []
es_train = []
for i in [2,3,4,5,6,7,9,10,11,12,13,14]:
	en_train.append(np.load("sdc/en_train.fold{0}.npz".format(i))[DATA_KEY])
	de_train.append(np.load("sdc/de_train.fold{0}.npz".format(i))[DATA_KEY])
	es_train.append(np.load("sdc/es_train.fold{0}.npz".format(i))[DATA_KEY])

en_train = np.concatenate(en_train);
de_train = np.concatenate(de_train);
es_train = np.concatenate(es_train);

assert len(en_train) == len(de_train)
assert len(de_train) == len(es_train)

partial = int(0.005 * len(en_train))
en_train = shuffle(en_train, random_state=SEED)[:partial]
de_train = shuffle(de_train, random_state=SEED)[:partial]
es_train = shuffle(es_train, random_state=SEED)[:partial]

print(en_train.shape)
print(de_train.shape)
print(es_train.shape)

en_gmm = GaussianMixture(n_components=K, covariance_type='diag')
de_gmm = GaussianMixture(n_components=K, covariance_type='diag')
es_gmm = GaussianMixture(n_components=K, covariance_type='diag')

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

	for language_idx, language in enumerate(LANGUAGES):
		samples = np.load("sdc/{0}_train.fold{1}.npz".format(language, fold))[DATA_KEY]
		print(samples.shape)

		correct = 0
		size = int(len(samples) / SAMPLE_LENGTH)
		for i in range(0, size):
			begin = SAMPLE_LENGTH * i
			end = SAMPLE_LENGTH * (i + 1)

			sample = samples[begin:end,:]

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
