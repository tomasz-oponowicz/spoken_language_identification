import os
from glob import glob
import shutil

import numpy as np
from keras.models import load_model

import common
import features
import folds
from audio_toolbox import ffmpeg, sox
from constants import *

MODEL_FILE = 'model.f1_95.h5'
FRAGMENT_DURATION = 10
INPUT_SHAPE = (FB_HEIGHT, WIDTH, COLOR_DEPTH)

input_file = 'KDP-Episode-005.mp3'

# normalize input

output_dir = '.temp'
transcoded_file = os.path.join(output_dir, 'test.transcoded.flac')
normalized_file = os.path.join(output_dir, 'test.normalized.flac')
trimmed_file = os.path.join(output_dir, 'test.trimmed.flac')
fragment_file = os.path.join(output_dir, 'test.fragment@n.flac')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

ffmpeg.transcode(input_file, transcoded_file)

sox.remove_silence(transcoded_file, trimmed_file)

duration = sox.get_duration(trimmed_file)
duration = int((duration // FRAGMENT_DURATION) * FRAGMENT_DURATION)
sox.normalize(trimmed_file, normalized_file, duration_in_sec=duration)

sox.split(normalized_file, fragment_file, FRAGMENT_DURATION)

os.remove(transcoded_file)
os.remove(normalized_file)
os.remove(trimmed_file)

# generate features

features.process_audio(output_dir)

samples = []

files = glob(os.path.join(output_dir, '*.npz'))
for file in files:
    sample = np.load(file)[DATA_KEY]
    sample = folds.normalize_fb(sample)

    assert sample.shape == INPUT_SHAPE
    assert sample.dtype == DATA_TYPE

    samples.append(sample)

samples = np.array(samples)
# shutil.rmtree(output_dir)

# predict language

label_binarizer, clazzes = common.build_label_binarizer()
model = load_model(MODEL_FILE)

results = model.predict(samples)

counters = np.zeros(len(LANGUAGES))
for result in results:
	counters[np.argmax(result)] += 1

language_idx = np.argmax(counters)
confidence = counters[language_idx] / np.sum(counters)

print('Counters:', counters)
print('Language:', clazzes[language_idx])
print('Confidence:', confidence)