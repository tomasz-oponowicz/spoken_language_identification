import os
import glob
import shutil
import tempfile

import numpy as np

import common
import features
import folds
from audio_toolbox import ffmpeg, sox
from constants import *


def normalize(input_file):
    temp_dir = tempfile.mkdtemp()

    transcoded_file = os.path.join(temp_dir, 'transcoded.flac')
    ffmpeg.transcode(input_file, transcoded_file)

    if not args.keep_silence:
        trimmed_file = os.path.join(temp_dir, 'trimmed.flac')
        sox.remove_silence(
            transcoded_file,
            trimmed_file,
            min_duration_sec=args.silence_min_duration_sec,
            threshold=args.silence_threshold)
    else:
        trimmed_file = transcoded_file

    duration = sox.get_duration(trimmed_file)
    duration = int((duration // FRAGMENT_DURATION) * FRAGMENT_DURATION)

    normalized_file = os.path.join(temp_dir, 'normalized.flac')
    sox.normalize(trimmed_file, normalized_file, duration_in_sec=duration)

    return normalized_file, temp_dir


def load_samples(normalized_file):
    temp_dir = tempfile.mkdtemp()

    fragmented_file = os.path.join(temp_dir, 'fragment@n.flac')
    sox.split(normalized_file, fragmented_file, FRAGMENT_DURATION)

    features.process_audio(temp_dir)

    samples = []
    for file in glob.glob(os.path.join(temp_dir, '*.npz')):
        sample = np.load(file)[DATA_KEY]
        sample = folds.normalize_fb(sample)

        assert sample.shape == INPUT_SHAPE
        assert sample.dtype == DATA_TYPE

        samples.append(sample)

    samples = np.array(samples)

    return samples, temp_dir


def predict(model_file):
    import keras.models

    _, languages = common.build_label_binarizer()

    model = keras.models.load_model(model_file)
    results = model.predict(samples)

    scores = np.zeros(len(languages))
    for result in results:
        scores[np.argmax(result)] += 1

    return scores, languages


def clean(paths):
    for path in paths:
        shutil.rmtree(path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test the model.')
    parser.add_argument(
        'input',
        help='a path to an audio file')
    parser.add_argument(
        '--model',
        dest='model',
        help='a path to the H5 model file; the default is `model.h5`')
    parser.add_argument(
        '--silence-threshold',
        dest='silence_threshold',
        type=float,
        help=("indicates what sample value you should treat as silence; "
              "the default is `0.5`"))
    parser.add_argument(
        '--silence-min-duration',
        dest='silence_min_duration_sec',
        type=float,
        help=("specifies a period of silence that must exist before audio is "
              "not copied any more; the default is `0.1`"))
    parser.add_argument(
        '--keep-silence',
        dest='keep_silence',
        action='store_true',
        help='don\'t remove silence from samples')
    parser.add_argument(
        '--keep-temp-files',
        dest='keep_temp_files',
        action='store_true',
        help='don\'t remove temporary files when done')
    parser.add_argument(
        '--verbose',
        dest='verbose',
        action='store_true',
        help='print more logs')

    parser.set_defaults(
        model='model.h5',
        keep_silence=False,
        silence_min_duration_sec=0.1,
        silence_threshold=0.5,
        keep_temp_files=False,
        verbose=False)

    args = parser.parse_args()

    if not args.verbose:

        # supress all warnings
        import warnings
        warnings.filterwarnings("ignore")

        # supress tensorflow warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    normalized_file, normalized_dir = normalize(args.input)
    samples, samples_dir = load_samples(normalized_file)

    if not args.keep_temp_files:
        clean((normalized_dir, samples_dir))

    scores, languages = predict(args.model)

    total = np.sum(scores)
    for language_idx, language in enumerate(languages):
        score = scores[language_idx]
        print("{language}: {percent:.2f}% ({amount:.0f})".format(
            language=language,
            percent=(score / total) * 100,
            amount=score))
