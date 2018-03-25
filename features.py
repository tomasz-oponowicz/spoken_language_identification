import librosa as lr
import glob
import imageio
import os
import librosa.display
import matplotlib.pyplot as plt
import time
import numpy as np
import soundfile as sf

from constants import *

def audio_to_spectrogram(file):

    # loading samples with soundfile is much faster than librosa
    signal, sample_rate = sf.read(file)

    assert sample_rate == 22050

    # defaults
    n_mels = 128
    n_fft = 2048
    hop_length = int(n_fft / 4)

    spectrogram = lr.feature.melspectrogram(signal, sr=sample_rate,
        n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)

    log_spectrogram = lr.amplitude_to_db(spectrogram)

    # trim
    log_spectrogram = log_spectrogram[:, 0:WIDTH]

    assert log_spectrogram.shape[0] == HEIGHT
    assert log_spectrogram.shape[1] >= WIDTH

    return log_spectrogram, sample_rate, hop_length

def normalize(spectrogram):

    # scale values between (0,1)
    normalized = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram))
    normalized = normalized.astype(DATA_TYPE)
    normalized = normalized.reshape(HEIGHT, WIDTH, COLOR_DEPTH)

    assert normalized.shape == (HEIGHT, WIDTH, COLOR_DEPTH)
    assert normalized.dtype == DATA_TYPE

    # ignore precision issues
    assert abs(1.0 - np.max(normalized)) < 0.01
    assert abs(0.0 - np.min(normalized)) < 0.01

    return normalized

def process_audio(input_dir, debug=False):
    files = []

    extensions = ['*.flac']
    for extension in extensions:
        files.extend(glob.glob(os.path.join(input_dir, extension)))

    for file in files:
        print(file)

        start = time.time()

        spectrogram, sample_rate, hop_length = audio_to_spectrogram(file)

        file_without_ext = os.path.splitext(file)[0]
        normalized = normalize(spectrogram)

        # .npz extension is added automatically
        np.savez_compressed(file_without_ext, data=normalized)

        if debug:
            end = time.time()
            print("It took [s]: ", end - start)

            # data is casted to uint8, i.e. (0, 255)
            imageio.imwrite('spectrogram_image.png', spectrogram)

            # sample rate and hop length parameters
            # are used to render the time axis
            plt.figure()
            lr.display.specshow(spectrogram, sr=sample_rate, hop_length=hop_length,
                                x_axis='time', y_axis='mel')
            plt.title('mel power spectrogram')
            plt.colorbar(format='%+02.0f dB')
            plt.savefig('spectrogram_chart.png')

            exit(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate spectrograms from audio samples.')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.set_defaults(debug=False)

    args = parser.parse_args()

    if args.debug:
        process_audio('build/train', debug=True)
    else:
        process_audio('build/valid')
        process_audio('build/test')
        process_audio('build/train')
