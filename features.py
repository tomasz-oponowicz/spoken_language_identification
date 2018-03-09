import librosa as lr
import glob
import imageio
import os
import librosa.display
import matplotlib.pyplot as plt
import time


def audio_to_spectrogram(path, height=192, width=192):
    signal, sample_rate = lr.load(path, res_type='kaiser_fast')

    hop_length = signal.shape[0] // float(width)
    spectrogram = lr.feature.melspectrogram(signal, n_mels=height, hop_length=int(hop_length))
    log_spectrogram = lr.amplitude_to_db(spectrogram)

    # trim
    log_spectrogram = log_spectrogram[:, 0:width]

    return log_spectrogram, sample_rate, hop_length


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

        # use loseless image format, e.g. png
        imageio.imwrite(file_without_ext + '.png', spectrogram, compress_level=6)

        if debug:
            end = time.time()
            print("It took [s]: ", end - start)

            # Make a new figure
            plt.figure(figsize=(12, 4))

            # Display the spectrogram on a mel scale
            # sample rate and hop length parameters are used to render the time axis
            lr.display.specshow(spectrogram, sr=sample_rate, hop_length=int(hop_length),
                                x_axis='time', y_axis='mel')

            # Put a descriptive title on the plot
            plt.title('mel power spectrogram')

            # draw a color bar
            plt.colorbar(format='%+02.0f dB')

            # Make the figure layout compact
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    process_audio('build/valid')
    # process_audio('build/test')
    # process_audio('build/train')
