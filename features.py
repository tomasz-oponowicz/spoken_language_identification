import glob
import os
import time

import numpy as np
import soundfile as sf

from constants import *
import common


# source1: https://bit.ly/2GSsEgw
# source2: https://bit.ly/2sWxHIc
def generate_fb_and_mfcc(signal, sample_rate):

    # Pre-Emphasis
    pre_emphasis = 0.97
    emphasized_signal = np.append(
        signal[0],
        signal[1:] - pre_emphasis * signal[:-1])

    # Framing
    frame_size = 0.025
    frame_stride = 0.01

    # Convert from seconds to samples
    frame_length, frame_step = (
        frame_size * sample_rate,
        frame_stride * sample_rate)
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))

    # Make sure that we have at least 1 frame
    num_frames = int(
        np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))

    # Pad Signal to make sure that all frames have equal
    # number of samples without truncating any samples
    # from the original signal
    pad_signal = np.append(emphasized_signal, z)

    indices = (
        np.tile(np.arange(0, frame_length), (num_frames, 1)) +
        np.tile(
            np.arange(0, num_frames * frame_step, frame_step),
            (frame_length, 1)
        ).T
    )
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # Window
    frames *= np.hamming(frame_length)

    # Fourier-Transform and Power Spectrum
    NFFT = 512

    # Magnitude of the FFT
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))

    # Power Spectrum
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))

    # Filter Banks
    nfilt = 40

    low_freq_mel = 0

    # Convert Hz to Mel
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))

    # Equally spaced in Mel scale
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)

    # Convert Mel to Hz
    hz_points = (700 * (10**(mel_points / 2595) - 1))
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)

    # Numerical Stability
    filter_banks = np.where(
        filter_banks == 0,
        np.finfo(float).eps,
        filter_banks)

    # dB
    filter_banks = 20 * np.log10(filter_banks)

    # MFCCs
    # num_ceps = 12
    # cep_lifter = 22

    # ### Keep 2-13
    # mfcc = dct(
    #     filter_banks,
    #     type=2,
    #     axis=1,
    #     norm='ortho'
    # )[:, 1 : (num_ceps + 1)]

    # (nframes, ncoeff) = mfcc.shape
    # n = np.arange(ncoeff)
    # lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    # mfcc *= lift

    return filter_banks


def process_audio(input_dir, debug=False):
    files = []

    extensions = ['*.flac']
    for extension in extensions:
        files.extend(glob.glob(os.path.join(input_dir, extension)))

    for file in files:
        if debug:
            file = ('build/test/'
                    'de_f_63f5b79c76cf5a1a4bbd1c40f54b166e.fragment1.flac')
            start = time.time()

        print(file)

        signal, sample_rate = sf.read(file)
        assert len(signal) > 0
        assert sample_rate == 22050

        fb = generate_fb_and_mfcc(signal, sample_rate)
        fb = fb.astype(DATA_TYPE, copy=False)

        assert fb.dtype == DATA_TYPE
        assert fb.shape == (WIDTH, FB_HEIGHT)

        # .npz extension is added automatically
        file_without_ext = os.path.splitext(file)[0]

        np.savez_compressed(file_without_ext + '.fb', data=fb)

        if debug:
            end = time.time()
            print("It took [s]: ", end - start)

            # data is casted to uint8, i.e. (0, 255)
            import imageio
            imageio.imwrite('fb_image.png', fb)

            exit(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate various features from audio samples.')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.set_defaults(debug=False)

    args = parser.parse_args()

    if args.debug:
        process_audio(os.path.join(common.DATASET_DIST, 'train'), debug=True)
    else:
        process_audio(os.path.join(common.DATASET_DIST, 'test'))
        process_audio(os.path.join(common.DATASET_DIST, 'train'))
