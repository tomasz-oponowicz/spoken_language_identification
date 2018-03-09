import librosa as lr
import os
import glob
import imageio
import os


def mp3_to_img(path, height=192, width=192):
    signal, sr = lr.load(path, res_type='kaiser_fast')
    hl = signal.shape[0]//(width*1.1)  # this will cut away 5% from start and end
    spec = lr.feature.melspectrogram(signal, n_mels=height, hop_length=int(hl))
    img = lr.logamplitude(spec)**2
    start = (img.shape[1] - width) // 2
    return img[:, start:start+width]


def process_audio(input_dir):
    files = []

    extensions = ('*.mp3', '*.ogg')
    for extension in extensions:
        files.extend(glob.glob(os.path.join(input_dir, extension)))

    for file in files:
        image = mp3_to_img(file)

        file_without_ext = os.path.splitext(file)[0]
        imageio.imwrite(file_without_ext + '.jpg', image)


if __name__ == "__main__":
    process_audio('test')
    process_audio('train')
    process_audio('valid')
