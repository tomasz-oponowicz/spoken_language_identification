# spoken language identification

[![Build Status](https://travis-ci.org/tomasz-oponowicz/spoken_language_identification.svg?branch=master)](https://travis-ci.org/tomasz-oponowicz/spoken_language_identification)

Identify a spoken language using artificial intelligence (LID).
The solution uses [the convolutional neural network][cnn] in order to detect language specific phonemes.
It supports 3 languages: English, German and Spanish.
The inspiration for the project came from the TopCoder contest, [Spoken Languages 2][tc].

Take a look at the [Demo](#demo) section to try the project yourself against real life content.

## Dataset

New dataset was created from scratch.

[LibriVox recordings][lv] were used to prepare the dataset. 
Particular attention was paid to a big variety of unique speakers.
Big variance forces the network to concentrate more on language properties than a specific voice. 
Samples are equally balanced between languages, genders and speakers in order not to favour any subgroup.
Finally speakers present in the test set, are not present in the train set.
This helps estimate a generalization error.

More information at [tomasz-oponowicz/spoken_language_dataset][sld].

## Architecture

The first step is to normalize input audio. Each sample is an FLAC audio file with:

* sample rate: 22050
* bit depth: 16
* channels: 1
* duration: 10 seconds (sharp)

Next [filter banks][src_fb] are extracted from samples. 
[Mean and variance normalization][src_mvn] is applied.
Then data is scaled with [the Min/Max scaler][src_mms].

Finally preprocessed data is passed to [the convolutional neural network][src_cnn].
Please notice [the *AveragePooling2D* layer][src_apl] which improved the performance.
This strategy is called global average pooling.
It effectively forces the previous layers to produce the confidence maps.

The output is multiclass.

## Performance

The score against the test set (out-of-sample) is 97% (F1 metric). 
Additionally the network generalizes well and presents high score against real life content, for example podcasts or TV news.

Sound effects or languages other than English, German or Spanish may be badly classified.
If you want to work with noisy audio consider filtering noise out beforehand.

## Demo

### Prerequisites

* docker is installed (tested with 18.03.0)

### Steps

1. Create a temporary directory and change the current directory:

       $ mkdir examples && cd $_
1. Download samples:
    > NOTE: An audio file should contain speech and silence only. For example podcasts, interviews or audiobooks are a good fit. Sound effects or languages other than English, German or Spanish may be badly classified.
    * English (confidence 85.36%):

          $ wget "https://javascriptair.podbean.com/mf/player-preload/nkdkps/048_JavaScript_Air_-_JavaScript_and_the_Web_Platform_The_Grand_Finale_.mp3" -O en.mp3
    * German (confidence 85.53%):

          $ wget "http://mp3-download.ard.de/radio/radiofeature/auf-die-fresse-xa9c.l.mp3" -O de.mp3
    * Spanish (confidence 86.96%):

          $ wget "http://mvod.lvlt.rtve.es/resources/TE_SCINCOC/mp3/2/8/1526585716282.mp3" -O es.mp3
	  
    ...other examples of real life content are listed in the [EXAMPLES.md](./EXAMPLES.md).
1. Build the docker image:

       $ docker build -t sli --rm https://github.com/tomasz-oponowicz/spoken_language_identification.git
1. Mount the `examples` directory and classify an audio file, for example:

       $ docker run --rm -it -v $(pwd):/data sli /data/en.mp3

	...there are several [options available through command line][src_opt].
	For example you can tweak the noise reducer by increasing or decreasing the `silence-threshold` (`0.5` by default):

       $ docker run --rm -it -v $(pwd):/data sli --silence-threshold=1 /data/es.mp3

## Train

### Prerequisites

* ffmpeg is installed (tested with 3.4.2)
* sox is installed (tested with 14.4.2)
* docker is installed (tested with 18.03.0)

### Steps

1. Clone the repository:

       $ git clone git@github.com:tomasz-oponowicz/spoken_language_identification.git
1. Go to the newly created directory:

       $ cd spoken_language_identification
1. Generate samples:
    1. Fetch the *spoken_language_dataset* dataset:
    
           $ git submodule update --init --recursive
    1. Go to the dataset directory:

           $ cd spoken_language_dataset
    1. Generate samples:
		> NOTE: Alternatively you can [download the pregenerated dataset][kg]. Depending on your hardware it can save you 1-2 hours. After downloading, extract contents into `build/train` and `build/test` directories.

           $ make build
    1. Fix file permission of newly generated samples:
    
           $ make fix_permissions
    1. Return to the `spoken_language_identification` directory

           $ cd ..
1. Install dependencies

       $ pip install -r requirements.txt
    ...the `tensorflow` package is installed by default (i.e. CPU support only). In order to speed up the training, install the `tensorflow-gpu` package instead (i.e. GPU support). More information at [*Installing TensorFlow*](https://www.tensorflow.org/install/install_linux).
1. Generate features from samples:

       $ python features.py
1. Normalize features and build folds:

       $ python folds.py
1. Train the model:
       
       $ python model.py
    ...new model is stored at `model.h5`.

## Release history

* 2018-07-06 / v1.0 / Initial version

[tc]: https://community.topcoder.com/longcontest/?module=ViewProblemStatement&rd=16555&pm=13978
[cnn]: https://en.wikipedia.org/wiki/Convolutional_neural_network
[sld]: https://github.com/tomasz-oponowicz/spoken_language_dataset
[lv]: https://librivox.org
[src_fb]: https://github.com/tomasz-oponowicz/spoken_language_identification/blob/8f886bc2ca54f22b693d46264fb19aadfb30dc97/features.py#L14
[src_mvn]: https://github.com/tomasz-oponowicz/spoken_language_identification/blob/8f886bc2ca54f22b693d46264fb19aadfb30dc97/folds.py#L128
[src_mms]: https://github.com/tomasz-oponowicz/spoken_language_identification/blob/8f886bc2ca54f22b693d46264fb19aadfb30dc97/folds.py#L133
[src_cnn]: https://github.com/tomasz-oponowicz/spoken_language_identification/blob/master/model.py#L61-L131
[src_apl]: https://github.com/tomasz-oponowicz/spoken_language_identification/blob/master/model.py#L114
[kg]: https://www.kaggle.com/toponowicz/spoken-language-identification
[src_opt]: https://github.com/tomasz-oponowicz/spoken_language_identification/blob/master/cli.py#L86
