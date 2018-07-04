# spoken language identification

[![Build Status](https://travis-ci.org/tomasz-oponowicz/spoken_language_identification.svg?branch=master)](https://travis-ci.org/tomasz-oponowicz/spoken_language_identification)

Analyze audio to identify a language.
The solution uses the CNN network in order to detect language specific phonems.
It supports 3 languages: English, German and Spanish.

## Performance

The score against the test set (out-of-sample) is 97% (F1 metric). Additionally the network generalizes well and presents high score against real life content, for example podcasts or TV news.

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
1. Build the docker image:

       $ docker build -t sli --rm https://github.com/tomasz-oponowicz/spoken_language_identification.git
1. Mount the `examples` directory and classify an audio file, for example:

       $ docker run --rm -it -v $(pwd):/data sli /data/en.mp3

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
