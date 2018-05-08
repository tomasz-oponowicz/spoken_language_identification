# spoken language identification

Analyze audio to identify a language.
The solution uses the CNN network in order to detect language specific phonems.
It supports 3 languages: English, German and Spanish.

## Performance

The score against the test set (out-of-sample) is 97% (F1 score). Additionally the network generalizes well and presents high score against random speech samples, for example podcasts or audiobooks.

## Test

1. Install dependencies:

       $ pip install -r requirements.txt
1. Classify audio file

       $ python cli.py foo.mp3
    ...an audio file should contain speech and silence. For example podcasts, interviews or audiobooks are good fit. Sound effects or languages other than English, German or Spanish may be badly classified.

## Train

### Prerequisites

* ffmpeg is installed (tested with 3.4.2)
* sox is installed (tested with 14.4.2)

### Steps

1. Fetch the dataset
1. Generate samples
1. Generate features
1. Normalize features and build folds
1. Train the model
