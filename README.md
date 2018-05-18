# spoken language identification

Analyze audio to identify a language.
The solution uses the CNN network in order to detect language specific phonems.
It supports 3 languages: English, German and Spanish.

## Performance

The score against the test set (out-of-sample) is 97% (F1 score). Additionally the network generalizes well and presents high score against random speech samples, for example podcasts or audiobooks.

## Demo

1. Create a temporary directory and change directory:

       $ mkdir examples && cd $_
1. Download samples:
    > NOTE: An audio file should contain speech and silence only. For example podcasts, interviews or audiobooks are a good fit. Sound effects or languages other than English, German or Spanish may be badly classified.
    * English
    
          $ wget "https://s102.podbean.com/pb/e19e826a5c0e755683b154195e22127a/5afe956e/data1/fs145/862611/uploads/039_jsAir_-_Node_js_and_Community.mp3" -O en.mp3
    * German
    
          $ wget "http://1m40ti5.podcaster.de/Ernaehrungspsychologie/media/052_Am_Arsch_vorbei_geht_auch_ein_Weg.mp3" -O de.mp3
1. Build docker image:

       $ docker build -t sli --rm https://github.com/tomasz-oponowicz/spoken_language_identification.git
1. Mount the `examples` directory and classify an audio file, for example:

       $ docker run --rm -it -v $(pwd):/data sli /data/en.mp3

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
