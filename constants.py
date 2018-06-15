SEED = 42

FB_HEIGHT = 40  # filter banks
WIDTH = 1000
COLOR_DEPTH = 1
INPUT_SHAPE = (FB_HEIGHT, WIDTH, COLOR_DEPTH)

DATA_TYPE = 'float32'
DATA_KEY = 'data'

LANGUAGES = ['en', 'de', 'es']
GENDERS = ['m', 'f']

LANGUAGE_INDEX = 0
GENDER_INDEX = 1

THRESHOLD = 0.8

FRAGMENT_DURATION = 10  # seconds

DATASET_DIST = 'spoken_language_dataset/build'
