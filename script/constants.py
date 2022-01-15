DATASET_DIR = '../dataset'
SAMPLE_RATE = 8000  # 8 kHz sample rate;
# we down sample because we want 4KHz making the model more robust against noise in the higher frequencies.
IMAGE_HEIGHT = 500
IMAGE_WIDTH = 157
SPLIT_SECOND = 10   # second
WAVE_SAMPLE_LENGTH = int(SAMPLE_RATE * 0.25)
SAMPLE_LENGTH = SAMPLE_RATE * 10
