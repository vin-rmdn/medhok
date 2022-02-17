"""
Constants fill out all the constants needed for the Dialect Identification
project. Those wishing to modify the execution of the model should be able to
change values from this Python module.
"""

from pathlib import Path

import tensorflow as tf

# DEBUG
DEBUG = True


# Directories
PROJECT_ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = Path(PROJECT_ROOT_DIR / 'dataset/')
FEATURES_DIR = Path(PROJECT_ROOT_DIR / 'dataset/features/')
RAW_DIR = Path(PROJECT_ROOT_DIR/'dataset/raw/')
VISUALIZATION_DIR = Path(PROJECT_ROOT_DIR / 'visualization/')
CHECKPOINT_DIR = Path(PROJECT_ROOT_DIR / 'model/checkpoints/')
FEATURES_UNIFIED_DIR = Path(PROJECT_ROOT_DIR / 'dataset/unified/')
TFRECORDS_DIR = Path(PROJECT_ROOT_DIR / 'dataset/tfrecords/')
TENSORBOARD_LOG_DIR = Path(PROJECT_ROOT_DIR / 'log' / 'tensorboard/')


# Metadata
# DIALECTS = os.listdir(DATASET_DIR + 'raw/')
DIALECTS = (DATASET_DIR / 'raw').iterdir()


# === AUDIO PROPERTIES
SAMPLE_RATE = 8000  # 8 kHz sample rate;
# we down sample because we want 4KHz making the model more robust against
# noise in the higher frequencies.
IMAGE_HEIGHT = 500
MEL_SPEC_SECOND = 30.1
SPLIT_SECOND = 10   # second
WAVE_SAMPLE_LENGTH = int(SAMPLE_RATE * 0.25)
SAMPLE_LENGTH = SAMPLE_RATE * 10
FFT_NUMBER = 256
MFCC_NUMBER = 40
HOP_LENGTH = 266    # Shon et al. (2018) used 160. Ours were taken from a website.
FEATURES = [
    'mel_spectrogram', 'mfcc', 'spectrogram'
]
WINDOW_SECOND = 5
WINDOW_SIZE = int(MEL_SPEC_SECOND * WINDOW_SECOND)
USE_BOTH_NORMALISATION = True

FIGURE_SIZE = (70, 5)


# TensorFlow
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 200
EPOCH_STEPS = 128


# Functions
def bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
