"""
Constants fill out all the constants needed for the Dialect Identification
project. Those wishing to modify the execution of the model should be able to
change values from this Python module.
"""

import os
from pathlib import Path

# DEBUG
DEBUG = True


# Directories
DATASET_DIR = Path('../dataset/')
FEATURES_DIR = Path('../dataset/features/')
RAW_DIR = Path('../dataset/raw/')
VISUALIZATION_DIR = Path('../visualization/')
CHECKPOINT_DIR = Path('../model/checkpoints/')
FEATURES_UNIFIED_DIR = Path('../dataset/unified/')


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
# def HOP_LENGTH(x): return int(x / 300)
# def WINDOW_SIZE()
