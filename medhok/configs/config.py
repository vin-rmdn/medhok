"""
Constants fill out all the constants needed for the Dialect Identification
project. Those wishing to modify the execution of the model should be able to
change values from this Python module.
"""

from pathlib import Path

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import librosa
# import matplotlib


# Random
np.random.seed(42)


# DEBUG
DEBUG = True
POOL_SIZE = 12  # Number of threads
# matplotlib.use('svg')


# Directories
# PROJECT_DIR = Path()
PROJECT_ROOT_DIR = Path(__file__).parents[2].resolve()
DATASET_DIR = Path(PROJECT_ROOT_DIR / 'data')
FEATURES_DIR = Path(DATASET_DIR / 'features')
RAW_DIR = Path(DATASET_DIR / 'audio')
VISUALIZATION_DIR = Path(PROJECT_ROOT_DIR / 'visualization')
CHECKPOINT_DIR = Path(PROJECT_ROOT_DIR / 'model/checkpoints/')
FEATURES_UNIFIED_DIR = Path(PROJECT_ROOT_DIR / 'dataset/unified/')
TFRECORDS_DIR = Path(PROJECT_ROOT_DIR / 'dataset/tfrecords/')
LOG_DIR = PROJECT_ROOT_DIR / 'log'
TENSORBOARD_LOG_DIR = LOG_DIR / 'tensorboard/'
MODEL_DIR = Path(PROJECT_ROOT_DIR / 'model')

# Metadata
# DIALECTS = os.listdir(DATASET_DIR + 'raw/')
DIALECTS = list(RAW_DIR.iterdir())
FEATURE_AMOUNT = {
    'mel_spectrogram': 128,
    'spectrogram': 190,
    'mfcc': 40
}
MODEL_ARCHITECTURES = {
    'warohma18',
    'chatfield14',
    'shon18',
    'draghici20_crnn'
}
USE_PATHLIB = True     # True for Generator, False for tf.data.Dataset


# === AUDIO PROPERTIES
SAMPLE_RATE = 16000  # 16kHz sample rate; at this setting, 8000Hz reports empty frequencies
# we down sample because we want 4KHz making the model more robust against
# noise in the higher frequencies.
IMAGE_HEIGHT = 500
MEL_SPEC_SECOND = 30.1
SPLIT_SECOND = 5   # second
WAVE_SAMPLE_LENGTH = int(SAMPLE_RATE * 0.25)
SEGMENT_LENGTH = SAMPLE_RATE * SPLIT_SECOND
SEGMENT_STRIDE = SEGMENT_LENGTH
FEATURES = [
    'mel_spectrogram', 'mfcc', 'spectrogram'
]
# WINDOW_SECOND = 5
USE_BOTH_NORMALISATION = True
MONO = True
RESAMPLER_TYPE = 'soxr_vhq'
POLYNOMIAL_ORDER = 6


# Window Properties
FRAME_SIZE_SECOND = 0.025  # seconds
FRAME_STRIDE_SECOND = 0.01     # stride
FRAME_SIZE = int(SAMPLE_RATE * FRAME_SIZE_SECOND)   # for n_fft
FRAME_STRIDE = int(SAMPLE_RATE * FRAME_STRIDE_SECOND)   # for hop_length


# === Feature properties
# Shon et al. (2018) used 160. Ours were taken from a website.
HOP_LENGTH = 266
FFT_AMOUNT = 800
MEL_AMOUNT = 128
MFCC_AMOUNT = 40
F_MIN = 200
F_MAX = 4000
DEFAULT_FEATURE = 'mel_spectrogram'
PRE_EMPHASIS_ALPHA = 0.97
WINDOW_SIZE = 501
FREQUENCY_RESOLUTION = 20
SPEC_LOWER_BOUND = F_MIN // FREQUENCY_RESOLUTION
SPEC_UPPER_BOUND = F_MAX // FREQUENCY_RESOLUTION

FIGURE_SIZE = (15, 5)


# ====== TensorFlow
BATCH_SIZE = 32     # 32 normal, 16 for spectrograms
LEARNING_RATE = 1e-4
FINETUNE_LEARNING_RATE = 1e-5
EPOCHS = 100
FINETUNE_EPOCHS = 20
EPOCH_STEPS = 128   # 16 for shon18 due to limited vram, 64 for spectrograms, 128 for usual
FINETUNE_STEPS = 512
VALIDATION_STEPS = 32

# Parameters
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
FINETUNE_OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=FINETUNE_LEARNING_RATE)
LOSS = tf.keras.losses.CategoricalCrossentropy()
METRICS = [
    tf.keras.metrics.CategoricalAccuracy(name='acc'),
    tf.keras.metrics.Precision(name='prec'),
    tf.keras.metrics.Recall(name='rec'),
    tfa.metrics.F1Score(num_classes=len(DIALECTS), name='f1')
]

# Generator parameters
GENERATOR_PARAMS = {
    'dim': (128, WINDOW_SIZE),
    'batch_size': BATCH_SIZE,
    'n_dialects': len(DIALECTS),
    'n_channels': 1,
    'shuffle': True
}

WEIGHT_DECAY = 1e-2


# Functions
def bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def mel_spectrogram(wav):
    return librosa.feature.melspectrogram(
        y=wav,
        sr=SAMPLE_RATE,
        win_length=FRAME_SIZE,
        hop_length=FRAME_STRIDE,
        n_mels=MEL_AMOUNT,
        fmin=F_MIN,
        fmax=F_MAX
    )


def MFCC(wav):
    return librosa.feature.mfcc(
        y=wav,
        sr=SAMPLE_RATE,
        n_mfcc=MFCC_AMOUNT,
        n_fft=FRAME_SIZE,
        hop_length=FRAME_STRIDE
    )


def spectrogram(wav):
    return np.abs(
        librosa.core.stft(
            y=wav,
            n_fft=FFT_AMOUNT,
            hop_length=FRAME_STRIDE,
            win_length=FRAME_SIZE
        )
    )


def INPUT_SHAPE(feature='mel_spectrogram', generator=False, squeeze=False) -> list:
    """Returns input shape based on the feature for the model input.

    mel_spectrogram has 128 features per time frame, spectrogram has 201 features and MFCC contains 40 coefficients. On top of that, the function will return the shape of window length (as set by c.WINDOW_SIZE) and channels (1).

    Args:
        feature (str, optional): Audio features wished to be used. Defaults to 'mel_spectrogram'.
        generator (bool, optional): Whether the input shape is for a generator or not. Defaults to False.

    Returns:
        list: Input shape for Keras models
    """
    if generator or squeeze:
        return (FEATURE_AMOUNT[feature], WINDOW_SIZE)
    return (FEATURE_AMOUNT[feature], WINDOW_SIZE, 1)


def CHECKPOINT_CALLBACK(model_name, feature_name, finetune=False):
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_DIR / f'{model_name}-{feature_name}' / f'{model_name}-{feature_name}{"-finetune" if finetune else ""}',
        save_weights_only=True,
        verbose=1
    )


def CONFLICT_CHECK(arch, feature):
    if (
        arch == 'chatfield14' and feature == 'mfcc'
    ) or (
        arch == 'draghici20_crnn' and feature == 'mfcc'
    ):
        return True
    else:
        return False
