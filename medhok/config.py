"""
Constants fill out all the constants needed for the Dialect Identification
project. Those wishing to modify the execution of the model should be able to
change values from this Python module.
"""

from pathlib import Path

import tensorflow as tf
import numpy as np
import librosa
# import matplotlib

from model_helper.time_delay import TimeDelay

# DEBUG
DEBUG = True
POOL_SIZE = 12  # Number of threads
# matplotlib.use('svg')


# Directories
# PROJECT_DIR = Path()
PROJECT_ROOT_DIR = Path(__file__).parents[1].resolve()
DATASET_DIR = Path(PROJECT_ROOT_DIR / 'data')
FEATURES_DIR = Path(DATASET_DIR / 'features')
RAW_DIR = Path(DATASET_DIR / 'audio')
VISUALIZATION_DIR = Path(PROJECT_ROOT_DIR / 'visualization')
CHECKPOINT_DIR = Path(PROJECT_ROOT_DIR / 'model/checkpoints/')
FEATURES_UNIFIED_DIR = Path(PROJECT_ROOT_DIR / 'dataset/unified/')
TFRECORDS_DIR = Path(PROJECT_ROOT_DIR / 'dataset/tfrecords/')
TENSORBOARD_LOG_DIR = Path(PROJECT_ROOT_DIR / 'log' / 'tensorboard/')


# Metadata
# DIALECTS = os.listdir(DATASET_DIR + 'raw/')
DIALECTS = list(RAW_DIR.iterdir())


# === AUDIO PROPERTIES
SAMPLE_RATE = 16000  # 16kHz sample rate; at this setting, 8000Hz reports empty frequencies
# we down sample because we want 4KHz making the model more robust against
# noise in the higher frequencies.
IMAGE_HEIGHT = 500
MEL_SPEC_SECOND = 30.1
SPLIT_SECOND = 5   # second
WAVE_SAMPLE_LENGTH = int(SAMPLE_RATE * 0.25)
SEGMENT_LENGTH = SAMPLE_RATE * SPLIT_SECOND
SEGMENT_STRIDE = SEGMENT_LENGTH // 2
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
FFT_AMOUNT = 256
MFCC_AMOUNT = 40
F_MIN = 200
F_MAX = 4000
DEFAULT_FEATURE = 'mel_spectrogram'
PRE_EMPHASIS_ALPHA = 0.97


FIGURE_SIZE = (15, 5)


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


def mel_spectrogram(wav):
    return librosa.feature.melspectrogram(
        wav, sr=SAMPLE_RATE, n_fft=FRAME_SIZE,
        hop_length=FRAME_STRIDE
    )


def MFCC(wav):
    return librosa.feature.mfcc(
        wav,
        sr=SAMPLE_RATE,
        n_mfcc=MFCC_AMOUNT,
        n_fft=FRAME_SIZE,
        hop_length=FRAME_STRIDE
    )


def spectrogram(wav):
    return np.abs(
        librosa.core.stft(
            wav,
            n_fft=FRAME_SIZE,
            hop_length=FRAME_STRIDE,
            win_length=FRAME_SIZE
        )
    )


# === TensorFlow models
def baseline(_shape) -> tf.keras.Model:
    """Baseline model that is proposed for my final project. Consists of three convolutional layers alongside a max pooling layer after each layer, a Flattening layer, several fully-connected layers of 32, 64 and 128 layers, and a softmax layers of 16 classes.

    Args:
        _shape (_type_): Shape of the input tensor.

    Returns:
        tf.keras.Model: A Keras sequential model.
    """
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu, input_shape=_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation=tf.nn.relu),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(16, activation=tf.nn.softmax)
    ])


def chatfield14(_shape) -> tf.keras.Model:
    """Also known as CNN-M, or VGG-M (Nagrani et al., 2017)

    Args:
        _shape (_type_): Shape of the input tensor.

    Returns:
        tf.keras.Model: A Keras sequential model.
    """
    return tf.keras.Sequential([
        # conv1
        tf.keras.layers.Conv2D(96, (7, 7), 2,
                               activation=tf.nn.relu, input_shape=_shape),
        tf.keras.layers.Lambda(
            lambda x: tf.nn.local_response_normalization(
                x, alpha=1e-4, beta=0.75, bias=2, depth_radius=5)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # conv2
        tf.keras.layers.ZeroPadding2D(1),
        tf.keras.layers.Conv2D(256, (5, 5), 2, activation=tf.nn.relu),
        tf.keras.layers.Lambda(
            lambda x: tf.nn.local_response_normalization(
                x, alpha=1e-4, beta=0.75, bias=2, depth_radius=5)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # conv3
        tf.keras.layers.ZeroPadding2D(1),
        tf.keras.layers.Conv2D(512, (3, 3), 1, activation=tf.nn.relu),
        # conv4
        tf.keras.layers.ZeroPadding2D(1),
        tf.keras.layers.Conv2D(512, (3, 3), 1, activation=tf.nn.relu),
        # conv5
        tf.keras.layers.ZeroPadding2D(1),
        tf.keras.layers.Conv2D(512, (3, 3), 1, activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(2, 2),
        # full6
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.5),
        # full7
        tf.keras.layers.Dense(4096, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.5),
        # full8
        tf.keras.layers.Dense(16, activation=tf.nn.softmax)
    ])


def voxceleb(_shape, t=5) -> tf.keras.Model:
    """Quoting from Nagrani et al. (2017): "...we base our model on the VGG-M" (Chatfield, 2014) "CNN, known for good classification performance on input data, with modifications to adapt to the spectrogram input. The fully-connected fc6 layer of dimension 9x8 (support in both dimensions) is replaced by two layers -- a fully connected layer of 9x1 (support in the frequency domain) and an average pool layer with support 1xn, where n depends on the length of the input speech segment (for example for a 3 second segment, n=8)"

    Adjusting to the 5-second window, we will have an average pool layer of 1x13.

    Args:
        _shape (list): Shape of the input tensor.
        t (int): Length of the input speech segment.

    Returns:
        tf.keras.Model: A Keras Sequential model.
    """
    return tf.keras.Sequential([
        # conv1
        tf.keras.layers.Conv2D(96, (7, 7), 2,
                               activation=tf.nn.relu, input_shape=_shape),
        tf.keras.layers.Lambda(
            lambda x: tf.nn.local_response_normalization(
                x, alpha=1e-4, beta=0.75, bias=2, depth_radius=5)),
        # mpool1
        tf.keras.layers.MaxPooling2D(2, 2),
        # conv2
        tf.keras.layers.ZeroPadding2D(1),
        tf.keras.layers.Conv2D(256, (5, 5), 2, activation=tf.nn.relu),
        tf.keras.layers.Lambda(
            lambda x: tf.nn.local_response_normalization(
                x, alpha=1e-4, beta=0.75, bias=2, depth_radius=5)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # conv3
        tf.keras.layers.ZeroPadding2D(1),
        tf.keras.layers.Conv2D(384, (3, 3), 1, activation=tf.nn.relu),
        # conv4
        tf.keras.layers.ZeroPadding2D(1),
        tf.keras.layers.Conv2D(256, (3, 3), 1, activation=tf.nn.relu),
        # conv5
        tf.keras.layers.ZeroPadding2D(1),
        tf.keras.layers.Conv2D(256, (3, 3), 1, activation=tf.nn.relu),
        # mpool5
        tf.keras.layers.MaxPooling2D(2, 2),
        # full6
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.5),
        # apool6
        tf.keras.layers.AveragePooling2D((1, int(2.6 * t))),
        # full7
        tf.keras.layers.Dense(1024, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.5),
        # full8 - with adjusted shape (original: 1024 for thousand speakers)
        tf.keras.layers.Dense(16, activation=tf.nn.softmax)
    ])


def warohma18(_shape) -> tf.keras.Model:
    """Per Warohma et al. (2018):
    "The process of classification and the recognition of dialect proposed in this study use backpropagation neural network with three hidden layers. These layers are sufficient to handle complex pattern of classification problem (sic.). The number of neurons in each hidden layers are altered and the testing process is conducted four times with different number of hidden units. In the hidden layer, a log sigmoid activation function is utilized. The function is most commonly used in the classification of a pattern and has range of 0 and 1. The learning rate is 0.01, a large learning rate may affect the system performance and too low learning rate may cause the long duration of the learning process."

    Additional notes: Testing 4 (50, 75, 100) performs the best result for Javanese dataset, with accuracy ranging from 76.7% up to 83.4%. The model is trained for 30,000 epochs. Tested with MFCC with Discrete Cosine Transfer as follows:

    c(n) = Σ(m-1 to M) (Y(m) * cos(πn(m - ½) / M))

    Args:
        _shape (list): Shape of the input tensor

    Returns:
    tf.keras.Model: A Keras Sequential model.
    """
    return tf.keras.Sequential([
        tf.keras.layers.Dense(50, activation=tf.nn.sigmoid, input_shape=_shape),
        tf.keras.layers.Dense(75, activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(100, activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(16, activation=tf.nn.softmax)
    ])


def snyder17(_shape) -> tf.keras.Model:
    """Per Snyder et al. (2017):
    "The first 5 layers of the network work at the frame level, with a time-delay architecture. Suppose t is the current time step. At the input, we splice together frames at {t-2, t-1, t, t+1, t+2}. The next two layers splice together the output of the previous layer at times {t-2, t, t+2} and {t-3, t, t+3}, respectively. The next two layers also operate at the frame-level, but without any added temporal context."

    P.S.: Do not forget to omit the softmax layer and get either the first or second embeddings after training.

    Args:
        _shape (list): Shape of the input tensor

    Returns:
        tf.keras.Model: A Keras Sequential model.
    """
    return tf.keras.Sequential([
        TimeDelay([-2, 2], dilation_rate=1),
        TimeDelay([-2, 2], dilation_rate=2),
        TimeDelay([-3, 3], dilation_rate=3),
        TimeDelay(),
        TimeDelay(),
        # TODO: complete TDNN
    ])


time_delay = snyder17   # alias


def draghici20(_shape, type='crnn') -> tf.keras.Model:
    """Per Draghici et al. (2020):
    "We re-implemented the CNN model and the CRNN model based on a freely-available implementation (https://github.com/HPI-DeepLearning/crnn-lid) in which some of the model parameters such as number of layers and filters differ from the original paper."

    "...both models share a feature learning front-end of seven convolutional blocks each consisting of 2D convoultional layers, batch normalization, and max pooling for downsampling along time and frequency..."

    Args:
        _shape (list): Shape of the input tensor

    Returns:
        tf.keras.Model: A Keras Sequential model.
    """
    model = tf.keras.Sequential()

    # ConvBlock(64) - Input
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu, input_shape=_shape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # ConvBlock(128)
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation=tf.nn.relu))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # ConvBlock(256)
    model.add(tf.keras.layers.Conv2D(256, (3, 3), activation=tf.nn.relu))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # ConvBlock(256)
    model.add(tf.keras.layers.Conv2D(256, (3, 3), activation=tf.nn.relu))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # ConvBlock(512)
    model.add(tf.keras.layers.Conv2D(512, (3, 3), activation=tf.nn.relu))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # ConvBlock(512)
    model.add(tf.keras.layers.Conv2D(512, (3, 3), activation=tf.nn.relu))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # ConvBlock(512)
    model.add(tf.keras.layers.Conv2D(512, (3, 3), activation=tf.nn.relu))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # Global Max Pooling
    model.add(tf.keras.layers.GlobalMaxPool2D())
    if type == 'crnn':
        # Bidirectional LSTM(256)
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)))
        # TODO: check for model summary
    elif type == 'cnn':
        # Dense(1024)
        model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu))
    # Output
    model.add(tf.keras.layers.Dense(16, activation=tf.nn.softmax))
    return model
