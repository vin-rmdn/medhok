#!/usr/bin/env python3
"""Module containing models to train in TensorFlow.
"""

import tensorflow as tf

from configs import config as c
from model_helper.time_delay import TimeDelay
# from model_helper.local_response_normalization import LocalResponseNormalization


def baseline(_shape):
    """Baseline model that is proposed for my final project. Consists of three convolutional layers alongside a max pooling layer after each layer, a Flattening layer, several fully-connected layers of 32, 64 and 128 layers, and a softmax layers of 16 classes.

    Args:
        _shape (_type_): Shape of the input tensor.

    Returns:
        tf.keras.Model: A Keras sequential model.
    """
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu, input_shape=_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(256, (3, 3), activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(1024, activation=tf.nn.relu),
        tf.keras.layers.Dense(1024, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.5),
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
        tf.keras.layers.Conv2D(96, (7, 7), 2, input_shape=_shape),
        # tf.keras.layers.Lambda(
        #     lambda x: tf.nn.local_response_normalization(
        #         x, alpha=1e-4, beta=0.75, bias=2, depth_radius=5)),
        # LocalResponseNormalization(alpha=1e-4, beta=0.75, bias=2, depth_radius=5),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(2, 2),
        # conv2
        tf.keras.layers.ZeroPadding2D(1),
        tf.keras.layers.Conv2D(256, (5, 5), 2),
        # tf.keras.layers.Lambda(
        #     lambda x: tf.nn.local_response_normalization(
        #         x, alpha=1e-4, beta=0.75, bias=2, depth_radius=5)),
        # LocalResponseNormalization(alpha=1e-4, beta=0.75, bias=2, depth_radius=5),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
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
        # tf.keras.layers.Lambda(
        #     lambda x: tf.nn.local_response_normalization(
        #         x, alpha=1e-4, beta=0.75, bias=2, depth_radius=5)),

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
        # tf.keras.layers.Flatten(),
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
        tf.keras.layers.InputLayer(input_shape=_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(50, activation=tf.nn.sigmoid, kernel_regularizer=tf.keras.regularizers.l2(c.WEIGHT_DECAY), input_shape=_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(75, kernel_regularizer=tf.keras.regularizers.l2(c.WEIGHT_DECAY), activation=tf.nn.sigmoid),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l2(c.WEIGHT_DECAY), activation=tf.nn.sigmoid),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
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


def draghici20(_shape, arch_type='crnn') -> tf.keras.Model:
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
    model.add(tf.keras.layers.Conv2D(64, (3, 3), kernel_regularizer=tf.keras.regularizers.l2(c.WEIGHT_DECAY), activation=tf.nn.relu, input_shape=_shape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # ConvBlock(128)
    model.add(tf.keras.layers.Conv2D(128, (3, 3), kernel_regularizer=tf.keras.regularizers.l2(c.WEIGHT_DECAY), activation=tf.nn.relu))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # ConvBlock(256)
    model.add(tf.keras.layers.Conv2D(256, (3, 3), activation=tf.nn.relu))
    model.add(tf.keras.layers.BatchNormalization())
    # ConvBlock(256)
    model.add(tf.keras.layers.Conv2D(256, (3, 3), kernel_regularizer=tf.keras.regularizers.l2(c.WEIGHT_DECAY), activation=tf.nn.relu))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # ConvBlock(512)
    model.add(tf.keras.layers.Conv2D(512, (3, 3), kernel_regularizer=tf.keras.regularizers.l2(c.WEIGHT_DECAY), activation=tf.nn.relu))
    model.add(tf.keras.layers.BatchNormalization())
    # ConvBlock(512)
    model.add(tf.keras.layers.Conv2D(512, (3, 3), kernel_regularizer=tf.keras.regularizers.l2(c.WEIGHT_DECAY), activation=tf.nn.relu))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # ConvBlock(512)
    model.add(tf.keras.layers.Conv2D(512, (3, 3), kernel_regularizer=tf.keras.regularizers.l2(c.WEIGHT_DECAY), activation=tf.nn.relu))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(tf.keras.layers.GlobalMaxPooling2D())
    # On Global Average Pooling: this is technically GMP/GAP because while we have 12 temporal units, we only have 1 Mel feature (for mel_spec).
    # Permute — (bs, y, x, c) -> (bs, x, y, c)
    model.add(tf.keras.layers.Permute((2, 1, 3)))
    # # Reshape — (bs, x, y, c) -> (bs, x, y*c)
    bs, x, y, _c = model.layers[-1].output_shape
    model.add(tf.keras.layers.Reshape((x, y * _c)))
    if arch_type == 'crnn':
        # Bidirectional LSTM(256)
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=False), merge_mode='concat'))
    elif arch_type == 'cnn':
        # Dense(1024)
        model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu))
    # Output
    model.add(tf.keras.layers.Dense(16, activation=tf.nn.softmax))
    return model


def shon18(_shape) -> tf.keras.Model:
    """Per Shon et al. (2018):
    Our end-to-end system is based on [14, 13], but instead of a VGG [24] or Time Delayed Neural Network, we used four 1-dimensional CNN (1d-CNN) layers (40x5 - 500x7 - 500x1 - 500x1 filter sizes with 1-2-1-1 strides and the number of filters is 500-500-500-3000) and two FC layers (1500-600) that are connected with a Global average pooling layer which avrerages the CNN outputs to produce a fixed output size of 3000x1. After global average pooling, the fixed length output is fed into two FC layers and a Softmax layer.

    Implementation in pytorch and Python 2 as described in this GitHub code: https://github.com/swshon/dialectID_e2e/blob/master/models/e2e_model.py

    Args:
        _shape (list): Shape of the input tensor

    Returns:
        tf.keras.Model: A Keras Sequential model.
    """
    return tf.keras.Sequential([
        tf.keras.layers.Permute((2, 1), input_shape=_shape),
        tf.keras.layers.Conv1D(500, 5, strides=1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv1D(500, 7, strides=2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv1D(500, 1, strides=1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv1D(3000, 1, strides=1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(1500),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(600),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(16, activation=tf.nn.softmax)
    ])
