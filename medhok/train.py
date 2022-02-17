#!/usr/bin/env python3

# import librosa
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from matplotlib import pyplot as plt

from . import get_dataset as ds
from . import constants

# Setup
np.random.seed(42)
tf.random.set_seed(42)


def create_model(
    feature='mel_spectrogram',
    arch='cnn',
    base_neuron=32,
    classes=len(constants.DIALECTS)
):
    """Creates a TensorFlow model.

    Args:
        feature (str, optional): Audio feature used. Defaults to 'mel_spec'.
        arch (str, optional): Architecture used. Defaults to 'cnn'.
    """
    _feat_amount = 128 if feature == 'mel_spectrogram' else (129 if feature == 'spectrogram' else 40)
    _shape = (_feat_amount, constants.WINDOW_SIZE, 1)

    print(f'Using shape: {_shape}')

    if arch == 'cnn':
        model = tf.keras.Sequential()
        # Input
        model.add(tf.keras.layers.Conv2D(
            base_neuron, (3, 3), activation=tf.nn.relu, input_shape=_shape))
        model.add(tf.keras.layers.MaxPooling2D(2, 2))
        model.add(tf.keras.layers.Conv2D(base_neuron << 1, (3, 3), activation=tf.nn.relu))
        model.add(tf.keras.layers.MaxPooling2D(2, 2))
        model.add(tf.keras.layers.Conv2D(base_neuron << 2, (3, 3), activation=tf.nn.relu))
        model.add(tf.keras.layers.MaxPooling2D(2, 2))
        if feature != 'mfcc':
            model.add(tf.keras.layers.Conv2D(base_neuron << 3, (3, 3), activation=tf.nn.relu))
            model.add(tf.keras.layers.MaxPooling2D(2, 2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(base_neuron, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(
            base_neuron << 1, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(
            base_neuron << 2, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(classes, activation=tf.nn.softmax))
    elif arch == 'vgg19':   # Transfer Learning: VGG19
        model = tf.keras.application.vgg19.VGG19(
            input_shape=_shape,
            include_top=False,
            weights='imagenet'
        )
        model.trainable = False
    elif arch == 'chatfield2014':
        # Chatfield: article mentioned by Shon et al. (2018)
        model = tf.keras.Sequential([
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

    return model


def main(feature_name='mel_spectrogram', arch='cnn'):
    """Main function of the training module.

    Args:
        feature_name (str, optional): Selected audio feature to be used as an input. Defaults to 'mel_spectrogram'.
        arch (str, optional): Preferred neural network architecture. Defaults to 'cnn'.
    """
    # ex: ./train.py mel_spectrogram cnn
    print(f"Using {feature_name} and {arch}.")

    # Getting model-ready windowed dataset
    (x_train, x_test), (y_train, y_test) = ds.load_windowed_dataset(
        feat_name=feature_name,
        split=True,
        onehot=True,
        normalised=True
    )

    print('[TF] Using', tf.config.list_physical_devices('GPU'))
    model = create_model(feature=feature_name, arch=arch)
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=constants.LEARNING_RATE
        ),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tfa.metrics.F1Score(num_classes=len(constants.DIALECTS))
        ]
    )

    # Checkpoint initialisation
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=constants.CHECKPOINT_DIR + 'model_' +
        feature_name + '-' + arch + '/',
        save_weights_only=True,
        verbose=1
    )

    # Training
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=constants.BATCH_SIZE,
        epochs=constants.EPOCHS,
        validation_data=(x_test, y_test),
        steps_per_epoch=constants.EPOCH_STEPS,
        verbose=1,
        callbacks=[ckpt_callback]
    )

    # Creating model report
    fig, ax = plt.subplots(2, 1, figsize=(16, 12))
    fig.suptitle('Plots of model performance: ' + feature_name + ', ' + arch)

    ax[0].plot(history.history['loss'], 'r')
    ax[0].plot(history.history['val_loss'], 'g')
    ax[1].plot(history.history['categorical_accuracy'], 'r')
    ax[1].plot(history.history['val_categorical_accuracy'], 'g')
    ax[0].grid()
    ax[1].grid()

    plt.savefig(
        '../visualization/performance/'
        + str(datetime.now()) + '-' + feature_name + '-' + arch + '.png')

    # Saving model
    model.save('../model/model-' + feature_name + '-' + arch + '.h5')
    model.save_weights(
        '../model/weights/model-' + feature_name + '-' + arch + '/')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Running without custom parameters. Defaulting to mel_spectrogram\
            and CNN.')
        main()
        exit()
    elif len(sys.argv) > 3:
        print('Too many parameters.')
        exit()
    else:
        main(sys.argv[1], sys.argv[2])
