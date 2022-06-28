#!/usr/bin/env python3

import logging
import sys
import pickle
from datetime import datetime
from typing import Tuple

import tensorflow as tf
import numpy as np
# import visualkeras

from loader.feature_metadata import FeatureMetadata
from loader.dataset import create_dataset
from loader.generator import MedhokGenerator
from configs import config as c, models as m


# Setup
np.random.seed(42)
tf.random.set_seed(42)


def set_mixed_precision() -> None:
    # Keras: setting up mixed precision
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    logging.info('Compute dtype: %s', policy.compute_dtype)
    logging.info('Variable dtype: %s', policy.variable_dtype)


def select_model(arch: str = 'baseline', feature: str = 'mel_spectrogram', finetuned=False, pre_finetuned=False):
    # Determining the right model builder
    if arch == 'baseline':
        model = m.baseline(c.INPUT_SHAPE(feature))
    elif arch == 'chatfield14':
        model = m.chatfield14(c.INPUT_SHAPE(feature))
    elif arch == 'voxceleb':
        model = m.voxceleb(c.INPUT_SHAPE(feature))
    elif arch == 'warohma18':
        model = m.warohma18(c.INPUT_SHAPE(feature))
    elif arch == 'snyder17':
        model = m.snyder17(c.INPUT_SHAPE(feature))
    elif arch == 'draghici20_crnn':
        model = m.draghici20(c.INPUT_SHAPE(feature), arch_type='crnn')
    elif arch == 'draghici20_cnn':
        model = m.draghici20(c.INPUT_SHAPE(feature), arch_type='cnn')
    elif arch == 'shon18':
        model = m.shon18(c.INPUT_SHAPE(feature, squeeze=True))
    else:
        logging.error('Unknown model: %s', sys.argv[2])
        sys.exit(1)

    model.compile(
        optimizer=c.OPTIMIZER,
        loss=c.LOSS,
        metrics=c.METRICS
    )

    if pre_finetuned or finetuned:
        logging.info('Loading checkpoint weights and evaluating...')
        model.load_weights(c.CHECKPOINT_DIR / f'{arch}-{feature}' / f'{arch}-{feature}{"-finetune" if finetuned else ""}')

    return model


def __load_generator(arch: str = 'baseline', feature: str = 'mel_spectrogram') -> Tuple[MedhokGenerator, MedhokGenerator]:
    # Getting features metadata
    X_train, X_test, y_train, y_test = FeatureMetadata.load_feature_metadata(feature, split=True, pathlib=c.USE_PATHLIB)
    logging.info('Dataset loaded with %d on training and %d on validation.', X_train.shape[0], X_test.shape[0])
    y_train = FeatureMetadata.transform_dialects(y_train)
    y_test = FeatureMetadata.transform_dialects(y_test)

    # Keras Generator
    logging.info('Loading Keras generator...')
    use_squeeze = True if arch == 'shon18' else False

    train_generator = MedhokGenerator(
        X_train,
        batch_size=c.BATCH_SIZE,
        dim=c.INPUT_SHAPE(feature, generator=True),
        squeeze=use_squeeze
    )
    valid_generator = MedhokGenerator(
        X_test,
        dim=c.INPUT_SHAPE(feature, generator=True),
        batch_size=c.BATCH_SIZE,
        squeeze=use_squeeze
    )

    return train_generator, valid_generator


def __load_dataset(arch: str = 'baseline', feature: str = 'mel_spectrogram'):
    use_squeeze = True if arch == 'shon18' else False
    X_train, X_test, y_train, y_test = FeatureMetadata.load_feature_metadata(feature, split=True, pathlib=False)

    use_squeeze = True if arch == 'shon18' else False

    train_data = create_dataset(X_train, y_train, squeeze=use_squeeze)
    valid_data = create_dataset(X_test, y_test, squeeze=use_squeeze)

    return train_data, valid_data


# def __create_graph(model, arch: str = 'baseline') -> None:
#     # ann_viz(model, title=arch, view=False, filename=f'visualization/graph/{arch}.gv')
#     visualkeras.layered_view(model, to_file=f'visualization/graph/{arch}.png')


def train_model(arch: str = 'baseline', feature: str = 'mel_spectrogram') -> dict:
    set_mixed_precision()
    train_generator, valid_generator = __load_generator(arch, feature)
    model = select_model(arch, feature)

    logging.info('Showing model summary for %s', arch)
    model.summary()

    # __create_graph(model, arch)

    logging.info('Training model: %s...', arch)
    history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=c.EPOCHS,
        steps_per_epoch=c.EPOCH_STEPS,
        verbose=1,
        callbacks=[c.CHECKPOINT_CALLBACK(arch, feature)]
    )

    logging.info("Saving model.")
    model.save(c.MODEL_DIR / f'{arch}-{feature}.h5')
    model.save_weights(c.MODEL_DIR / 'weights' / f'{arch}-{feature}')

    logging.info('Saving metrics with pickle.')
    with open(c.LOG_DIR / 'metrics' / f'{arch}-{feature}-lr{c.LEARNING_RATE}-b{c.BATCH_SIZE}-{c.EPOCH_STEPS}_per_epoch-{datetime.now()}.txt', 'wb') as f:
        pickle.dump(history.history, f)

    return history


def finetune(arch: str = 'baseline', feature: str = 'mel_spectrogram') -> dict:
    set_mixed_precision()
    train_generator, valid_generator = __load_generator(arch, feature)
    model = select_model(arch, feature, pre_finetuned=True)

    model.evaluate(valid_generator)

    logging.info('Fine-tuning...')
    history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=c.FINETUNE_EPOCHS,
        steps_per_epoch=c.FINETUNE_STEPS,
        verbose=1,
        callbacks=[c.CHECKPOINT_CALLBACK(arch, feature, finetune=True)]
    )

    logging.info("Saving fine-tuned model.")
    model.save(c.MODEL_DIR / f'{arch}-{feature}-finetuned.h5')
    model.save_weights(c.MODEL_DIR / 'weights' / f'{arch}-{feature}-finetuned')

    logging.info('Saving metrics with pickle.')
    with open(c.LOG_DIR / 'metrics' / f'{arch}-{feature}-lr{c.LEARNING_RATE}-b{c.BATCH_SIZE}-{c.EPOCH_STEPS}_per_epoch-{datetime.now()}-ft', 'wb') as f:
        pickle.dump(history.history, f)

    return history
