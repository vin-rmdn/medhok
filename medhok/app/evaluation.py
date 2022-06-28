#!/usr/bin/env python3

import gc
import pickle
import logging

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from sklearn.metrics import classification_report
# from sklearn.preprocessing import OneHotEncoder

from app.train import set_mixed_precision, select_model
from loader.feature_metadata import FeatureMetadata
from loader.greedy_loader import GreedyLoader
from configs import config as c


def __get_encoder():
    # One-hot Encoder
    with open(c.MODEL_DIR / 'dialects-encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    return encoder


def __get_finetuned_model(arch: str = 'chatfield14', feature: str = 'mel_spectrogram'):
    model = tf.keras.models.load_model(c.MODEL_DIR / f'{arch}-{feature}-finetuned.h5', compile=False)
    return model


def get_validation_dataset(feature='mel_spectrogram'):
    _, X_valid, _, y_valid = FeatureMetadata.load_feature_metadata(feature, split=True, pathlib=c.USE_PATHLIB)
    del _
    valid_dataset = GreedyLoader.load_features(X_valid, feature_name=feature)
    return valid_dataset, y_valid


def evaluate_model(model='draghici20_crnn', feature='mel_spectrogram'):
    encoder = __get_encoder()
    valid_dataset, y_valid = get_validation_dataset(feature)
    model = select_model(arch=model, feature=feature, finetuned=True)

    y_pred_raw = model.predict(valid_dataset)
    y_pred = encoder.inverse_transform(y_pred_raw)

    print(classification_report(y_valid, y_pred))


def evaluate_models():
    set_mixed_precision()
    encoder = __get_encoder()

    for feature in c.FEATURES:
        logging.info('============ Evaluating model by %s ============', feature)
        valid_dataset, y_valid = get_validation_dataset(feature)
        for arch in c.MODEL_ARCHITECTURES:
            if c.CONFLICT_CHECK(arch, feature):
                continue
            logging.info('Model: %s', arch)
            # Loading model
            model = __get_finetuned_model(arch=arch, feature=feature)

            if arch == 'shon18':
                y_pred_raw = model.predict(np.squeeze(valid_dataset))
            y_pred_raw = model.predict(valid_dataset)
            y_pred = encoder.inverse_transform(y_pred_raw)

            print(classification_report(y_valid, y_pred))

            del model
            gc.collect()
