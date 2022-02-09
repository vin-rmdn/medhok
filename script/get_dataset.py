#!/usr/bin/env python3
"""Get Dataset — provides the developer with the features extracted with
the Extract Feature module.
"""
import time
import gc

import numpy as np
import tensorflow as tf
from pympler.asizeof import asizeof
import preprocessing as pre
import constants

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# TODO: CMVN, Consider putting data augmentation


def get_dataset() -> dict:
    """Get Dataset function — gets the full dataset location.

    Returns:
        dict: a list of dialects and its respective .wav files
    """
    dialects = (constants.RAW_DIR).iterdir()

    wavs = {}
    for dialect in dialects:
        wavs[dialect.parts[-1]] = [wav for wav in dialect.iterdir() if wav.parts[-1][-3:].lower() == 'wav']

    return wavs


def load_features(feature_name='mel_spectrogram', return_tensor=False, normalised=False):
    """
    Loads precached features to RAM.
    :param feature_name: feature name (defaults to mel_spectrogram)
    :return: numpy.ndarray
    """
    return_feature = []
    dialects = []
    dataset = get_dataset()
    if constants.DEBUG:
        print('Using normalisation method:', normalised)
    for dialect, data in dataset.items():
        if constants.DEBUG:
            print(f'Loading {dialect}: ')

        for datum in data:
            if constants.DEBUG:
                print('-', datum.parts[-1], end=' ')
            #     print(constants.FEATURES_DIR / dialect)
            #     print(str(datum.parts[-1]) + '-' + feature_name + '.npy')
            time_start = time.time()
            _buffer = np.load(
                constants.FEATURES_DIR / dialect / (str(datum.parts[-1]) + '-' + feature_name + '.npy'))
            if normalised:
                # _buffer = pre.normalise_feature(_buffer, mean_var=True)   # non-CUDA
                pre.normalise_feature(_buffer, mean_var=True)
            return_feature.append(_buffer)
            dialects.append(dialect)
            print(f'(time: {time.time() - time_start:.1f})')
            del _buffer
            gc.collect()

    print('Done!')

    # Cleanup
    del dialect, data, datum, time_start, dataset
    gc.collect()

    if return_tensor:
        return tf.data.Dataset.from_tensor_slices(return_feature), dialects
    return return_feature, dialects


# TODO: consider whether or not to split the dataset into train/test for the TensorFlow dataset generator.
def load_windowed_dataset(
    feat_name='mel_spectrogram',
    split=False,
    onehot=True,
    normalised=True
):
    """
    Loads windowed features. Provides convenience for the developer to be able
    to load features straight to the model.
    """
    # Variables
    feats_split = []
    dialects_split = []

    feats, dialects = load_features(feature_name=feat_name, normalised=normalised)
    print(f"Loaded dataset size in RAM: {asizeof(feats) / 1e9:2.2f}GB")

    # Windowing features
    for feature, dialect in zip(feats, dialects):
        temp = pre.split_window(feature)
        for window in temp:
            feats_split.append(window)
            dialects_split.append(dialect)
        del temp
    del feature, dialect
    gc.collect()

    feats_split = np.array(feats_split, dtype=np.float32)[:, :, :, np.newaxis]
    dialects_split = np.array(dialects_split)[:, np.newaxis]

    if onehot:
        # Transform dialects to onehot representations with sklearn
        onehot_encoder = OneHotEncoder(handle_unknown='error')
        onehot_encoder.fit(dialects_split)
        dialects_split = onehot_encoder.transform(dialects_split).toarray()
        del onehot_encoder
        gc.collect()

    if split:
        # Split into training and testing with sklearn, 80:20 split
        feats_train, feats_test, dialects_train, dialects_test = train_test_split(
            feats_split, dialects_split,
            train_size=0.8, random_state=42
        )
        return (feats_train, feats_test), (dialects_train, dialects_test)

    return feats_split, dialects_split
