#!/usr/bin/env python3
"""Get Dataset — provides the developer with the features extracted with
the Extract Feature module.
"""
import time
import gc
import pickle

import numpy as np
import tensorflow as tf
from pympler.asizeof import asizeof
from . import preprocessing as pre
from . import constants as c
from . import tf_helper as tfh

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# TODO: CMVN, Consider putting data augmentation


def get_dataset() -> dict:
    """Get Dataset function — gets the full dataset location.

    Returns:
        dict: a list of dialects and its respective .wav files
    """
    dialects = (c.RAW_DIR).iterdir()

    wavs = {}
    for dialect in dialects:
        wavs[dialect.parts[-1]] = [wav for wav in dialect.iterdir() if wav.parts[-1][-3:].lower() == 'wav']

    return wavs


def load_features(feature_name='mel_spectrogram', normalised=False):
    """
    Loads precached features to RAM.
    :param feature_name: feature name (defaults to mel_spectrogram)
    :return: numpy.ndarray
    """
    return_feature = []
    dialects = []
    dataset = get_dataset()
    if c.DEBUG:
        print('Using normalisation method:', normalised)
    for dialect, data in dataset.items():
        if c.DEBUG:
            print(f'Loading {dialect}: ')

        for datum in data:
            if c.DEBUG:
                print('-', datum.parts[-1], end=' ')
            #     print(constants.FEATURES_DIR / dialect)
            #     print(str(datum.parts[-1]) + '-' + feature_name + '.npy')
            time_start = time.time()
            _buffer = np.load(
                c.FEATURES_DIR / dialect / (str(datum.parts[-1]) + '-' + feature_name + '.npy'))
            if normalised:
                # _buffer = pre.normalise_feature(_buffer, mean_var=True)   # non-CUDA
                pre.normalise_feature(_buffer, mean_var=True)
            return_feature.append(_buffer)
            dialects.append(dialect)
            print('(time: {time:.1f})'.format(time=time.time() - time_start))
            del _buffer
            gc.collect()

    print('Done!')

    # Cleanup
    del dialect, data, datum, time_start, dataset
    gc.collect()

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
        return [feats_train, feats_test], [dialects_train, dialects_test]

    return feats_split, dialects_split

def __create_tf_record(filename, features, dialects):
    """
    Creates a TFRecord from a serialized feature and dialect.
    """
    print(f"Saving TFRecords to {filename}.")
    with tf.io.TFRecordWriter(filename) as writer:
        for i in range(features.shape[0]):
            print(f"\rWriting {i}/{features.shape[0]}...", end='')
            tf_feature_list = {
                'feature': c.bytes_feature(tf.io.serialize_tensor(features[i]).numpy()),
                'dialect': c.bytes_feature(tf.io.serialize_tensor(dialects[i]))
            }
            tf_features = tf.train.Features(feature=tf_feature_list)

            record = tf.train.Example(features=tf_features)
            record_bytes=record.SerializeToString()
            writer.write(record_bytes)

def write_tf_records(feature_name='mel_spectrogram', split=False, normalised=True):
    """ Write TFRecord files to disk.

    Args:
        feature_name (str, optional): Pre-extracted audio feature name. Available values: mel_spectrogram, spectrogram, mfcc. Defaults to 'mel_spectrogram'.
    """
    print("Loading dataset...", end=' ')
    features, dialects = load_windowed_dataset(feature_name, split=split, onehot=False, normalised=normalised)
    print('Done!')

    # Reducing dialect dimension
    if split:
        dialects[0] = dialects[0].reshape(-1)
        dialects[1] = dialects[1].reshape(-1)
    else:
        dialects = dialects.reshape(-1)

    # Writing additional information to metadata files
    print("Writing metadata...", end=' ')
    if split:
        total_train_size = features[0].shape[0]
        total_test_size = features[1].shape[0]
        with open(c.TFRECORDS_DIR / 'train_metadata.pickle', 'wb') as f:
            pickle.dump(total_train_size, f)
        with open(c.TFRECORDS_DIR / 'test_metadata.pickle', 'wb') as f:
            pickle.dump(total_test_size, f)
    else:
        total_size = features.shape[0]
        with open(c.TFRECORDS_DIR / 'metadata.pickle', 'wb') as f:
            pickle.dump(total_size, f)
    print('Done!')

    # Creating TensorFlow dataset
    print("Creating TensorFlow dataset...", end=' ')
    if split:
        filename_train = (c.TFRECORDS_DIR / (feature_name + ('-normalised' if normalised else '') + '-train.tfrecords')).as_posix()
        filename_test = (c.TFRECORDS_DIR / (feature_name + ('-normalised' if normalised else '') + '-test.tfrecords')).as_posix()

        __create_tf_record(filename_train, features[0], dialects[0])
        __create_tf_record(filename_test, features[1], dialects[1])
        del filename_test
        gc.collect()


    else:
        filename = (c.TFRECORDS_DIR / (feature_name + ('-normalised' if normalised else '') + '.tfrecords')).as_posix()
        print("\n")
        __create_tf_record(filename, features, dialects)
    print('\nDone!')
