#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
from typing import Tuple
import pickle

from configs import config as c


with open('model/dialects-encoder.pkl', 'rb') as f:
    dialects_encoder = pickle.load(f)


def squeeze_feature(feature, label):
    return tf.squeeze(feature), label


def transform_data(filename, label) -> Tuple[np.ndarray, np.ndarray]:
    # Load numpy data
    feat = np.load(filename.numpy())[:, :, np.newaxis]

    # Onehot
    label_onehot = dialects_encoder.transform([label.numpy().astype('U13')]).reshape(16)

    # print(feat.shape)
    # print(label_onehot)
    return feat, label_onehot


def transform_data_tf(filename, label):
    feat, label_onehot = tf.py_function(transform_data, [filename, label], (tf.float16, tf.float16))
    return feat, label_onehot


def create_dataset(X, y, squeeze=False) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.map(transform_data_tf, num_parallel_calls=tf.data.AUTOTUNE)
    # dataset = dataset.map(onehot_label)
    # dataset = dataset.map(load_numpy_as_tf_tensor, num_parallel_calls=tf.data.AUTOTUNE)

    if squeeze:
        dataset = dataset.map(squeeze_feature, num_parallel_calls=tf.data.AUTOTUNE)

    # Performance setting
    dataset = (
        dataset
        .cache()
        .shuffle(buffer_size=len(dataset), reshuffle_each_iteration=True)
        .batch(c.BATCH_SIZE, drop_remainder=True)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    return dataset
