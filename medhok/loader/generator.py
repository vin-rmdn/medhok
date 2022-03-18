#!/usr/bin/env python3
import pickle
# import logging

import numpy as np
import tensorflow as tf


class MedhokGenerator(tf.keras.utils.Sequence):
    """Medhok Generator: a generator that loads (split) files from disk (cached in data/features/) to memory.
    """
    def __init__(self, list_wavs: list, batch_size=32, dim=(128, 667, 1), n_channels=1, n_dialects=16, shuffle=True, squeeze=False):
        """Initialization function.

        Args:
            list_wavs (list): List of audio wave directories (of type pathlib.Path)
            dialects (list): List of dialects corresponding to each value on list_wavs
            batch_size (int, optional): _description_. Defaults to 32.
            dim (tuple, optional): _description_. Defaults to (128, 667).
            n_channels (int, optional): _description_. Defaults to 1.
            n_dialects (int, optional): _description_. Defaults to 16.
            shuffle (bool, optional): _description_. Defaults to True.
            squeeze (bool, optional): _description_. Defaults to False.
        """
        self.list_wavs = list_wavs
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_dialects = n_dialects
        self.shuffle = shuffle
        self.squeeze = squeeze
        with open('model/dialects-encoder.pkl', 'rb') as f:
            self.dialects_encoder = pickle.load(f)
        self.on_epoch_end()

    def __data_generation(self, list_wavs_temp):
        """Data generation function.

        Args:
            list_wavs_temp (list): Wav file list
        """
        _shape = (self.batch_size, *self.dim)
        if not self.squeeze:
            _shape += (self.n_channels, )
        X = np.empty(_shape, dtype=np.float16)
        y = np.empty((self.batch_size, self.n_dialects), dtype=np.uint8)

        for i, filename in enumerate(list_wavs_temp):
            if self.squeeze:
                X[i, ] = np.load(filename).astype(np.float16)
            else:
                X[i, ] = np.load(filename)[:, :, np.newaxis].astype(np.float16)
            y[i, ] = self.dialects_encoder.transform([[filename.parent.name]]).astype(np.uint8)
            # logging.info(f'Got size {X[i].shape} and {y[i].shape} of dtype {X[i].dtype} and {y[i].dtype}')

        return X, y

    def __len__(self):
        'Denotes the number of batches in an epoch'
        return int(np.floor(len(self.list_wavs) / self.batch_size))

    def __getitem__(self, index):
        'Loads item of a batch from list_wavs and returns it'
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_wavs_temp = [self.list_wavs[k] for k in indexes]
        X, y = self.__data_generation(list_wavs_temp)
        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch. Shuffles the data.
        """
        self.indexes = np.arange(len(self.list_wavs))
        if self.shuffle:
            np.random.shuffle(self.indexes)
