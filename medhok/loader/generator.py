import numpy as np
import tensorflow as tf


class MedhokGenerator(tf.keras.utils.Sequence):
    """Medhok Generator: a generator that loads (split) files from disk (cached in data/features/) to memory.
    """
    def __init__(self, list_wavs: list, batch_size=32, dim=(128, 667), n_channels=1, n_dialects=16, shuffle=True):
        """Initialization function.

        Args:
            list_wavs (list): List of audio wave directories (of type pathlib.Path)
            dialects (list): List of dialects corresponding to each value on list_wavs
            batch_size (int, optional): _description_. Defaults to 32.
            dim (tuple, optional): _description_. Defaults to (128, 667).
            n_channels (int, optional): _description_. Defaults to 1.
            n_dialects (int, optional): _description_. Defaults to 16.
            shuffle (bool, optional): _description_. Defaults to True.
        """
        self.list_wavs = list_wavs
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_dialects = n_dialects
        self.shuffle = shuffle
        self.on_epoch_end()

    def __data_generation(self, list_wavs_temp):
        """Data generation function.

        Args:
            list_wavs_temp (list): Wav file list
        """
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.n_dialects), dtype=int)

        for i, filename in enumerate(list_wavs_temp):
            X[i,] = np.load(filename)
            y[i,] = filename.parent.name

    def on_epoch_end(self):
        """Updates indexes after each epoch. Shuffles the data.
        """
        self.indexes = np.arange(len(self.list_wavs))
        if self.shuffle:
            np.random.shuffle(self.indexes)
