#!/usr/bin/env python3
import logging
import pickle
import numpy as np
import config as c


class GreedyLoader:
    """Greedy Loader loads the whole dataset directly to RAM. Make sure to have at least  24GB of RAM when operating GreedyLoader.
    """
    @staticmethod
    def load_features(metadata, feature_name='mel_spectrogram', split=False):
        """Loads all features from the cached file.
        """
        features = np.empty((len(metadata), c.FEATURE_AMOUNT[feature_name], c.WINDOW_SIZE, 1), dtype=np.float32)
        for i, f in enumerate(metadata):
            print(f'\rLoading {i + 1} out of {len(metadata)} files.', end='')
            features[i, ] = np.load(f)[:, :, np.newaxis]
        print()
        logging.info('Greedy loading finished.')
        return features

    @staticmethod
    def transform_dialects(dialects):
        with open('model/dialects-encoder.pkl', 'rb') as f:
            dialects_encoder = pickle.load(f)
        dialects_onehot = dialects_encoder.transform(dialects).astype(np.float32)
        return dialects_onehot
