#!/usr/bin/env python3

import pickle

import numpy as np
from sklearn.model_selection import train_test_split

from configs import config as c


class FeatureMetadata:
    """Feature metadata class. Provides several functions such as the loading of feature metadata into a list.
    """

    @staticmethod
    def load_feature_metadata(feature_name='mel_spectrogram', split=False, sample=False, pathlib=True):
        """Loads feature metadata from the database.

        Args:
            feature_name (str): Name of the feature to load.
            split (bool): Whether to split the data into training and test set.
            sample (bool): Whether to sample the data.
            pathlib (bool): Whether to return pathlib.Path as the metadata

        Returns:
            list: List of feature metadata.
        """
        features = (c.FEATURES_DIR / feature_name).glob(f'**/*-{feature_name}-*.npy')
        metadata_feat = np.array([f for f in features])
        # Sampling
        if sample:
            metadata_feat = np.random.choice(metadata_feat, size=5000, replace=False)
        metadata_dial = np.array([f.parent.name for f in metadata_feat])[:, np.newaxis]

        # String
        if not pathlib:
            metadata_feat = np.array([str(f) for f in metadata_feat])

        if split:
            train_feat, test_feat, train_dial, test_dial = train_test_split(metadata_feat, metadata_dial, test_size=0.1)
            return train_feat, test_feat, train_dial, test_dial
        return metadata_feat, metadata_dial

    @staticmethod
    def transform_dialects(dialects):
        with open('model/dialects-encoder.pkl', 'rb') as f:
            dialects_encoder = pickle.load(f)
        dialects_onehot = dialects_encoder.transform(dialects).astype(np.float16)
        return dialects_onehot
