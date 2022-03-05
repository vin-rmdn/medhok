#!/usr/bin/env python3
import logging
import os
import time
import gc

import librosa
import numpy as np

import config as c
from loader.audio_numba import normalize


class Audio:
    """The Audio class of Medhok. Provides data to recorded instances of Javanese language interviews as a class.
    """
    def __init__(self):
        pass

    def __load_audio(self, filename):
        return librosa.load(filename, sr=c.SAMPLE_RATE, mono=c.MONO, res_type=c.RESAMPLER_TYPE)

    def __trim_silence(self, wave, feature):
        _rms = librosa.feature.rms(wave, hop_length=c.HOP_LENGTH)
        _threshold = np.mean(_rms) / 2 * 1.04   # formula from somewhere
        _mask = np.nonzero(_rms > _threshold)[1]

        return feature[:, _mask]

    def load_metadata(self) -> None:
        logging.info('Initializing audio metadata...')
        self.dialects = c.DIALECTS
        self.metadata = {}

        # Iterate through dialects folder to get WAV filenames
        for dialect_path in self.dialects:
            dialect = dialect_path.name
            self.metadata[dialect] = []
            for file in (c.RAW_DIR / dialect).iterdir():
                if file.suffix == '.wav':
                    logging.info(f'Found {file.name}. Registering as {file.stem}')
                    self.metadata[dialect].append(file.stem)

    def get_features(self, feature_name, preloaded=True) -> None:
        """Get features from loaded audio files and load it to memory.

        @param feature_name: Option between 'mel_spectrogram', 'spectrogram', or MFCC.
        @return none
        """
        logging.info(f'Getting {feature_name} features with preloaded={preloaded}...')

        self.features = []
        self.dialects = []
        dialects = c.DIALECTS

        # Feature function selection
        try:
            assert feature_name in c.FEATURES
        except AssertionError:
            logging.error(f'feature_name parameter cannot be outside of these choices: {c.FEATURES}')
            exit(1)

        if feature_name == 'mel_spectrogram':
            extractor = c.mel_spectrogram
        elif feature_name == 'spectrogram':
            extractor = c.spectrogram
        elif feature_name == 'mfcc':
            extractor = c.MFCC

        for dialect_path in dialects:
            dialect = dialect_path.name
            if not os.path.exists(c.FEATURES_DIR / dialect):
                os.mkdir(c.FEATURES_DIR / dialect)
            for file in self.metadata[dialect]:
                logging.info(f'Loading {file}.')
                if not preloaded:
                    time_start = time.time()
                    wav = self.__load_audio(str(dialect_path / (file + '.wav')))[0]
                    feature = extractor(wav)
                    feature = self.__trim_silence(wav, feature)
                    # TODO: visualize and save image
                    feature = normalize(feature)
                    filename = str(c.FEATURES_DIR / dialect / (file + '-' + feature_name))
                    np.save(filename, feature)
                    self.features.append(feature)
                    logging.info(f"\tTime taken: {time.time() - time_start}")
                else:
                    logging.info(f'Getting {feature_name} from {file}.')
                    time_start = time.time()
                    _buffer = np.load(c.FEATURES_DIR / dialect / (file + '-' + feature_name + '.npy'))
                    self.features.append(_buffer)
                    self.dialects.append(dialect)
                    time_taken = time.time() - time_start
                    logging.info(f"\tTime taken: {time_taken:.1f} second{'s' if time_taken > 1 else ''}.")
                    del _buffer, time_start
                    gc.collect()
        logging.info('Function get_feature finished.')

    def __window_split(self, feature):
        windows = []
        l_p = 0
        r_p = c.WINDOW_SIZE
        while l_p < feature.shape[1]:
            temp = feature[:, l_p:r_p]
            if temp.shape[1] != c.WINDOW_SIZE:
                while temp.shape[1] < c.WINDOW_SIZE:
                    temp = np.append(temp, [[0]] * feature.shape[0], axis=1)
            windows.append(temp)
            l_p += c.WINDOW_SIZE
            r_p += c.WINDOW_SIZE
        return np.array(windows, dtype=np.float32)

    def feature_split(self):
        """Split features into fixed windows of several seconds."""
        logging.info('Splitting features into fixed windows...')
        feats_split = []
        dialects_split = []

        for feature, dialect in zip(self.features, self.dialects):
            _buffer = self.__window_split(feature)
            for window in _buffer:
                feats_split.append(window)
                dialects_split.append(dialect)
        del _buffer, feature, dialect
        gc.collect()

        logging.info('Converting features and dialects to numpy representations...')
        feats_split = np.array(feats_split, dtype=np.float32)[:, :, :, np.newaxis]
        dialects_split = np.array(dialects_split)[:, np.newaxis]

        self.features = feats_split
        self.dialects = dialects_split

    def encode_variable(self, variable='dialect'):
        """Encode variables to one-hot encoding.

        @param variable: Option defaults to 'dialect'. Further developments are to be seen.
        """
        logging.info(f'Encoding {variable} to one-hot encoding...')
        if variable == 'dialect':
            dialects = np.unique(self.dialects)
            self.dialects = np.array([np.where(dialects == d)[0][0] for d in self.dialects])
            self.dialects = self.dialects[:, np.newaxis]
            self.dialects = np.array([[1 if d == i else 0 for i in range(len(dialects))] for d in self.dialects])
        else:
            logging.error(f'Variable {variable} not supported.')
            exit(1)

    def create_tfrecords(self):
        """Create TFRecords from loaded features with the additional dialect information."""
        logging.info('Creating TFRecords...')
        # TODO: create TFRecords
