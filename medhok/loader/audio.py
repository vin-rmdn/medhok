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
        logging.info(f'Getting {feature_name} features...')

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
                logging.info(f'Loading {file} with preloaded={preloaded}.')
                if not preloaded:
                    time_start = time.time()
                    wav = self.__load_audio(str(dialect_path / (file + '.wav')))[0]
                    feature = extractor(wav)
                    feature = self.__trim_silence(wav, feature)
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
                    logging.info(f"\tTime taken: {time.time() - time_start}")
                    del _buffer, time_start
                    gc.collect()

        logging.info('Done!')
