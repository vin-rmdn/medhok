#!/usr/bin/env python3
"""This module serves as a module for the Audio class, a class that prepares the data for the neural network."""

import logging
import os
import time
import gc

import librosa
from librosa.display import specshow
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from dsp.filter import butter_highpass_filter, butter_lowpass_filter

import config as c
from loader.audio_numba import CMVN

matplotlib.use('svg')
plt.style.use('seaborn')


class Audio:
    """The Audio class of Medhok. Provides data to recorded instances of
    Javanese language interviews as a class.
    """
    def __init__(self):
        self.dialects = []
        self.features = []
        self.metadata = {}

    @staticmethod
    def trim_silence(wave, feature=None):
        """Employs Voice Activity Detection on a wavefile to cut silence throughout the wavefile."""
        if feature is not None:
            frame_length = c.FRAME_SIZE
            hop_length = c.FRAME_STRIDE
        else:
            frame_length = 1
            hop_length = 1
        _rms = librosa.feature.rms(wave, hop_length=hop_length, frame_length=frame_length)
        _threshold = np.mean(_rms) / 2 * 1.04   # formula from Shon et al. (2018)
        _mask = np.nonzero(_rms > _threshold)[1]
        if feature is not None:
            return wave[:, _mask]
        return wave[_mask]

    def __pre_emphasize(self, wav):
        return np.append(wav[0], wav[1:] - c.PRE_EMPHASIS_ALPHA * wav[:-1])

    def __filter_frequency(self, wav) -> np.ndarray:
        """Filter the wave frequency to only include frequencies audible to humans.

        Args:
            wav (numpy.ndarray): wave file

        Return:
            numpy.ndarray: filtered wave file
        """
        # Low-pass filter
        wav = butter_lowpass_filter(wav, cutoff=c.F_MAX, fs=c.SAMPLE_RATE, order=c.POLYNOMIAL_ORDER)
        wav = butter_highpass_filter(wav, cutoff=c.F_MIN, fs=c.SAMPLE_RATE, order=c.POLYNOMIAL_ORDER)
        return wav

    def __load_audio(self, filename):
        return librosa.load(filename, sr=c.SAMPLE_RATE, mono=c.MONO, res_type=c.RESAMPLER_TYPE)

    def __save_visualization(self, feature, filepath, feature_name='mel_spectrogram') -> None:
        """Save visualization of features.

        @param feature: Feature to be visualized.
        @param filename: Filename to be saved.
        @return none
        """
        fig = plt.figure(figsize=c.FIGURE_SIZE)
        ax = fig.add_subplot(1, 1, 1)
        if feature_name == 'mel_spectrogram':
            plt.title(f'Mel spectrogram - {filepath.stem}')
            spec = specshow(
                librosa.power_to_db(feature, ref=np.max),
                sr=c.SAMPLE_RATE,
                hop_length=c.FRAME_STRIDE,
                y_axis='mel',
                x_axis='time',
                fmin=c.F_MIN,
                fmax=c.F_MAX,
                ax=ax
            )
            fig.colorbar(spec, format='%+2.0f dB', ax=ax)
        elif feature_name == 'spectrogram':
            plt.title(f'Spectrogram - {filepath.stem}')
            spec = ax.imshow(
                librosa.amplitude_to_db(
                    feature
                ),
                aspect='auto',
                origin='lower',
                extent=[0, feature.shape[1] / c.SAMPLE_RATE, c.F_MIN, c.F_MAX],
                cmap='viridis'
            )
            # spec = specshow(
            #     librosa.power_to_db(feature, ref=np.max),
            #     sr=c.SAMPLE_RATE,
            #     hop_length=c.FRAME_STRIDE,
            #     y_axis='linear',
            #     x_axis='time',
            #     ax=ax
            # )
            fig.colorbar(spec, format='%+2.0f dB', ax=ax)
        elif feature_name == 'mfcc':
            plt.title(f'MFCC - {filepath.stem}')
            specshow(
                feature,
                sr=c.SAMPLE_RATE,
                y_axis='mel',
                x_axis='time',
                ax=ax
            )
        if not filepath.parent.parent.resolve().exists():
            os.mkdir(filepath.parent.parent.resolve())
        if not filepath.parent.resolve().exists():
            os.mkdir(filepath.parent.resolve())
        plt.savefig(filepath)
        fig.clear()
        plt.close(fig)
        plt.clf()
        del fig, ax
        gc.collect()

    def __split_segments(self, wav) -> np.ndarray:
        """Splits a wav file into several segments of decided seconds.

        Args:
            wav (np.ndarray): input wav to be split

        Returns:
            np.ndarray: split segments
        """
        segments = []
        l_p = 0
        r_p = c.SEGMENT_LENGTH
        while l_p < wav.shape[0]:
            temp = wav[l_p:r_p]
            while temp.shape[0] < c.SEGMENT_LENGTH:
                temp = np.append(temp, [0] * (c.SEGMENT_LENGTH - temp.shape[0]))
            segments.append(temp)
            if l_p + c.SEGMENT_LENGTH >= wav.shape[0]:
                break
            l_p += c.SEGMENT_STRIDE
            r_p += c.SEGMENT_STRIDE
        return np.array(segments, dtype=np.float32)

    def load_metadata(self) -> None:
        logging.info('Initializing audio metadata...')
        dialects = c.DIALECTS

        # Iterate through dialects folder to get WAV filenames
        for dialect_path in dialects:
            dialect = dialect_path.name
            self.metadata[dialect] = []
            for file in (c.RAW_DIR / dialect).iterdir():
                if file.suffix == '.wav':
                    logging.info(f'Found {file.name}. Registering as {file.stem}')
                    self.metadata[dialect].append(file.stem)

    def __save_feature(self, feature, filename):
        """Save feature to file.

        Args:
            feature (_type_): feature to be saved
            filename (_type_): filename to be saved in
        """
        np.save(filename, feature)

    def get_features(self, feature_name, preloaded=True) -> None:
        """Get features from loaded audio files and load it to memory.

        @param feature_name: Option between 'mel_spectrogram', 'spectrogram', or MFCC.
        @param preloaded: Option to use cached data (as provided in data/features).
        @return none
        """
        logging.info(f'Getting {feature_name} features with preloaded={preloaded}...')

        self.features = []
        self.dialects = []
        dialects = c.DIALECTS

        if feature_name == 'mel_spectrogram':
            extractor = c.mel_spectrogram
        elif feature_name == 'spectrogram':
            extractor = c.spectrogram
        elif feature_name == 'mfcc':
            extractor = c.MFCC
        else:
            logging.error(f'feature_name parameter cannot be outside of these choices: {c.FEATURES}')
            exit(1)

        dial_iter = 1
        for dialect_path in dialects:
            dialect = dialect_path.name
            if not os.path.exists(c.FEATURES_DIR / feature_name):
                os.mkdir(c.FEATURES_DIR / feature_name)
            if not os.path.exists(c.FEATURES_DIR / feature_name/ dialect):
                os.mkdir(c.FEATURES_DIR / feature_name / dialect)
            wav_iter = 1
            for file in self.metadata[dialect]:
                logging.info(f'Loading {file}. (dial {dial_iter}/{len(dialects)}, file {wav_iter}/{len(self.metadata[dialect])})')

                if not preloaded:
                    time_start = time.time()

                    # Loading audio and doing wave signal modification
                    # (such as pre-emphasis, trimming silence, and segmenting)
                    wav = self.__load_audio(str(dialect_path / (file + '.wav')))[0]
                    wav = self.__pre_emphasize(wav)
                    wav = Audio.trim_silence(wav, feature=None)
                    wav = self.__filter_frequency(wav)
                    feature = extractor(wav)
                    vis_timer = time.time()
                    self.__save_visualization(
                        feature, c.VISUALIZATION_DIR / feature_name / dialect / f'{file}.svg',
                        feature_name=feature_name
                    )
                    logging.info('Saving %s visualization took %.2f second(s).', file, time.time() - vis_timer)

                    # Segments
                    segments = self.__split_segments(wav)
                    for iteration in range(segments.shape[0]):
                        split_start = time.time()
                        segment = segments[iteration]
                        feature = extractor(segment)
                        feature = CMVN(feature)
                        feature = np.nan_to_num(feature)    # anticipate divide by 0
                        if feature_name == 'spectrogram':
                            feature = feature[c.SPEC_LOWER_BOUND:c.SPEC_UPPER_BOUND, :]
                        feature = feature.astype(np.float16)    # FP16 after numba

                        # Cache features
                        filepath = str(c.FEATURES_DIR / feature_name / dialect / (file + '-' + feature_name + '-' + str(iteration)))

                        # Check for null values
                        if np.isnan(feature).any():
                            logging.error(f'Iteration {iteration + 1} ({iteration}) of {file} has null values. Start: {c.SEGMENT_STRIDE * iteration}, end: {c.SEGMENT_STRIDE * iteration + c.SEGMENT_LENGTH}. Exiting...')
                            exit(1)

                        # Saving
                        np.save(filepath, feature)
                        # self.features.append(feature)
                        logging.info('Iteration %d of %d, took %.2f seconds.', iteration + 1, segments.shape[0], time.time() - split_start)
                    logging.info("\tTime taken: %.2f seconds.", time.time() - time_start)
                    del time_start, wav, segments, split_start, segment, feature, filepath
                    gc.collect()
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
                wav_iter += 1
            dial_iter += 1
        logging.info('Function get_feature finished.')

    def __window_split(self, feature):
        windows = []
        l_p = 0
        r_p = c.FRAME_SIZE * c.SAMPLE_RATE
        while l_p < feature.shape[1]:
            temp = feature[:, l_p:r_p]
            if temp.shape[1] != c.WINDOW_SIZE:
                while temp.shape[1] < c.WINDOW_SIZE:
                    temp = np.append(temp, [[0]] * feature.shape[0], axis=1)
            windows.append(temp)
            l_p += c.FRAME_STRIDE * c.SAMPLE_RATE
            r_p += c.FRAME_STRIDE * c.SAMPLE_RATE
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
