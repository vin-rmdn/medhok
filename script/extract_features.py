#!/usr/bin/env python3
"""Extract Features module — contains functions to extract features from
wave files. Options are as follows: Mel spectrogram, Spectrogram
(through Short-time Fourier Transform), and Mel Frequency Cepstral
Coefficients.
"""
import os
import time
import matplotlib
from matplotlib import pyplot as plt
import librosa
from librosa import display
import numpy as np

import constants
import get_dataset

matplotlib.use('agg')


def extract_feature(
    wave, feature, sr=constants.SAMPLE_RATE, n_mfcc=constants.MFCC_NUMBER,
    n_fft=constants.FFT_NUMBER, fmin=200, fmax=4000,
    hop_length=constants.HOP_LENGTH
) -> np.ndarray:
    if feature == 'mel_spectrogram':
        return librosa.feature.melspectrogram(
            wave, sr=sr, n_fft=n_fft,
            hop_length=hop_length,
            fmin=fmin, fmax=fmax
        )
    elif feature == 'mfcc':
        return librosa.feature.mfcc(
            wave,
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            fmin=fmin,
            fmax=fmax)
    elif feature == 'spectrogram':
        return np.abs(
            librosa.core.stft(
                wave,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft))


def trim_silence(wave, feature):
    _rms = librosa.feature.rms(wave, hop_length=constants.HOP_LENGTH)
    _threshold = np.mean(_rms) / 2 * 1.04   # formula from somewhere
    _mask = np.nonzero(_rms > _threshold)[1]

    return feature[:, _mask]


def save_figure(feature, filename='untitled', dialect='', feature_name=''):
    fig = plt.figure(figsize=constants.FIGURE_SIZE)
    ax = fig.add_subplot(1, 1, 1)
    if feature_name == 'mel_spectrogram':
        display.specshow(
            librosa.core.power_to_db(feature),
            sr=constants.SAMPLE_RATE,
            x_axis='time',
            y_axis='log',
            ax=ax
        )
    elif feature_name == 'spectrogram':
        display.specshow(
            librosa.amplitude_to_db(feature),
            sr=constants.SAMPLE_RATE,
            hop_length=constants.HOP_LENGTH,
            x_axis='time',
            y_axis='hz',
            ax=ax
        )
    elif feature_name == 'mfcc':
        display.specshow(
            feature,
            sr=constants.SAMPLE_RATE,
            x_axis='time',
            ax=ax
        )

    if not os.path.exists(
        constants.VISUALIZATION_DIR + feature_name + '/' + dialect
    ):
        os.mkdir(constants.VISUALIZATION_DIR + feature_name + '/' + dialect)
    fig.savefig(
        constants.VISUALIZATION_DIR + feature_name + '/' + dialect + '/' + filename + '-' + feature_name + '.png'
    )
    fig.clear()
    plt.close(fig)
    plt.clf()
    del fig, ax


def main():
    """Main function for Extract Features module
    """
    for dialect, wavs in get_dataset.get_dataset().items():
        for wav in wavs:
            time_start = time.time()
            print('Extracting features for:', wav, end='')
            wave = librosa.load(
                constants.DATASET_DIR + '/raw/' + dialect + '/' + wav,
                sr=constants.SAMPLE_RATE,
                res_type='soxr_vhq',
                mono=True
            )[0]

            # Save to file
            if not os.path.exists('../dataset/features/' + dialect):
                os.mkdir('../dataset/features/' + dialect)

            for feature_name in constants.FEATURES:
                feature = extract_feature(wave, feature_name)
                feature = trim_silence(wave, feature)   # shorten with VAD
                # feature = feature.tolist()

                # Save numpy file
                np.save(
                    '../dataset/features/' + dialect + '/' + wav
                    + '-' + feature_name, feature
                )

            # Append
            del wave
            print(f'; Done! ({time.time() - time_start:2.1f})')


if __name__ == '__main__':
    main()
