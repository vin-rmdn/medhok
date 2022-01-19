#!/usr/bin/env python3

# TODO: CMVN, Consider putting data augmentation

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

def extract_feature(wave, feature, sr=constants.SAMPLE_RATE, n_mfcc=constants.MFCC_NUMBER, n_fft=constants.FFT_NUMBER,
                    fmin=200, fmax=4000) -> np.ndarray:
    if feature == 'mel_spectrogram':
        return librosa.feature.melspectrogram(wave, sr=sr, n_fft=n_fft, fmin=fmin, fmax=fmax)
    elif feature == 'mfcc':
        return librosa.feature.mfcc(wave, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft,
                                    fmin=fmin, fmax=fmax)
    elif feature == 'spectrogram':
        return np.abs(librosa.core.stft(wave))


def trim_silence(wave, feature):
    _rms = librosa.feature.rms(wave, frame_length=constants.FFT_NUMBER)
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
            x_axis='time',
            y_axis='log',
            ax=ax
        )
    elif feature_name == 'mfcc':
        display.specshow(
            feature,
            sr=constants.SAMPLE_RATE,
            x_axis='time'
        )

    if not os.path.exists(constants.VISUALIZATION_DIR + feature_name + '/' + dialect):
        os.mkdir(constants.VISUALIZATION_DIR + feature_name + '/' + dialect)
    fig.savefig(
        constants.VISUALIZATION_DIR + feature_name + '/' + dialect + '/' + filename + '-' + feature_name + '.png'
    )
    fig.clear()
    plt.close(fig)
    plt.clf()
    del fig, ax


def main():
    for dialect, wavs in get_dataset.get_dataset().items():
        for wav in wavs:
            lr_start = time.time()
            print('Extracting features for:', wav, end='')
            wave = librosa.load(
                constants.DATASET_DIR + '/raw/' + dialect + '/' + wav,
                sr=constants.SAMPLE_RATE,
                res_type='soxr_vhq',
                mono=True
            )[0]

            # Save to file
            if not os.path.exists('../dataset/features/'+dialect):
                os.mkdir('../dataset/features/'+dialect)

            sv_start = time.time()
            for feature_name in constants.FEATURES:
                feature = extract_feature(wave, feature_name)
                feature = trim_silence(wave, feature)   # shorten with VAD

                # Save numpy file
                feature.tofile('../dataset/features/'+dialect+'/'+wav+'-'+feature_name)

                # Save spectrogram image
                save_figure(feature=feature, filename=wav, dialect=dialect, feature_name=feature_name)

            # Append
            del wave
            print('; Done! (np: {np:2.1f}, plt: {plt:2.1f})'.format(np=time.time() - lr_start, plt=time.time() - sv_start))
            # return
    # features_df = cudf.DataFrame(features, columns=constants.FEATURE_COLUMNS)
    # features_df.to_json('../dataset/features.json', orient='index')


if __name__ == '__main__':
    main()
