import os
import constants
import numpy as np


def get_dataset():
    dialects = os.listdir(constants.DATASET_DIR + '/raw')

    wavs = dict()
    for dialect in dialects:
        wavs[dialect] = [wav for wav in os.listdir(constants.DATASET_DIR + '/raw/' + dialect) if wav[-3:] == 'wav']

    return wavs


def get_spectrograms(local=True, params='mel_spectrogram'):
    dataset = get_dataset()
    return_data = []
    if local:   # TODO: finish loading spectrogram function
        for dialect, data in dataset:
            for datum in data:
                if constants.DEBUG:
                    print('Getting', dialect + ':', datum)
                return_data.append(np.fromfile(constants.FEATURES_DIR + dialect + '/' + datum + '-' + params))
    return return_data

