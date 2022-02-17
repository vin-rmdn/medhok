#!/usr/bin/env python3

import os
import librosa
from librosa import display
import matplotlib
from matplotlib import pyplot as plt
import time
import warnings

import constants
warnings.filterwarnings('ignore')

matplotlib.use('agg')     # agg leaks. need fix
# matplotlib.rcParams['agg.path.chunksize'] = 20000
plt.ioff()

dialects = os.listdir(constants.DATASET_DIR + '/dialects')
wav_metadata = dict()

for dialect in dialects:
    wav_metadata[dialect] = [wav for wav in os.listdir(constants.DATASET_DIR + '/dialects/' + dialect) if wav[-3:] == 'wav']

for dialect, wavs in wav_metadata.items():
    for wav in wavs:
        time_start = time.time()
        print('Processing', dialect, '-', wav, end='')
        wav_float = librosa.load(
            constants.DATASET_DIR + '/dialects/' + dialect + '/' + wav,
            sr=constants.SAMPLE_RATE,
            mono=True,
            res_type='soxr_qq'
        )[0]
        print('; Saving plot...', end='')
        fig = plt.figure(figsize=(70, 5))
        ax = fig.add_subplot(1, 1, 1)
        display.waveshow(wav_float, sr=constants.SAMPLE_RATE, ax=ax)
        if not os.path.exists('../visualization/' + dialect):
            os.mkdir('../visualization/' + dialect)
        fig.tight_layout()
        fig.savefig('../visualization/' + dialect + '/' + wav + '.png', pad_inches=0)
        plt.pause(.1)
        fig.clear()
        plt.close('all')
        plt.cla()
        plt.clf()
        del wav_float
        del fig, ax
        print(' Done! (' + str(time.time() - time_start) + ')', end='\n')


print('Done!')
