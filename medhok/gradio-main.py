#!/usr/bin/env python3

import gradio as gr
import tensorflow as tf
import librosa
import numpy as np
import pickle

from configs import config as c
from loader.audio import Audio
from loader.audio_numba import CMVN

model = tf.keras.models.load_model('model/shon18-mel_spectrogram-finetuned.h5')
with open('model/dialects-encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)


def dialect_recognition(wave):
    sr, wave = wave[0], librosa.util.buf_to_float(wave[1])
    wave = librosa.resample(wave, orig_sr=sr, target_sr=16000)
    wave = Audio.pre_emphasize(wave)
    wave = Audio.trim_silence(wave, feature=None)
    wave = Audio.filter_frequency(wave)
    waves = Audio.split_segments(wave)

    features = []
    for wav in waves:
        feature = librosa.feature.melspectrogram(
            y=wav,
            sr=c.SAMPLE_RATE,
            win_length=c.FRAME_SIZE,
            hop_length=c.FRAME_STRIDE,
            n_mels=c.MEL_AMOUNT,
            fmin=c.F_MIN,
            fmax=c.F_MAX
        )
        feature = CMVN(feature)
        feature = np.nan_to_num(feature)
        feature = feature.astype(np.float16)
        features.append(feature)

    features = np.array(features, dtype=np.float16)

    predictions = model.predict(features)

    return encoder.inverse_transform(predictions)


def main():
    # Preparation steps
    # model = tf.keras.models.load_model('model/shon18-mfcc-finetuned.h5')

    interface = gr.Interface(fn=dialect_recognition, inputs='audio', outputs='text')
    interface.launch(share=True)


if __name__ == '__main__':
    main()
