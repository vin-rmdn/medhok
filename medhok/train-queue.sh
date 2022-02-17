#!/usr/bin/env sh

#./train.py mel_spectrogram cnn | tee ../log/train-mel_spectrogram-cnn.txt
./train.py spectrogram cnn | tee ../log/train-spectrogram-cnn.txt
#./train.py mfcc cnn | tee ../log/train-mfcc-cnn.txt

echo Done!
