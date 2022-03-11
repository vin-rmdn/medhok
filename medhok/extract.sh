#!/usr/bin/env sh

python -u ./medhok/main.py extract new mel_spectrogram 2>&1 | tee log/extract-mel_spectrogram.txt
python -u ./medhok/main.py extract new spectrogram 2>&1 | tee log/extract-spectrogram.txt
python -u ./medhok/main.py extract new mfcc 2>&1 | tee log/extract-mfcc.txt

echo Done!
