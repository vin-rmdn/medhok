#!/usr/bin/env sh

# Baseline - not available for MFCC
# python -u ./medhok/main.py train baseline mel_spectrogram 2>&1 | tee log/baseline-mel_spectrogram.txt
# python -u ./medhok/main.py train baseline spectrogram 2>&1 | tee log/baseline-spectrogram.txt

# Chatfield et al. (2014)
# python -u ./medhok/main.py train chatfield14 mel_spectrogram 2>&1 | tee log/chatfield14-mel_spectrogram.txt
python -u ./medhok/main.py train chatfield14 spectrogram 2>&1 | tee log/chatfield14-spectrogram.txt

# Shon et al. (2018)
# python -u ./medhok/main.py train shon18 spectrogram 2>&1 | tee log/shon18-spectrogram.txt     # too memory-exhaustive
# python -u ./medhok/main.py train shon18 mel_spectrogram 2>&1 | tee log/shon18-mel_spectrogram.txt
# python -u ./medhok/main.py train shon18 mfcc 2>&1 | tee log/shon18-mfcc.txt

# Warohma et al. (2018)
# python -u ./medhok/main.py train warohma18 mel_spectrogram 2>&1 | tee log/warohma18-mel_spectrogram.txt
# python -u ./medhok/main.py train warohma18 spectrogram 2>&1 | tee log/warohma18-spectrogram.txt
# python -u ./medhok/main.py train warohma18 mfcc 2>&1 | tee log/warohma18-mfcc.txt

# Draghici et al. (2020) - CRNN
# python -u ./medhok/main.py train draghici20_crnn spectrogram 2>&1 | tee log/draghici20_crnn-spectrogram.txt
# python -u ./medhok/main.py train draghici20_crnn mel_spectrogram 2>&1 | tee log/draghici20_crnn-mel_spectrogram.txt
# python -u ./medhok/main.py train draghici20_crnn mfcc 2>&1 | tee log/draghici20_crnn-mfcc.txt

echo Done!
