#!/usr/bin/env sh

# Baseline - not available for MFCC
# python -u ./medhok/main.py train baseline mel_spectrogram 2>&1 | tee log/baseline-mel_spectrogram.txt
# python -u ./medhok/main.py train baseline spectrogram 2>&1 | tee log/baseline-spectrogram.txt

# Chatfield et al. (2014)
# python -u ./medhok/main.py finetune chatfield14 mel_spectrogram 2>&1 | tee log/chatfield14-mel_spectrogram-finetune.txt
# python -u ./medhok/main.py finetune chatfield14 spectrogram 2>&1 | tee log/chatfield14-spectrogram-finetune.txt
# python -u ./medhok/main.py train chatfield14 mfcc 2>&1 | tee log/chatfield14-mfcc.txt

# Shon et al. (2018)
# python -u ./medhok/main.py finetune shon18 mel_spectrogram 2>&1 | tee log/shon18-mel_spectrogram-finetune.txt
# python -u ./medhok/main.py finetune shon18 spectrogram 2>&1 | tee log/shon18-spectrogram-finetune.txt     # too memory-exhaustive
# python -u ./medhok/main.py finetune shon18 mfcc 2>&1 | tee log/shon18-mfcc-finetune.txt

# Warohma et al. (2018)
# python -u ./medhok/main.py finetune warohma18 mel_spectrogram 2>&1 | tee log/warohma18-mel_spectrogram-finetune.txt
python -u ./medhok/main.py finetune warohma18 spectrogram 2>&1 | tee log/warohma18-spectrogram-finetune.txt
# python -u ./medhok/main.py finetune warohma18 mfcc 2>&1 | tee log/warohma18-mfcc-finetune.txt

# Draghici et al. (2020) - CRNN
# python -u ./medhok/main.py finetune draghici20_crnn mel_spectrogram 2>&1 | tee log/draghici20_crnn-mel_spectrogram-finetune.txt
# python -u ./medhok/main.py finetune draghici20_crnn spectrogram 2>&1 | tee log/draghici20_crnn-spectrogram-finetune.txt
# python -u ./medhok/main.py train draghici20_crnn mfcc 2>&1 | tee log/draghici20_crnn-mfcc.txt

echo Done!
