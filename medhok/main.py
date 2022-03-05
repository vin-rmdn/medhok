#!/usr/bin/env python3
from sys import argv
import logging

from loader.audio import Audio
import config as c

# STARTUP PARAMETERS
SERIALIZE_NORMALIZE = False

if __name__ == '__main__':
    # Debug
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    if len(argv) > 1:
        if argv[1] == 'extract':
            logging.info('Loading dataset to memory...')
            audio = Audio()
            logging.info('Loading metadata...')
            audio.load_metadata()
            if len(argv) > 2 and argv[2] == 'preloaded':
                logging.info(f'Loading {c.DEFAULT_FEATURE} features...')
                audio.get_features(c.DEFAULT_FEATURE, preloaded=True)
                logging.info('Done.')
            else:
                logging.info('Extracting features from wav files...')
                audio.get_features('mel_spectrogram', preloaded=False)
                audio.get_features('spectrogram', preloaded=False)
                audio.get_features('mfcc', preloaded=False)
            audio.feature_split()
            logging.info('Getting one-hot-encoded dialect labels...')
            audio.encode_variable('dialect')
        elif argv[2] == 'serialize':
            pass
        elif argv[2] == 'train':
            if len(argv) > 3:
                if argv[3] == 'chatfield14':
                    model = c.chatfield14
                elif argv[3] == 'voxceleb':


    print('Done!')

