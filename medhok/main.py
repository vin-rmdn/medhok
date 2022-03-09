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
            if len(argv) > 2:
                if argv[2] == 'new':
                    if len(argv) > 3:
                        logging.info(f'Extracting {argv[3]} feature from wav files...')
                        audio.get_features(argv[3], preloaded=False)
                        logging.info('Done.')
                    else:
                        logging.error('Please specify a feature to extract.')
                        exit(1)
                else:
                    logging.info(f'Loading {c.DEFAULT_FEATURE} features...')
                    audio.get_features(c.DEFAULT_FEATURE, preloaded=True)
                    logging.info('Done.')
                # logging.info('Extracting features from wav files...')
                # audio.get_features('mel_spectrogram', preloaded=False)
                # audio.get_features('spectrogram', preloaded=False)
                # audio.get_features('mfcc', preloaded=False)
            # audio.feature_split()
            # logging.info('Getting one-hot-encoded dialect labels...')
            # audio.encode_variable('dialect')
        elif argv[2] == 'serialize':
            pass
        elif argv[2] == 'train':
            if len(argv) > 3:
                if argv[3] == 'baseline':
                    model = c.baseline
                elif argv[3] == 'chatfield14':
                    model = c.chatfield14
                elif argv[3] == 'voxceleb':
                    model = c.voxceleb
                elif argv[3] == 'warohma18':
                    model = c.warohma18
                elif argv[3] == 'snyder17':
                    model = c.snyder17
                elif argv[3] == 'dragichi2020':
                    model = c.draghici20
                else:
                    logging.error(f'Unknown model: {argv[3]}')
                    exit(1)
            else:
                model = c.baseline
            if len(argv) > 4:
                if argv[4] == 'mel_spectrogram':
                    feature = 'mel_spectrogram'
                elif argv[4] == 'spectrogram':
                    feature = 'spectrogram'
                elif argv[4] == 'mfcc':
                    feature = 'mfcc'
                else:
                    logging.error(f'Unknown feature: {argv[4]}')
                    exit(1)
            logging.info(f'Training model: {model}')
            logging.info(f'Features selected: {feature}')

