#!/usr/bin/env python3
import sys
import logging

from loader.audio import Audio
from app.train import train_model, finetune
from app.visualization import visualize_train_hist
from app.evaluation import evaluate_models, evaluate_model
from configs import config as c

# STARTUP PARAMETERS
SERIALIZE_NORMALIZE = False

if __name__ == '__main__':
    # Debug
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    if len(sys.argv) > 1:
        if sys.argv[1] == 'extract':
            logging.info('Loading dataset to memory...')
            audio = Audio()
            logging.info('Loading metadata...')
            audio.load_metadata()
            if len(sys.argv) > 2:
                if sys.argv[2] == 'new':
                    if len(sys.argv) > 3:
                        logging.info('Extracting %s feature from wav files...', sys.argv[3])
                        audio.get_features(sys.argv[3], preloaded=False)
                        logging.info('Done.')
                    else:
                        logging.error('Please specify a feature to extract.')
                        sys.exit(1)
                else:
                    logging.info('Loading %s features...', c.DEFAULT_FEATURE)
                    audio.get_features(c.DEFAULT_FEATURE, preloaded=True)
                    logging.info('Done.')
                # logging.info('Extracting features from wav files...')
                # audio.get_features('mel_spectrogram', preloaded=False)
                # audio.get_features('spectrogram', preloaded=False)
                # audio.get_features('mfcc', preloaded=False)
            # audio.feature_split()
            # logging.info('Getting one-hot-encoded dialect labels...')
            # audio.encode_variable('dialect')
        elif sys.argv[1] == 'serialize':
            pass
        elif sys.argv[1] == 'train':
            if len(sys.argv) > 3:
                history = train_model(sys.argv[2], sys.argv[3])
                visualize_train_hist(history, arch=sys.argv[2], feature=sys.argv[3])
            elif len(sys.argv) > 2:
                history = train_model(arch=sys.argv[2])
                visualize_train_hist(history, arch=sys.argv[2])
            else:
                logging.info('No parameters given. Using default options')
                history = train_model()
                visualize_train_hist(history)
        elif sys.argv[1] == 'finetune':
            if len(sys.argv) > 3:
                history = finetune(sys.argv[2], sys.argv[3])
                visualize_train_hist(history, arch=sys.argv[2], feature=sys.argv[3], finetune=True)
            elif len(sys.argv) > 2:
                history = finetune(arch=sys.argv[2])
                visualize_train_hist(history, arch=sys.argv[2], finetune=True)
            else:
                logging.info('No parameters given. Using default options')
                history = finetune()
                visualize_train_hist(history, finetune=True)
        elif sys.argv[1] == 'evaluate':
            if len(sys.argv) > 2:
                logging.info('Evaluating model %s', sys.argv[2])
                evaluate_model(sys.argv[2])
            else:
                logging.info('Evaluating model.')
                evaluate_models()
