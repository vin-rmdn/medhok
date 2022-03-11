#!/usr/bin/env python3
from sys import argv
import logging
import datetime

from matplotlib import pyplot as plt

from loader.audio import Audio
from loader.generator import MedhokGenerator
from loader.greedy_loader import GreedyLoader
from loader.feature_metadata import FeatureMetadata
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
        elif argv[1] == 'serialize':
            pass
        elif argv[1] == 'train':
            # Processing args 4: feature parameter
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
            else:
                feature = 'mel_spectrogram'
            logging.info(f'Features selected: {feature}')

            # Processing args 3: training parameters. Needs to be done after specifying features.
            if len(argv) > 3:
                if argv[3] == 'baseline':
                    model = c.baseline(c.INPUT_SHAPE(feature))
                elif argv[3] == 'chatfield14':
                    model = c.chatfield14(c.INPUT_SHAPE(feature))
                elif argv[3] == 'voxceleb':
                    model = c.voxceleb(c.INPUT_SHAPE(feature))
                elif argv[3] == 'warohma18':
                    model = c.warohma18(c.INPUT_SHAPE(feature))
                elif argv[3] == 'snyder17':
                    model = c.snyder17(c.INPUT_SHAPE(feature))
                elif argv[3] == 'dragichi2020':
                    model = c.draghici20(c.INPUT_SHAPE(feature))
                else:
                    logging.error(f'Unknown model: {argv[3]}')
                    exit(1)
                model_name = argv[3]
            else:
                model = c.baseline(c.INPUT_SHAPE(feature))
                model_name = 'baseline'
            logging.info(f'Training model: {model_name}...')

            # Getting features metadata
            # X_train, X_test, y_train, y_test = FeatureMetadata.load_feature_metadata(feature, split=True)
            X_train, X_test, y_train, y_test = FeatureMetadata.load_feature_metadata(feature, split=True, sample=True)
            y_train = GreedyLoader.transform_dialects(y_train)
            y_test = GreedyLoader.transform_dialects(y_test)

            # Loading Keras generator
            logging.info('Loading Keras generator...')
            train_generator = MedhokGenerator(
                X_train,
                # y_train,
                **c.GENERATOR_PARAMS
            )
            valid_generator = MedhokGenerator(
                X_test,
                # y_test,
                **c.GENERATOR_PARAMS
            )

            model.summary()

            model.compile(
                optimizer=c.OPTIMIZER,
                loss=c.LOSS,
                metrics=c.METRICS
            )
            history = model.fit(
                train_generator,
                validation_data=valid_generator,
                epochs=c.EPOCHS,
                # steps_per_epoch=c.EPOCH_STEPS,
                verbose=1,
                callbacks=[c.CHECKPOINT_CALLBACK(model_name, feature)]
            )

            # Creating model report
            fig, ax = plt.subplots(2, 1, figsize=(16, 12))
            fig.suptitle('Plots of model performance: ' + feature + ', ' + model_name)

            ax[0].plot(history.history['loss'], 'r')
            ax[0].plot(history.history['val_loss'], 'g')
            ax[1].plot(history.history['categorical_accuracy'], 'r')
            ax[1].plot(history.history['val_categorical_accuracy'], 'g')
            ax[0].grid()
            ax[1].grid()

            plt.savefig(
                '../visualization/performance/' + str(datetime.now()) + '-' + feature + '-' + model_name + '.png')

            # Saving model
            model.save('../model/model-' + feature + '-' + model_name + '.h5')
            model.save_weights('../model/weights/model-' + feature + '-' + model_name + '/')
