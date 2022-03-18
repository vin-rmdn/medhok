#!/usr/bin/env python3
import sys
import logging
from datetime import datetime
import pickle

from matplotlib import pyplot as plt
import tensorflow as tf

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
            # Processing args 3: feature parameter
            if len(sys.argv) > 3:
                if sys.argv[3] == 'mel_spectrogram':
                    feature = 'mel_spectrogram'
                elif sys.argv[3] == 'spectrogram':
                    feature = 'spectrogram'
                elif sys.argv[3] == 'mfcc':
                    feature = 'mfcc'
                else:
                    logging.error('Unknown feature: %s', sys.argv[3])
                    exit(1)
            else:
                feature = 'mel_spectrogram'
            logging.info('Features selected: %s', feature)

            # Getting features metadata
            X_train, X_test, y_train, y_test = FeatureMetadata.load_feature_metadata(feature, split=True)
            logging.info('Dataset loaded with %d on training and %d on validation.', X_train.shape[0], X_test.shape[0])
            # X_train, X_test, y_train, y_test = FeatureMetadata.load_feature_metadata(feature, split=True, sample=True)
            y_train = GreedyLoader.transform_dialects(y_train)
            y_test = GreedyLoader.transform_dialects(y_test)

            # Processing args 2: training parameters. Needs to be done after specifying features.
            if len(sys.argv) > 2:
                if sys.argv[2] == 'baseline':
                    model = c.baseline(c.INPUT_SHAPE(feature))
                elif sys.argv[2] == 'chatfield14':
                    model = c.chatfield14(c.INPUT_SHAPE(feature))
                elif sys.argv[2] == 'voxceleb':
                    model = c.voxceleb(c.INPUT_SHAPE(feature))
                elif sys.argv[2] == 'warohma18':
                    model = c.warohma18(c.INPUT_SHAPE(feature))
                elif sys.argv[2] == 'snyder17':
                    model = c.snyder17(c.INPUT_SHAPE(feature))
                elif sys.argv[2] == 'draghici20_crnn':
                    model = c.draghici20(c.INPUT_SHAPE(feature), type='crnn')
                elif sys.argv[2] == 'draghici20_cnn':
                    model = c.draghici20(c.INPUT_SHAPE(feature), type='cnn')
                elif sys.argv[2] == 'shon18':
                    model = c.shon18(c.INPUT_SHAPE(feature))
                    logging.info('Loading Keras generator for 1d convolution...')
                    train_generator = MedhokGenerator(
                        X_train,
                        16,
                        c.INPUT_SHAPE(feature, generator=True),
                        squeeze=True
                    ).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
                    valid_generator = MedhokGenerator(
                        X_test,
                        16,
                        c.INPUT_SHAPE(feature, generator=True),
                        squeeze=True
                    ).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
                else:
                    logging.error('Unknown model: %s', sys.argv[2])
                    exit(1)

                # Setting up Keras generator
                if sys.argv[2] != 'shon18':
                    logging.info('Loading Keras generator...')
                    train_generator = MedhokGenerator(
                        X_train,
                        dim=c.INPUT_SHAPE(feature, generator=True),
                        batch_size=c.BATCH_SIZE
                    )
                    valid_generator = MedhokGenerator(
                        X_test,
                        dim=c.INPUT_SHAPE(feature, generator=True),
                        batch_size=c.BATCH_SIZE
                    )

                model_name = sys.argv[2]
            else:
                model = c.baseline(c.INPUT_SHAPE(feature))
                model_name = 'baseline'

            model.summary()
            model.compile(
                optimizer=c.OPTIMIZER,
                loss=c.LOSS,
                metrics=c.METRICS
            )

            logging.info('Training model: %s...', model_name)
            history = model.fit(
                train_generator,
                validation_data=valid_generator,
                epochs=c.EPOCHS,
                steps_per_epoch=c.EPOCH_STEPS,
                verbose=1,
                callbacks=[c.CHECKPOINT_CALLBACK(model_name, feature)]
            )

            # Creating model report
            fig, ax = plt.subplots(2, 1, figsize=(12, 9))
            fig.suptitle(f'Plots of model performance: {feature}, {model_name} - lr {c.LEARNING_RATE}, batch {c.BATCH_SIZE}, {c.EPOCH_STEPS} steps/epoch')

            ax[0].plot(history.history['loss'], 'r')
            ax[0].plot(history.history['val_loss'], 'g')
            ax[1].plot(history.history['acc'], 'r')
            ax[1].plot(history.history['val_acc'], 'g')
            ax[0].grid()
            ax[1].grid()

            plt.savefig(
                c.VISUALIZATION_DIR / 'performance' / f'{model_name}-{feature}-lr{c.LEARNING_RATE}-b{c.BATCH_SIZE}-{c.EPOCH_STEPS}_per_epoch-{datetime.now()}.svg'
            )

            # Saving model
            model.save(c.MODEL_DIR / f'{model_name}-{feature}.h5')
            model.save_weights(c.MODEL_DIR / 'weights' / f'{model_name}-{feature}')

            # Save history to log/metrics/
            with open(c.LOG_DIR / 'metrics' / f'{model_name}-{feature}-lr{c.LEARNING_RATE}-b{c.BATCH_SIZE}-{c.EPOCH_STEPS}_per_epoch-{datetime.now()}.txt', 'wb') as f:
                pickle.dump(history.history, f)
