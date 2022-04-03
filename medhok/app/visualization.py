#!/usr/bin/env python3

from datetime import datetime

from matplotlib import pyplot as plt

from configs import config as c


def visualize_train_hist(history: dict, arch: str = 'baseline', feature: str = 'mel_spectrogram', finetune=False) -> None:
    # Creating model report
    fig, ax = plt.subplots(2, 1, figsize=(12, 9))
    fig.suptitle(f'Plots of model performance: {feature}, {arch} - lr {c.LEARNING_RATE}, batch {c.BATCH_SIZE}, {c.EPOCH_STEPS} steps/epoch')

    ax[0].plot(history.history['loss'], 'r', label='loss')
    ax[0].plot(history.history['val_loss'], 'g', label='val_loss')
    ax[1].plot(history.history['acc'], 'r', label='acc')
    ax[1].plot(history.history['val_acc'], 'g', label='val_acc')
    ax[0].grid()
    ax[1].grid()
    ax[0].legend()
    ax[1].legend()

    filename = (
        c.VISUALIZATION_DIR / 'performance' / f'{arch}-{feature}-lr{c.LEARNING_RATE}-b{c.BATCH_SIZE}-{c.EPOCH_STEPS}_per_epoch-{datetime.now()}{"-finetune" if finetune else ""}.svg'
    )

    plt.savefig(filename)
