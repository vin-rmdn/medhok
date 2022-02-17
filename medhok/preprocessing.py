#!/usr/bin/env python3
"""Preprocessing module — provides the developer with the functions required for preprocessing the audio dataset.
"""

from . import constants
import numpy as np
from numba import jit

# TODO: do augmentation (low)


def split_window(feature, window_size=constants.WINDOW_SIZE):
    windows = []
    l_p = 0
    r_p = window_size
    while l_p < feature.shape[1]:
        temp = feature[:, l_p:r_p]
        if temp.shape[1] != window_size:
            while temp.shape[1] < window_size:
                temp = np.append(temp, [[0]] * feature.shape[0], axis=1)
        windows.append(temp)
        l_p += window_size
        r_p += window_size
    # print('end')
    return np.array(windows, dtype=np.float32)


# Normalise features with respect to the mean and standard deviation.
# Provide a mutually-exclusive option to choose between both mean and
# both mean and variance.
@jit(nopython=True, fastmath=True)
def normalise_feature(
    feats: np.ndarray,
    mean_var=constants.USE_BOTH_NORMALISATION
) -> np.ndarray:
    """
    Normalises features.
    :param feats: features to normalise
    :param mean_var: whether to normalise with respect to the mean and variance
    :return: normalised features
    # """
    feats_new = feats.copy()
    for i in range(feats.shape[1]):
        feats_new[:, i] = feats_new[:, i] - np.mean(feats_new[:, i])
        if mean_var:
            feats_new[:, i] /= np.std(feats_new[:, i])

    return feats_new
