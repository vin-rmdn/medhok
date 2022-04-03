#!/usr/bin/env python3
from numba import jit
import numpy as np

from configs import config as c


@jit(nopython=True, fastmath=True)
def CMVN(
    feats: np.ndarray,
    mean_var=c.USE_BOTH_NORMALISATION
) -> np.ndarray:
    """
    Normalises features.
    :param feats: features to normalise
    :param mean_var: whether to normalise with respect to the mean and variance
    :return: normalised features
    # """
    # feats_new = feats.copy()
    for i in range(feats.shape[1]):
        feats[:, i] = feats[:, i] - np.mean(feats[:, i])
        if mean_var:
            # /home/vin/Projects/medhok/medhok/loader/audio_numba.py:23: RuntimeWarning: invalid value encountered in true_divide
            feats[:, i] /= np.std(feats[:, i])

    return feats
