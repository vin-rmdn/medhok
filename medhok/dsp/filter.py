#!/usr/bin/env python3
"""Filter module
This module is developed to help with filtering of frequency in waves.
"""

import numpy as np
from scipy.signal import butter, lfilter, filtfilt

from configs import config as c


def butter_lowpass(cutoff=c.F_MAX, fs=c.SAMPLE_RATE, order=c.POLYNOMIAL_ORDER):
    """Creates a butter low-pass.

    Args:
        cutoff (int): Cutoff frequency. Defaults to c.F_MAX
        fs (int): Sample rate. Defaults to c.SAMPLE_RATE
        order (_type_, optional): _description_. Defaults to c.POLYNOMIAL_ORDER.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(wav, cutoff=c.F_MAX, fs=c.SAMPLE_RATE, order=c.POLYNOMIAL_ORDER):
    """Filters a wave data with a butter low-pass filter.

    Args:
        wav (_type_): Wave data wished to be processed with the low-pass filter.
        cutoff (_type_, optional): Frequency cutoff. Defaults to c.F_MAX.
        fs (_type_, optional): Sample rate. Defaults to c.SAMPLE_RATE.
        order (_type_, optional): Polynomial order. Defaults to c.POLYNOMIAL_ORDER.

    Returns:
        _type_: _description_
    """
    b, a = butter_lowpass(cutoff, fs, order)
    y = lfilter(b, a, wav)
    return y


def butter_highpass(cutoff=c.F_MIN, fs=c.SAMPLE_RATE, order=c.POLYNOMIAL_ORDER):
    """Creates a butter high-pass.

    Args:
        cutoff (_type_, optional): Frequency cutoff. Defaults to c.F_MIN.
        fs (_type_, optional): Sample rate. Defaults to c.SAMPLE_RATE.
        order (_type_, optional): Polynomial order. Defaults to c.POLYNOMIAL_ORDER.

    Returns:
        _type_: _description_
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(wav, cutoff=c.F_MIN, fs=c.SAMPLE_RATE, order=c.POLYNOMIAL_ORDER) -> np.ndarray:
    """Filters a wave data with a butter high pass filter.

    Args:
        wav (_type_): Wave data wished to be processed with the high-pass filter
        cutoff (_type_, optional): Frequency cutoff for the high-pass filter. Defaults to c.F_MIN.
        fs (_type_, optional): Sample rate. Defaults to c.SAMPLE_RATE.
        order (_type_, optional): Polynomial order. Defaults to c.POLYNOMIAL_ORDER.

    Returns:
        numpy.ndarray: The filtered wave data.
    """
    b, a = butter_highpass(cutoff, fs, order)
    y = filtfilt(b, a, wav)
    return y
