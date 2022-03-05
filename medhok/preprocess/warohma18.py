#!/usr/bin/env python3
import numpy as np

class Warohma18:
    def __init__(self):
        pass

    @staticmethod
    def pre_emphasis(wave):
        """From Warohma et al.,(2018):
        "Pre-emphasis aims to allow the signal spectrum to be evenly distributed across all frequencies. It removes noise and maximizes its energy. In this process, the filter suppresses the lower frequency while leaving the higher frequency. Pre-emphasis filter is calculated as follows:

        H(z) = 1 - ∂z⁻¹, 0.95 ≥ α ≥ 1 (1)

        Here, we use α of 0.97. The pre-emphasis output with the filter model as in (1) can be written as:

        g(n) - x(n) - α(n-1)

        where x(n) is the symmetrical window function."
        """
