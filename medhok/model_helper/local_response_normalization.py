#!/usr/bin/env python3

import tensorflow as tf


class LocalResponseNormalization(tf.keras.layers.Layer):
    def __init__(self, alpha=1e-4, beta=0.75, bias=2, depth_radius=5, **kwargs):
        super(LocalResponseNormalization, self).__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.bias = 2
        self.depth_radius = depth_radius

    def call(self, x):
        return tf.nn.local_response_normalization(x, self.alpha, self.beta, self.bias, self.depth_radius)

    # def compute_output_shape(self, input_shape):
    #     return input_shape
