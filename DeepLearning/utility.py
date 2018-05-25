#!/usr/bin/python
'''
Author: SARBAJIT MUKHERJEE
Email: sarbajit.mukherjee@aggiemail.usu.edu

This is the utility file that contains all the function used in raw_wave_cnn_n.py
'''

import tensorflow as tf
import numpy as np
import warnings
import os
from tensorflow.python.client import device_lib
import tflearn

_FLOATX = 'float32'
_MANUAL_VAR_INIT = False


def floatx():
    return _FLOATX


def tfmean(x, axis=None, keepdims=False):
    """Mean of a tensor, alongside the specified axis.
    # Arguments
        x: A tensor or variable.
        axis: A list of integer. Axes to compute the mean.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1 for each entry in `axis`. If `keep_dims` is `True`,
            the reduced dimensions are retained with length 1.
    # Returns
        A tensor with the mean of elements of `x`.
    """
    if x.dtype.base_dtype == tf.bool:
        x = tf.cast(x, floatx())
    return tf.reduce_mean(x, axis=axis, keep_dims=keepdims)


def dtype(x):

    return x.dtype.base_dtype.name