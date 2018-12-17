#!usr/bin/env python  
#-*- coding:utf-8 _*- 
"""
@version: python3.6
@author: QLMX
@contact: wenruichn@gmail.com
@time: 2018/11/17 10:42 PM
"""

import numpy as np
import tensorflow as tf
import six
import tools

from tensorflow.python.training import moving_averages


 # Convert the step value to the required step array of tf.nn.conv
def _stride_arr(stride):
    return [1, stride, stride, 1]

def build_renet(inputs, num_classes, use_bottleneck=False, num_residual_units=5, relu_leakiness=0.0):
    with tf.variable_scope('init'):
        x = inputs
        x = tools.conv('init_conv', x, 3, 3, 16, _stride_arr(1))

    strides = [1, 2, 2]
    activate_before_residual = [True, False, False]
    if use_bottleneck:
        res_func = tools.bottleneck_residual
        filters = [16, 64, 128, 256]
    else:
        res_func = tools.residual
        filters = [16, 16, 32, 64]

    # first group
    with tf.variable_scope('unit_1_0'):
        x = res_func(x, filters[0], filters[1], _stride_arr(strides[0]), activate_before_residual[0])

    for i in six.moves.range(1, num_residual_units):
        with tf.variable_scope('unit_1_%d' % i):
            x = res_func(x, filters[1], filters[1], _stride_arr(1), False)

    # second group
    with tf.variable_scope('unit_2_0'):
        x = res_func(x, filters[1], filters[2], _stride_arr(strides[1]), activate_before_residual[1])
    for i in six.moves.range(1, num_residual_units):
        with tf.variable_scope('unit_2_%d' % i):
            x = res_func(x, filters[2], filters[2], _stride_arr(1), False)

    # third group
    with tf.variable_scope('unit_3_0'):
        x = res_func(x, filters[2], filters[3], _stride_arr(strides[2]),
                     activate_before_residual[2])
    for i in six.moves.range(1, num_residual_units):
        with tf.variable_scope('unit_3_%d' % i):
            x = res_func(x, filters[3], filters[3], _stride_arr(1), False)

    # all pool layer
    with tf.variable_scope('unit_last'):
        x = tools.batch_norm('final_bn', x)
        x = tools.relu(x, relu_leakiness)
        x = tools.global_avg_pool(x)

    # fc_layer + softmax
    with tf.variable_scope('logit'):
        logits = tools.fully_connected(x, num_classes)
        predictions = tf.nn.softmax(logits)

    return logits, predictions