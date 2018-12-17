#!/usr/bin/env python
# encoding: utf-8
"""
@version: python3.6
@author: QLMX
@contact: wenruichn@gmail.com
@time: 18-10-18 上午12:23
"""
import tensorflow as tf
import numpy as np

from tensorflow.python.training import moving_averages


def conv(name, x, filter_size, in_filters, out_filters, strides):
    with tf.variable_scope(name):
        n = filter_size * filter_size * out_filters
        # 获取或新建卷积核，正态随机初始化
        kernel = tf.get_variable('DW',
                                 [filter_size, filter_size, in_filters, out_filters],
                                 tf.float32,
                                 initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
        # 计算卷积
        return tf.nn.conv2d(x, kernel, strides, padding='SAME')

def relu(x, leakiness=0.0):
    '''
    if leakiness is not zero, It is leakiness relu
    :param x:
    :param leakiness:
    :return:
    '''
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

def global_avg_pool(x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])

def fully_connected(x, out_dim):
    shape = x.get_shape()
    if len(shape) == 4:
        size = shape[1].value * shape[2].value * shape[3].value
    else:
        size = shape[-1].value
    w = tf.get_variable('DW',
                        shape=[size, out_dim],
                        initializer=tf.initializers.variance_scaling(distribution="uniform"))
    b = tf.get_variable('biases', shape=[out_dim], initializer=tf.constant_initializer())
    return tf.nn.xw_plus_b(x, w, b)


  # ((x-mean)/var)*gamma+beta

def batch_norm(name, x, mode='train'):
    extra_train_ops = []

    with tf.variable_scope(name):
        params_shape = [x.get_shape()[-1]]
        # offset
        beta = tf.get_variable('beta',
                               params_shape,
                               tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
        # cale
        gamma = tf.get_variable('gamma',
                                params_shape,
                                tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))

        if mode == 'train':
            mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')
            # 新建或建立测试阶段使用的batch均值、标准差
            moving_mean = tf.get_variable('moving_mean',
                                          params_shape, tf.float32,
                                          initializer=tf.constant_initializer(0.0, tf.float32),
                                          trainable=False)
            moving_variance = tf.get_variable('moving_variance',
                                              params_shape, tf.float32,
                                              initializer=tf.constant_initializer(1.0, tf.float32),
                                              trainable=False)

            extra_train_ops.append(moving_averages.assign_moving_average(moving_mean, mean, 0.9))
            extra_train_ops.append(moving_averages.assign_moving_average(moving_variance, variance, 0.9))
        else:
            # 获取训练中积累的batch均值、标准差
            mean = tf.get_variable('moving_mean',
                                   params_shape, tf.float32,
                                   initializer=tf.constant_initializer(0.0, tf.float32),
                                   trainable=False)
            variance = tf.get_variable('moving_variance',
                                       params_shape, tf.float32,
                                       initializer=tf.constant_initializer(1.0, tf.float32),
                                       trainable=False)

      # BN层：((x-mean)/var)*gamma+beta
        y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
        y.set_shape(x.get_shape())
        return y

def bottleneck_residual(x, in_filter, out_filter, stride,
                        relu_leakiness=0.0, activate_before_residual=False):
    if activate_before_residual:
        with tf.variable_scope('common_bn_relu'):
            # 先做BN和ReLU激活
            x = batch_norm('init_bn', x)
            x = relu(x, relu_leakiness)
            # 获取残差直连
            orig_x = x
    else:
        with tf.variable_scope('residual_bn_relu'):
            # 获取残差直连
            orig_x = x
            # 后做BN和ReLU激活
            x = batch_norm('init_bn', x)
            x = relu(x, relu_leakiness)

    # 第1子层
    with tf.variable_scope('sub1'):
        # 1x1卷积，使用输入步长，通道数(in_filter -> out_filter/4)
        x = conv('conv1', x, 1, in_filter, out_filter / 4, stride)

    # 第2子层
    with tf.variable_scope('sub2'):
        # BN和ReLU激活
        x = batch_norm('bn2', x)
        x = relu(x, relu_leakiness)
        # 3x3卷积，步长为1，通道数不变(out_filter/4)
        x = conv('conv2', x, 3, out_filter / 4, out_filter / 4, [1, 1, 1, 1])

    # 第3子层
    with tf.variable_scope('sub3'):
        # BN和ReLU激活
        x = batch_norm('bn3', x)
        x = relu(x, relu_leakiness)
        # 1x1卷积，步长为1，通道数不变(out_filter/4 -> out_filter)
        x = conv('conv3', x, 1, out_filter / 4, out_filter, [1, 1, 1, 1])

    # 合并残差层
    with tf.variable_scope('sub_add'):
        # 当通道数有变化时
        if in_filter != out_filter:
            # 1x1卷积，使用输入步长，通道数(in_filter -> out_filter)
            orig_x = conv('project', orig_x, 1, in_filter, out_filter, stride)

        # 合并残差
        x += orig_x

    return x

def residual(x, in_filter, out_filter, stride, relu_leakiness=0.0, activate_before_residual=False):
    if activate_before_residual:
        with tf.variable_scope('shared_activation'):
            # 先做BN和ReLU激活
            x = batch_norm('init_bn', x)
            x = relu(x, relu_leakiness)
            # 获取残差直连
            orig_x = x
    else:
        with tf.variable_scope('residual_only_activation'):
            # 获取残差直连
            orig_x = x
            # 后做BN和ReLU激活
            x  = batch_norm('init_bn', x)
            x = relu(x, relu_leakiness)

    with tf.variable_scope('sub1'):
        x = conv('conv1', x, 3, in_filter, out_filter, stride)

    with tf.variable_scope('sub2'):
        x = batch_norm('bn2', x)
        x = relu(x, relu_leakiness)
        x = conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
        if in_filter != out_filter:
            orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
            # 通道补零()
            orig_x = tf.pad(orig_x,
                            [[0, 0],
                             [0, 0],
                             [0, 0],
                             [(out_filter - in_filter) // 2, (out_filter - in_filter) // 2]])
        x += orig_x

    return x


