#!/usr/bin/env python
# encoding: utf-8
"""
@version: python3.6
@author: QLMX
@contact: wenruichn@gmail.com
@time: 18-10-17 下午6:32
"""
import os

class Config:
    base_dir = os.path.dirname(os.path.abspath('__file__'))
    data_dir = base_dir + '/data'
    test_dir = base_dir + '/test'
    result_dir = base_dir + '/result'

    #model parameter
    mode = 'train'
    num_classes = 10
    class_balancing = False
    threshold = 0.5
    model_dir = None
    write_summary = True
    freq_summary = 20

    # the data preprocess parameter
    height = 224
    width = 224
    p_flip = 0.5
    p_rotate = 1.0
    p_hsv = 1.0
    p_gamma = 1.0

    #train parameter
    num_epochs = 100
    batch_size = 48
    test_batch_size = 20
    num_val_images = 10
    lr = 0.001
    num_keep = 1000
    checkpoint_step = 10
    validation_step = 1
    continue_training = False

    #test parameter
    test_bath_size = 8


cfg = Config()