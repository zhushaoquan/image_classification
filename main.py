#!/usr/bin/env python
# encoding: utf-8
"""
@version: python3.6
@author: QLMX
@contact: wenruichn@gmail.com
@time: 18-10-17 下午6:34
"""
from __future__ import print_function

import cv2
import tensorflow as tf
import numpy as np
import time
import os, sys
import random

from config import cfg
from network import NetWork


def main(mode):
    # Specify operating environment
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    fashionAI = NetWork(lr=cfg.lr,
                        height=cfg.height,
                        width=cfg.width,
                        num_classes=cfg.num_classes,
                        mode=mode)

    if mode == 'train':
        fashionAI.train(max_epochs=cfg.num_epochs,
                        batch_size=cfg.batch_size,
                        write_summary=cfg.write_summary,
                        freq_summary=cfg.freq_summary,
                        train_dir=os.path.join(cfg.data_dir, 'train.txt'),
                        data_scale=os.path.join(cfg.data_dir, 'category_scale.txt'),
                        val_dir=os.path.join(cfg.data_dir, 'val.txt'),
                        model_dir=cfg.model_dir)
    elif mode == 'test':
        fashionAI.test(data_dir=cfg.test_dir,
                       model_dir='checkpoints/checkpoint/11_19_21/checkpoint_10',
                       output_dir=cfg.result_dir,
                       batch_size=cfg.test_batch_size)

if __name__ == "__main__":
    main('train')