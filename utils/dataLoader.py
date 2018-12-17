#!/usr/bin/env python
# encoding: utf-8
"""
@version: python3.6
@author: QLMX
@contact: wenruichn@gmail.com
@time: 18-10-18 上午1:22
"""
import cv2, sys
import numpy as np
import tensorflow as tf
import random
from dataset import read_data, load_image, resize_image, onehot
from dataset import rotate, flip, random_hsv_transform, random_gamma_transform

sys.path.append("..")
from config import cfg

class DataLoader(object):
    """
    A data generator for preprocessing on CPU
    """
    def __init__(self, data_dir='train.txt', num_classes=10, mode='train', height=256, width=256):
        """
        init
        :param data_dir: str
        :param mode: str, train or test
        """
        self.curr = 0
        self.mode = mode
        self.height = height
        self.width = width
        self.num = num_classes
        self.img_paths, self.label = read_data(data_dir, mode)

        self.n = len(self.img_paths)

    def generator(self, n=0):
        i = 0
        name = None
        if n == 0:
            n = self.n

        while i < n:
            img_path = self.img_paths[i]

            if self.mode != 'test':
                label_val = int(self.label[i])
                label = np.array(onehot(label_val, self.num))
            else:
                label = self.label

            img = load_image(img_path)
            ori_img  = resize_image(img, (self.width, self.height))
            img = np.float32(ori_img) / 255.0


            if self.mode == 'train':
                #flip image

                    #rotate image in range of [-30, 30]
                    if random.random() < cfg.p_rotate:
                        img_rotate = rotate(img, self.width, self.height)
                        yield img_rotate, label


                    # # hsv image
                    # if random.random() < cfg.p_hsv:
                    #     img_hsv = random_hsv_transform(ori_img)
                    #     img_hsv = np.float32(img_hsv) / 255.0
                    #     yield img_hsv, label
                    #
                    #
                    # #gamma
                    # if random.random() < cfg.p_gamma:
                    #     img_gamma = random_gamma_transform(ori_img)
                    #     img_gamma = np.float32(img_gamma) / 255.0
                    #     yield img_gamma, label

            if self.mode == 'test':
                yield img_path, img
            else:
                yield img, label

            i += 1



