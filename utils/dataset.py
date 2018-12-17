#!/usr/bin/env python
# encoding: utf-8
"""
@version: python3.6
@author: QLMX
@contact: wenruichn@gmail.com
@time: 18-10-17 下午7:11
"""
import cv2, os
import numpy as np
import sys
import csv
import random

sys.path.append('../')
from config import cfg


def rotate(img, width, height):
    angle = np.random.uniform(-30, 30)
    center = (width / 2, height / 2)
    rot_mat = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1.0)
    new_img = cv2.warpAffine(img, rot_mat, (width, height), flags=cv2.INTER_NEAREST)

    return new_img


def flip(img):
    new_img = cv2.flip(img, 1)
    return new_img


def _hsv_transform(img, hue_delta, sat_mult, val_mult):
    """
    define hsv transformation function
    :param img: original image
    :param hue_delta: Tonal scale
    :param sat_mult: Saturation ratio
    :param val_mult: Proportion of brightness change
    :return:
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
    img_hsv[:, :, 1] *= sat_mult
    img_hsv[:, :, 2] *= val_mult
    img_hsv[img_hsv > 255] = 255
    return cv2.cvtColor(np.round(img_hsv).astype(np.uint8), cv2.COLOR_HSV2RGB)


def random_hsv_transform(img, hue_vari=10, sat_vari=0.1, val_vari=0.1):
    """
    random transform hsv
    :param img:
    :param hue_vari:
    :param sat_vari:
    :param val_vari:
    :return:
    """
    hue_delta = np.random.randint(-hue_vari, hue_vari)
    sat_mult = 1 + np.random.uniform(-sat_vari, sat_vari)
    val_mult = 1 + np.random.uniform(-val_vari, val_vari)
    return _hsv_transform(img, hue_delta, sat_mult, val_mult)


def _gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table).astype(np.uint8))
    return cv2.LUT(img, gamma_table)


def random_gamma_transform(img, gamma_vari=2.0):
    """
    random gamma transform
    gamma in range of [1/gamma_vari, gamma_vari]
    :param img:
    :param gamma_vari:
    :return:
    """
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return _gamma_transform(img, gamma)

# Takes an absolute file path and returns the name of the file without th extension
def filepath_to_name(full_name):
    file_name = os.path.basename(full_name)
    file_name = os.path.splitext(file_name)[0]
    return file_name

def read_data(path, mode='train'):
    img = []
    ann = []
    if mode == 'test':
        for item in os.listdir(path):
            # item_name = os.path.splitext(item)[0]
            img.append(path + '/' + item)
        return img, ann

    else:
        with open(path, 'r') as f:
            for line in f.readlines():
                item = line.strip().split(',')
                if len(item) == 1:
                    img.append(item[0])
                else:
                    img.append(item[0])
                    ann.append(item[1])
    return img, ann



def load_image(path):
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return image

def resize_image(image, size):
    input_image = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
    return input_image

def onehot(data, num):
    res = np.array([0] * num)
    res[data] = 1
    return res

def reverse_one_hot(data):
    np_data = np.array(data)
    return int(np.where(np_data==1)[0])