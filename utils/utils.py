#!/usr/bin/env python
# encoding: utf-8
"""
@version: python3.6
@author: QLMX
@contact: wenruichn@gmail.com
@time: 18-10-18 上午12:34
"""
import numpy as np
import sys, datetime
import os, cv2
import tensorflow as tf

from scipy.misc import imread
from sklearn.metrics import precision_score, recall_score, f1_score

# Print with time. To console or file
def LOG(X, f=None):
    time_stamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    if not f:
        print(time_stamp + " " + X)
    else:
        f.write(time_stamp + " " + X)


# Takes an absolute file path and returns the name of the file without th extension
def filepath_to_name(full_name):
    file_name = os.path.basename(full_name)
    file_name = os.path.splitext(file_name)[0]
    return file_name

def writer(output_dir,batch_size, queue, stop_token='stop'):

    while True:
        token, img_name_dir, pre_label = queue.get()
        if token == stop_token:
            return
        for i in range(batch_size):

            name_dir = img_name_dir[i].decode('ascii')
            base_name = os.path.basename(name_dir)
            label = pre_label[i]

            save_dir = output_dir + '/' + str(label)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            img = cv2.imread(name_dir)

            cv2.imwrite(save_dir + "/%s" % (base_name), img)
