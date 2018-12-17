#!usr/bin/env python  
#-*- coding:utf-8 _*- 
"""
@version: python3.6
@author: QLMX
@contact: wenruichn@gmail.com
@time: 2018/11/17 08:49 PM
"""
import cv2, os
import numpy as np
import random
import sys

sys.path.append("../utils")
from dataset import random_hsv_transform, random_gamma_transform


path = "train/"
base_path = os.path.dirname(os.path.abspath('__file__'))
image_num = 0

def rand_color():
    return (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

def rand_cos(x_start, x_end, y_start, y_end):
    return (np.random.randint(x_start, x_end),
            np.random.randint(y_start, y_end))

def get_image_name(index):
    if index < 10:
        return "00" + str(index)
    elif index < 100:
        return "0" + str(index)
    else:
        return str(index)

def write_file(data, path):
    with open(path, 'w') as f:
        for item in data:
            f.write(item + '\n')

def generate_image(n_class, nx, ny, path):
    if not os.path.exists(path):
        os.mkdir(path)

    for i in range(n_class):

        image_path = path + '/' + str(i)
        if not os.path.exists(image_path):
            os.mkdir(image_path)

        if i % 2 == 0:
            nums = random.randint(500, 600)
        else:
            nums = random.randint(50, 100)

        for num in range(nums):

            img_name = get_image_name(num)

            # 生成一个随机矩阵
            img = np.random.randint(10, 120, (nx, ny, 3), np.uint8)

            #随机生成数字坐标
            x_pos = np.random.randint(nx/2-50, nx/2-20)
            y_pos = np.random.randint(ny/2+20, ny/2+50)

            put_num = str(i)
            cv2.putText(img, put_num,
                        (x_pos,y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        random.uniform(4,8),
                        rand_color(),
                        6,
                        cv2.LINE_AA)

            cv2.imwrite(image_path + '/' +img_name + ".jpg", img)

def split_data(path, val_rate=0.1, balance=False):
    train = []
    val = []
    image = []
    label = []
    paths = os.listdir(path)
    category_scale = {}


    for i in paths:
        num = 0
        item = []
        item_img = []
        item_ann = []
        data_path = base_path + '/' + path + i + '/'
        for j in os.listdir(data_path):
            num += 1
            item.append(data_path + j + ',' + i)
            item_img.append(data_path + j)
            item_ann.append(i)
        val_len = int(len(item) * val_rate)
        category_scale[int(i)] = num - val_len
        random.shuffle(item)
        train.extend(item[:-val_len])
        val.extend(item[-val_len:])
        image.extend(item_img[:-val_len])
        label.extend(item_ann[:-val_len])

    if balance:
        cate_preprop = get_scale_dict(category_scale)
        for i, value in enumerate(image):
            if cate_preprop[int(label[i])]:
                train.extend(data_balance(value, label[i], cate_preprop[int(label[i])]))

    #shuffle data
    random.shuffle(train)
    random.shuffle(val)

    #write data
    write_file(train, os.path.join(base_path, 'train.txt'))
    write_file(val, os.path.join(base_path, 'val.txt'))

def get_scale_dict(data_dict):
    '''
    Find the proportion of the largest number
    :param path:
    :return:the proportion
    '''
    max_value = data_dict[max(data_dict, key=data_dict.get)]

    for key, value in data_dict.items():
        data_dict[key] = int(round(float(max_value)/float(value))) - 1
    return data_dict

def data_balance(data_path, label, num):
    global image_num
    img = cv2.imread(data_path)
    save_path = base_path + '/expansion/'
    train = []

    while num:
        img_hsv = random_hsv_transform(img)
        img_dir = save_path + str(image_num) + ".jpg"
        cv2.imwrite(img_dir, img_hsv)
        image_num += 1
        train.append(img_dir + ',' + label)
        num -= 1
        if num <= 0:
            break

        img_gamma = random_gamma_transform(img)
        img_dir2 = save_path + str(image_num) + ".jpg"
        cv2.imwrite(img_dir2, img_gamma)
        image_num += 1
        train.append(img_dir2 + ',' + label)
        num -= 1
        if num <= 0:
            break
    return train

if __name__=="__main__":
    generate_image(10, 360, 363, path)
    split_data(path, 0.1, balance=True)