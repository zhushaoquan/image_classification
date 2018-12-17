#!/usr/bin/env python
# encoding: utf-8
"""
@version: python3.6
@author: QLMX
@contact: wenruichn@gmail.com
@time: 18-10-17 下午8:59
"""
import tensorflow as tf
import os, sys
import datetime, time
import multiprocessing
import numpy as np
import cv2

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
    create_pairwise_gaussian, softmax_to_unary

sys.path.append("utils")
from dataLoader import DataLoader
from dataset import resize_image, filepath_to_name
from utils import LOG, writer

sys.path.append("models")
from resnet import build_renet
from vgg19 import VGG19



class NetWork(object):

    def __init__(self, lr, height=384, width=384, num_classes=10, use_bottleneck=False, num_residual_units=5, relu_leakiness=0.1, mode='train'):

        self.starter_learning_rate = lr
        self.height = height
        self.width = width
        self.num_classes = num_classes
        self.use_bottleneck = use_bottleneck
        self.num_residual_units = num_residual_units
        self.relu_leakiness = relu_leakiness
        self.mode = mode

        if self.mode == 'test':
            self.output_types = (tf.string, tf.float32)
            self.output_shapes = (tf.TensorShape([None]),
                                  tf.TensorShape([None, self.height, self.width, 3]))
        else:
            self.output_types = (tf.float32, tf.int32)
            self.output_shapes = (tf.TensorShape([None, self.height, self.width, 3]),
                              tf.TensorShape([None, self.num_classes]))
        self._build_model()

    def _build_input(self):

        self.it = tf.data.Iterator.from_structure(self.output_types,
                                                  self.output_shapes)
        if self.mode == 'test':
            self.img_name, self.img= self.it.get_next()
        else:
            self.img, self.label = self.it.get_next()


    def _build_solver(self):
        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = self.starter_learning_rate
        self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                   2000, 0.96, staircase=True)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

            # 权重衰减，L2正则loss

    def _decay(self):
        costs = []
        # 遍历所有可训练变量
        for var in tf.trainable_variables():
            # 只计算标有“DW”的变量
            if var.op.name.find(r'DW') > 0:
                costs.append(tf.nn.l2_loss(var))
        # 加和，并乘以衰减因子
        return tf.multiply(0.0002, tf.add_n(costs))

    def _build_model(self):
        self._build_input()

        # model = VGG19(self.img, 1, self.num_classes, [])
        # self.logits = model.fc8
        self.logits, _ = build_renet(self.img, self.num_classes, self.use_bottleneck, self.num_residual_units, self.relu_leakiness)
        self.prediction_value = tf.argmax(self.logits, 1)

        if self.mode == 'train':
            correct_prediction = tf.equal(tf.argmax(self.label, 1), tf.argmax(self.logits, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.label)
            self.loss = tf.reduce_mean(loss)
            self.loss += self._decay()

            self._build_solver()
            self._build_summary()


    def _build_summary(self):
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("learning_rate", self.learning_rate)
        tf.summary.scalar("accuracy", self.accuracy)


        for v in tf.trainable_variables():
            tf.summary.histogram(v.name, v)

        tf.summary.image("image", self.img)

        self.merged = tf.summary.merge_all()

    def _build_data(self, data_dir='train_dir', num_classes=10, mode='train'):
        loader = DataLoader(data_dir=data_dir, num_classes=num_classes,
                            mode=mode, height=self.height, width=self.width)

        dataset = tf.data.Dataset.from_generator(generator=loader.generator,
                                                 output_types=(tf.float32,
                                                               tf.int32),
                                                 output_shapes=(tf.TensorShape([self.height, self.width, 3]),
                                                                tf.TensorShape([self.num_classes])))
        return dataset

    def _bulid_save_path(self):
        now = datetime.datetime.now()
        self.model_dir = 'checkpoints/checkpoint/{}_{}_{}/'.format(now.month, now.day, now.hour)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.summary_dir = 'checkpoints/summary/{}_{}_{}'.format(now.month, now.day, now.hour)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

    def train(self, max_epochs=20, model_dir=None, train_dir='train_set', data_scale='category_scale',
              val_dir='val_set', hreshold=0.5, batch_size=4, write_summary=False, freq_summary=200):

        #load train data
        dataset = self._build_data(train_dir, self.num_classes, 'train')


        dataset = dataset.shuffle(100)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(20)
        train_init = self.it.make_initializer(dataset)


        #load val data
        valset = self._build_data(val_dir, self.num_classes, 'val')
        valset = valset.batch(20)
        valset = valset.prefetch(10)
        val_init = self.it.make_initializer(valset)

        print("training starts.")
        self._bulid_save_path()

        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            train_writer = tf.summary.FileWriter(self.summary_dir, sess.graph)
            # continue training
            if model_dir:
                print("continue training from " + model_dir)
                saver.restore(sess, model_dir)
            else:
                sess.run(tf.global_variables_initializer())
            # train
            for epoch in range(max_epochs):
                cnt = 0
                sess.run(train_init)
                st = time.time()
                print("epoch {} begins:".format(epoch))
                try:
                    while True:
                        if write_summary:
                            _, loss, acc, summary, step = sess.run([self.train_op,
                                                               self.loss,
                                                               self.accuracy,
                                                               self.merged,
                                                               self.global_step])
                            lo = sess.run(self.logits)
                            # print(lo)
                            # print('\n')


                            # evaluator.evaluate(pred_heatmap, label, img_path)
                            if step % freq_summary == 0:
                                # summary
                                train_writer.add_summary(summary, step)
                        else:
                            _, loss, acc, step = sess.run([self.train_op, self.loss, self.accuracy, self.global_step])
                        cnt += batch_size
                        if cnt % (batch_size*2) == 0:
                            string_print = "Epoch = %d Nums = %d Loss = %.4f Train_Acc = %.4f  Time = %.2f" % ( epoch, cnt, loss, acc, time.time() - st)
                            LOG(string_print)
                            st = time.time()

                except tf.errors.OutOfRangeError:
                    print('saving checkpoint......')
                    saver.save(sess, os.path.join(self.model_dir, str('checkpoint_' + str(epoch + 1))))
                    print('checkpoint saved.')
                    self.val_out(sess=sess, val_init=val_init)

    def val_out(self, sess, val_init):
        print('\n')
        print("validation starts.")

        sess.run(val_init)
        times = 0

        try:
            while True:
                acc = sess.run(self.accuracy)

                print("the %d times Validation Validation_Acc = %.4f" % (times, acc))

                times += 1

        except tf.errors.OutOfRangeError:
            print("...validation completed")
            print('\n')


    def test(self, data_dir='test', model_dir=None, output_dir='result', batch_size=10):
        print("testing starts.")

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        # load test data
        loader = DataLoader(data_dir=data_dir, num_classes=self.num_classes,
                            mode='test', height=self.height, width=self.width)

        testset = tf.data.Dataset.from_generator(generator=loader.generator,
                                                 output_types=(tf.string,
                                                               tf.float32),
                                                 output_shapes=(tf.TensorShape([]),
                                                                tf.TensorShape([self.height, self.width, 3])))
        testset = testset.shuffle(100)
        testset = testset.batch(batch_size)
        testset = testset.prefetch(20)
        test_init = self.it.make_initializer(testset)

        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            saver.restore(sess, model_dir)
            sess.run(test_init)
            queue = multiprocessing.Queue(maxsize=30)
            writer_process = multiprocessing.Process(target=writer, args=[output_dir, batch_size, queue, 'stop'])
            writer_process.start()
            print('writing predictions...')
            try:
                while True:
                    img_name, pre_label = sess.run([self.img_name, self.prediction_value])
                    queue.put(('continue', img_name, pre_label))
            except tf.errors.OutOfRangeError:
                queue.put(('stop', None, None))

        print('testing finished.')



