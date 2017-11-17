# -*- coding:utf-8 -*-
# Created Time: Oct 13 Apr 2017 04:07:50 PM CST
# Author: Taihong Xiao <xiaotaihong@126.com>

import numpy as np
import tensorflow as tf
import glob, os, time
from scipy import misc
from functools import partial


class Config:
    @property
    def base_dir(self):
        return os.path.abspath(os.curdir)

    @property
    def data_dir(self):
        data_dir = os.path.join('./datasets/celebA/')
        if not os.path.exists(data_dir):
            raise ValueError('Please specify a data dir.')
        return data_dir

    @property
    def exp_dir(self):
        exp_dir = os.path.join('train_log')
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        return exp_dir

    @property
    def model_dir(self):
        model_dir = os.path.join(self.exp_dir, 'model')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        return model_dir

    @property
    def log_dir(self):
        log_dir = os.path.join(self.exp_dir, 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    @property
    def sample_img_dir(self):
        sample_img_dir = os.path.join(self.exp_dir, 'sample_img')
        if not os.path.exists(sample_img_dir):
            os.makedirs(sample_img_dir)
        return sample_img_dir

    def g_lr(self, init_lr=0.00005, decay_rate=1, decay_step=10000, epoch=0):
        return init_lr * decay_rate ** (epoch / np.float(decay_step))

    def d_lr(self, init_lr=0.00005, decay_rate=1, decay_step=10000, epoch=0):
        return init_lr * decay_rate ** (epoch / np.float(decay_step))

    nhwc = [64,64,64,3]

    num_threads = 10

    shuffle = True

    buffer_size = 640

    max_iter = 200000

    weight_decay = 5e-5

    second_ratio = 0.25


config = Config()


class Dataset(object):

    def __init__(self, feature_list, data_dir=config.data_dir, nhwc=config.nhwc, num_threads=config.num_threads, shuffle=config.shuffle, buffer_size=config.buffer_size):
        super(Dataset, self).__init__()
        self.data_dir = data_dir
        self.feature_list = feature_list
        self.n_feat = len(self.feature_list)

        self.batch_size, self.height, self.width, self.channel = nhwc
        self.num_threads = num_threads
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        self.tfrecords_dir = os.path.join(self.data_dir, 'align_5p_tfrecords')
        self.filenames = [os.path.join(self.tfrecords_dir, name) for name in sorted(os.listdir(self.tfrecords_dir)) if name.endswith('.tfrecords')]

        with open(os.path.join(self.data_dir, 'list_attr_celeba.txt'), 'r') as f:
            lines = f.read().strip().split('\n')
            self.attribute_names = lines[1].split()

    def parse_fn(self, serialized_example):
        features={
            'image/id_name': tf.FixedLenFeature([], tf.string),
            'image/height' : tf.FixedLenFeature([], tf.int64),
            'image/width'  : tf.FixedLenFeature([], tf.int64),
            'image/encoded': tf.FixedLenFeature([], tf.string),
        }
        for name in self.feature_list:
            features[name] = tf.FixedLenFeature([], tf.int64)

        example = tf.parse_single_example(serialized_example, features=features)
        image = tf.decode_raw(example['image/encoded'], tf.uint8)
        raw_height = tf.cast(example['image/height'], tf.int32)
        raw_width = tf.cast(example['image/width'], tf.int32)
        image = tf.reshape(image, [raw_height, raw_width, 3])
        image = tf.image.resize_images(image, size=[self.height, self.width])
        # from IPython import embed; embed(); exit()

        feature_val_list = [tf.cast(example[name], tf.float32) for name in self.feature_list]
        return image, feature_val_list

    def filter_fn(self, feat_id, pos, image, feature_val):
        if pos:
            return tf.equal(feature_val[feat_id],  tf.ones_like(feature_val[feat_id]))
        else:
            return tf.equal(feature_val[feat_id], -tf.ones_like(feature_val[feat_id]))

    def get_filter_fn(self, feat_id, pos):
        return partial(self.filter_fn, feat_id, pos)

    def input(self):
        datasets = [tf.contrib.data.TFRecordDataset(self.filenames) for i in range(2*self.n_feat)]
        datasets = [dataset.map(self.parse_fn, num_threads=self.num_threads) for dataset in datasets]
        if self.shuffle:
            datasets = [dataset.shuffle(self.buffer_size) for dataset in datasets]

        datasets = [datasets[2*i+j].filter(self.get_filter_fn(i, pos)).repeat().batch(self.batch_size) for i in range(self.n_feat) for j, pos in enumerate([True, False])]

        iterators = [dataset.make_one_shot_iterator() for dataset in datasets]
        batchs = [iterator.get_next()[0] for iterator in iterators]
        labels = [iterator.get_next()[1] for iterator in iterators]
        return batchs, labels

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    config = Config()
    celebA = Dataset(['Bangs', 'Smiling'])

    batchs, labels = celebA.input()
    # batch = celebA.input1()

    X1 = tf.placeholder(tf.float32, config.nhwc)
    X2 = tf.placeholder(tf.float32, config.nhwc)
    Y = tf.reduce_mean(X1) + tf.reduce_mean(X2)
    Z1 = tf.reduce_mean(X1)
    Z2 = tf.reduce_mean(X2)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    t1 = time.time()
    for i in range(100):
        # print(i, sess.run(Y, feed_dict={X1: sess.run(batch1), X2: sess.run(batch2)}))
        print(i)
        batch_images, batch_labels = sess.run([batchs, labels])
        print(batch_images[0].shape, batch_images[1].shape, batch_labels[0].shape, batch_labels[1].shape)

    t2 = time.time()
    print(t2-t1)


    coord.request_stop()
    coord.join(threads)



