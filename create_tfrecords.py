# -*- coding:utf-8 -*-
# Created Time: 2017/10/07 10:31:10
# Author: Taihong Xiao <xiaotaihong@126.com>

import tensorflow as tf
from tqdm import tqdm
import os, math
from scipy import misc

from functools import partial
from multiprocessing import Pool


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def create_tf_example(line, attribute_name, img_dir):
    info = line.split()
    img_name = os.path.join(img_dir, info[0])
    img = misc.imread(img_name)
    # from IPython import embed; embed();exit()
    feature={
        'image/id_name': bytes_feature(info[0]),
        'image/height' : int64_feature(img.shape[0]),
        'image/width'  : int64_feature(img.shape[1]),
        'image/encoded': bytes_feature(tf.compat.as_bytes(img.tostring())),
    }
    for j, val in enumerate(info[1:]):
        feature[attribute_name[j]] = int64_feature(int(val))

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example

def work(list_id):
    data_dir = './datasets/celebA/'
    img_dir = os.path.join(data_dir, 'align_5p')
    attri_file = os.path.join(data_dir, 'list_attr_celeba.txt')
    tfrecords_dir = os.path.join(data_dir, 'align_5p_tfrecords')

    with open(attri_file, 'r') as f:
        lines = f.read().strip().split('\n')
        attribute_name = lines[1].split()
        lines = lines[2:]

        writer = tf.python_io.TFRecordWriter(os.path.join(tfrecords_dir, '{:02d}.tfrecords'.format(list_id)))
        if (list_id + 1) * 20000 <= len(lines):
            id_list = range(list_id * 20000, (list_id + 1) * 20000)
        else:
            id_list = range(list_id * 20000, len(lines))

        for i in id_list:
            example = create_tf_example(lines[i], attribute_name, img_dir)
            writer.write(example.SerializeToString())
        writer.close()


def main():
    data_dir = './datasets/celebA/'
    img_dir = os.path.join(data_dir, 'align_5p')
    attri_file = os.path.join(data_dir, 'list_attr_celeba.txt')
    tfrecords_dir = os.path.join(data_dir, 'align_5p_tfrecords')
    if not os.path.exists(tfrecords_dir):
        os.makedirs(tfrecords_dir)

    with open(attri_file, 'r') as f:
        lines = f.read().strip().split('\n')
        attribute_name = lines[1].split()
        print(len(lines))
        # from IPython import embed; embed(); exit()

    pool = Pool(11)
    # partial_work = partial(work, lines)
    pool.map(work, list(range(int(math.ceil((len(lines)-2) / 20000.)))))
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
