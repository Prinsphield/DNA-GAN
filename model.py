# -*- coding:utf-8 -*-
# Created Time: Oct 13 Apr 2017 04:07:50 PM CST
# Author: Taihong Xiao <xiaotaihong@126.com>

from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
from dataset import config, Dataset
from six.moves import reduce


class Model(object):
    def __init__(self, feature_list, is_train=True, nhwc=config.nhwc, max_iter=config.max_iter, weight_decay=config.weight_decay, second_ratio=config.second_ratio):
        super(Model, self).__init__()
        self.feature_list = feature_list
        self.n_feat = len(self.feature_list)
        self.is_train = is_train
        self.batch_size, self.height, self.width, self.channel = nhwc
        self.max_iter = max_iter
        self.g_lr = tf.placeholder(tf.float32)
        self.d_lr = tf.placeholder(tf.float32)
        self.weight_decay = weight_decay
        self.second_ratio = second_ratio
        self.reuse = {}
        self.build_model()

    def leakyRelu(self, x, alpha=0.2):
        return tf.maximum(alpha * x, x)

    def make_conv(self, name, X, shape, strides):
        with tf.variable_scope(name) as scope:
            W = tf.get_variable('W',
                                shape=shape,
                                initializer=tf.random_normal_initializer(stddev=0.02),
                                )
            return tf.nn.conv2d(X, W, strides=strides, padding='SAME')


    def make_conv_bn(self, name, X, shape, strides):
        with tf.variable_scope(name) as scope:
            W = tf.get_variable('W',
                                shape=shape,
                                initializer=tf.random_normal_initializer(stddev=0.02),
                                )
            return tf.layers.batch_normalization(
                        tf.nn.conv2d(X, W, strides=strides, padding='SAME'),
                        training=self.is_train
                    )

    def make_fc(self, name, X, out_dim):
        in_dim = X.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            W = tf.get_variable('W',
                                shape=[in_dim, out_dim],
                                initializer=tf.random_normal_initializer(stddev=0.02),
                                )
            b = tf.get_variable('b',
                                shape=[out_dim],
                                initializer=tf.zeros_initializer(),
                                )
            return tf.add(tf.matmul(X, W), b)

    def make_fc_bn(self, name, X, out_dim):
        in_dim = X.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            W = tf.get_variable('W',
                                shape=[in_dim, out_dim],
                                initializer=tf.random_normal_initializer(stddev=0.02),
                                )
            b = tf.get_variable('b',
                                shape=[out_dim],
                                initializer=tf.zeros_initializer(),
                                )
            X = tf.add(tf.matmul(X, W), b)
            return tf.layers.batch_normalization(X, training=self.is_train)

    def make_deconv(self, name, X, filter_shape, out_shape, strides):
        with tf.variable_scope(name) as scope:
            W = tf.get_variable('W',
                                shape=filter_shape,
                                initializer=tf.random_normal_initializer(stddev=0.02),
                                )
            return tf.nn.conv2d_transpose(X, W, output_shape=out_shape, strides=strides, padding='SAME')

    def make_deconv_bn(self, name, X, filter_shape, out_shape, strides):
        with tf.variable_scope(name) as scope:
            W = tf.get_variable('W',
                                shape=filter_shape,
                                initializer=tf.random_normal_initializer(stddev=0.02),
                                )
            return tf.layers.batch_normalization(
                        tf.nn.conv2d_transpose(X, W,
                            output_shape=out_shape, strides=strides, padding='SAME'
                        ), training=self.is_train
                    )

    def discriminator(self, name, image, label, feat_id):
        if name in self.reuse.keys():
            reuse = self.reuse[name]
        else:
            self.reuse[name] = True
            reuse = False

        X1 = image / 127.5 - 1
        label_concat_list = [tf.tile(tf.expand_dims(tf.expand_dims(tf.expand_dims(label[:,j],-1),-1),-1), [1,self.height,self.width,1]) for j in range(self.n_feat)]
        X2 = tf.concat(label_concat_list, -1)
        X = tf.concat([X1, X2], -1)

        with tf.variable_scope(name, reuse=reuse) as scope:
            X = self.make_conv('conv1', X, shape=[4,4,3+self.n_feat,128], strides=[1,2,2,1])
            X = self.leakyRelu(X, 0.2)
            # print(name, X.get_shape())

            X = self.make_conv_bn('conv2', X, shape=[4,4,128,256], strides=[1,2,2,1])
            X = self.leakyRelu(X, 0.2)
            # print(name, X.get_shape())

            X = self.make_conv_bn('conv3', X, shape=[4,4,256,512], strides=[1,2,2,1])
            X = self.leakyRelu(X, 0.2)
            # print(name, X.get_shape())

            X = self.make_conv_bn('conv4', X, shape=[4,4,512,512], strides=[1,2,2,1])
            X = self.leakyRelu(X, 0.2)
            # print(name, X.get_shape())

            flat_dim = reduce(lambda x,y: x*y, X.get_shape().as_list()[1:])
            X = tf.reshape(X, [-1, flat_dim])
            X = self.make_fc('fct', X, self.n_feat)
            # X = tf.nn.sigmoid(X)
            return X[:,feat_id]

    def splitter(self, name, image):
        X = image / 127.5 - 1
        if name in self.reuse.keys():
            reuse = self.reuse[name]
        else:
            self.reuse[name] = True
            reuse = False

        with tf.variable_scope(name, reuse=reuse) as scope:
            X = self.make_conv('conv1', X, shape=[4,4,3,128], strides=[1,2,2,1])
            X = self.leakyRelu(X, 0.2)

            X = self.make_conv_bn('conv2', X, shape=[4,4,128,256], strides=[1,2,2,1])
            X = self.leakyRelu(X, 0.2)

            X = self.make_conv_bn('conv3', X, shape=[4,4,256,512], strides=[1,2,2,1])
            X = self.leakyRelu(X, 0.2)

            num_ch = int(X.get_shape().as_list()[-1] * self.second_ratio)
            self.num_ch = num_ch
            # encode = [X[:,:,:,:-self.n_feat*num_ch]] + [X[:,:,:,-i*num_ch:(-i-1)*num_ch] for i in range(self.n_feat,0,-1)]
            encode = [X[:,:,:,i*num_ch:(i+1)*num_ch] for i in range(self.n_feat)] + [X[:,:,:,self.n_feat*num_ch:]]
            return encode

    def joiner(self, name, encode):
        X = tf.concat(encode, axis=-1)
        # X0 = X
        if name in self.reuse.keys():
            reuse = self.reuse[name]
        else:
            self.reuse[name] = True
            reuse = False

        with tf.variable_scope(name, reuse=reuse) as scope:
            X = self.make_deconv_bn('deconv1', X, filter_shape=[4,4,512,512],
                                    out_shape=[self.batch_size, int(self.height/4), int(self.width/4), 512],
                                    strides=[1,2,2,1])
            X = tf.nn.relu(X)

            X = self.make_deconv_bn('deconv2', X, filter_shape=[4,4,256,512],
                                    out_shape=[self.batch_size, int(self.height/2), int(self.width/2), 256],
                                    strides=[1,2,2,1])
            X = tf.nn.relu(X)

            X = self.make_deconv('deconv3', X, filter_shape=[4,4,self.channel,256],
                                    out_shape=[self.batch_size, self.height, self.width, self.channel],
                                    strides=[1,2,2,1])
            b = tf.get_variable('b', shape=[1,1,1,self.channel], initializer=tf.zeros_initializer())
            X = X + b

            X = (tf.tanh(X) + 1) * 127.5
            return X

    def zeros_encode(self, enc, enc_id):
        '''
        enc: a list of latent encoding, [enc_1, ..., enc_n, z]
        enc_id: the id of the latent to be null
        '''
        enc_nil = [tf.identity(x) for x in enc]
        enc_nil[enc_id] = tf.zeros_like(enc_nil[enc_id], tf.float32)
        return enc_nil

    def swap_attribute(self, enc_Ax, enc_Be, enc_ids):
        '''
        enc_Ax: a list of latent encoding of A, [enc_A1, ..., enc_An, z_A]
        enc_Be: a list of latent encoding of B, [enc_B1, ..., enc_Bn, z_B]
        enc_ids: a list of id indicating the swapping id numbers
        '''
        enc_Ae = [tf.identity(enc) for enc in enc_Ax]
        enc_Bx = [tf.identity(enc) for enc in enc_Be]
        for enc_id in enc_ids:
            enc_Ae[enc_id] = tf.zeros_like(enc_Be[enc_id])
            enc_Bx[enc_id] = enc_Ax[enc_id]
        return enc_Ae, enc_Bx

    def build_model(self):
        self.Axs = [tf.placeholder(tf.float32, [self.batch_size,self.height,self.width,self.channel], name='data_Ax{}'.format(i)) for i in range(self.n_feat)]
        self.Bes = [tf.placeholder(tf.float32, [self.batch_size,self.height,self.width,self.channel], name='data_Be{}'.format(i)) for i in range(self.n_feat)]
        self.label_Axs = [tf.placeholder(tf.float32, [self.batch_size,self.n_feat], name='label_A{}'.format(i)) for i in range(self.n_feat)]
        self.label_Bes = [tf.placeholder(tf.float32, [self.batch_size,self.n_feat], name='label_B{}'.format(i)) for i in range(self.n_feat)]

        self.enc_Axs = [self.splitter('encoder', A) for A in self.Axs]
        self.enc_Bes = [self.splitter('encoder', B) for B in self.Bes]

        self.Axs2 = [self.joiner('decoder', enc_Ax) for enc_Ax in self.enc_Axs]
        self.Bes2 = [self.joiner('decoder', self.zeros_encode(enc_Be, i)) for i, enc_Be in enumerate(self.enc_Bes)]

        # crossover
        self.enc_Aes, self.enc_Bxs = zip(*[self.swap_attribute(self.enc_Axs[enc_id], self.enc_Bes[enc_id], [enc_id]) for enc_id in range(self.n_feat)])

        self.Aes = [self.joiner('decoder', enc_A) for enc_A in self.enc_Aes]
        self.Bxs = [self.joiner('decoder', enc_B) for enc_B in self.enc_Bxs]

        # discriminate
        self.real_Axs = [self.discriminator('D', Ax, self.label_Axs[i], i) for i, Ax in enumerate(self.Axs)]
        self.fake_Bxs = [self.discriminator('D', Bx, self.label_Axs[i], i) for i, Bx in enumerate(self.Bxs)]

        self.real_Bes = [self.discriminator('D', Be, self.label_Bes[i], i) for i, Be in enumerate(self.Bes)]
        self.fake_Aes = [self.discriminator('D', Ae, self.label_Bes[i], i) for i, Ae in enumerate(self.Aes)]

        # variable list
        self.g_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder') \
                        + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')

        self.d_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D') \
                        + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D')

        # G loss
        self.G_loss = {}
        self.G_loss['loss_G/cycle_A'] = sum([tf.reduce_mean(tf.abs(self.Axs[i] - self.Axs2[i])) / 255.0 for i in range(self.n_feat)])
        self.G_loss['loss_G/cycle_B'] = sum([tf.reduce_mean(tf.abs(self.Bes[i] - self.Bes2[i])) / 255.0 for i in range(self.n_feat)])

        self.G_loss['loss_G/Bx'] = -tf.reduce_mean(sum(self.fake_Bxs))
        self.G_loss['loss_G/Ae'] = -tf.reduce_mean(sum(self.fake_Aes))
        self.loss_G_nodecay = sum(self.G_loss.values())

        self.loss_G_decay = 0.0
        for w in self.g_var_list:
            if w.name.startswith('G') and w.name.endswith('W:0'):
                self.loss_G_decay += 0.5 * self.weight_decay * tf.reduce_mean(tf.square(w))

        self.loss_G = self.loss_G_decay + self.loss_G_nodecay

        # D loss
        self.D_loss = {}
        self.D_loss['loss_D/x'] = tf.reduce_mean(sum(self.fake_Bxs) - sum(self.real_Axs))
        self.D_loss['loss_D/e'] = tf.reduce_mean(sum(self.fake_Aes) - sum(self.real_Bes))
        self.loss_D = sum(self.D_loss.values())


        # G, D optimizer
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.g_opt = tf.train.RMSPropOptimizer(self.g_lr, decay=0.8).minimize(self.loss_G, var_list=self.g_var_list)
            self.d_opt = tf.train.RMSPropOptimizer(self.d_lr, decay=0.8).minimize(self.loss_D, var_list=self.d_var_list)

        # clip weights in D
        with tf.name_scope('clip_d'):
            self.clip_d = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in self.d_var_list]

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ''

    celebA = Dataset(['Eyeglasses', 'Smiling'])
    image_batch = celebA.input()

    DNA_GAN = Model(['Eyeglasses', 'Smiling'])
    # from IPython import embed;embed();exit()

