# -*- coding:utf-8 -*-
# Created Time: 2017/11/04 21:19:32
# Author: Taihong Xiao <xiaotaihong@126.com>

import tensorflow as tf
import numpy as np
import os
from scipy import misc

from model import Model
from dataset import Dataset
import argparse

def get_representation(img, model, model_dir, out_path):
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(model_dir)
        # print(ckpt)
        # print(ckpt.model_checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        Ax = tf.placeholder(tf.float32, [model.batch_size,model.height,model.width,model.channel],name='Ax')
        enc_Ax = model.splitter('encoder', Ax)
        grad_att_1 =[tf.gradients(enc_Ax[0][:,:,:,i], Ax)[0] for i in range(128)]
        grad_att_2 =[tf.gradients(enc_Ax[1][:,:,:,i], Ax)[0] for i in range(128)]
        grad_att_3 =[tf.gradients(enc_Ax[2][:,:,:,i], Ax)[0] for i in range(256)]
        # from IPython import embed;embed();exit()
        grad_1 = sess.run(grad_att_1, feed_dict={Ax: img})
        grad_2 = sess.run(grad_att_2, feed_dict={Ax: img})
        grad_3 = sess.run(grad_att_3, feed_dict={Ax: img})
        for i in range(128):
            misc.imsave(os.path.join(out_path, '0_{:03d}.jpg'.format(i)), grad_1[i][0])
            misc.imsave(os.path.join(out_path, '1_{:03d}.jpg'.format(i)), grad_2[i][0])
            np.save(os.path.join(out_path, '0_{:03d}.npy'.format(i)), grad_2[i][0])
            np.save(os.path.join(out_path, '1_{:03d}.npy'.format(i)), grad_2[i][0])

        for i in range(256):
            misc.imsave(os.path.join(out_path, '2_{:03d}.jpg'.format(i)), grad_3[i][0])
            np.save(os.path.join(out_path, '2_{:03d}.npy'.format(i)), grad_3[i][0])

def main():
    parser = argparse.ArgumentParser(description='test', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', '--input', type=str, help='image path')
    parser.add_argument('-a', '--attributes', type=str, nargs='+', help='attribute list')
    parser.add_argument('--model_dir', type=str, default='train_log/model/', help='path to the model')
    parser.add_argument('--latent_path', type=str, default='latent', help='path to the model')
    parser.add_argument('-g', '--gpu', type=str, default='', help='gpu ids')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not os.path.exists(args.latent_path): os.makedirs(args.latent_path)

    DNA_GAN = Model(feature_list=args.attributes, is_train=False, nhwc=[1,64,64,3])
    img = np.expand_dims(misc.imresize(misc.imread(args.input), (DNA_GAN.height, DNA_GAN.width)), axis=0)
    get_representation(img, DNA_GAN, args.model_dir, args.latent_path)


if __name__ == "__main__":
    main()
