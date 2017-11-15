# -*- coding:utf-8 -*-
# Created Time: Tue 02 May 2017 09:42:27 PM CST
# $Author: Taihong Xiao <xiaotaihong@126.com>

import tensorflow as tf
import numpy as np
from model import Model
from dataset import Dataset
import os
# import cv2
from scipy import misc
import argparse



def swap_attribute(src_img, att_img, swap_list, model_dir, model, gpu):
    '''
    Input
        src_img: the source image that you want to change its attribute
        att_img: the attribute image that has certain attribute
        swap_list: the swap id list
        model_dir: the directory that contains the checkpoint, ckpt.* files
        model: the DNA_GAN network that defined in train.py
        gpu: for example, '0,1'. Use '' for cpu mode
    Output
        out1: src_img with attributes
        out2: att_img without attributes
    '''
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(model_dir)
        # print(ckpt)
        # print(ckpt.model_checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        Ax = tf.placeholder(tf.float32, [model.batch_size,model.height,model.width,model.channel],name='Ax')
        Be = tf.placeholder(tf.float32, [model.batch_size,model.height,model.width,model.channel],name='Be')

        enc_Ax = model.splitter('encoder', Ax)
        enc_Be = model.splitter('encoder', Be)
        enc_Ae, enc_Bx = model.swap_attribute(enc_Ax, enc_Be, swap_list)
        Ae = model.joiner('decoder', enc_Ae)
        Bx = model.joiner('decoder', enc_Bx)
        out2, out1 = sess.run([Ae, Bx], feed_dict={Ax: att_img, Be:src_img})
        swap = np.concatenate((src_img[0], att_img[0], out1[0], out2[0]), 1)
        misc.imsave('swap.jpg', swap)
        # misc.imsave('out1.jpg', out1[0])
        # misc.imsave('out2.jpg', out2[0])


def interpolation(src_img, att_img, swap_id, inter_num, model_dir, model, gpu):
    '''
    Input
        src_img: the source image that you want to change its attribute
        att_img: the attribute image that has certain attribute
        swap_id: the attribute id
        inter_num: number of interpolation points
        model_dir: the directory that contains the checkpoint, ckpt.* files
        model: the DNA_GAN network that defined in train.py
        gpu: for example, '0,1'. Use '' for cpu mode
    Output
        out: [src_img, inter1, inter2, ..., inter_{inter_num}]
    '''
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(model_dir)
        # print(ckpt)
        # print(ckpt.model_checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        Ax = tf.placeholder(tf.float32, [model.batch_size,model.height,model.width,model.channel],name='Ax')
        Be = tf.placeholder(tf.float32, [model.batch_size,model.height,model.width,model.channel],name='Be')

        enc_Ax = model.splitter('encoder', Ax)
        enc_Be = model.splitter('encoder', Be)
        out = src_img[0]
        for i in range(1, inter_num + 1):
            lambda_i = i / float(inter_num)
            enc_Bx_i = [tf.identity(enc) for enc in enc_Be]
            enc_Bx_i[swap_id] = enc_Ax[swap_id] * lambda_i
            Bx_i = model.joiner('decoder', enc_Bx_i)
            out_i = sess.run(Bx_i, feed_dict={Ax: att_img, Be: src_img})
            out = np.concatenate((out, out_i[0]), axis=1)
        return out

def interpolation2(src_img, att_imgs, swap_list, size, model_dir, model, gpu):
    '''
    Input
        src_img: the source image that you want to change its attribute
        att_imgs: the attribute images that has certain attribute
        swap_list: the attributes list
        size: size of output matrix
        model_dir: the directory that contains the checkpoint, ckpt.* files
        model: the DNA_GAN network that defined in train.py
        gpu: for example, '0,1'. Use '' for cpu mode
    Output
        out: [src_img, inter1, inter2, ..., inter_{inter_num}]
    '''
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(model_dir)
        # print(ckpt)
        # print(ckpt.model_checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        m, n = size
        h, w = model.height, model.width

        Ax1 = tf.placeholder(tf.float32, [model.batch_size,model.height,model.width,model.channel],name='Ax1')
        Ax2 = tf.placeholder(tf.float32, [model.batch_size,model.height,model.width,model.channel],name='Ax2')
        Be = tf.placeholder(tf.float32, [model.batch_size,model.height,model.width,model.channel],name='Be')

        enc_Ax1 = model.splitter('encoder', Ax1)
        enc_Ax2 = model.splitter('encoder', Ax2)
        enc_Be = model.splitter('encoder', Be)

        out = np.zeros((h * m, w * n, model.channel))
        canvas = np.ones((h * m, w * (n+2), model.channel)) * 255
        for i in range(m):
            for j in range(n):
                lambda_i = i / float(m-1)
                lambda_j = j / float(n-1)
                enc_Bx_i = [tf.identity(enc) for enc in enc_Be]
                enc_Bx_i[0] = enc_Ax1[0] * lambda_i + enc_Be[0] * (1 - lambda_i)
                enc_Bx_i[1] = enc_Ax2[1] * lambda_j + enc_Be[1] * (1 - lambda_j)

                Bx_i_j = model.joiner('decoder', enc_Bx_i)
                out_i_j = sess.run(Bx_i_j, feed_dict={Ax1: att_imgs[:1], Ax2: att_imgs[1:], Be: src_img})
                out[i*h:(i+1)*h, j*w:(j+1)*w, :] = out_i_j[0]
        canvas[:h,:w,:] = src_img
        canvas[:h*m, w:w*(n+1),:] = out
        canvas[:h,w*(n+1):w*(n+2),:] = att_imgs[1]
        canvas[-h:,:w,:] = att_imgs[0]
        return out, canvas

def interpolation1_(src_img, att_img, inter_num, model_dir, model, gpu):
    '''
    Input
        src_img: the source image that you want to change its attribute
        att_img: the attribute image that has certain attribute
        inter_num: number of interpolation points
        model_dir: the directory that contains the checkpoint, ckpt.* files
        model: the DNA_GAN network that defined in train.py
        gpu: for example, '0,1'. Use '' for cpu mode
    Output
        out: [src_img, inter1, inter2, ..., inter_{inter_num}]
    '''
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(model_dir)
        # print(ckpt)
        # print(ckpt.model_checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        B, src_feat = sess.run([model.B, model.e], feed_dict={model.Be: src_img})
        att_feat = sess.run(model.x, feed_dict={model.Ax: att_img})

        out = src_img[0]
        for i in range(1, inter_num + 1):
            lambda_i = i / float(inter_num)
            out_i = sess.run(model.joiner('G_joiner', B, src_feat + (att_feat - src_feat) * lambda_i) )
            out = np.concatenate((out, out_i[0]), axis=1)
        # print(out.shape)
        misc.imsave('interpolation2.jpg', out)

def interpolation_matrix(src_img, att_imgs, swap_id, size, model_dir, model, gpu):
    '''
    Input
        src_img: the source image that you want to change its attribute [1, h, w, c]
        att_imgs: four attribute images that has certain attribute [4, h, w, c]
        swap_id: the attribute id
        size: the size of output matrix
        model_dir: the directory that contains the checkpoint, ckpt.* files
        model: the DNA_GAN network that defined in train.py
        gpu: for example, '0,1'. Use '' for cpu mode
    Output
        out: image matrix
    '''
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(model_dir)
        # print(ckpt)
        # print(ckpt.model_checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        m, n = size
        h, w = model.height, model.width

        rows = [[1 - i/float(m-1), i/float(m-1)] for i in range(m)]
        cols = [[1 - i/float(n-1), i/float(n-1)] for i in range(n)]
        four_tuple = []
        for row in rows:
            for col in cols:
                four_tuple.append([row[0]*col[0], row[0]*col[1], row[1]*col[0], row[1]*col[1]])

        Axs = [tf.placeholder(tf.float32, [model.batch_size,model.height,model.width,model.channel]) for i in range(4)]
        Be = tf.placeholder(tf.float32, [model.batch_size,model.height,model.width,model.channel],name='Be')
        feed_dict = {Be: src_img}
        for i in range(4):
            feed_dict[Axs[i]] = att_imgs[i:i+1]

        enc_Axs = [model.splitter('encoder', Ax) for Ax in Axs]
        enc_Be = model.splitter('encoder', Be)

        out = np.zeros((h * m, w * n, model.channel))
        canvas = np.ones((h * m, w * (n+2), model.channel)) * 255
        cnt = 0
        for i in range(m):
            for j in range(n):
                four = four_tuple[cnt]
                cnt += 1

                enc_Bx_i_j = [tf.identity(enc) for enc in enc_Be]
                enc_Bx_i_j[swap_id] = sum([four[k] * enc_Axs[k][swap_id] for k in range(4)])

                Bx_i_j = model.joiner('decoder', enc_Bx_i_j)
                out_i_j = sess.run(Bx_i_j, feed_dict=feed_dict)
                out[i*h:(i+1)*h, j*w:(j+1)*w, :] = out_i_j[0]
                # misc.imsave('out_{:02d}.jpg'.format(cnt), out_i_j[0])

        canvas[:h,:w,:] = att_imgs[0]
        canvas[:h,-w:,:] = att_imgs[1]
        canvas[-h:,:w,:] = att_imgs[2]
        canvas[-h:,-w:,:] = att_imgs[3]
        canvas[:,w:-w,:] = out
        return out, canvas

def main():
    parser = argparse.ArgumentParser(description='test', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-m', '--mode',
        default='swap',
        type=str,
        choices=['swap', 'interpolation', 'interpolation2', 'matrix'],
        help='Specify mode.'
    )
    parser.add_argument(
        '-a', '--attributes',
        nargs='+',
        type=str,
        help='attributes list'
    )
    parser.add_argument(
        '--swap_list',
        nargs='+',
        type=int,
        help='0/1 list'
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        help='Specify source image name.'
    )
    parser.add_argument(
        '-t', '--target',
        metavar='target image with attributes',
        type=str,
        help='Specify target image name.'
    )
    parser.add_argument(
        '--targets',
        nargs='+',
        type=str,
        help='Specify target image name.'
    )
    parser.add_argument(
        '--model_dir',
        default='train_log/model/',
        type=str,
        help='Specify model_dir. \ndefault: %(default)s.'
    )
    parser.add_argument(
        '-n', '--num',
        default='2',
        type=int,
        help='Specify number of interpolations.'
    )
    parser.add_argument(
        '-s', '--size',
        nargs=2,
        default=[3,3],
        type=int,
        help='Specify number of interpolations.'
    )
    parser.add_argument(
        '-g', '--gpu',
        default='',
        type=str,
        help='Specify GPU id. \ndefault: %(default)s. \nUse comma to seperate several ids, for example: 0,1'
    )
    args = parser.parse_args()

    DNA_GAN = Model(feature_list=args.attributes, is_train=False, nhwc=[1,64,64,3])
    if args.mode == 'swap':
        src_img = np.expand_dims(misc.imresize(misc.imread(args.input), (DNA_GAN.height, DNA_GAN.width)), axis=0)
        att_img = np.expand_dims(misc.imresize(misc.imread(args.target), (DNA_GAN.height, DNA_GAN.width)), axis=0)
        swap_attribute(src_img, att_img, args.swap_list, args.model_dir, DNA_GAN, args.gpu)

    elif args.mode == 'interpolation':
        src_img = np.expand_dims(misc.imresize(misc.imread(args.input), (DNA_GAN.height, DNA_GAN.width)), axis=0)
        att_img = np.expand_dims(misc.imresize(misc.imread(args.target), (DNA_GAN.height, DNA_GAN.width)), axis=0)
        out = interpolation(src_img, att_img, args.swap_list[0], args.num, args.model_dir, DNA_GAN, args.gpu)
        misc.imsave('interpolation.jpg', out)

    elif args.mode == 'interpolation2':
        src_img = np.expand_dims(misc.imresize(misc.imread(args.input), (DNA_GAN.height, DNA_GAN.width)), axis=0)
        att_imgs = np.array([misc.imresize(misc.imread(img), (DNA_GAN.height, DNA_GAN.width)) for img in args.targets])
        out, canvas = interpolation2(src_img, att_imgs, args.swap_list, args.size, args.model_dir, DNA_GAN, args.gpu)
        misc.imsave('interpolation2.jpg', canvas)

    elif args.mode == 'matrix':
        src_img = np.expand_dims(misc.imresize(misc.imread(args.input), (DNA_GAN.height, DNA_GAN.width)), axis=0)
        att_imgs = np.array([misc.imresize(misc.imread(img), (DNA_GAN.height, DNA_GAN.width)) for img in args.targets])
        out, canvas = interpolation_matrix(src_img, att_imgs, args.swap_list[0], args.size, args.model_dir, DNA_GAN, args.gpu)
        misc.imsave('four_matrix.jpg', canvas)

    else:
        raise NotImplementationError()

if __name__ == "__main__":
    main()
