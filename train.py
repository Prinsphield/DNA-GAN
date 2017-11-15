# -*- coding:utf-8 -*-
# Created Time: Oct 13 Apr 2017 04:07:50 PM CST
# Author: Taihong Xiao <xiaotaihong@126.com>

import tensorflow as tf
import os
from model import Model
from dataset import config, Dataset
import numpy as np
from scipy import misc
import argparse


def run(config, dataset, model, gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    batchs, labels = dataset.input()

    saver = tf.train.Saver()

    # image summary
    image_summry_op = []
    image_summry_op += [tf.summary.image('Ax_{}'.format(i), model.Axs[i], max_outputs=30) for i in range(model.n_feat)]
    image_summry_op += [tf.summary.image('Be_{}'.format(i), model.Bes[i], max_outputs=30) for i in range(model.n_feat)]
    image_summry_op += [tf.summary.image('Ax2_{}'.format(i), model.Axs2[i], max_outputs=30) for i in range(model.n_feat)]
    image_summry_op += [tf.summary.image('Be2_{}'.format(i), model.Bes2[i], max_outputs=30) for i in range(model.n_feat)]
    image_summry_op += [tf.summary.image('Ae_{}'.format(i), model.Aes[i], max_outputs=30) for i in range(model.n_feat)]
    image_summry_op += [tf.summary.image('Bx_{}'.format(i), model.Bxs[i], max_outputs=30) for i in range(model.n_feat)]

    # G loss summary
    for key in model.G_loss.keys():
        tf.summary.scalar(key, model.G_loss[key])

    loss_G_nodecay_op = tf.summary.scalar('loss_G_nodecay', model.loss_G_nodecay)
    loss_G_decay_op = tf.summary.scalar('loss_G_decay', model.loss_G_decay)
    loss_G_op = tf.summary.scalar('loss_G', model.loss_G)

    # D loss summary
    for key in model.D_loss.keys():
        tf.summary.scalar(key, model.D_loss[key])

    loss_D_op = tf.summary.scalar('loss_D', model.loss_D)

    # learning rate summary
    g_lr_op = tf.summary.scalar('g_learning_rate', model.g_lr)
    d_lr_op = tf.summary.scalar('d_learning_rate', model.d_lr)

    # merged_op = tf.contrib.deprecated.merge_all_summaries()
    merged_op = tf.summary.merge_all()

    # start training
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state(config.model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)


    writer = tf.summary.FileWriter(config.log_dir, sess.graph)
    writer.add_graph(sess.graph)


    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(config.max_iter):
        d_num = 100 if i % 500 == 0 else 1

        batch_images, batch_labels = sess.run([batchs, labels])
        feed_dict = {model.g_lr: config.g_lr(epoch=i),
                     model.d_lr: config.d_lr(epoch=i),
                    }
        for j in range(model.n_feat):
            feed_dict[model.Axs[j]] = batch_images[2*j]
            feed_dict[model.Bes[j]] = batch_images[2*j+1]
            feed_dict[model.label_Axs[j]] = batch_labels[2*j]
            feed_dict[model.label_Bes[j]] = batch_labels[2*j+1]

        # from IPython import embed; embed();exit()
        # update D with clipping
        for j in range(d_num):
            _, loss_D_sum, _ = sess.run([model.d_opt, model.loss_D, model.clip_d],feed_dict=feed_dict)

        # update G
        _, loss_G_sum = sess.run([model.g_opt, model.loss_G], feed_dict=feed_dict)

        print('iter: {:06d},   g_loss: {}    d_loss: {}'.format(i, loss_D_sum, loss_G_sum))

        if i % 20 == 0:
            merged_summary = sess.run(merged_op, feed_dict=feed_dict)
            writer.add_summary(merged_summary, i)

        if i % 500 == 0:
            saver.save(sess, os.path.join(config.model_dir, 'model_{:06d}.ckpt'.format(i)))

            img_Axs, img_Bes, img_Aes, img_Bxs, img_Axs2, img_Bes2 = sess.run([model.Axs, model.Bes, model.Aes, model.Bxs, model.Axs2, model.Bes2],
                                                                            feed_dict=feed_dict)

            for k in range(model.n_feat):
                for j in range(5):
                    img = np.concatenate((img_Axs[k][j], img_Bes[k][j], img_Aes[k][j], img_Bxs[k][j], img_Axs2[k][j], img_Bes2[k][j]), axis=1)
                    misc.imsave(os.path.join(config.sample_img_dir, 'iter_{:06d}_{}_{}.jpg'.format(i,j, model.feature_list[k])), img)

    writer.close()
    saver.save(sess, os.path.join(config.model_dir, 'model.ckpt'))

    coord.request_stop()
    coord.join(threads)

def main():
    parser = argparse.ArgumentParser(description='test', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-a', '--attributes',
        nargs='+',
        type=str,
        help='Specify attribute name for training. \nAll attributes can be found in list_attr_celeba.txt'
    )
    parser.add_argument(
        '-g', '--gpu',
        default='0',
        type=str,
        help='Specify GPU id. \ndefault: %(default)s. \nUse comma to seperate several ids, for example: 0,1'
    )
    args = parser.parse_args()

    celebA = Dataset(args.attributes)
    DNA_GAN = Model(args.attributes, is_train=True)
    run(config, celebA, DNA_GAN, gpu=args.gpu)


if __name__ == "__main__":
    main()
