"""
Created on Mon July 11 18:50:39 2020

@author: Haifeng Luo
"""

import os
import sys
import numpy as np
import tensorflow as tf
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'backbone_network'))
import PDE_net


def restore_trained_model(NUM_POINT, MODEL_DIR, BATCH_SIZE=1):

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        batch = tf.Variable(0, trainable=False)

        with tf.variable_scope(tf.get_variable_scope()):
            with tf.device('/gpu:0'):
                pointclouds = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, 3))
                is_training = tf.placeholder(tf.bool, shape=())

                #####DirectionEmbedding
                PDE = PDE_net.get_model_RRFSegNet('PDE_net',
                                        pointclouds,
                                        is_training=is_training,
                                        k=20)

                # PDE = PDE_net.get_model_DGCNN('PDE_net',
                #                         pointclouds,
                #                         is_training=is_training,
                #                         k=20)

                PDE = tf.nn.l2_normalize(PDE, axis=2, epsilon=1e-20)

        saver = tf.train.Saver(tf.global_variables())
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        saver.restore(sess, tf.train.latest_checkpoint(MODEL_DIR))

        PDE_net_ops = {'pointclouds': pointclouds,
                       'is_training': is_training,
                       'PDE': PDE,
                       'step': batch}
        return sess, PDE_net_ops


def prediction(sess, testdata, ops):

    testdata = np.expand_dims(testdata, axis=0)
    feed_dict = {ops['pointclouds']: testdata,
                 ops['is_training']: False}

    pde_ = sess.run(ops['PDE'], feed_dict=feed_dict)
    testdata = np.squeeze(testdata)
    pde_ = np.squeeze(pde_)
    ####################
    xyz_direction = np.concatenate([testdata, pde_], -1)
    return xyz_direction



