"""
In theory, the backbone network can be any semantic segmentation
framework that directly takes discrete points as input.

Created on Mon July 11 18:50:39 2020

@author: Haifeng Luo
"""
import tensorflow as tf
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import tf_util


def relation_reasoning_layers(name, inputs, is_training, bn_decay, nodes_list, weight_decay, is_dist):
    '''
    relation feature reasoning layers
    '''
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        net = tf_util.conv2d(inputs, nodes_list[0], [1, 1],
                              padding='VALID', stride=[1, 1],
                              bn=True, is_training=is_training,
                              scope='mpl_g1', bn_decay=bn_decay,
                              weight_decay=weight_decay, is_dist=is_dist)
        net = tf_util.conv2d(net, nodes_list[1], [1, 1],
                              padding='VALID', stride=[1, 1],
                              bn=True, is_training=is_training,
                              scope='mpl_g2', bn_decay=bn_decay,
                              weight_decay=weight_decay, is_dist=is_dist)
        # net = tf.reduce_sum(net, axis=-2, keep_dims=True)
        net = tf.reduce_max(net, axis=-2, keep_dims=True)

        net = tf_util.conv2d(net, nodes_list[2], [1, 1],
                              padding='VALID', stride=[1, 1],
                              bn=True, is_training=is_training,
                              scope='mpl_f1', bn_decay=bn_decay,
                              weight_decay=weight_decay, is_dist=is_dist)
        return net

def get_model_RRFSegNet(name, points, is_training,  k=20, is_dist=True, weight_decay=0.0004,
                     bn_decay=None, reuse=tf.AUTO_REUSE):
    ''' RRFSegNet-based Backbone Network (PDE-net) '''

    with tf.variable_scope(name, reuse=reuse):
        num_point = points.get_shape()[1].value
        Position = points[:, :, :3]
        adj = tf_util.pairwise_distance(Position)
        nn_idx = tf_util.knn(adj, k=k)
        ### layer_1
        relation_features1 = tf_util.get_relation_features(points,nn_idx=nn_idx, k=k)
        net_1 = relation_reasoning_layers('layer_1', relation_features1,
                                             is_training=is_training, bn_decay=bn_decay,
                                             nodes_list=[64, 64, 64],
                                             weight_decay=weight_decay,
                                             is_dist=is_dist)
        ### layer_2
        relation_features1 = tf_util.get_relation_features(net_1, nn_idx=nn_idx, k=k)
        net_2 = relation_reasoning_layers('layer_2', relation_features1,
                                             is_training=is_training, bn_decay=bn_decay,
                                             nodes_list=[128, 128, 128],
                                             weight_decay=weight_decay,
                                             is_dist=is_dist)

        ###generate global features
        global_net = tf_util.conv2d(tf.concat([net_1, net_2], axis=-1), 1024, [1, 1],
                                             padding='VALID', stride=[1, 1], weight_decay=weight_decay,
                                             bn=True, is_training=is_training,
                                             scope='mpl_global', bn_decay=bn_decay, is_dist=is_dist)

        global_net = tf.reduce_max(global_net, axis=1, keep_dims=True)
        global_net = tf.tile(global_net, [1, num_point, 1, 1])

        ###
        concat = tf.concat(axis=3, values=[global_net, net_1, net_2])

        # CONV
        net = tf_util.conv2d(concat, 256, [1, 1], padding='VALID', stride=[1, 1],
                                             bn=True, is_training=is_training, scope='dir/conv1',
                                             weight_decay=weight_decay, is_dist=is_dist, bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp1')
        net = tf_util.conv2d(net, 64, [1, 1], padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training, scope='dir/conv2', is_dist=is_dist)
        net = tf_util.conv2d(net, 3, [1, 1], padding='VALID', stride=[1, 1],
                             bn=True, activation_fn=None, is_training=is_training,
                             scope='dir/conv3', is_dist=is_dist)
        net = tf.squeeze(net, axis=2)

        return net


def get_model_DGCNN(name, point_cloud, is_training, is_dist=False,
              weight_decay=0.0001, bn_decay=None, k=20, reuse=tf.AUTO_REUSE):
    '''DGCNN-based backbone network (PDE-net)'''

    with tf.variable_scope(name, reuse=reuse):

        num_point = point_cloud.get_shape()[1].value
        input_image = tf.expand_dims(point_cloud, -1)
        input_point_cloud = tf.expand_dims(point_cloud, -2)
        adj = tf_util.pairwise_distance(point_cloud[:, :, :3])
        nn_idx = tf_util.knn(adj, k=k)
        ###
        edge_feature1 = tf_util.get_edge_feature(input_image, nn_idx=nn_idx, k=k)
        net = tf_util.conv2d(edge_feature1, 64, [1, 1],
                           padding='VALID', stride=[1, 1],
                           bn=True, is_training=is_training, weight_decay=weight_decay,
                           scope='adj_conv1', bn_decay=bn_decay, is_dist=is_dist)
        net_1 = tf.reduce_max(net, axis=-2, keep_dims=True)

        edge_feature2 = tf_util.get_edge_feature(net_1, nn_idx=nn_idx, k=k)
        net = tf_util.conv2d(edge_feature2, 64, [1,1],
                           padding='VALID', stride=[1,1],
                           bn=True, is_training=is_training, weight_decay=weight_decay,
                           scope='adj_conv3', bn_decay=bn_decay, is_dist=is_dist)
        net_2 = tf.reduce_max(net, axis=-2, keep_dims=True)

        edge_feature3 = tf_util.get_edge_feature(net_2, nn_idx=nn_idx, k=k)
        net = tf_util.conv2d(edge_feature3, 64, [1,1],
                           padding='VALID', stride=[1,1],
                           bn=True, is_training=is_training, weight_decay=weight_decay,
                           scope='adj_conv5', bn_decay=bn_decay, is_dist=is_dist)
        net_3 = tf.reduce_max(net, axis=-2, keep_dims=True)

        net = tf_util.conv2d(tf.concat([net_1, net_2, net_3], axis=-1), 1024, [1, 1],
                           padding='VALID', stride=[1,1],
                           bn=True, is_training=is_training,
                           scope='adj_conv7', bn_decay=bn_decay, is_dist=is_dist)
        out_max = tf_util.max_pool2d(net, [num_point, 1], padding='VALID', scope='maxpool')
        expand = tf.tile(out_max, [1, num_point, 1, 1])

        ##############
        net = tf.concat(axis=3, values=[expand, net_1, net_2, net_3, input_point_cloud])
        ############
        net = tf_util.conv2d(net, 512, [1, 1], padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training, scope='dir/conv1', is_dist=is_dist)
        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp1')
        net = tf_util.conv2d(net, 64, [1, 1], padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training, scope='dir/conv2', is_dist=is_dist)
        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp2')
        net = tf_util.conv2d(net, 3, [1, 1], padding='VALID', stride=[1, 1],
                             bn=True, activation_fn=None, is_training=is_training,
                             scope='dir/conv3', is_dist=is_dist)
        net = tf.squeeze(net, axis=2)
        return net
