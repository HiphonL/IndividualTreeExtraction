"""
Loss functions for training the PDE-net

Created on Mon July 11 18:50:39 2020

@author: Haifeng Luo
"""
import tensorflow as tf


def slack_based_direction_loss(pre_direction, gt_direction, sigma=0.955):
    '''
    Error Slack-based Direction Loss
    '''
    gt_direction = tf.nn.l2_normalize(gt_direction, axis=2, epsilon=1e-20)
    pre_direction = tf.nn.l2_normalize(pre_direction, axis=2, epsilon=1e-20)

    loss = tf.subtract(sigma, tf.reduce_sum(tf.multiply(pre_direction, gt_direction), axis=2))
    tmp = tf.zeros_like(loss)
    condition = tf.greater(loss, 0.0)
    loss = tf.where(condition, loss, tmp)
    loss = tf.reduce_mean(loss)
    return loss


def direction_loss(pre_direction, gt_direction):
    '''
    Plain Direction Loss
    '''
    gt_direction = tf.nn.l2_normalize(gt_direction, axis=2, epsilon=1e-20)
    pre_direction = tf.nn.l2_normalize(pre_direction, axis=2, epsilon=1e-20)
    loss = -tf.reduce_mean(tf.reduce_sum(tf.multiply(pre_direction, gt_direction), axis=2))

    return loss