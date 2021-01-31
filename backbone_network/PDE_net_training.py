"""
Created on Mon July 11 18:50:39 2020

@author: Haifeng Luo
"""
import argparse
import os
import sys
import tensorflow as tf
from tqdm import tqdm
import BatchSampleGenerator as BSG
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import py_util
import PDE_net
import Loss


parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default='pre_trained_PDE_net', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 4096]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 100]')
parser.add_argument('--batch_size', type=int, default=12, help='Batch Size during training for each GPU [default: 12]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--decay_step', type=int, default=50000, help='Decay step for lr decay [default: 50000]')
parser.add_argument('--decay_rate', type=float, default=0.95, help='Decay rate for lr decay [default: 0.95]')
parser.add_argument('--training_data_path',
                    default='./data/training_data/',
                    help='Make sure the source training-data files path')
parser.add_argument('--validating_data_path',
                    default='./data/validating_data/',
                    help='Make sure the source validating-data files path')

FLAGS = parser.parse_args()
TRAIN_DATA_PATH = FLAGS.training_data_path
VALIDATION_PATH = FLAGS.validating_data_path


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,  # Decay step.
        DECAY_RATE,  # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=False)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        batch = tf.Variable(0, trainable=False)
        bn_decay = get_bn_decay(batch)
        learning_rate = get_learning_rate(batch)

        with tf.variable_scope(tf.get_variable_scope()):
            with tf.device('/gpu:0'):
                pointclouds = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, 3))
                direction_labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, 3))
                is_training = tf.placeholder(tf.bool, shape=())

                #####DirectionEmbedding
                DeepPointwiseDirections = PDE_net.get_model_RRFSegNet('PDE_net',
                                                            pointclouds,
                                                            is_training=is_training,
                                                            weight_decay=0.0001,
                                                            bn_decay=bn_decay,
                                                            k=20)

                #####DirectionEmbedding
                # DeepPointwiseDirections = PDE_net.get_model_DGCNN('PDE_net',
                #                                             pointclouds,
                #                                             is_training=is_training,
                #                                             weight_decay=0.0001,
                #                                             bn_decay=bn_decay,
                #                                             k=20)
                ######
                loss_esd = Loss.slack_based_direction_loss(DeepPointwiseDirections, direction_labels)
                loss_pd = Loss.direction_loss(DeepPointwiseDirections, direction_labels)
                loss = 1 * loss_esd + 0 * loss_pd + tf.add_n(tf.get_collection('losses'))

                ###optimizer--Adam
                train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=batch)

        saver = tf.train.Saver(tf.global_variables(),  max_to_keep=3)
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        #####
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)

        ops = {'learning_rate': learning_rate,
               'pointclouds': pointclouds,
               'direction_labels': direction_labels,
               'is_training': is_training,
               'loss': loss,
               'loss_esd': loss_esd,
               'loss_pd': loss_pd,
               'train_op': train_op,
               'step': batch}

        init_loss = 999.999
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            ####training data generator
            train_set = py_util.get_data_set(TRAIN_DATA_PATH)
            generator_training = BSG.minibatch_generator(TRAIN_DATA_PATH,BATCH_SIZE, train_set, NUM_POINT)

            ####validating data generator
            val_set = py_util.get_data_set(VALIDATION_PATH)
            generator_val = BSG.minibatch_generator(TRAIN_DATA_PATH, BATCH_SIZE, val_set, NUM_POINT)

            #####trainging steps
            temp_loss = train_one_epoch(sess, epoch, train_set, generator_training, ops)
            #####validating steps
            validation(sess, val_set, generator_val, ops)

            ####saving the trianed models
            if temp_loss < init_loss:
                saver.save(sess, os.path.join(LOG_DIR, 'epoch_' + str(epoch) + '.ckpt'))
                init_loss = temp_loss

def train_one_epoch(sess, epoch, train_set, generator, ops):
    """ ops: dict mapping from string to tf ops """

    num_batches_training = len(train_set) // (BATCH_SIZE)
    print('-----------------training--------------------')
    print('training steps: %d'%num_batches_training)

    total_loss = 0
    total_loss_esd = 0
    total_loss_pd = 0
    for i in tqdm(range(num_batches_training)):
        ###
        batch_train_data, batch_direction_label_data, _ = next(generator)
        ###
        feed_dict = {ops['pointclouds']: batch_train_data,
                     ops['direction_labels']: batch_direction_label_data,
                     ops['is_training']: True}

        _, lr, loss, loss_esd_, loss_pd_ = sess.run([ops['train_op'], ops['learning_rate'], ops['loss'],
                                                     ops['loss_esd'], ops['loss_pd']], feed_dict=feed_dict)
        total_loss += loss
        total_loss_esd += loss_esd_
        total_loss_pd += loss_pd_

        if i % 20 == 0:
            print('lr: %f, loss: %f, loss_esd: %f,loss_pd: %f'%(lr, loss, loss_esd_, loss_pd_))

    print('trianing_log_epoch_%d'%epoch)
    log_string('epoch: %d, loss: %f, loss_esd: %f,loss_pd: %f'%(epoch, total_loss/(num_batches_training),
                                                                total_loss_esd/(num_batches_training),
                                                                total_loss_pd/(num_batches_training)))
    return total_loss/(num_batches_training)


def validation(sess, test_set, generator, ops):

    num_batches_testing = len(test_set) // (BATCH_SIZE)
    total_loss = 0
    total_loss_esd = 0
    total_loss_pd = 0
    for _ in tqdm(range(num_batches_testing)):
        ###
        batch_test_data, batch_direction_label_data, _ = next(generator)
        ###
        feed_dict = {ops['pointclouds']: batch_test_data,
                     ops['direction_labels']: batch_direction_label_data,
                     ops['is_training']: False,
                     }
        loss_, loss_esd_, loss_pd_ = sess.run([ops['loss'], ops['loss_esd'], ops['loss_pd']], feed_dict=feed_dict)
        total_loss += loss_
        total_loss_esd += loss_esd_
        total_loss_pd += loss_pd_

    log_string('val loss: %f, loss_esd: %f, loss_pd: %f'%(total_loss/num_batches_testing,
                                                          total_loss_esd/num_batches_testing,
                                                          total_loss_pd/num_batches_testing))

if __name__ == "__main__":
    train()
    LOG_FOUT.close()
