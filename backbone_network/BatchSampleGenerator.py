"""
Created on Mon July 11 18:50:39 2020

@author: Haifeng Luo
"""

import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import py_util

def minibatch_generator(trainingdata_path, batch_size, train_set, num_points):
    '''
        Generator for PDE-net training and validation
    '''

    while True:
        train_xyz_data = []
        direction_label_data = []
        object_label_data = []
        batch = 0
        for i in (range(len(train_set))):
            batch += 1
            url = train_set[i]
            temp_point_set = py_util.load_data(trainingdata_path+url)

            #####
            temp_xyz = temp_point_set[:, :3]
            temp_xyz = py_util.normalize(temp_xyz)

            object_label = temp_point_set[:, 3]
            unique_object_label = np.unique(object_label)

            temp_multi_objects_sample = []
            for j in range(np.size(unique_object_label)):
                ###for each object
                temp_index = np.where(object_label == unique_object_label[j])
                temp_index_object_xyz = temp_xyz[temp_index[0], :]
                ###object_label
                temp_object_label = np.expand_dims(object_label[temp_index[0]], axis=-1)
                ###center point
                temp_object_center_xyz = py_util.compute_object_center(temp_index_object_xyz)
                ###deta_x + x = center_point ---->deta_x = center_point - x
                temp_direction_label = temp_object_center_xyz - temp_index_object_xyz
                ####[x, y, z, deta_x, deta_y, deta_z]
                temp_xyz_direction_label_concat = np.concatenate([temp_index_object_xyz,
                                                             temp_direction_label,
                                                             temp_object_label],
                                                             axis=-1)
                ####
                temp_multi_objects_sample.append(temp_xyz_direction_label_concat)

            temp_multi_objects_sample = np.vstack(temp_multi_objects_sample)
            ###
            temp_multi_objects_sample = py_util.shuffle_data(temp_multi_objects_sample)
            temp_multi_objects_sample = temp_multi_objects_sample[:num_points, :]
            ###
            training_xyz = temp_multi_objects_sample[:, :3]
            training_direction_label = temp_multi_objects_sample[:, 3:-1]
            training_object_label = temp_multi_objects_sample[:, -1]

            train_xyz_data.append(training_xyz)
            direction_label_data.append(training_direction_label)
            object_label_data.append(training_object_label)

            if batch % batch_size == 0:
                train_xyz_data = np.array(train_xyz_data)
                direction_label_data = np.array(direction_label_data)
                object_label_data = np.array(object_label_data)
                yield [train_xyz_data, direction_label_data, object_label_data]
                train_xyz_data = []
                direction_label_data = []
                object_label_data = []
                batch = 0