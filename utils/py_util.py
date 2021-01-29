"""
Created on Mon July 11 18:50:39 2020

@author: Haifeng Luo
"""
import numpy as np
import random
import math
import os

def load_data(path):
    try:
        return np.load(path)
    except:
        return np.loadtxt(path)

def get_data_set(data_path):
    files_set = os.listdir(data_path)
    random.shuffle(files_set)
    return files_set


def get_train_val_set(trainingdata_path, val_rate=0.20):
    train_set = []
    val_set = []
    all_train_set = os.listdir(trainingdata_path)
    random.shuffle(all_train_set)
    total_num = len(all_train_set)
    val_num = int(val_rate * total_num)
    for j in range(len(all_train_set)):
        if j < val_num:
            val_set.append(all_train_set[j])
        else:
            train_set.append(all_train_set[j])
    return train_set, val_set


def normalize(sample_xyz):
    min_xyz = np.min(sample_xyz, axis=0)
    max_xyz = np.max(sample_xyz, axis=0)
    deta_central_xyz = (max_xyz - min_xyz)/2.0
    central_xyz = deta_central_xyz + min_xyz
    n_data = sample_xyz - central_xyz
    # normalize into unit sphere
    n_data /= np.max(np.linalg.norm(n_data, axis=1))
    return n_data

def compute_object_center(sample_xyz):
    min_xyz = np.min(sample_xyz, axis=0)
    max_xyz = np.max(sample_xyz, axis=0)
    deta_central_xyz = (max_xyz - min_xyz) / 2.0
    central_xyz = deta_central_xyz + min_xyz
    return central_xyz


def jitter_point_cloud(sample_xyz, Jitter_argument, sigma=0.001, clip=0.05):
    if np.random.random() < Jitter_argument:
        N, C = sample_xyz.shape
        assert(clip > 0)
        jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
        sample_xyz += jittered_data
    return sample_xyz

def shuffle_data(data):
    idx = np.arange(np.size(data, 0))
    np.random.shuffle(idx)
    return data[idx, ...]

def ratation(sample_xyz, Rotation_argument):
    if np.random.random() < Rotation_argument:
        ###
        rot = random.uniform(0, 2 * math.pi)
        rotation_matrix = [[math.cos(rot), math.sin(rot), 0],
                           [-math.sin(rot), math.cos(rot), 0],
                           [0, 0, 1]]
        sample_xyz = np.dot(sample_xyz, rotation_matrix)
    return sample_xyz

def ratation_angle(sample_xyz, angel):
    rot = angel/180.0
    rotation_matrix = [[math.cos(rot), math.sin(rot), 0],
                       [-math.sin(rot), math.cos(rot), 0],
                       [0, 0, 1]]
    sample_xyz = np.dot(sample_xyz, rotation_matrix)
    return sample_xyz

def transfer_xy(sample_xyz, x_d, y_d):
    temp_ones = np.ones([np.size(sample_xyz, 0), 1])
    sample_xyz = np.concatenate([sample_xyz, temp_ones], axis=-1)

    transfer_matrix = [[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 1],
                       [x_d, y_d, 0, 1]]
    sample_xyz = np.dot(sample_xyz, transfer_matrix)
    return sample_xyz[:, :3]

def farthest_point_sample(xyz, npoint):
    N, _ = xyz.shape
    centroids = []
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids.append(farthest)
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = int(np.where(distance == np.max(distance))[0][0])
    return centroids
