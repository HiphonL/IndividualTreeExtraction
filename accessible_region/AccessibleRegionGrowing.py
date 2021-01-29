"""
Created on Mon July 11 18:50:39 2020

@author: Haifeng Luo
"""

import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'voxel_region_grow'))
import VoxelRegionGrow


def detect_accessible_region(input_xyz, point_directions, center_xyz, voxel_size, angle_threshold=np.pi / 9):
    """
    Generate Accessible Region
    """
    #####
    temp_point2center_vector = center_xyz - input_xyz
    ######
    temp_point2center_vector_L2 = np.linalg.norm(temp_point2center_vector, axis=1)
    temp_point2center_vector_L2[temp_point2center_vector_L2 == 0.0] = 1e-25
    point_directions_L2 = np.linalg.norm(point_directions, axis=1)
    point_directions_L2[point_directions_L2 == 0.0] = 1e-25

    ######
    angles_point2center_vector_and_vote = np.arccos(np.sum(np.multiply(point_directions, temp_point2center_vector), axis=1) /
                                                    (point_directions_L2 * temp_point2center_vector_L2))
    #####
    accessible_index = np.where(angles_point2center_vector_and_vote < angle_threshold)
    #####
    x_range_bottom = center_xyz[0] - 1.5 * voxel_size
    x_range_up = center_xyz[0] + 1.5 * voxel_size
    y_range_bottom = center_xyz[1] - 1.5 * voxel_size
    y_range_up = center_xyz[1] + 1.5 * voxel_size
    vertical_neighbors_index_x = np.where((input_xyz[:, 0] >= x_range_bottom) == (input_xyz[:, 0] <= x_range_up))
    vertical_neighbors_index_y = np.where((input_xyz[:, 1] >= y_range_bottom) == (input_xyz[:, 1] <= y_range_up))
    vertical_neighbors_index_xy = np.intersect1d(vertical_neighbors_index_x[0], vertical_neighbors_index_y[0])
    ###
    accessible_index = list(accessible_index[0]) + list(vertical_neighbors_index_xy)
    accessible_index = list(set(accessible_index))
    #####
    accessible_region = input_xyz[accessible_index, :]

    return accessible_region, accessible_index

############################################################
def voxelization(accessible_region, accessible_index, voxel_size, center_xyz, min_xyz, num_voxel_xyz):


    ###seed position
    seed_x = center_xyz[0]
    seed_y = center_xyz[1]
    seed_z = center_xyz[2]
    seed_voxel_id_x = int(np.floor((seed_x - min_xyz[0]) / voxel_size))
    seed_voxel_id_y = int(np.floor((seed_y - min_xyz[1]) / voxel_size))
    seed_voxel_id_z = int(np.floor((seed_z - min_xyz[2]) / voxel_size))
    seed_voxel = [seed_voxel_id_x, seed_voxel_id_y, seed_voxel_id_z]

    #######init voxels
    output_voxels = np.zeros((int(num_voxel_xyz[0]), int(num_voxel_xyz[1]), int(num_voxel_xyz[2])), dtype=int)
    ######
    valid_voxel_position = []
    voxel2point_index_list = []
    for i in range(int(num_voxel_xyz[0])):
        if i == 0:
            temp_x_range = np.where((accessible_region[:, 0] >= min_xyz[0] + i * voxel_size)
                                    == (accessible_region[:, 0] <= min_xyz[0] + (i + 1) * voxel_size))
        else:
            temp_x_range = np.where((accessible_region[:, 0] > min_xyz[0] + i * voxel_size)
                                    == (accessible_region[:, 0] <= min_xyz[0] + (i + 1) * voxel_size))

        if np.size(temp_x_range[0]) == 0:
            continue
        else:
            for j in range(int(num_voxel_xyz[1])):
                if j == 0:
                    temp_y_range = np.where((accessible_region[:, 1] >= min_xyz[1] + j * voxel_size)
                                            == (accessible_region[:, 1] <= min_xyz[1] + (j + 1) * voxel_size))
                else:
                    temp_y_range = np.where((accessible_region[:, 1] > min_xyz[1] + j * voxel_size)
                                            == (accessible_region[:, 1] <= min_xyz[1] + (j + 1) * voxel_size))
                if np.size(temp_y_range[0]) == 0:
                    continue
                else:
                    xy_intersect = np.intersect1d(temp_x_range[0], temp_y_range[0])
                    if np.size(xy_intersect) == 0:
                        continue
                    else:
                        for k in range(int(num_voxel_xyz[2])):
                            if k == 0:
                                temp_z_range = np.where((accessible_region[:, 2] >= min_xyz[2] + k * voxel_size)
                                                        == (accessible_region[:, 2] <= min_xyz[2] + (
                                        k + 1) * voxel_size))
                            else:
                                temp_z_range = np.where((accessible_region[:, 2] > min_xyz[2] + k * voxel_size)
                                                        == (accessible_region[:, 2] <= min_xyz[2] + (
                                        k + 1) * voxel_size))

                            if np.size(temp_z_range[0]) == 0:
                                continue
                            else:
                                xy_z_intersect = np.intersect1d(xy_intersect, temp_z_range[0])
                                if np.size(xy_z_intersect) != 0:
                                    valid_voxel_position.append([i, j, k])
                                    ######
                                    temp_voxel2point_index_list = [accessible_index[l] for l in list(xy_z_intersect)]
                                    voxel2point_index_list.append(temp_voxel2point_index_list)
                                    output_voxels[i, j, k] = 1

    return output_voxels, seed_voxel, valid_voxel_position, voxel2point_index_list

############################################################
def voxel_region_grow(output_voxels, seed):
    voxelRG = VoxelRegionGrow.Build(output_voxels)
    objcetMask = voxelRG.Run(seed)
    objcetMaskVoxelIndex = np.vstack(np.where(np.array(objcetMask) == 1)).T
    objcetMaskVoxelIndex = [list(tempMaskVoxelIndex) for tempMaskVoxelIndex in objcetMaskVoxelIndex]
    return objcetMask, objcetMaskVoxelIndex






