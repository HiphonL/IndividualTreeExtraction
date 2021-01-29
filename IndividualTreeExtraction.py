"""
Created on Mon July 11 18:50:39 2020

@author: Haifeng Luo
"""
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'voxel_traversal'))
sys.path.append(os.path.join(BASE_DIR, 'accessible_region'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import py_util
import VoxelTraversalAlgorithm as VTA
import AccessibleRegionGrowing as ARG
import PointwiseDirectionPrediction as PDE_net


def show_AR_RG(voxels1, voxels2):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ####accessible region
    ax.voxels(voxels2, facecolors='red', edgecolor='k', alpha=0.9)
    ####region growing results
    ax.voxels(voxels1, facecolors='green', edgecolor='k')
    plt.show()

############################################################
def compute_object_center(sample_xyz):
    min_xyz = np.min(sample_xyz, axis=0)
    max_xyz = np.max(sample_xyz, axis=0)
    deta_central_xyz = (max_xyz - min_xyz) / 2.0
    central_xyz = deta_central_xyz + min_xyz
    return central_xyz

############################################################
def object_xoy_bounding(xyz, object_xyz, sphere_level, bounding_order=1):

    min_xy = np.min(object_xyz[:, :2], axis=0)
    max_xy = np.max(object_xyz[:, :2], axis=0)
    delta_xy = (max_xy - min_xy) / sphere_level
    min_xy += bounding_order * delta_xy
    max_xy -= bounding_order * delta_xy
    modify_object_index_x = np.where((xyz[:, 0] >= min_xy[0]) == (xyz[:, 0] < max_xy[0]))
    modify_object_index_y = np.where((xyz[:, 1] >= min_xy[1]) == (xyz[:, 1] < max_xy[1]))
    modify_object_index_xy = np.intersect1d(modify_object_index_x[0], modify_object_index_y[0])
    modify_object_index_xy = list(modify_object_index_xy)
    return modify_object_index_xy

def direction_vote_voxels(points, directions, voxel_size, num_voxel_xyz, min_xyz):
    numpints = np.size(points, 0)
    output_voxel_direction_count = np.zeros((int(num_voxel_xyz[0]), int(num_voxel_xyz[1]), int(num_voxel_xyz[2])), dtype=int)

    ######
    per_voxel_direction_start_points = [[[[] for _ in range(int(num_voxel_xyz[2]))] for _ in range(int(num_voxel_xyz[1]))] for _ in range(int(num_voxel_xyz[0]))]
    ####
    for i in range(numpints):
        visited_voxels = VTA.voxel_traversal(points[i, :], directions[i, :], min_xyz, num_voxel_xyz, voxel_size)
        for j in range(len(visited_voxels)):
            output_voxel_direction_count[int(visited_voxels[j][0]), int(visited_voxels[j][1]), int(visited_voxels[j][2])] += 1
            per_voxel_direction_start_points[int(visited_voxels[j][0])][int(visited_voxels[j][1])][int(visited_voxels[j][2])].append(i)

    return output_voxel_direction_count, per_voxel_direction_start_points

def center_detection_xoy(voxel_direction_count, num_voxel_xyz, center_direction_count_th):

    numVoxel_x = num_voxel_xyz[0]
    numVoxel_y = num_voxel_xyz[1]
    object_center_voxel_list = []

    for i in range(int(numVoxel_x - 2)):
        for j in range(int(numVoxel_y - 2)):
            temp_object_voxel_dir_count = voxel_direction_count[i + 1, j + 1]

            if temp_object_voxel_dir_count < center_direction_count_th:
                continue

            temp_neighbors = [voxel_direction_count[i, j], voxel_direction_count[i + 1, j],
                              voxel_direction_count[i + 2, j],
                              voxel_direction_count[i, j + 1], voxel_direction_count[i + 2, j + 1],
                              voxel_direction_count[i, j + 2], voxel_direction_count[i + 1, j + 2],
                              voxel_direction_count[i + 2, j + 2]]
            max_neighbors = np.max(np.array(temp_neighbors))

            if temp_object_voxel_dir_count > max_neighbors:
                object_center_voxel_list.append([i + 1, j + 1])

    return np.vstack(object_center_voxel_list)

############################################################
def center_detection(data, voxel_size, angle_threshold, center_direction_count_th=20):
    '''detect the tree centers'''

    object_xyz_list = []
    xyz = data[:, :3]
    directions = data[:, 3:]
    min_xyz = np.min(xyz, axis=0)
    max_xyz = np.max(xyz, axis=0)
    delta_xyz = max_xyz - min_xyz
    num_voxel_xyz = np.ceil(delta_xyz / voxel_size)

    #######################################################################
    ############################Center Detection###########################
    #######################################################################
    output_voxel_direction_count, per_voxel_direction_start_points = direction_vote_voxels(xyz,
                                                                                           directions,
                                                                                           voxel_size,
                                                                                           num_voxel_xyz,
                                                                                           min_xyz)
    #####centers in xoy plane
    output_voxel_direction_count_xoy = np.sum(output_voxel_direction_count, axis=2)
    object_centers_xoy = center_detection_xoy(output_voxel_direction_count_xoy,
                                                     num_voxel_xyz[:2],
                                                     center_direction_count_th)

    ####centers in z-axis
    for i in range(np.size(object_centers_xoy, 0)):
        temp_object_center_xoy = object_centers_xoy[i, :]
        ####
        temp_centre_xyz = np.array([temp_object_center_xoy[0], temp_object_center_xoy[1]])
        temp_centre_xyz = temp_centre_xyz * voxel_size + min_xyz[:2] # + voxel_size / 2.0
        ####
        center_xbottom = temp_centre_xyz[0] - voxel_size / 2.0
        center_xup = temp_centre_xyz[0] + voxel_size / 2.0
        center_ybottom = temp_centre_xyz[1] - voxel_size / 2.0
        center_yup = temp_centre_xyz[1] + voxel_size / 2.0
        x_vaild_range = np.where((xyz[:, 0] > center_xbottom) == (xyz[:, 0] < center_xup))
        y_vaild_range = np.where((xyz[:, 1] > center_ybottom) == (xyz[:, 1] < center_yup))
        xy_intersection_index = list(set(x_vaild_range[0]).intersection(set(y_vaild_range[0])))

        ####discard the fake centers
        if len(xy_intersection_index) == 0:
            continue
        #####
        output_voxel_direction_count_z = output_voxel_direction_count[temp_object_center_xoy[0], temp_object_center_xoy[1], :]
        temp_index = np.where(output_voxel_direction_count_z == np.max(output_voxel_direction_count_z))
        object_xyz_list.append([temp_object_center_xoy[0], temp_object_center_xoy[1], temp_index[0][0]])

    object_xyz_list = np.vstack(object_xyz_list)
    object_xyz_list = object_xyz_list * voxel_size + min_xyz # + voxel_size / 2.0

    ####### further refine detected centers using intersection directions
    ####### Note that the following steps have not been discussed in our paper #############
    ####### If higher efficiency is required, these steps can be discarded ###############
    objectVoxelMask_list = []
    for i in range(np.size(object_xyz_list, 0)):

        center_xyz = object_xyz_list[i, :]
        _, _, objectVoxelMask = individual_tree_separation(xyz,
                                                         directions,
                                                         center_xyz,
                                                         voxel_size,
                                                         min_xyz,
                                                         num_voxel_xyz,
                                                         angle_threshold)

        objectVoxelMask_index = np.where(objectVoxelMask == True)
        if np.size(objectVoxelMask_index[0], 0) == 0:
            continue
        temp_objectvoxels = []
        for j in range(np.size(objectVoxelMask_index[0], 0)):
            temp_objectvoxel_index = [objectVoxelMask_index[0][j], objectVoxelMask_index[1][j], objectVoxelMask_index[2][j]]
            temp_objectvoxels.append(temp_objectvoxel_index)
        objectVoxelMask_list.append(temp_objectvoxels)

    #######
    final_object_center_index = []
    for i in range(len(objectVoxelMask_list)):
        #####
        temp_object_voxels = np.vstack(objectVoxelMask_list[i])
        #####copy array
        temp_all_object_voxels = objectVoxelMask_list[:]
        del temp_all_object_voxels[i]

        #######
        for j in range(len(temp_all_object_voxels)):

            temp_remain_object_voxels = np.vstack(temp_all_object_voxels[j])
            temp_intersection = np.array([x for x in set(tuple(x) for x in temp_object_voxels) & set(tuple(x) for x in temp_remain_object_voxels)])

            if np.size(temp_intersection, 0) > 0:
                temp_object_voxels = set(tuple(x) for x in temp_object_voxels).difference(set(tuple(x) for x in temp_intersection))
                temp_object_voxels = np.array([list(x) for x in temp_object_voxels])

                if np.size(temp_object_voxels, 0) == 0:
                    break        #
        if np.size(temp_object_voxels, 0) >= 3:
            final_object_center_index.append(i)

    object_xyz_list = object_xyz_list[final_object_center_index, :]
    print('Num of Tree Centers: %d'%int(np.size(object_xyz_list, 0)))
    return object_xyz_list

############################################################
def individual_tree_separation(xyz, directions, center_xyz, voxel_size, min_xyz, num_voxel_xyz,
                               angle_threshold, visulization=False):

    #####generate accessible region
    accessible_region, accessible_index = ARG.detect_accessible_region(xyz, directions, center_xyz,
                                                                       voxel_size, angle_threshold)
    #####
    #####voxelize accessible region
    accessible_region_voxels, seed_voxel, valid_voxels, voxel2point_index_list = ARG.voxelization(accessible_region,
                                                                                              accessible_index,
                                                                                              voxel_size,
                                                                                              center_xyz,
                                                                                              min_xyz,
                                                                                              num_voxel_xyz)
    ###########
    output_voxels_v2 = np.array(accessible_region_voxels)
    output_voxels_v2 = output_voxels_v2.astype(bool)

    ####voxel-based region growing
    objcetMask, objcetMaskVoxelIndex = ARG.voxel_region_grow(accessible_region_voxels, seed_voxel)

    ###########visualization
    objcetMask = np.array(objcetMask)
    objcetMask = objcetMask.astype(bool)
    if visulization == True:
        show_AR_RG(objcetMask, output_voxels_v2)

    ######refine seed voxels
    index_voxel2point = [valid_voxels.index(tempMaskIndex) for tempMaskIndex in objcetMaskVoxelIndex]
    ######
    temp_object_xyz_index = []
    for temp_index_voxel2point in index_voxel2point:
        temp_object_xyz_index += voxel2point_index_list[temp_index_voxel2point]
    #####
    object_result = xyz[temp_object_xyz_index, :]
    return object_result, temp_object_xyz_index, objcetMask

def individual_tree_extraction(PDE_net_model_path, test_data_path, result_path, voxel_size, Nd, ARe):
    '''Individual Tree Extraction'''
    ####restore trained PDE-net
    sess, PDE_net_ops = PDE_net.restore_trained_model(NUM_POINT, PDE_net_model_path)
    ####
    file_list = os.listdir(test_data_path)
    for i in range(len(file_list)):
        tree_index = 0
        filename, _ = os.path.splitext(file_list[i])
        print('Separating ' + filename + '...')
        #### data[x, y, z] original coordinates
        testdata = py_util.load_data(test_data_path + file_list[i])[:, :3]
        ####normalized coordinates
        nor_testdata = py_util.normalize(testdata)
        ####Pointwise direction prediction
        xyz_direction = PDE_net.prediction(sess, nor_testdata, PDE_net_ops)
        ####tree center detection
        object_center_list = center_detection(xyz_direction, voxel_size, ARe, Nd)

        ####for single tree clusters
        if np.size(object_center_list, axis=0) <= 1:
            ####random colors
            num_pointIntree = np.size(xyz_direction, axis=0)
            color = np.random.randint(0, 255, size=3)
            ####assign tree labels
            temp_tree_label = np.ones([num_pointIntree, 1]) * tree_index
            color = np.ones([num_pointIntree, 3]) * color
            ######
            individualtree = np.concatenate([testdata[:, :3], color, temp_tree_label], axis=-1)
            np.savetxt(result_path + file_list[i], individualtree, fmt='%.4f')
            tree_index += 1
            continue

        ####for multi tree clusters
        extracted_object_list = []
        object_color_list = []
        temp_tree_id = 0
        for j in range(np.size(object_center_list, 0)):

            xyz = xyz_direction[:, :3]
            directions = xyz_direction[:, 3:]
            ####
            min_xyz = np.min(xyz, axis=0)
            max_xyz = np.max(xyz, axis=0)
            delta_xyz = max_xyz - min_xyz
            num_voxel_xyz = np.ceil(delta_xyz / voxel_size)
            ####
            center_xyz = object_center_list[j, :]
            ####using padding to fix the situation where the tree center voxel is empty
            center_xyz_padding = np.array([[center_xyz[0], center_xyz[1], center_xyz[2]],
                                           [center_xyz[0], center_xyz[1], center_xyz[2] - voxel_size],
                                           [center_xyz[0], center_xyz[1], center_xyz[2] + voxel_size]])
            directions_padding = np.array([[0.0, 0.0, 0.0],
                                           [0.0, 0.0, 1.0],
                                           [0.0, 0.0, -1.0]])
            center_direction_padding = np.concatenate([center_xyz_padding, directions_padding], axis=-1)

            xyz = np.concatenate([center_xyz_padding, xyz], axis=0)
            directions = np.concatenate([directions_padding, directions], axis=0)
            xyz_direction = np.concatenate([center_direction_padding, xyz_direction], axis=0)

            ####only for align the indexes
            testdata = np.concatenate([testdata[:3, :], testdata], axis=0)
            ####
            object_result, temp_object_xyz_index, _ = individual_tree_separation(xyz,
                                                                                 directions,
                                                                                 center_xyz,
                                                                                 voxel_size,
                                                                                 min_xyz,
                                                                                 num_voxel_xyz,
                                                                                 ARe,
                                                                                 visulization=False)
            ####refine the NULL growing results
            if np.size(object_result, 0) == 0: continue
            ###fix the discontinuity of the voxel in the vertical direction of tree centers
            modify_object_index_xy = object_xoy_bounding(xyz, object_result, 8, bounding_order=1)
            temp_object_xyz_index += modify_object_index_xy
            temp_object_xyz_index = list(set(temp_object_xyz_index))

            #####remove padding points
            real_object_xyz_index = [i for i in temp_object_xyz_index if i > 2]
            object_result = testdata[real_object_xyz_index, :3]
            ####generate random color for extracted individual tree points
            num_pointInObject = np.size(object_result, axis=0)
            color = np.random.randint(0, 255, size=3)
            object_color_list.append(color)
            ####assign a tree label for each individual tree
            temp_object_label = np.ones([num_pointInObject, 1]) * temp_tree_id
            color = np.ones([num_pointInObject, 3]) * color
            extracted_object_list.append(np.concatenate([object_result, color, temp_object_label], axis=-1))
            ####
            temp_tree_id += 1
            ####delete the extracted individual tree points
            testdata = np.delete(testdata, temp_object_xyz_index, axis=0)
            xyz_direction = np.delete(xyz_direction, temp_object_xyz_index, axis=0)

        ####using the nearest neighbor assignment to refine those points with large direction errors
        for k in range(np.size(xyz_direction, 0)):
            temp_remain_xyz_nor = xyz_direction[k, :3]
            temp_remain_xyz = testdata[k, :3]
            temp_distances = np.sqrt(np.sum(np.asarray(temp_remain_xyz_nor - object_center_list) ** 2, axis=1))
            nearestObjectCenter = np.where(temp_distances == np.min(temp_distances))
            color = object_color_list[int(nearestObjectCenter[0])]
            temp_remain_xyz_label = np.expand_dims(np.concatenate([temp_remain_xyz, color, nearestObjectCenter[0]], axis=-1), axis=0)
            extracted_object_list.append(temp_remain_xyz_label)
        ####output the final results
        np.savetxt(result_path + filename + '.txt', np.vstack(extracted_object_list), fmt='%.4f')


if __name__ == '__main__':

    NUM_POINT = 4096
    Nd = 80
    ARe = np.pi / 9.0
    voxel_size = 0.08
    #######
    PDE_net_model_path ='./backbone_network/pre_trained_PDE_net/'
    test_data_path = './data/test/'
    result_path = './result/'
    if not os.path.exists(result_path): os.mkdir(result_path)

    #######extracting individual trees from tree clusters
    individual_tree_extraction(PDE_net_model_path, test_data_path, result_path, voxel_size, Nd, ARe)
