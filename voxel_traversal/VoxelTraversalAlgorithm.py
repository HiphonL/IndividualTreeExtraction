"""
Created on Mon July 11 18:50:39 2020

@author: Haifeng Luo
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def show_voxel(voxels):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxels, facecolors='red', edgecolor='k')
    plt.show()

############################################################
def voxel_traversal(start_point, directions, min_xyz, num_voxel_xyz, voxel_size):
    '''
    voxel traversal for tree center detection
    '''
    current_voxel_x = int(np.floor((start_point[0] - min_xyz[0]) / voxel_size))
    current_voxel_y = int(np.floor((start_point[1] - min_xyz[1]) / voxel_size))
    current_voxel_z = int(np.floor((start_point[2] - min_xyz[2]) / voxel_size))

    stepX = 1 if directions[0] >= 0 else -1
    stepY = 1 if directions[1] >= 0 else -1
    stepZ = 1 if directions[2] >= 0 else -1

    next_voxel_boundary_x = (current_voxel_x + stepX) * voxel_size + min_xyz[0]
    next_voxel_boundary_y = (current_voxel_y + stepY) * voxel_size + min_xyz[1]
    next_voxel_boundary_z = (current_voxel_z + stepZ) * voxel_size + min_xyz[2]

    tMaxX = (next_voxel_boundary_x - start_point[0]) / directions[0] if directions[0] != 0 else float('inf')
    tMaxY = (next_voxel_boundary_y - start_point[1]) / directions[1] if directions[1] != 0 else float('inf')
    tMaxZ = (next_voxel_boundary_z - start_point[2]) / directions[2] if directions[2] != 0 else float('inf')

    tDeltaX = voxel_size / directions[0] * stepX if directions[0] != 0 else float('inf')
    tDeltaY = voxel_size / directions[1] * stepY if directions[1] != 0 else float('inf')
    tDeltaZ = voxel_size / directions[2] * stepZ if directions[2] != 0 else float('inf')

    visited_voxels = []
    visited_voxels.append([current_voxel_x, current_voxel_y, current_voxel_z])
    while current_voxel_x <= (num_voxel_xyz[0] - 1) and current_voxel_x >= 0 and current_voxel_y <= (
            num_voxel_xyz[1] - 1) \
            and current_voxel_y >= 0 and current_voxel_z <= (num_voxel_xyz[2] - 1) and current_voxel_z >= 0:

        if tMaxX < tMaxY:
            if tMaxX < tMaxZ:
                current_voxel_x += stepX
                tMaxX += tDeltaX
            else:
                current_voxel_z += stepZ
                tMaxZ += tDeltaZ
        else:
            if tMaxY < tMaxZ:
                current_voxel_y += stepY
                tMaxY += tDeltaY
            else:
                current_voxel_z += stepZ
                tMaxZ += tDeltaZ
        if current_voxel_x < num_voxel_xyz[0] and current_voxel_x >= 0 and \
                current_voxel_y < num_voxel_xyz[1] and current_voxel_y >= 0 and \
                current_voxel_z < num_voxel_xyz[2] and current_voxel_z >= 0:
            visited_voxels.append([current_voxel_x, current_voxel_y, current_voxel_z])

    return visited_voxels


def show_direction_aggregation(points, directions, voxel_size, num_voxel_xyz, min_xyz):
    numpints = np.size(points, 0)
    ####
    output_voxels = np.zeros((int(num_voxel_xyz[0]), int(num_voxel_xyz[1]), int(num_voxel_xyz[2])), dtype=int)
    for i in range(numpints):
        visited_voxels = voxel_traversal(points[i, :], directions[i, :], min_xyz, num_voxel_xyz, voxel_size)

        for j in range(len(visited_voxels)):
            output_voxels[int(visited_voxels[j][0]), int(visited_voxels[j][1]), int(visited_voxels[j][2])] = 1
        # #####
        if i == 5: #show 5 point directions
            output_voxels_v2 = np.array(output_voxels)
            output_voxels_v2 = output_voxels_v2.astype(bool)
            show_voxel(output_voxels_v2)
