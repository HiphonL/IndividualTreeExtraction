# -*- coding: utf-8 -*-
from collections import deque
import numpy as np
cimport numpy as np
cimport cython

cdef class Build:

    cdef int[:,:,:] voxels
    cdef int[:,:,:] outputMask
    cdef int num_voxel_x
    cdef int num_voxel_y
    cdef int num_voxel_z
    cdef queue

    def __cinit__(self, int[:,:,:] voxels):
        self.voxels = voxels
        self.outputMask = np.zeros_like(self.voxels)
        self.queue = deque()
        self.num_voxel_x = voxels.shape[0]
        self.num_voxel_y = voxels.shape[1]
        self.num_voxel_z = voxels.shape[2]

    def Run(self, seed):
        cdef int newItem[3]
        cdef int neighbors[26][3]
        self.queue.append((seed[0], seed[1], seed[2]))

        if self.voxels[seed[0], seed[1], seed[2]] == 1:
            self.outputMask[seed[0], seed[1], seed[2]] = 1

        while len(self.queue) != 0:
            newItem = self.queue.pop()
            neighbors = [[newItem[0]-1, newItem[1]-1, newItem[2]-1],   [newItem[0]-1, newItem[1]-1, newItem[2]],   [newItem[0]-1, newItem[1]-1, newItem[2]+1],
                         [newItem[0]-1, newItem[1], newItem[2]-1],     [newItem[0]-1, newItem[1], newItem[2]],     [newItem[0]-1, newItem[1], newItem[2]+1],
                         [newItem[0]-1, newItem[1]+1, newItem[2]-1],   [newItem[0]-1, newItem[1]+1, newItem[2]],   [newItem[0]-1, newItem[1]+1, newItem[2]+1],
                         [newItem[0], newItem[1]-1, newItem[2]-1],     [newItem[0], newItem[1]-1, newItem[2]],     [newItem[0], newItem[1]-1, newItem[2]+1],
                         [newItem[0], newItem[1], newItem[2]-1],       [newItem[0], newItem[1], newItem[2]+1],     [newItem[0], newItem[1]+1, newItem[2]-1],
                         [newItem[0], newItem[1]+1, newItem[2]],       [newItem[0], newItem[1]+1, newItem[2]+1],   [newItem[0]+1, newItem[1]-1, newItem[2]-1],
                         [newItem[0]+1, newItem[1]-1, newItem[2]],     [newItem[0]+1, newItem[1]-1, newItem[2]+1], [newItem[0]+1, newItem[1], newItem[2]-1],
                         [newItem[0]+1, newItem[1], newItem[2]],       [newItem[0]+1, newItem[1], newItem[2]+1],   [newItem[0]+1, newItem[1]+1, newItem[2]-1],
                         [newItem[0]+1, newItem[1]+1, newItem[2]],     [newItem[0]+1, newItem[1]+1, newItem[2]+1]
                         ]
            for neighbor in neighbors:
                self.checkNeighbour(neighbor[0], neighbor[1], neighbor[2])

        return self.outputMask

    cdef checkNeighbour(self, int x, int y, int z):
        cdef int voxelValue

        if (x < self.num_voxel_x and y < self.num_voxel_y and z < self.num_voxel_z and x > -1 and y > -1 and z > -1):

            voxelValue = self.voxels[x, y, z]
            if self.isVoxelAcceptable(voxelValue) and self.outputMask[x,y,z] == 0:
                self.outputMask[x,y,z] = 1
                self.queue.append((x, y, z))

    cdef isVoxelAcceptable(self, int voxelValue):
        if voxelValue == 1:
            return True
        return False
