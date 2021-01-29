"""
Created on Mon July 11 18:50:39 2020

@author: Haifeng Luo
"""

from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("VoxelRegionGrow.pyx"),
    include_dirs=[numpy.get_include()]
)