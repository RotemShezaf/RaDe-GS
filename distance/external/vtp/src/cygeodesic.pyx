#!python
# cython: language_level=3
# distutils: language = c++
libraries=["stdc++"]
# distutils: sources = main.cpp

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector

cdef extern from "main.cpp":
    vector[double] gds(double*, int, int*, int, unsigned)

cdef extern from "main.cpp":
    void gds_matrix(double*, int, int*, int, double*)

def geodesic_distance(np.ndarray[double, ndim=2] vertices, np.ndarray[int, ndim=2] triangles,
                      unsigned source_index):

    cdef np.ndarray[double, ndim=1] c_v = vertices.flatten()
    cdef np.ndarray[int, ndim=1] c_t = triangles.flatten()

    distances = gds(&c_v[0], vertices.shape[0], &c_t[0], triangles.shape[0], source_index)

    distances = np.asarray(distances)
    return distances

def geodesic_matrix(np.ndarray[np.float64_t, ndim=2] vertices, np.ndarray[np.int32_t, ndim=2] triangles):

    cdef np.ndarray[double, ndim=1] c_v = vertices.flatten()
    cdef np.ndarray[int, ndim=1] c_t = triangles.flatten()
    cdef np.ndarray[double, ndim=1] result = np.zeros(vertices.shape[0]**2, dtype=np.float64)
    distances = gds_matrix(&c_v[0], vertices.shape[0], &c_t[0], triangles.shape[0], &result[0])
    return result.reshape((vertices.shape[0], -1))