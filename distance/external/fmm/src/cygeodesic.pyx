#!python
# cython: language_level=3
# distutils: language = c++
libraries=["stdc++"]
# distutils: sources = main.cpp

MAX_DISTANCE = 9999999.0


import numpy as np
cimport numpy as np
from libcpp.vector cimport vector

cdef extern from "main.cpp":
    void gds(double*, double*, double*, int*, int, int, double*, double*)

cdef extern from "main.cpp":
    void gds_matrix(double*, double*, double*, int*, int, int, double*)

def geodesic_distance(np.ndarray[double, ndim=2] vertices, np.ndarray[int, ndim=2] triangles,
                      np.ndarray[int, ndim=1] source_indices):

    cdef np.ndarray[double, ndim=1] c_x = np.array(vertices[:,0])
    cdef np.ndarray[double, ndim=1] c_y = np.array(vertices[:,1])
    cdef np.ndarray[double, ndim=1] c_z = np.array(vertices[:,2])

    cdef np.ndarray[int, ndim=1] c_t = np.concatenate((triangles[:,0],triangles[:,1],triangles[:,2]))

    cdef np.ndarray[double, ndim=1] sources_array = np.full(vertices.shape[0], MAX_DISTANCE)
    sources_array[source_indices] = 0

    cdef np.ndarray[double, ndim=1] result = np.zeros(vertices.shape[0], dtype=np.float64)


    gds(&c_x[0], &c_y[0], &c_z[0], &c_t[0], vertices.shape[0], triangles.shape[0], &sources_array[0],
                    &result[0])
    return result

def geodesic_matrix(np.ndarray[np.float64_t, ndim=2] vertices, np.ndarray[np.int32_t, ndim=2] triangles):

    cdef np.ndarray[double, ndim=1] c_x = np.array(vertices[:,0])
    cdef np.ndarray[double, ndim=1] c_y = np.array(vertices[:,1])
    cdef np.ndarray[double, ndim=1] c_z = np.array(vertices[:,2])

    cdef np.ndarray[int, ndim=1] c_t = np.concatenate((triangles[:,0],triangles[:,1],triangles[:,2]))
    cdef np.ndarray[double, ndim=1] result = np.zeros(vertices.shape[0]**2, dtype=np.float64)

    gds_matrix(&c_x[0], &c_y[0], &c_z[0], &c_t[0], vertices.shape[0], triangles.shape[0],
                    &result[0])

    return result.reshape((vertices.shape[0], -1))