cimport cython

import multiprocessing as mp
from math import ceil

import numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def compile_voxel_data_string(
    unsigned char[:, :, :] materials,
    float[:, :, :] densities,
) -> str:
    cdef Py_ssize_t n_i = materials.shape[0]
    cdef Py_ssize_t n_j = materials.shape[1]
    cdef Py_ssize_t n_k = materials.shape[2]

    cdef Py_ssize_t i, j, k

    data = []

    for k in range(n_k):
        for j in range(n_j):
            for i in range(n_i):
                data.append(f'{materials[i, j, k]} {densities[i, j, k]:.6f}\n')
            data.append('\n')
        data.append('\n')

    return ''.join(data).strip()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def compile_voxel_data_string_no_strip(
    unsigned char[:, :, :] materials,
    float[:, :, :] densities,
) -> str:
    cdef Py_ssize_t n_i = materials.shape[0]
    cdef Py_ssize_t n_j = materials.shape[1]
    cdef Py_ssize_t n_k = materials.shape[2]

    cdef Py_ssize_t i, j, k

    data = []

    for k in range(n_k):
        for j in range(n_j):
            for i in range(n_i):
                data.append(f'{materials[i, j, k]} {densities[i, j, k]:.6f}\n')
            data.append('\n')
        data.append('\n')

    return ''.join(data)

def compile_voxel_data_string_fast(materials: np.ndarray, densities: np.ndarray, n_processes: int = mp.cpu_count()):
    def chunk_3d_array(arr, chunk_size: int = 16):
        for i in range(0, arr.shape[-1], chunk_size):
            yield arr[..., i: i + chunk_size]

    chunk_size = ceil(materials.shape[0] / n_processes)
    with mp.Pool() as pool:
        results = pool.starmap(
            compile_voxel_data_string_no_strip,
            zip(
                chunk_3d_array(materials, chunk_size=chunk_size),
                chunk_3d_array(densities, chunk_size=chunk_size),
            ),
        )

        return "".join(results).strip()
