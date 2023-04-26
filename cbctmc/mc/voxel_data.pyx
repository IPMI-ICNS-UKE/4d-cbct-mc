cimport cython


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
