import multiprocessing as mp
import time
from math import ceil

import numpy as np

from cbctmc.mc.voxel_data import (
    compile_voxel_data_string,
    compile_voxel_data_string_fast,
)

# def compile_fast(materials, densities, n_processes: int = mp.cpu_count()):
#     def chunk_3d_array(arr, chunk_size: int = 16):
#         for i in range(0, arr.shape[-1], chunk_size):
#             yield arr[..., i: i + chunk_size]
#
#     chunk_size = ceil(materials.shape[0] / n_processes)
#     with mp.Pool() as pool:
#         results = pool.starmap(
#             compile_voxel_data_string2,
#             zip(
#                 chunk_3d_array(materials, chunk_size=chunk_size),
#                 chunk_3d_array(densities, chunk_size=chunk_size),
#             ),
#         )
#
#         return "".join(results).strip()


densities = np.random.random((512, 512, 320)).astype(np.float32)
materials = (densities > 0.5).astype(np.uint8)

t = time.monotonic()
c1 = compile_voxel_data_string(materials, densities)
print(time.monotonic() - t)

t = time.monotonic()
c2 = compile_voxel_data_string_fast(materials, densities)
print(time.monotonic() - t)

# print(c1 == c2)
# print('***')
# print(c1)
# print('***')
# print(c2)
