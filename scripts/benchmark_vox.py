import multiprocessing as mp
import time
from math import ceil

import numpy as np

from cbctmc.mc.voxel_data import (
    compile_voxel_data_string,
    compile_voxel_data_string_fast,
)

densities = np.random.random((512, 512, 320)).astype(np.float32)
materials = (densities > 0.5).astype(np.uint8)

t = time.monotonic()
c2 = compile_voxel_data_string_fast(materials, densities)
print(time.monotonic() - t)

t = time.monotonic()
c1 = compile_voxel_data_string(materials, densities)
print(time.monotonic() - t)


print(c1 == c2)
# print('***')
# print(c1)
# print('***')
# print(c2)
