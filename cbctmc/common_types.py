import os
from typing import Union

import numpy as np
import torch

# generic
PathLike = Union[os.PathLike, str]


# NumPy / PyTorch stuff
ArrayOrTensor = Union[np.ndarray, torch.Tensor]
