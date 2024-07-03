import os
from typing import Any, Callable, Sequence, Tuple, TypeVar, Union

import numpy as np
import torch

T = TypeVar("T")

# generic
PathLike = Union[os.PathLike, str]
Function = Callable[..., Any]

# numbers
Number = Union[int, float]
PositiveNumber = Number

# sequences
MaybeSequence = Union[T, Sequence[T]]

IntTuple = Tuple[int, ...]
FloatTuple = Tuple[float, ...]

IntTuple2D = Tuple[int, int]
FloatTuple2D = Tuple[float, float]
SlicingTuple2D = Tuple[slice, slice]


IntTuple3D = Tuple[int, int, int]
FloatTuple3D = Tuple[float, float, float]
SlicingTuple3D = Tuple[slice, slice, slice]

# NumPy / PyTorch stuff
ArrayOrTensor = Union[np.ndarray, torch.Tensor]

# PyTorch
TorchDevice = Union[str, torch.device]
