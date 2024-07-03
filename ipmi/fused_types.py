import os
from typing import Any, Callable, Union

# generic
PathLike = Union[os.PathLike, str]
Function = Callable[..., Any]

# numbers
Number = Union[int, float]
PositiveNumber = Number
