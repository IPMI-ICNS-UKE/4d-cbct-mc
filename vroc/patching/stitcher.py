from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


class PatchStitcher:
    def __init__(
        self,
        array_shape: Tuple[int, ...],
        color_axis: int = 0,
        dtype=np.float32,
        max_expected_overlap: int = 255,
    ):
        self.array_shape = array_shape
        self.color_axis = color_axis

        self.dtype = dtype
        self.max_expected_overlap = max_expected_overlap
        if self.max_expected_overlap < 2**8:
            self._n_dtype = np.uint8
        elif self.max_expected_overlap < 2**16:
            self._n_dtype = np.uint16
        else:
            self._n_dtype = np.uint32
        self._hard_overlap_limit = np.iinfo(self._n_dtype).max
        self._unsigned_dtype = None

        self.n_patches_added = 0

        self.reset()

    @property
    def array_shape(self):
        return self.__array_shape

    @array_shape.setter
    def array_shape(self, value):
        self.__array_shape = value

    @property
    def color_axis(self):
        return self.__color_axis

    @color_axis.setter
    def color_axis(self, value):
        if value is None:
            self.__color_axis = value
        else:
            assert -self.n_total_dims <= value < self.n_total_dims
            self.__color_axis = value if value >= 0 else value + self.n_total_dims

    @property
    def n_total_dims(self):
        return len(self.array_shape)

    def reset(self):
        self.k = np.zeros(self.array_shape, dtype=self.dtype)
        self.n = np.zeros(self.array_shape, dtype=self._n_dtype)
        self.sum = np.zeros(self.array_shape, dtype=self.dtype)
        self.sum_squared = np.zeros(
            self.array_shape, dtype=self._unsigned_dtype or self.dtype
        )

        self.n_patches_added = 0

    def print_internal_stats(self):
        stats = (
            f"internal min/max stats:\n"
            f"n: min={self.n.min()}, max={self.n.max()}\n"
            f"k: min={self.k.min()}, max={self.k.max()}\n"
            f"sum: min={self.sum.min()}, max={self.sum.max()}\n"
            f"sum_squared: min={self.sum_squared.min()}, max={self.sum_squared.max()}"
        )
        print(stats)

    @property
    def coverage(self):
        return self.n

    def add_patch(self, data: np.ndarray, slicing: Tuple[slice, ...]):
        with np.errstate(over="raise", under="raise"):
            n_masking = self.n[slicing] == 0
            self.k[slicing][n_masking] = data[n_masking]
            self.n[slicing] += 1
            diff = data - self.k[slicing]

            self.sum[slicing] += diff
            self.sum_squared[slicing] += diff**2

    def add_patches(
        self, data: Sequence[np.ndarray], slicings: Sequence[Tuple[slice, ...]]
    ):
        for patch, slicing in zip(data, slicings):
            self.add_patch(patch, slicing)

    def calculate_mean(self, default_value: float = 0.0):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.nan_to_num(self.k + self.sum / self.n, nan=default_value)

    def calculate_variance(self, ddof: int = 0):
        return (self.sum_squared - self.sum**2 / self.n) / (self.n - ddof)
