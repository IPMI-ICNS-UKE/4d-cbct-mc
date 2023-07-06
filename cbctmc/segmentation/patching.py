from __future__ import annotations

from abc import ABC, abstractmethod
from math import floor
from typing import Any, List, Sequence, Tuple

import numpy as np

from cbctmc.utils import rescale_range


class BaseIndexer(ABC):
    @abstractmethod
    def calculate_index(self, reference_index: Tuple[int, ...]) -> Tuple[int, ...]:
        """Calculates one index/multiple indices based on one reference index.

        :param reference_index:
        :return:
        """


class SingleScaleIndexer(BaseIndexer):
    def calculate_index(self, reference_index: Tuple[int, ...]) -> Tuple[int, ...]:
        return reference_index


class MultiScaleIndexer(BaseIndexer):
    def __init__(
        self,
        array_shapes: Tuple[Tuple[int, ...], ...],
        reference_array_shape: Tuple[int, ...] = None,
    ):
        self.array_shapes = array_shapes
        if not reference_array_shape:
            reference_array_shape = array_shapes[0]
        self.reference_array_shape = reference_array_shape

    def calculate_index(
        self, reference_index: Tuple[int, ...]
    ) -> Tuple[Tuple[int, ...], ...]:
        multi_scale_indices = []
        for array_shape in self.array_shapes:
            scaled_index = []
            if array_shape == self.reference_array_shape:
                scaled_index = reference_index
            else:
                for idx, upper_bound, ref_upper_bound in zip(
                    reference_index, array_shape, self.reference_array_shape
                ):
                    scaled_idx = int(
                        floor(
                            rescale_range(idx, (0, ref_upper_bound), (0, upper_bound))
                        )
                    )
                    scaled_index.append(scaled_idx)
            multi_scale_indices.append(tuple(scaled_index))
        return tuple(multi_scale_indices)


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
        mean = np.full_like(self.n, fill_value=default_value)
        mean[self.n > 0] = self.k + self.sum / self.n

        return mean

    def calculate_variance(self, ddof: int = 0):
        return (self.sum_squared - self.sum**2 / self.n) / (self.n - ddof)


class PatchExtractor:
    def __init__(
        self,
        patch_shape: Tuple[int, ...],
        array_shape: Tuple[int, ...],
        color_axis: int | None = 0,
        indexer: SingleScaleIndexer | MultiScaleIndexer = None,
    ):
        self.array_shape = array_shape
        self.color_axis = color_axis
        self.patch_shape = patch_shape

        if not indexer:
            indexer = SingleScaleIndexer()
        self.indexer = indexer

    def central_to_lower_index(
        self, central_index: Tuple[int, ...], correct: bool = True
    ):
        lower_index = tuple(
            c - hps for c, hps in zip(central_index, self.half_patch_shape)
        )
        if correct:
            return self.correct_index(lower_index)
        return lower_index

    def lower_to_cental_index(self, lower_index: Tuple[int, ...], correct: bool = True):
        central_index = tuple(
            lower + half_ps
            for lower, half_ps in zip(lower_index, self.half_patch_shape)
        )
        if correct:
            return self.correct_index(central_index)
        return central_index

    def correct_index(self, index: Tuple[int, ...]):
        index = list(index)
        low_corrections = tuple(idx - l for idx, l in zip(index, self.min_lower_index))
        high_corrections = tuple(idx - h for idx, h in zip(index, self.max_upper_index))

        if all(lc == 0 for lc in low_corrections) and all(
            hc == 0 for hc in high_corrections
        ):
            return index

        for i_dim, low_correction in enumerate(low_corrections):
            if low_correction < 0:
                index[i_dim] -= low_correction

        for i_dim, high_correction in enumerate(high_corrections):
            if high_correction > 0:
                index[i_dim] -= high_correction

        return tuple(index)

    @property
    def half_patch_shape(self) -> Tuple[int, ...]:
        return tuple(ps // 2 for ps in self.patch_shape)

    @property
    def color_axis(self) -> int | None:
        return self.__color_axis

    @color_axis.setter
    def color_axis(self, value):
        if value is None:
            self.__color_axis = value
        else:
            assert -self.n_total_dims <= value < self.n_total_dims
            self.__color_axis = value if value >= 0 else value + self.n_total_dims

    @property
    def array_shape(self) -> Tuple[int, ...]:
        return self.__array_shape

    @array_shape.setter
    def array_shape(self, value: Tuple[int, ...]):
        self.__array_shape = value

    @property
    def patch_shape(self) -> Tuple[int, ...]:
        return self.__patch_shape

    @patch_shape.setter
    def patch_shape(self, value: Tuple[int, ...]):
        n_color_dim = 1 if self.color_axis is not None else 0
        assert len(value) + n_color_dim == len(self.array_shape)
        self.__patch_shape = value

    @property
    def n_total_dims(self) -> int:
        return len(self.array_shape)

    @property
    def spatial_dims(self) -> Tuple[int, ...]:
        return tuple(i for i in range(self.n_total_dims) if i != self.color_axis)

    @property
    def n_spatial_dims(self) -> int:
        return len(self.spatial_dims)

    @property
    def spatial_shape(self) -> Tuple[int, ...]:
        return tuple(self.array_shape[i] for i in self.spatial_dims)

    @property
    def min_lower_index(self) -> Tuple[int, ...]:
        return (0,) * len(self.spatial_dims)

    @property
    def max_upper_index(self) -> Tuple[int, ...]:
        upper_index = tuple(
            self.array_shape[i_dim] - p_s
            for (p_s, i_dim) in zip(self.patch_shape, self.spatial_dims)
        )
        return upper_index

    @staticmethod
    def _convert_to_tuple(value: Any, n_dims: int = 1) -> Tuple[Any, ...]:
        if isinstance(value, tuple):
            return value
        return (value,) * n_dims

    def calculate_number_ordered_indices(
        self, stride: int | Tuple[int, ...] = 1
    ) -> Tuple[int, ...]:
        stride = self._convert_to_tuple(stride, n_dims=len(self.spatial_dims))
        n = []
        for i in range(len(self.spatial_dims)):
            n.append(
                1 + (self.max_upper_index[i] - self.min_lower_index[i]) // stride[i]
            )
        return tuple(n)

    def _get_ordered_indices(
        self, stride: int | Tuple[int, ...], flush: bool = False
    ) -> List[Tuple[int, ...]]:
        stride = self._convert_to_tuple(stride, n_dims=self.n_spatial_dims)
        ranges = []
        lower = self.min_lower_index
        upper = self.max_upper_index

        for i in range(len(self.spatial_dims)):
            if flush:
                ranges.append(
                    np.arange(
                        lower[i], upper[i] + stride[i] + 1, stride[i], dtype=np.uint16
                    )
                )
            else:
                ranges.append(
                    np.arange(lower[i], upper[i] + 1, stride[i], dtype=np.uint16)
                )

        mesh = np.meshgrid(*ranges, indexing="ij", sparse=False)
        indices = np.concatenate(
            tuple(m[..., np.newaxis] for m in mesh), axis=-1
        ).reshape((-1, len(self.spatial_dims)))
        return [self.correct_index(idx) for idx in indices]

    def _calculate_multi_scale_slicings(self, reference_slicing: Tuple[slice, ...]):
        if isinstance(self.indexer, SingleScaleIndexer):
            return reference_slicing
        elif isinstance(self.indexer, MultiScaleIndexer):
            spatial_reference_slicing = self.full_to_spatial_slicing(reference_slicing)
            central_index = self.lower_to_cental_index(
                tuple(s.start for s in spatial_reference_slicing)
            )
            multi_scale_indices = self.indexer.calculate_index(central_index)

            return tuple(
                self.get_patch_slicing(
                    self.central_to_lower_index(index, correct=False)
                )
                for index in multi_scale_indices
            )

    def get_patch_slicing(self, lower_index: Tuple[int, ...]) -> Tuple[slice, ...]:
        slicing = [slice(None, None, None)] * self.n_total_dims
        for i, spatial_dim in enumerate(self.spatial_dims):
            slicing[spatial_dim] = slice(
                lower_index[i], lower_index[i] + self.patch_shape[i]
            )

        return tuple(slicing)

    def full_to_spatial_slicing(
        self, full_slicing: Tuple[slice, ...]
    ) -> Tuple[slice, ...]:
        spatial_slicing = []
        for i, spatial_dim in enumerate(self.spatial_dims):
            spatial_slicing.append(full_slicing[spatial_dim])

        return tuple(spatial_slicing)

    def extract_ordered(
        self,
        stride: int | Tuple[int, ...] | None = None,
        flush: bool = True,
        mask: np.ndarray | None = None,
    ):
        if stride is None:
            stride = self.patch_shape

        ordered_indices = self._get_ordered_indices(stride=stride, flush=flush)

        for idx in ordered_indices:
            slicing = self.get_patch_slicing(idx)
            if mask is not None:
                spatial_slicing = self.full_to_spatial_slicing(slicing)
                if not mask[spatial_slicing].any():
                    continue
            yield self._calculate_multi_scale_slicings(slicing)


if __name__ == "__main__":
    ps = PatchStitcher(array_shape=(1, 100, 200, 300), color_axis=0)
    ps.add_patch(np.ones((1, 10, 20, 30)), slicing=np.index_exp[:, :10, :20, :30])

    extractor = PatchExtractor(
        patch_shape=(10, 20, 30),
        array_shape=(1, 500, 500, 300),
        color_axis=0,
    )
