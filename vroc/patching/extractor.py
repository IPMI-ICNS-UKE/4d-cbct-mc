from __future__ import annotations

import warnings
from typing import Any, List, Tuple

import numpy as np

from vroc.patching.indexer import MultiScaleIndexer, SingleScaleIndexer


class PatchExtractor:
    def __init__(
        self,
        patch_shape: Tuple[int, ...] | None,
        array_shape: Tuple[int, ...],
        color_axis: int | None = 0,
        indexer: SingleScaleIndexer | MultiScaleIndexer = None,
    ):
        self.array_shape = array_shape
        self.color_axis = color_axis
        # patch must be <= array shape
        self.patch_shape = self._get_patch_shape(patch_shape)

        if not indexer:
            indexer = SingleScaleIndexer()
        self.indexer = indexer

    def _get_patch_shape(self, patch_shape: Tuple[int, ...]):
        if patch_shape is None:
            _patch_shape = self.array_shape
        else:
            _patch_shape = tuple(
                min(p_s, a_s) for p_s, a_s in zip(patch_shape, self.array_shape)
            )

            if _patch_shape != patch_shape:
                warnings.warn(
                    f"Given patch shape of {patch_shape} was reduced to {_patch_shape} fit array shape of {self.array_shape}"
                )

        return _patch_shape

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
            return tuple(index)

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
        indices = [self.correct_index(idx) for idx in indices]
        # remove duplicates while keeping order
        indices = list(dict.fromkeys(indices))

        return indices

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
