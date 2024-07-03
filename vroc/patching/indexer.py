from __future__ import annotations

from abc import ABC, abstractmethod
from math import floor
from typing import Tuple

from vroc.helper import rescale_range


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
