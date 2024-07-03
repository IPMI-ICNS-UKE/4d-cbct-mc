from __future__ import annotations

import itertools
import logging
import pickle
import random
from functools import cache, partial
from glob import glob
from itertools import combinations
from math import prod
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import lz4.frame
import numpy as np
import SimpleITK as sitk
import torch
import yaml
from scipy.ndimage.morphology import binary_dilation
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, IterableDataset

from vroc.common_types import (
    FloatTuple2D,
    FloatTuple3D,
    IntTuple3D,
    Number,
    PathLike,
    SlicingTuple3D,
)
from vroc.dataset import DatasetMixin
from vroc.decorators import convert
from vroc.hashing import hash_path
from vroc.helper import (
    LazyLoadableList,
    nearest_factor_pow_2,
    read_landmarks,
    rescale_range,
    torch_prepare,
)
from vroc.preprocessing import (
    crop_background,
    crop_or_pad,
    resample_image_size,
    resample_image_spacing,
)

logger = logging.getLogger(__name__)


class KeypointMatcherDataset(IterableDataset, DatasetMixin):
    def __init__(
        self,
        image_filepaths: List[PathLike],
        segmentation_filepaths: List[PathLike] | List[List[PathLike]],
        patch_shape: IntTuple3D | None = None,
        image_spacing_range: Tuple | None = None,
        patches_per_image: int | float = 1,
        center_crop: bool = False,
        input_value_range: Tuple[Number, Number] | None = None,
        output_value_range: Tuple[Number, Number] | None = None,
    ):
        self.images = LazyLoadableList(
            image_filepaths, loader=KeypointMatcherDataset.load_and_preprocess
        )
        self.segmentations = LazyLoadableList(
            segmentation_filepaths,
            loader=partial(KeypointMatcherDataset.load_and_preprocess, is_mask=True),
        )

        self.patch_shape = patch_shape
        self.image_spacing_range = image_spacing_range
        self.patches_per_image = patches_per_image
        self.center_crop = center_crop
        self.input_value_range = input_value_range
        self.output_value_range = output_value_range

        if (
            self.center_crop
            and not isinstance(self.patches_per_image, int)
            and self.patches_per_image != 1
        ):
            raise ValueError("Center crop implies 1 patch per image")

    @staticmethod
    def _resample_image_spacing(
        image: sitk.Image, segmentation: sitk.Image, image_spacing: FloatTuple3D
    ) -> Tuple[sitk.Image, sitk.Image]:
        image = resample_image_spacing(
            image, new_spacing=image_spacing, default_voxel_value=-1000
        )
        segmentation = resample_image_spacing(
            segmentation,
            new_spacing=image_spacing,
            resampler=sitk.sitkNearestNeighbor,
            default_voxel_value=0,
        )

        return image, segmentation

    @staticmethod
    def sample_random_patch_3d(
        patch_shape: IntTuple3D, image_shape: IntTuple3D
    ) -> SlicingTuple3D:
        if len(patch_shape) != len(image_shape) != 3:
            raise ValueError("Please pass 3D shapes")
        upper_left_index = tuple(
            random.randint(0, s - ps) for (s, ps) in zip(image_shape, patch_shape)
        )

        return tuple(
            slice(ul, ul + ps) for (ul, ps) in zip(upper_left_index, patch_shape)
        )

    @staticmethod
    def random_rotate_image_and_segmentation(
        image: np.ndarray,
        segmentation: np.ndarray | None = None,
        spacing: Tuple[int, ...] | None = None,
    ):
        rotation_plane = random.choice(list(combinations(range(-image.ndim, 0), 2)))

        n_rotations = random.randint(0, 3)

        image = np.rot90(image, k=n_rotations, axes=rotation_plane)
        if segmentation is not None:
            segmentation = np.rot90(segmentation, k=n_rotations, axes=rotation_plane)

        if spacing:
            spacing = list(spacing)
            if n_rotations % 2:
                spacing[rotation_plane[0]], spacing[rotation_plane[1]] = (
                    spacing[rotation_plane[1]],
                    spacing[rotation_plane[0]],
                )
            spacing = tuple(spacing)

        return image, segmentation, spacing

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

            # do worker subselection for IteratableDataset
            worker_images = self.images.copy()
            worker_images.items = worker_images.items[worker_id::num_workers]
            worker_segmentations = self.segmentations.copy()
            worker_segmentations.items = worker_segmentations.items[
                worker_id::num_workers
            ]

            logger.debug(
                f"Dataset length for worker {worker_id+1}/{num_workers}: "
                f"{len(worker_images)}"
            )

        else:
            worker_images = self.images
            worker_segmentations = self.segmentations

            logger.debug(f"Dataset length: {len(worker_images)}")

        for (
            image,
            image_filepath,
            segmentation,
            segmentation_filepath,
        ) in zip(
            worker_images,
            worker_images.items,
            worker_segmentations,
            worker_segmentations.items,
        ):
            if self.image_spacing_range is not None:
                # resample to random image spacing
                image_spacing = tuple(
                    float(np.random.uniform(*spacing_range))
                    for spacing_range in self.image_spacing_range
                )
                image, segmentation = KeypointMatcherDataset._resample_image_spacing(
                    image, segmentation, image_spacing=image_spacing
                )

            else:
                image_spacing = image.GetSpacing()

            image_arr = sitk.GetArrayFromImage(image)
            segmentation_arr = sitk.GetArrayFromImage(segmentation)

            if (
                isinstance(self.patches_per_image, float)
                and 0.0 < self.patches_per_image <= 1.0
            ):
                # interpret as fraction of image volume
                image_volume = prod(image_arr.shape)
                patch_volume_volume = prod(self.patch_shape)

                patches_per_image = round(
                    (image_volume / patch_volume_volume) * self.patches_per_image
                )

                # at least 1 patch
                patches_per_image = max(patches_per_image, 1)
            else:
                patches_per_image = self.patches_per_image

            # segmentation_arr has shape of (z, y, x, n_labels)
            image_arr = image_arr.transpose((2, 1, 0))
            if segmentation_arr.ndim == 4:
                segmentation_arr = segmentation_arr.transpose((3, 2, 1, 0))
            elif segmentation_arr.ndim == 3:
                segmentation_arr = segmentation_arr.transpose((2, 1, 0))
            else:
                raise NotImplementedError

            if not self.patch_shape:
                # no patching, feed full image: find nearest pow 2 shape
                self.patch_shape = tuple(
                    nearest_factor_pow_2(s) for s in image_arr.shape
                )

            # pad if (rotated) image shape < patch shape
            # also performs center cropping if specified
            image_arr, segmentation_arr = crop_or_pad(
                image=image_arr,
                mask=segmentation_arr,
                target_shape=self.patch_shape,
                no_crop=not self.center_crop,
            )

            for i_patch in range(patches_per_image):
                while True:
                    patch_slicing = KeypointMatcherDataset.sample_random_patch_3d(
                        patch_shape=self.patch_shape, image_shape=image_arr.shape
                    )
                    segmentation_patch_slicing = patch_slicing
                    if segmentation_arr.ndim > image_arr.ndim:
                        segmentation_patch_slicing = (..., *segmentation_patch_slicing)

                    if segmentation_arr[segmentation_patch_slicing].any():
                        # patch contains segmentation
                        break

                # copy for PyTorch (negative strides are not currently supported)
                image_arr_patch = image_arr[patch_slicing].astype(np.float32, order="C")

                mask_arr_patch = segmentation_arr[segmentation_patch_slicing].astype(
                    np.float32, order="C"
                )

                image_arr_patch = rescale_range(
                    image_arr_patch,
                    input_range=self.input_value_range,
                    output_range=self.output_value_range,
                    clip=True,
                )

                image_arr_patch = image_arr_patch[np.newaxis]
                if segmentation_arr.ndim < image_arr_patch.ndim:
                    mask_arr_patch = mask_arr_patch[np.newaxis]

                data = {
                    "id": hash_path(image_filepath)[:7],
                    "image_filepath": str(image_filepath),
                    "image_filename": image_filepath.name,
                    "image": image_arr_patch,
                    "segmentation": mask_arr_patch,
                    "image_spacing": image_spacing,
                    "full_image_shape": image_arr.shape,
                    "i_patch": i_patch,
                    "n_patches": patches_per_image,
                    "patch_slicing": patch_slicing,
                }

                yield data
