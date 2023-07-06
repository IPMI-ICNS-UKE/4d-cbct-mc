from __future__ import annotations

import logging
import pickle
import random
from collections.abc import MutableSequence
from functools import partial
from itertools import combinations
from math import prod
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import lz4.frame
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, IterableDataset

from cbctmc.common_types import (
    FloatTuple3D,
    Function,
    IntTuple3D,
    Number,
    PathLike,
    SlicingTuple3D,
)
from cbctmc.segmentation.labels import LABELS, N_LABELS
from cbctmc.utils import (
    crop_or_pad,
    hash_path,
    nearest_factor_pow_2,
    resample_image_spacing,
    rescale_range,
)

logger = logging.getLogger(__name__)


class LazyLoadableList(MutableSequence):
    def __init__(self, sequence, loader: Function | None = None):
        super().__init__()

        self._loader = loader
        self.items = list(sequence)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        if self._loader:
            item = self._loader(item)

        return item

    def __setitem__(self, index, value):
        self.items[index] = value

    def __delitem__(self, index):
        del self.items[index]

    def insert(self, index, value):
        self.items.insert(index, value)

    def append(self, value):
        self.insert(len(self) + 1, value)

    def __repr__(self):
        return repr(self.items)


class PickleDataset(Dataset):
    def __init__(self, filepaths: Sequence[PathLike], lz4_compressed: bool = True):
        self.filepaths = filepaths
        self.lz4_compressed = lz4_compressed

        super().__init__()

    def __getitem__(self, item) -> dict:
        if self.lz4_compressed:
            open_file = lz4.frame.open(self.filepaths[item], mode="rb")
        else:
            open_file = open(self.filepaths[item], mode="rb")
        with open_file as f:
            data = pickle.load(f)

        return data

    def __len__(self):
        return len(self.filepaths)


class DatasetMixin:
    @staticmethod
    def merge_segmentations(
        segmentation_filepaths: Sequence[PathLike], multi_label: bool = True
    ) -> Tuple[sitk.Image, dict]:
        """Merges multiple segmentations read from `segmentation_filepaths`.

        Order determines the class label (returned as `classes`)
        """
        segmentation_filepaths = [Path(p) for p in segmentation_filepaths]
        segmentations = None
        spacing, origin, direction = None, None, None
        classes = {}
        for i, segmentation_filepath in enumerate(segmentation_filepaths, start=1):
            segmentation = sitk.ReadImage(str(segmentation_filepath), sitk.sitkUInt8)
            if segmentations is None:
                spatial_shape = segmentation.GetSize()[::-1]
                spacing = segmentation.GetSpacing()
                origin = segmentation.GetOrigin()
                direction = segmentation.GetDirection()

                segmentations = np.zeros(
                    (*spatial_shape, len(segmentation_filepaths) + 1),
                    dtype=np.uint8,
                )
            segmentation = sitk.GetArrayFromImage(segmentation)

            segmentations[..., i] = segmentation
            classes[i] = Path(segmentation_filepath).name

        # add background class to index 0
        is_background = np.logical_not(segmentations.any(axis=-1))
        segmentations[is_background, 0] = 1

        if not multi_label:
            segmentations = segmentations.argmax(axis=-1).astype(np.uint8)

        segmentations = sitk.GetImageFromArray(segmentations)
        segmentations.SetSpacing(spacing)
        segmentations.SetOrigin(origin)
        segmentations.SetDirection(direction)

        return segmentations, classes

    @staticmethod
    def load_and_preprocess(
        filepath: PathLike | dict[PathLike],
        is_mask: bool = False,
        segmentation_merge_function: Callable | None = None,
    ) -> sitk.Image:
        if isinstance(filepath, dict) and is_mask:
            if not segmentation_merge_function:
                raise RuntimeError("No segmentation_merge_function given")
            # merge segmentations
            segmentations = segmentation_merge_function(segmentation_filepaths=filepath)
            return segmentations
        else:
            filepath = str(filepath)
            image = sitk.ReadImage(filepath)

            return image


class SegmentationDataset(IterableDataset, DatasetMixin):
    def __init__(
        self,
        image_filepaths: List[Path],
        segmentation_filepaths: List[Path] | List[dict[str, Path]],
        segmentation_merge_function: Callable,
        patch_shape: IntTuple3D | None = None,
        image_spacing_range: Tuple[
            Tuple[float, float], Tuple[float, float], Tuple[float, float]
        ]
        | None = None,
        random_rotation: bool = True,
        patches_per_image: int | float = 1,
        force_non_background: bool = False,
        force_balanced_sampling: bool = False,
        center_crop: bool = False,
        input_value_range: Tuple[Number, Number] | None = None,
        output_value_range: Tuple[Number, Number] | None = None,
    ):
        self.images = LazyLoadableList(
            image_filepaths, loader=SegmentationDataset.load_and_preprocess
        )
        self.segmentations = LazyLoadableList(
            segmentation_filepaths,
            loader=partial(
                SegmentationDataset.load_and_preprocess,
                is_mask=True,
                segmentation_merge_function=segmentation_merge_function,
            ),
        )

        self.patch_shape = patch_shape
        self.image_spacing_range = image_spacing_range
        self.random_rotation = random_rotation
        self.patches_per_image = patches_per_image
        self.force_non_background = force_non_background
        self.force_balanced_sampling = force_balanced_sampling
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
    def merge_mc_segmentations(
        segmentation_filepaths: dict[str, PathLike]
    ) -> Tuple[sitk.Image, list]:
        """Merges multiple segmentations read from `segmentation_filepaths`.

        Order determines the class label (returned as `classes`)
        """
        segmentations = {}
        spacing, origin, direction = None, None, None

        for label_name, filepath in segmentation_filepaths.items():
            try:
                segmentation = sitk.ReadImage(str(filepath), sitk.sitkUInt8)
            except RuntimeError:
                # not found, continue
                continue
            if not spacing:
                spatial_shape = segmentation.GetSize()[::-1]
                spacing = segmentation.GetSpacing()
                origin = segmentation.GetOrigin()
                direction = segmentation.GetDirection()

                segmentations_arr = np.zeros(
                    (*spatial_shape, N_LABELS),
                    dtype=np.uint8,
                )

            segmentation = sitk.GetArrayFromImage(segmentation)
            segmentations[label_name] = segmentation

        # calculate dynamic segmentations
        # background (not body) (index 0)
        segmentations["background"] = segmentations["body"] == 0

        # other tissue inside body
        segmentations["other"] = np.logical_not(
            np.stack(
                [
                    segmentations["upper_body_bones"],
                    segmentations["upper_body_muscles"],
                    segmentations["upper_body_fat"],
                    segmentations["liver"],
                    segmentations["stomach"],
                    segmentations["lung"],
                    segmentations["background"],
                ]
            ).any(axis=0)
        )

        for label_index, label_name in LABELS.items():
            segmentations_arr[..., label_index] = segmentations[label_name]

        segmentations = sitk.GetImageFromArray(segmentations_arr)
        segmentations.SetSpacing(spacing)
        segmentations.SetOrigin(origin)
        segmentations.SetDirection(direction)

        return segmentations

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

    # flake8: noqa: C901
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

            self.images.items = self.images.items[worker_id::num_workers]
            self.segmentations.items = self.segmentations.items[worker_id::num_workers]

        for (
            image,
            image_filepath,
            segmentation,
            segmentation_filepath,
        ) in zip(
            self.images,
            self.images.items,
            self.segmentations,
            self.segmentations.items,
        ):
            if self.image_spacing_range is not None:
                # resample to random image spacing
                image_spacing = tuple(
                    float(np.random.uniform(*spacing_range))
                    for spacing_range in self.image_spacing_range
                )
                image, segmentation = SegmentationDataset._resample_image_spacing(
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

            if self.random_rotation:
                (
                    image_arr,
                    segmentation_arr,
                    image_spacing,
                ) = SegmentationDataset.random_rotate_image_and_segmentation(
                    image_arr, segmentation=segmentation_arr, spacing=image_spacing
                )

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

            max_iterations = patches_per_image * 10
            i_patch = 0
            labels_already_sampled = {i_label: 0 for i_label in LABELS.keys()}
            for i in range(max_iterations):
                patch_slicing = SegmentationDataset.sample_random_patch_3d(
                    patch_shape=self.patch_shape, image_shape=image_arr.shape
                )

                # copy for PyTorch (negative strides are not currently supported)
                image_arr_patch = image_arr[patch_slicing].astype(np.float32, order="C")

                segmentation_patch_slicing = patch_slicing
                if segmentation_arr.ndim > image_arr.ndim:
                    segmentation_patch_slicing = (..., *segmentation_patch_slicing)
                mask_arr_patch = segmentation_arr[segmentation_patch_slicing].astype(
                    np.float32, order="C"
                )

                if self.force_non_background:
                    if not mask_arr_patch[..., 1:, :, :, :].any():
                        continue

                if self.force_balanced_sampling:
                    min_count = min(labels_already_sampled.values())
                    # sampling_probabilities = np.array(
                    #     [
                    #         1 / (count**2 + 1e-6)
                    #         for count in labels_already_sampled.values()
                    #     ]
                    # )
                    sampling_probabilities = np.array(
                        [
                            1 if count == min_count else 0
                            for count in labels_already_sampled.values()
                        ]
                    )
                    sampling_probabilities = (
                        sampling_probabilities / sampling_probabilities.sum()
                    )

                    selected_label = np.random.choice(
                        list(LABELS.keys()), p=sampling_probabilities
                    )

                    labels_present = set(
                        i_label
                        for i_label in LABELS.keys()
                        if mask_arr_patch[..., i_label, :, :, :].any()
                    )

                    if selected_label not in labels_present:
                        continue

                    for i_label in LABELS.keys():
                        if mask_arr_patch[..., i_label, :, :, :].any():
                            labels_already_sampled[i_label] += 1

                image_arr_patch = rescale_range(
                    image_arr_patch,
                    input_range=self.input_value_range,
                    output_range=self.output_value_range,
                    clip=True,
                )

                image_arr_patch = image_arr_patch[np.newaxis]
                if segmentation_arr.ndim < image_arr_patch.ndim:
                    mask_arr_patch = mask_arr_patch[np.newaxis]

                image_id = hash_path(image_filepath)[:7]
                slicing_str = "_".join(
                    f"{s.start:03d}:{s.stop:03d}" for s in patch_slicing
                )

                patch_id = f"{image_id}__{slicing_str}"
                data = {
                    "image_id": image_id,
                    "patch_id": patch_id,
                    "image_filepath": str(image_filepath),
                    "image_filename": image_filepath.name,
                    "image": image_arr_patch,
                    "segmentation": mask_arr_patch,
                    "image_spacing": image_spacing,
                    "full_image_shape": image_arr.shape,
                    "i_patch": i_patch,
                    "n_patches": patches_per_image,
                    "patch_slicing": patch_slicing,
                    "labels": LABELS,
                }

                yield data

                i_patch += 1
                if i_patch + 1 == patches_per_image:
                    break

    def compile_and_save(self, folder: PathLike):
        folder = Path(folder)
        folder.mkdir(exist_ok=True)
        for data in self:
            logger.info(
                f"Save image {data['image_id']}: "
                f"patch {data['i_patch'] + 1}/{data['n_patches']}"
            )
            filepath = folder / f"{data['patch_id']}.pkl"

            with lz4.frame.open(filepath, mode="wb", compression_level=0) as f:
                pickle.dump(data, f)
