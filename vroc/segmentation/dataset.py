from __future__ import annotations

import logging
import pickle
import random
from collections.abc import MutableSequence
from enum import Enum, auto
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

from vroc.common_types import (
    FloatTuple3D,
    Function,
    IntTuple3D,
    Number,
    PathLike,
    SlicingTuple3D,
)
from vroc.hashing import hash_path
from vroc.helper import nearest_factor_pow_2, rescale_range
from vroc.patching.extractor import PatchExtractor
from vroc.preprocessing import crop_or_pad, resample_image_spacing

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

    def __copy__(self):
        return LazyLoadableList(self.items.copy(), loader=self._loader)

    def copy(self):
        return self.__copy__()


class _SegmentationDatasetMixin:
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


class PatchExtractionMode(Enum):
    RANDOM = auto()
    RANDOM_BALANCED = auto()
    ORDERED = auto()


class LungCTSegmentationDataset(IterableDataset, _SegmentationDatasetMixin):
    def __init__(
        self,
        image_filepaths: List[Path],
        segmentation_filepaths: List[Path] | List[dict[str, Path]],
        labels: dict[int, str],
        patch_shape: IntTuple3D | None = None,
        image_spacing_range: Tuple[
            Tuple[float, float], Tuple[float, float], Tuple[float, float]
        ]
        | None = None,
        random_rotation: bool = True,
        add_noise: float = 100.0,
        shift_image_values: Tuple[float, float] | None = (-0.9, 1.1),
        patches_per_image: int | float = 1,
        patch_extraction_mode: PatchExtractionMode = PatchExtractionMode.RANDOM_BALANCED,
        center_crop: bool = False,
        input_value_range: Tuple[Number, Number] | None = None,
        output_value_range: Tuple[Number, Number] | None = None,
    ):
        self.images = LazyLoadableList(image_filepaths, loader=self.load_and_preprocess)
        self.segmentations = LazyLoadableList(
            segmentation_filepaths,
            loader=partial(
                self.load_and_preprocess,
                is_mask=True,
                segmentation_merge_function=self.merge_lung_ct_segmentations,
            ),
        )
        self.labels = labels
        self.n_labels = len(self.labels)

        self.patch_shape = patch_shape
        self.image_spacing_range = image_spacing_range
        # augmentation
        self.random_rotation = random_rotation
        self.add_noise = add_noise
        self.shift_image_values = shift_image_values

        self.patches_per_image = patches_per_image
        self.patch_extraction_mode = patch_extraction_mode
        self.center_crop = center_crop
        self.input_value_range = input_value_range
        self.output_value_range = output_value_range
        self._sampled_patches = []

        if (
            self.center_crop
            and not isinstance(self.patches_per_image, int)
            and self.patches_per_image != 1
        ):
            raise ValueError("Center crop implies 1 patch per image")

    def merge_lung_ct_segmentations(
        self, segmentation_filepaths: dict[str, PathLike]
    ) -> Tuple[sitk.Image, list]:
        """Merges multiple segmentations read from `segmentation_filepaths`.

        Order determines the class label (returned as `classes`)
        """
        segmentations = {}
        spacing, origin, direction = None, None, None

        for label_name, filepath in segmentation_filepaths.items():
            segmentation = sitk.ReadImage(str(filepath), sitk.sitkUInt8)
            if not spacing:
                spatial_shape = segmentation.GetSize()[::-1]
                spacing = segmentation.GetSpacing()
                origin = segmentation.GetOrigin()
                direction = segmentation.GetDirection()

                segmentations_arr = np.zeros(
                    (*spatial_shape, self.n_labels),
                    dtype=np.uint8,
                )

            segmentation = sitk.GetArrayFromImage(segmentation)
            segmentations[label_name] = segmentation

        # calculate dynamic segmentations
        # background (index 0) is everything not being any part of the lung
        segmentations["background"] = np.logical_not(
            np.stack(
                [
                    segmentations["lung_lower_lobe_left"],
                    segmentations["lung_upper_lobe_left"],
                    segmentations["lung_lower_lobe_right"],
                    segmentations["lung_middle_lobe_right"],
                    segmentations["lung_upper_lobe_right"],
                    segmentations["lung_vessels"],
                ]
            ).any(axis=0)
        )

        for label_index, label_name in self.labels.items():
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
    def add_noise_to_image(
        image: np.ndarray, max_noise_std: float = 50.0
    ) -> np.ndarray:
        if random.choice([True, False]):
            noise_std = np.random.uniform(1, max_noise_std)
            noise = np.random.normal(loc=0.0, scale=noise_std, size=image.shape)
            image = image + noise

            logger.debug(f"Added noise to image: {noise_std=}")

        return image

    @staticmethod
    def add_image_value_shift(
        image: np.ndarray, shift_relative_range: Tuple[float, float] | None
    ) -> np.ndarray:
        if shift_relative_range:
            shift = np.random.uniform(*shift_relative_range)
            image = image * shift

            logger.debug(f"Added value shift to image: {shift=}")

        return image

    @staticmethod
    def random_rotate_image_and_segmentation(
        image: np.ndarray,
        segmentation: np.ndarray | None = None,
        spacing: Tuple[int, ...] | None = None,
    ):
        rotation_plane = random.choice(list(combinations(range(-image.ndim, 0), 2)))

        n_rotations = random.randint(0, 3)

        if n_rotations > 0:
            image = np.rot90(image, k=n_rotations, axes=rotation_plane)
            if segmentation is not None:
                segmentation = np.rot90(
                    segmentation, k=n_rotations, axes=rotation_plane
                )

            if spacing:
                spacing = list(spacing)
                if n_rotations % 2:
                    spacing[rotation_plane[0]], spacing[rotation_plane[1]] = (
                        spacing[rotation_plane[1]],
                        spacing[rotation_plane[0]],
                    )
                spacing = tuple(spacing)

            logger.debug(f"Added rotation to image: {n_rotations=}, {rotation_plane=}")

        return image, segmentation, spacing

    def iterate_random_patches(
        self,
        image: np.ndarray,
        segmentation: np.ndarray,
        n_patches: int,
        balanced: bool = True,
    ):
        if not balanced:
            raise NotImplementedError

        max_iterations = n_patches * 10
        i_patch = 0
        labels_already_sampled = {i_label: 0 for i_label in self.labels.keys()}
        for i in range(max_iterations):
            patch_slicing = self.sample_random_patch_3d(
                patch_shape=self.patch_shape, image_shape=image.shape
            )
            logger.debug(f"Patch slicing: {patch_slicing}")

            # copy for PyTorch (negative strides are not currently supported)
            image_arr_patch = image[patch_slicing].astype(np.float32, order="C")

            segmentation_patch_slicing = patch_slicing
            if segmentation.ndim > image.ndim:
                segmentation_patch_slicing = (..., *segmentation_patch_slicing)
            segmentation_patch = segmentation[segmentation_patch_slicing].astype(
                np.float32, order="C"
            )

            min_count = min(labels_already_sampled.values())

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
                list(self.labels.keys()), p=sampling_probabilities
            )

            labels_present = set(
                i_label
                for i_label in self.labels.keys()
                if segmentation_patch[..., i_label, :, :, :].any()
            )

            if selected_label not in labels_present:
                logger.debug(
                    f"Skip patch without label {self.labels[selected_label]}, "
                    f"present labels: {labels_present}"
                )
                continue

            for i_label in self.labels.keys():
                if segmentation_patch[..., i_label, :, :, :].any():
                    labels_already_sampled[i_label] += 1

            yield image_arr_patch, segmentation_patch, patch_slicing

            i_patch += 1
            if i_patch >= n_patches:
                break

    def iterate_ordered_patches(
        self,
        image: np.ndarray,
        segmentation: np.ndarray,
        n_patches: int = -1,
    ):
        extractor = PatchExtractor(
            array_shape=image.shape,
            patch_shape=self.patch_shape,
            color_axis=None,
        )

        i_patch = 0
        for slicing in extractor.extract_ordered(
            stride=self.patch_shape,
            flush=True,
        ):
            image_patch = image[slicing].astype(np.float32, order="C")
            segmentation_patch = segmentation[slicing].astype(np.float32, order="C")

            yield image_patch, segmentation_patch, slicing

            i_patch += 1
            # break if n_patches is specified, i.e. >= 0, and reached
            if 0 <= n_patches <= i_patch:
                break

    # flake8: noqa: C901
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

            # do worker subselection for IteratableDataset
            images = self.images.copy()
            images.items = images.items[worker_id::num_workers]
            segmentations = self.segmentations.copy()
            segmentations.items = segmentations.items[worker_id::num_workers]

            logger.info(
                f"Dataset length for worker {worker_id+1}/{num_workers}: {len(images)}"
            )

        else:
            images = self.images
            segmentations = self.segmentations

            logger.info(f"Dataset length: {len(images)}")

        for (image, image_filepath, segmentation, segmentation_filepath,) in zip(
            images,
            images.items,
            segmentations,
            segmentations.items,
        ):
            if self.image_spacing_range is not None:
                # resample to random image spacing
                image_spacing = tuple(
                    float(np.random.uniform(*spacing_range))
                    for spacing_range in self.image_spacing_range
                )
                image, segmentation = self._resample_image_spacing(
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
                ) = self.random_rotate_image_and_segmentation(
                    image_arr, segmentation=segmentation_arr, spacing=image_spacing
                )

            if not self.patch_shape:
                # no patching, feed full image: find nearest pow 2 shape
                self.patch_shape = tuple(
                    nearest_factor_pow_2(s) for s in image_arr.shape
                )

            # pad if (rotated) image shape < patch shape
            # also performs center cropping if specified
            image_arr, seg_no_background = crop_or_pad(
                image=image_arr,
                mask=segmentation_arr[1:],
                mask_pad_value=0,
                target_shape=self.patch_shape,
                no_crop=not self.center_crop,
            )
            _, background = crop_or_pad(
                image=None,
                mask=segmentation_arr[:1],
                mask_pad_value=1,
                target_shape=self.patch_shape,
                no_crop=not self.center_crop,
            )

            segmentation_arr = np.concatenate([background, seg_no_background], axis=0)

            # iterate over patches
            if self.patch_extraction_mode == PatchExtractionMode.ORDERED:
                patch_iterator = self.iterate_ordered_patches(
                    image=image_arr,
                    segmentation=segmentation_arr,
                    n_patches=patches_per_image,
                )
            elif self.patch_extraction_mode == PatchExtractionMode.RANDOM_BALANCED:
                patch_iterator = self.iterate_random_patches(
                    image=image_arr,
                    segmentation=segmentation_arr,
                    n_patches=patches_per_image,
                    balanced=True,
                )
            else:
                raise NotImplementedError

            for i_patch, (image_patch, segmentation_patch, patch_slicing) in enumerate(
                patch_iterator
            ):
                if self.add_noise:
                    image_patch = self.add_noise_to_image(
                        image_patch, max_noise_std=self.add_noise
                    )
                if self.shift_image_values:
                    image_patch = self.add_image_value_shift(
                        image_patch, shift_relative_range=self.shift_image_values
                    )

                image_patch = rescale_range(
                    image_patch,
                    input_range=self.input_value_range,
                    output_range=self.output_value_range,
                    clip=True,
                )

                # adding color channels for PyTorch
                image_patch = image_patch[np.newaxis]
                if segmentation_arr.ndim < image_patch.ndim:
                    segmentation_patch = segmentation_patch[np.newaxis]

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
                    "image": image_patch,
                    "segmentation": segmentation_patch,
                    "image_spacing": image_spacing,
                    "full_image_shape": image_arr.shape,
                    "i_patch": i_patch,
                    "n_patches": patches_per_image,
                    "patch_slicing": patch_slicing,
                    "labels": self.labels,
                }

                yield data

                logger.debug(
                    f"Patch {i_patch}/{patches_per_image} for image {image_id}"
                )


if __name__ == "__main__":
    LABELS = {
        0: "background",  # softmax group 1
        1: "lung_lower_lobe_left",  # softmax group 1
        2: "lung_upper_lobe_left",  # softmax group 1
        3: "lung_lower_lobe_right",  # softmax group 1
        4: "lung_middle_lobe_right",  # softmax group 1
        5: "lung_upper_lobe_right",  # softmax group 1
        6: "lung_vessels",  # sigmoid
    }

    LABELS_TO_LOAD = [
        "lung_lower_lobe_left",
        "lung_upper_lobe_left",
        "lung_lower_lobe_right",
        "lung_middle_lobe_right",
        "lung_upper_lobe_right",
        "lung_vessels",
    ]

    # TOTALSEGMENTATOR
    ROOT_DIR_TOTALSEGMENTATOR = Path("/datalake_fast/totalsegmentator_mc")
    IMAGE_FILEPATHS_TOTALSEGMENTATOR = sorted(
        p for p in ROOT_DIR_TOTALSEGMENTATOR.glob("*/ct.nii.gz")
    )
    SEGMENTATION_FILEPATHS_TOTALSEGMENTATOR = [
        {
            segmentation_name: ROOT_DIR_TOTALSEGMENTATOR
            / image_filepath.parent.name
            / "segmentations"
            / f"{segmentation_name}.nii.gz"
            for segmentation_name in LABELS_TO_LOAD
        }
        for image_filepath in IMAGE_FILEPATHS_TOTALSEGMENTATOR
    ]

    dataset = LungCTSegmentationDataset(
        image_filepaths=IMAGE_FILEPATHS_TOTALSEGMENTATOR,
        segmentation_filepaths=SEGMENTATION_FILEPATHS_TOTALSEGMENTATOR,
        labels=LABELS,
        patch_shape=(128, 128, 64),
        image_spacing_range=((1.0, 2.0), (1.0, 2.0), (1.0, 2.0)),
        patches_per_image=16,
        patch_extraction_mode=PatchExtractionMode.ORDERED,
        random_rotation=True,
        add_noise=100.0,
        shift_image_values=(0.9, 1.1),
        input_value_range=(-1024, 3071),
        output_value_range=(0, 1),
    )

    for d in dataset:
        print(d["full_image_shape"])
        print(d["patch_slicing"])
