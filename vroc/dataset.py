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
from ipmi.common.logger import LoggerMixin
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


class DatasetMixin:
    @staticmethod
    def load_and_preprocess(
        filepath: PathLike | Sequence[PathLike], is_mask: bool = False
    ) -> sitk.Image:
        if isinstance(filepath, (tuple, list)):
            return [
                DatasetMixin.load_and_preprocess(f, is_mask=is_mask) for f in filepath
            ]
        else:
            filepath = str(filepath)
            image = sitk.ReadImage(filepath)

            return image


class LungCTRegistrationDataset(DatasetMixin):
    def __init__(self):
        super().__init__()

        self._image_pairs = []

    @convert("moving_image", converter=Path)
    @convert("fixed_image", converter=Path)
    @convert("moving_mask", converter=Path)
    @convert("fixed_mask", converter=Path)
    def append_image_pair(
        self,
        moving_image: PathLike,
        fixed_image: PathLike,
        moving_mask: PathLike,
        fixed_mask: PathLike,
    ):
        image_pair = {
            "moving_image": moving_image,
            "fixed_image": fixed_image,
            "moving_mask": moving_mask,
            "fixed_mask": fixed_mask,
        }
        self._image_pairs.append(image_pair)


class SegmentationDataset(IterableDataset, DatasetMixin):
    def __init__(
        self,
        image_filepaths: List[PathLike],
        segmentation_filepaths: List[PathLike] | List[List[PathLike]],
        segmentation_labels: List[Sequence[int | None]] | None = None,
        multi_label: bool = False,
        patch_shape: IntTuple3D | None = None,
        image_spacing_range: Tuple | None = None,
        random_rotation: bool = True,
        patches_per_image: int | float = 1,
        center_crop: bool = False,
        input_value_range: Tuple[Number, Number] | None = None,
        output_value_range: Tuple[Number, Number] | None = None,
    ):
        self.images = LazyLoadableList(
            image_filepaths, loader=SegmentationDataset.load_and_preprocess
        )
        self.segmentations = LazyLoadableList(
            segmentation_filepaths,
            loader=partial(SegmentationDataset.load_and_preprocess, is_mask=True),
        )

        self.segmentation_labels = segmentation_labels or [None] * len(
            self.segmentations
        )
        self.multi_label = multi_label
        self.patch_shape = patch_shape
        self.image_spacing_range = image_spacing_range
        self.random_rotation = random_rotation
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
            worker_segmentation_labels = self.segmentation_labels[
                worker_id::num_workers
            ]

            logger.debug(
                f"Dataset length for worker {worker_id+1}/{num_workers}: "
                f"{len(worker_images)}"
            )

        else:
            worker_images = self.images
            worker_segmentations = self.segmentations
            worker_segmentation_labels = self.segmentation_labels

            logger.debug(f"Dataset length: {len(worker_images)}")

        for (
            image,
            image_filepath,
            segmentation,
            segmentation_filepath,
            segmentation_labels,
        ) in zip(
            worker_images,
            worker_images.items,
            worker_segmentations,
            worker_segmentations.items,
            worker_segmentation_labels,
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

            if segmentation_labels is not None:
                segmentation_arr = np.isin(
                    segmentation_arr, segmentation_labels
                ).astype(np.uint8)

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

            for i_patch in range(patches_per_image):
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


class Lung4DArtifactBoostingDataset(DatasetMixin, Dataset):
    def __init__(
        self,
        patient_folders: List[Path],
        i_worker: Optional[int] = None,
        n_worker: Optional[int] = None,
        is_train: bool = True,
        train_size: float = 1.0,
    ):
        self.patient_folders = patient_folders

        if train_size < 1.0:
            train_patients, test_patients = train_test_split(
                patient_folders, train_size=train_size, random_state=1337
            )
            self.patient_folders = train_patients if is_train else test_patients
        else:
            self.patient_folders = patient_folders

        self.filepaths = [
            self.fetch_filepaths(patient_folder)
            for patient_folder in self.patient_folders
        ]
        # convert list if lists to flattened list
        self.filepaths = list(itertools.chain.from_iterable(self.filepaths))

        self.i_worker = i_worker
        self.n_worker = n_worker

        if self.i_worker is not None and self.n_worker is not None:
            self.filepaths = self.filepaths[self.i_worker :: self.n_worker]

    @staticmethod
    def fetch_filepaths(patient_folder: Path) -> List[dict]:
        image_data_filepath = patient_folder / "image_data.pkl.lz4"
        registration_data_filepaths = patient_folder.glob("registration*")

        return [
            {"image_data": image_data_filepath, "registration_data": registration_data}
            for registration_data in registration_data_filepaths
        ]

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, item):
        image_data_filepath = self.filepaths[item]["image_data"]
        registration_data_filepath = self.filepaths[item]["registration_data"]

        with lz4.frame.open(image_data_filepath, "rb") as f:
            image_data = pickle.load(f)
        with lz4.frame.open(registration_data_filepath, "rb") as f:
            registration_data = pickle.load(f)

        return image_data | registration_data


class Lung4DCTRegistrationDataset(DatasetMixin, LoggerMixin, Dataset):
    def __init__(
        self,
        patient_folders: Sequence[Path],
        phases: Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        input_value_range: FloatTuple2D | None = (-1024.0, 3071.0),
        output_value_range: FloatTuple2D | None = (-1024.0, 3071.0),
        output_image_spacing: FloatTuple3D | None = None,
        ignore_missing_files: bool = True,
    ):
        self.patient_folders = patient_folders
        self.phases = phases
        self.input_value_range = input_value_range
        self.output_value_range = output_value_range
        self.output_image_spacing = output_image_spacing
        self.ignore_missing_files = ignore_missing_files

        self.filepaths = self.collect_filepaths(self.patient_folders)

    @classmethod
    def from_folder(
        cls,
        folder: PathLike,
        phases: Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        input_value_range: FloatTuple2D | None = (-1024.0, 3071.0),
        output_value_range: FloatTuple2D | None = (-1024.0, 3071.0),
        output_image_spacing: FloatTuple3D | None = None,
        ignore_missing_files: bool = True,
    ):
        patient_folders = sorted([p for p in Path(folder).glob("*") if p.is_dir()])

        return cls(
            patient_folders=patient_folders,
            phases=phases,
            input_value_range=input_value_range,
            output_value_range=output_value_range,
            output_image_spacing=output_image_spacing,
            ignore_missing_files=ignore_missing_files,
        )

    def _assert_exists(
        self, path: Path, ignore_missing_files: bool = False
    ) -> Path | None:
        if not path.exists():
            if ignore_missing_files:
                self.logger.warn(f"File {path} not found. Skipping.")
                path = None
            else:
                raise FileNotFoundError(f"File {path} not found")

        return path

    def collect_filepaths(self, folders: Sequence[Path]) -> List[dict]:
        collected = []
        for folder in folders:
            _collected = {
                "patient": folder.name,
                "folder": folder,
                "images": {},
                "masks": {},
                "additional_masks": {},
                "keypoints": {},
                "landmarks": {},
                "meta": {},
            }

            # add phase images
            image_folder = folder  # / "images"
            for phase in self.phases:
                _collected["images"][phase] = self._assert_exists(
                    image_folder / f"phase_{phase:02d}.nii",
                    ignore_missing_files=self.ignore_missing_files,
                )

            # add masks
            masks_folder = folder / "masks"
            for phase in self.phases:
                _collected["masks"][phase] = self._assert_exists(
                    masks_folder / f"lung_phase_{phase:02d}.nii.gz",
                    ignore_missing_files=self.ignore_missing_files,
                )

            # add masks
            for phase in self.phases:
                _collected["additional_masks"][phase] = self._assert_exists(
                    # masks_folder / f"lung_phase_{phase:02d}.nii.gz",
                    masks_folder / f"lung_phase_{phase:02d}" / "lung_vessels.nii.gz",
                    ignore_missing_files=self.ignore_missing_files,
                )

            # add keypoints and landmarks
            for keypoint_or_landmark in ("keypoints", "landmarks"):
                for moving_phase, fixed_phase in itertools.product(
                    self.phases, self.phases
                ):
                    if moving_phase == fixed_phase:
                        # skip identity
                        continue

                    _collected[keypoint_or_landmark][(moving_phase, fixed_phase)] = []

                    for moving_or_fixed in ("moving", "fixed"):
                        filename = f"{moving_or_fixed}_{keypoint_or_landmark}_{moving_phase:02d}_to_{fixed_phase:02d}"
                        _collected[keypoint_or_landmark][
                            (moving_phase, fixed_phase)
                        ].append(
                            self._assert_exists(
                                folder / keypoint_or_landmark / f"{filename}.csv",
                                ignore_missing_files=self.ignore_missing_files,
                            )
                        )
                    _collected[keypoint_or_landmark][
                        (moving_phase, fixed_phase)
                    ] = tuple(
                        _collected[keypoint_or_landmark][(moving_phase, fixed_phase)]
                    )

            if (meta_filepath := folder / "metadata.yaml").exists():
                # load metadata if file exists, otherwise empty
                with open(meta_filepath, "r") as f:
                    _collected["meta"] = yaml.safe_load(f)

            collected.append(_collected)

        return collected

    @staticmethod
    def resample_image_spacing(
        image: sitk.Image,
        new_spacing: FloatTuple3D,
        resampler=sitk.sitkLinear,
        default_voxel_value: Number = 0.0,
    ):
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        new_size = [
            int(round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
            int(round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
            int(round(original_size[2] * (original_spacing[2] / new_spacing[2]))),
        ]
        resampled = sitk.Resample(
            image,
            new_size,
            sitk.Transform(),
            resampler,
            image.GetOrigin(),
            new_spacing,
            image.GetDirection(),
            default_voxel_value,
            image.GetPixelID(),
        )
        return resampled

    def _load_image(
        self, path: Path, is_mask: bool = False
    ) -> Tuple[np.ndarray | None, dict]:
        if not path:
            image, meta = None, {}
        else:
            image = sitk.ReadImage(str(path))

            meta = {
                "filepath": str(path),
                "original_image_spacing": image.GetSpacing(),
                "original_image_direction": image.GetDirection(),
                "original_image_origin": image.GetOrigin(),
            }

            if (
                self.output_image_spacing
                and self.output_image_spacing != image.GetSpacing()
            ):
                interpolator = sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
                default_voxel_value = 0 if is_mask else self.input_value_range[0]

                image = self.resample_image_spacing(
                    image=image,
                    new_spacing=self.output_image_spacing,
                    resampler=interpolator,
                    default_voxel_value=default_voxel_value,
                )

            meta.update(
                {
                    "image_spacing": image.GetSpacing(),
                    "image_direction": image.GetDirection(),
                    "image_origin": image.GetOrigin(),
                }
            )

            image = sitk.GetArrayFromImage(image)
            image = np.swapaxes(image, 0, 2)

            if is_mask:
                image = np.asarray(image, dtype=bool)
            else:
                image = np.asarray(image, dtype=np.float32)
                if self.input_value_range and self.output_value_range:
                    image = rescale_range(
                        image,
                        input_range=self.input_value_range,
                        output_range=self.output_value_range,
                        clip=True,
                    )

        return image, meta

    def _load_keypoints(self, path: Path) -> np.ndarray:
        return read_landmarks(path) if path else None

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, item):
        data = {
            "patient": self.filepaths[item]["patient"],
            "folder": self.filepaths[item]["folder"],
            "meta": self.filepaths[item]["meta"],
            "images": {},
            "masks": {},
            "additional_masks": {},
            "keypoints": {},
            "landmarks": {},
        }

        for phase in self.phases:
            image, meta = self._load_image(self.filepaths[item]["images"][phase])
            data["images"][phase] = {"data": image, "meta": meta}

            mask, meta = self._load_image(
                self.filepaths[item]["masks"][phase], is_mask=True
            )
            data["masks"][phase] = {"data": mask, "meta": meta}

            add_mask, meta = self._load_image(
                self.filepaths[item]["additional_masks"][phase], is_mask=True
            )
            data["additional_masks"][phase] = {"data": add_mask, "meta": meta}

            for other_phase in self.phases:
                if phase == other_phase:
                    continue
                # moving phase, fixed phase
                phase_combination = (phase, other_phase)
                for keypoint_or_landmark in ("keypoints", "landmarks"):
                    moving_keypoints = self._load_keypoints(
                        self.filepaths[item][keypoint_or_landmark][
                            (phase, other_phase)
                        ][0]
                    )
                    fixed_keypoints = self._load_keypoints(
                        self.filepaths[item][keypoint_or_landmark][
                            (phase, other_phase)
                        ][1]
                    )

                    if moving_keypoints is None and fixed_keypoints is None:
                        keypoint_pair = None
                    else:
                        keypoint_pair = (
                            moving_keypoints,
                            fixed_keypoints,
                        )
                    data[keypoint_or_landmark][phase_combination] = keypoint_pair

        return data


class NLSTDataset(DatasetMixin, Dataset):
    def __init__(
        self,
        root_dir: PathLike,
        i_worker: Optional[int] = None,
        n_worker: Optional[int] = None,
        is_train: bool = True,
        train_size: float = 1.0,
        dilate_masks: int = 0,
        as_sitk: bool = False,
        unroll_vector_fields: bool = False,
    ):
        self.root_dir = root_dir
        filepaths = self.fetch_filepaths(self.root_dir)

        if train_size < 1.0:
            train_filepaths, test_filepaths = train_test_split(
                filepaths, train_size=train_size, random_state=1337
            )
            self.filepaths = train_filepaths if is_train else test_filepaths
        else:
            self.filepaths = filepaths

        self.i_worker = i_worker
        self.n_worker = n_worker
        self.dilate_masks = dilate_masks
        self.as_sitk = as_sitk
        self.unroll_vector_fields = unroll_vector_fields

        if i_worker is not None and n_worker is not None:
            self.filepaths = self.filepaths[self.i_worker :: self.n_worker]

        if self.unroll_vector_fields:
            unrolled = []
            for _filepaths in self.filepaths:
                vector_fields = _filepaths.pop("precomputed_vector_fields")

                for vector_field in vector_fields:
                    unrolled.append(
                        {**_filepaths, "precomputed_vector_fields": [vector_field]}
                    )
            self.filepaths = unrolled

    @staticmethod
    @convert("root_dir", Path)
    def fetch_filepaths(
        root_dir: PathLike,
        image_folder: str = "imagesTr",
        mask_folder: str = "masksTr",
        keypoints_folder: str = "keypointsTr",
        vector_fields_folder: str = "detailed_boosting_dataa",
    ):
        root_dir: Path

        image_path = root_dir / image_folder
        mask_path = root_dir / mask_folder
        keypoints_path = root_dir / keypoints_folder
        vector_fields_path = root_dir / vector_fields_folder

        fixed_images = sorted(image_path.glob("*0000.nii.gz"))
        moving_images = sorted(image_path.glob("*0001.nii.gz"))
        fixed_masks = sorted(mask_path.glob("*0000.nii.gz"))
        moving_masks = sorted(mask_path.glob("*0001.nii.gz"))
        fixed_keypoints = sorted(keypoints_path.glob("*0000.csv"))
        moving_keypoints = sorted(keypoints_path.glob("*0001.csv"))
        precomputed_vector_fields = sorted(vector_fields_path.glob("*pkl"))

        filepath_lists = (
            fixed_images,
            moving_images,
            fixed_masks,
            moving_masks,
            fixed_keypoints,
            moving_keypoints,
        )

        if len(lengths := set(len(l) for l in filepath_lists)) > 1:
            raise RuntimeError("File mismatch")
        length = lengths.pop()

        def get_matching_vector_fields(image_filepath: Path) -> List[Path]:
            # this is NLST_0001, etc.
            name = image_filepath.name[:9]
            return [p for p in precomputed_vector_fields if p.name.startswith(name)]

        return [
            {
                "fixed_image": fixed_images[i],
                "moving_image": moving_images[i],
                "fixed_mask": fixed_masks[i],
                "moving_mask": moving_masks[i],
                "fixed_keypoints": fixed_keypoints[i],
                "moving_keypoints": moving_keypoints[i],
                "precomputed_vector_fields": get_matching_vector_fields(
                    fixed_images[i]
                ),
            }
            for i in range(length)
        ]

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, item):
        fixed_image = NLSTDataset.load_and_preprocess(
            self.filepaths[item]["fixed_image"]
        )
        moving_image = NLSTDataset.load_and_preprocess(
            self.filepaths[item]["moving_image"]
        )
        fixed_mask = NLSTDataset.load_and_preprocess(
            self.filepaths[item]["fixed_mask"], is_mask=True
        )
        moving_mask = NLSTDataset.load_and_preprocess(
            self.filepaths[item]["moving_mask"], is_mask=True
        )

        moving_keypoints = read_landmarks(
            self.filepaths[item]["moving_keypoints"], sep=","
        )
        fixed_keypoints = read_landmarks(
            self.filepaths[item]["fixed_keypoints"], sep=","
        )

        image_spacing = fixed_image.GetSpacing()
        image_shape = fixed_image.GetSize()
        if not self.as_sitk:
            image_spacing = np.array(image_spacing)

            moving_image = sitk.GetArrayFromImage(moving_image)
            fixed_image = sitk.GetArrayFromImage(fixed_image)
            moving_mask = sitk.GetArrayFromImage(moving_mask)
            fixed_mask = sitk.GetArrayFromImage(fixed_mask)

            # flip axes
            moving_image = np.swapaxes(moving_image, 0, 2)
            fixed_image = np.swapaxes(fixed_image, 0, 2)
            moving_mask = np.swapaxes(moving_mask, 0, 2)
            fixed_mask = np.swapaxes(fixed_mask, 0, 2)

            if self.dilate_masks:
                moving_mask = binary_dilation(
                    moving_mask.astype(np.uint8), iterations=1
                ).astype(np.uint8)
                fixed_mask = binary_dilation(
                    fixed_mask.astype(np.uint8), iterations=1
                ).astype(np.uint8)

            image_shape = np.array(image_shape)

            fixed_image = np.asarray(fixed_image[np.newaxis], dtype=np.float32)
            moving_image = np.asarray(moving_image[np.newaxis], dtype=np.float32)
            fixed_mask = np.asarray(fixed_mask[np.newaxis], dtype=np.float32)
            moving_mask = np.asarray(moving_mask[np.newaxis], dtype=np.float32)

            if self.filepaths[item]["precomputed_vector_fields"]:
                random_vector_field = random.choice(
                    self.filepaths[item]["precomputed_vector_fields"]
                )

                with open(random_vector_field, "rb") as f:
                    random_vector_field = pickle.load(f)
            else:
                random_vector_field = None

            data = {
                "moving_image_name": str(
                    self.filepaths[item]["moving_image"].relative_to(self.root_dir)
                ),
                "fixed_image_name": str(
                    self.filepaths[item]["fixed_image"].relative_to(self.root_dir)
                ),
                "moving_image": moving_image,
                "fixed_image": fixed_image,
                "moving_mask": moving_mask,
                "fixed_mask": fixed_mask,
                "moving_keypoints": moving_keypoints,
                "fixed_keypoints": fixed_keypoints,
                "image_shape": image_shape,
                "image_spacing": image_spacing,
                "precomputed_vector_field": random_vector_field,
            }

            return data


class AutoencoderDataset(DatasetMixin, Dataset):
    def __init__(self, filepaths: List[PathLike]):
        self.filepaths = filepaths

    @staticmethod
    def load_and_preprocess(filepath: PathLike, is_mask: bool = False) -> np.ndarray:
        filepath = str(filepath)
        dtype = sitk.sitkUInt8 if is_mask else sitk.sitkFloat32
        image = sitk.ReadImage(filepath, dtype)
        image = crop_background(image)
        image = resample_image_size(image, new_size=(128, 128, 128))
        return sitk.GetArrayFromImage(image)

    @staticmethod
    @convert("root_dirs", lambda paths: [Path(p) for p in paths])
    def fetch_filepaths(root_dirs: List[PathLike]):
        root_dirs: List[Path]

        filepaths = []
        allowed_extensions = [".nii.gz", ".mhd", ".mha"]
        for root_dir in root_dirs:
            for ext in allowed_extensions:
                filepaths.extend(
                    sorted(
                        Path(p) for p in glob(str(Path.joinpath(root_dir, "*" + ext)))
                    )
                )

        return filepaths

    def __len__(self):
        return len(self.filepaths)

    @cache
    def __getitem__(self, item):
        image_path = self.filepaths[item]
        image = self.load_and_preprocess(image_path)
        image = rescale_range(image, input_range=(-1024, 3071), output_range=(0, 1))

        image = torch_prepare(image)
        return image, str(image_path)


if __name__ == "__main__":
    dataset_dirlab_copdgene = Lung4DCTRegistrationDataset.from_folder(
        "/datalake/dirlab_copdgene/converted",
        phases=(0, 5),
        output_image_spacing=(1.0, 1.0, 1.0),
    )
    dataset_dirlab_4dct = Lung4DCTRegistrationDataset.from_folder(
        "/datalake/dirlab_4dct/converted", output_image_spacing=(1.0, 1.0, 1.0)
    )

    d = dataset_dirlab_copdgene[0]
    dd = dataset_dirlab_4dct[0]
