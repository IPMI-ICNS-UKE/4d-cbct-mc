from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
import torch

from cbctmc.common_types import ArrayOrTensor, IntTuple3D, Number

logger = logging.getLogger(__name__)


def hash_path(path: Path) -> str:
    return hashlib.sha1(str(path).encode()).hexdigest()


def iec61217_to_rsp(image):
    size = image.GetSize()
    spacing = image.GetSpacing()
    dimension = image.GetDimension()

    if dimension == 3:
        image.SetDirection((1, 0, 0, 0, 0, -1, 0, -1, 0))
        origin = np.subtract(
            image.GetOrigin(),
            image.TransformContinuousIndexToPhysicalPoint(
                (size[0] / 2, size[1] / 2, size[2] / 2)
            ),
        )
        origin = np.add(origin, (spacing[0] / 2, -spacing[1] / 2, -spacing[2] / 2))
    elif dimension == 4:
        image.SetDirection((1, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 1))
        origin = np.subtract(
            image.GetOrigin(),
            image.TransformContinuousIndexToPhysicalPoint(
                (size[0] / 2, size[1] / 2, size[2] / 2, size[3] / 2)
            ),
        )
        origin = np.add(
            origin, (spacing[0] / 2, -spacing[1] / 2, -spacing[2] / 2, spacing[0] / 2)
        )

    image.SetOrigin(origin)

    return image


def rescale_range(
    values: ArrayOrTensor | Number,
    input_range: Tuple,
    output_range: Tuple,
    clip: bool = True,
):
    if input_range and output_range and (input_range != output_range):
        is_tensor = isinstance(values, torch.Tensor)
        in_min, in_max = input_range
        out_min, out_max = output_range
        values = (
            ((values - in_min) * (out_max - out_min)) / (in_max - in_min)
        ) + out_min
        if clip:
            clip_func = torch.clip if is_tensor else np.clip
            values = clip_func(values, out_min, out_max)

    return values


def resample_image_spacing(
    image: sitk.Image,
    new_spacing: Tuple[float, float, float],
    resampler=sitk.sitkLinear,
    default_voxel_value=0.0,
):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [
        int(round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
        int(round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
        int(round(original_size[2] * (original_spacing[2] / new_spacing[2]))),
    ]
    resampled_img = sitk.Resample(
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
    return resampled_img


def pad_image(
    image: np.ndarray,
    target_shape: IntTuple3D,
    image_pad_value=-1000,
):
    pad_width = [(0, 0)] * image.ndim
    for i_axis in range(image.ndim):
        if target_shape[i_axis] is not None:
            if image.shape[i_axis] < target_shape[i_axis]:
                # pad
                padding = target_shape[i_axis] - image.shape[i_axis]
                padding_left = padding // 2
                padding_right = padding - padding_left
                pad_width[i_axis] = (padding_left, padding_right)

    image = np.pad(
        image,
        pad_width,
        mode="constant",
        constant_values=image_pad_value,
    )

    return image


def crop_or_pad(
    target_shape: IntTuple3D,
    image: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    image_pad_value=-1000,
    mask_pad_value=0,
    no_crop: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    n_dim = len(target_shape)
    if image is None:
        current_shape = mask.shape[-n_dim:]
    else:
        current_shape = image.shape

    pad_width = [(0, 0)] * n_dim
    cropping_slicing = [
        slice(None, None),
    ] * n_dim

    for i_axis in range(n_dim):
        if target_shape[i_axis] is not None:
            if current_shape[i_axis] < target_shape[i_axis]:
                # pad
                padding = target_shape[i_axis] - current_shape[i_axis]
                padding_left = padding // 2
                padding_right = padding - padding_left
                pad_width[i_axis] = (padding_left, padding_right)

            elif not no_crop and current_shape[i_axis] > target_shape[i_axis]:
                # crop
                cropping = current_shape[i_axis] - target_shape[i_axis]
                cropping_left = cropping // 2
                cropping_right = cropping - cropping_left

                cropping_slicing[i_axis] = slice(cropping_left, -cropping_right)

    if image is not None:
        image = np.pad(
            image,
            pad_width,
            mode="constant",
            constant_values=image_pad_value,
        )
    if mask is not None:
        extra_dims = mask.ndim - n_dim
        mask = np.pad(
            mask,
            [(0, 0)] * extra_dims + pad_width,
            mode="constant",
            constant_values=mask_pad_value,
        )

    cropping_slicing = tuple(cropping_slicing)
    if image is not None:
        image = image[cropping_slicing]
    if mask is not None:
        mask_cropping_slicing = cropping_slicing
        if mask.ndim > n_dim:
            mask_cropping_slicing = (..., *cropping_slicing)
        mask = mask[mask_cropping_slicing]

    return image, mask


def nearest_factor_pow_2(
    value: int,
    factors: Tuple[int, ...] = (2, 3, 5, 6, 7, 9),
    min_exponent: int | None = None,
    max_value: int | None = None,
    allow_smaller_value: bool = False,
) -> int:
    factors = np.array(factors)
    upper_exponents = np.ceil(np.log2(value / factors))
    lower_exponents = upper_exponents - 1

    if min_exponent:
        upper_exponents[upper_exponents < min_exponent] = np.inf
        lower_exponents[lower_exponents < min_exponent] = np.inf

    def get_distances(
        factors: Tuple[int, ...], exponents: Tuple[int, ...], max_value: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pow2_values = factors * 2**exponents
        if max_value:
            mask = pow2_values <= max_value
            pow2_values = pow2_values[mask]
            factors = factors[mask]
            exponents = exponents[mask]

        return np.abs(pow2_values - value), factors, exponents

    distances, _factors, _exponents = get_distances(
        factors=factors, exponents=upper_exponents, max_value=max_value
    )
    if len(distances) == 0:
        if allow_smaller_value:
            distances, _factors, _exponents = get_distances(
                factors=factors, exponents=lower_exponents, max_value=max_value
            )
        else:
            raise RuntimeError("Could not find a value")

    if len(distances):
        nearest_factor = _factors[np.argmin(distances)]
        nearest_exponent = _exponents[np.argmin(distances)]
    else:
        # nothing found
        pass

    return int(nearest_factor * 2**nearest_exponent)


def dict_collate(batch, noop_keys: Sequence[Any]) -> dict:
    from torch.utils.data import default_collate

    batch_torch = [
        {key: value for (key, value) in b.items() if key not in noop_keys}
        for b in batch
    ]

    batch_noop = [
        {key: value for (key, value) in b.items() if key in noop_keys} for b in batch
    ]

    batch_torch = default_collate(batch_torch)
    batch_noop = concat_dicts(batch_noop)

    return batch_torch | batch_noop


def concat_dicts(dicts: Sequence[dict], extend_lists: bool = False):
    concat = {}
    for d in dicts:
        for key, value in d.items():
            try:
                if extend_lists and isinstance(value, list):
                    concat[key].extend(value)
                else:
                    concat[key].append(value)
            except KeyError:
                if extend_lists and isinstance(value, list):
                    concat[key] = value
                else:
                    concat[key] = [value]

    return concat


def get_robust_bounding_box_3d(
    image: np.ndarray,
    bbox_range: Tuple[float, float] = (0.01, 0.99),
    padding: Tuple[int] = (0, 0, 0),
) -> Tuple[slice, slice, slice]:
    x = np.cumsum(image.sum(axis=(1, 2)))
    y = np.cumsum(image.sum(axis=(0, 2)))
    z = np.cumsum(image.sum(axis=(0, 1)))

    x = x / x[-1]
    y = y / y[-1]
    z = z / z[-1]

    x_min, x_max = np.searchsorted(x, bbox_range[0]), np.searchsorted(x, bbox_range[1])
    y_min, y_max = np.searchsorted(y, bbox_range[0]), np.searchsorted(y, bbox_range[1])
    z_min, z_max = np.searchsorted(z, bbox_range[0]), np.searchsorted(z, bbox_range[1])

    x_min, x_max = max(x_min - padding[0], 0), min(x_max + padding[0], image.shape[0])
    y_min, y_max = max(y_min - padding[1], 0), min(y_max + padding[1], image.shape[1])
    z_min, z_max = max(z_min - padding[2], 0), min(z_max + padding[2], image.shape[2])

    return np.index_exp[x_min : x_max + 1, y_min : y_max + 1, z_min : z_max + 1]


def get_folders_by_regex(root: Path, regex: str):
    pattern = re.compile(regex)

    for entry in root.iterdir():
        if entry.is_dir() and pattern.match(entry.name):
            yield entry
        else:
            logger.debug(
                f"Skipping {entry}, as it does not match "
                f"regex pattern {pattern.pattern}"
            )
