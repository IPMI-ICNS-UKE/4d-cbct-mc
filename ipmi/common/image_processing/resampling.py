from typing import Tuple

import SimpleITK as sitk

from ipmi.fused_types import PositiveNumber


def resample_itk_image(
    image: sitk.Image,
    new_spacing: Tuple[PositiveNumber, PositiveNumber, PositiveNumber],
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
