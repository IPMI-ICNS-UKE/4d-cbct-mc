import os
from multiprocessing import Pool
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

import ipmi.reconstruction.cbct.xim as xim
from ipmi.common.image_processing.resampling import resample_itk_image
from ipmi.fused_types import PathLike, PositiveNumber


def _read_xim_parallel(filepath: PathLike):
    projection, header = xim.read_xim(filepath)
    projection = projection.astype(np.float64)
    projection_mas = header["KVNormChamber"]

    return projection, projection_mas


def convert_xim(
    xim_files: Sequence[PathLike],
    air_scan_filepath: PathLike,
    detector_spacing: Tuple[float, float] = (0.388, 0.388),
    show_progress: bool = True,
) -> Tuple[sitk.Image, sitk.Image]:
    air_scan, air_header = xim.read_xim(air_scan_filepath)
    air_scan[air_scan == 0] = 1
    air_scan = air_scan.astype(np.float64)
    air_mas = air_header["KVNormChamber"]

    n_projections = len(xim_files)
    projections = np.zeros(
        (n_projections, air_scan.shape[0], air_scan.shape[1]), dtype=np.float32
    )
    projections_normalized = np.zeros_like(projections)

    with Pool() as pool:
        xim_files = pool.imap(_read_xim_parallel, xim_files, chunksize=1)
        if show_progress:
            xim_files = tqdm(
                xim_files, desc="Read and normalize projections", total=n_projections
            )
        for i, (projection, projection_mas) in enumerate(xim_files):
            # to prevent log(0) in the following normalization step
            projection[projection == 0] = 1
            projection_normalized = (
                np.log(air_scan) - np.log(projection) + np.log(projection_mas / air_mas)
            )

            projections_normalized[i] = projection_normalized
            projections[i] = projection

    projections_normalized = sitk.GetImageFromArray(projections_normalized)
    projections_normalized.SetSpacing((detector_spacing[0], detector_spacing[1], 1.0))
    size = projections_normalized.GetSize()

    projections_normalized.SetOrigin(
        (
            -size[0] * detector_spacing[0] / 2.0,
            -size[1] * detector_spacing[1] / 2.0,
            1.0,
        )
    )

    projections = sitk.GetImageFromArray(projections)
    projections.SetSpacing((detector_spacing[0], detector_spacing[1], 1.0))
    size = projections.GetSize()

    projections.SetOrigin(
        (
            -size[0] * detector_spacing[0] / 2.0,
            -size[1] * detector_spacing[1] / 2.0,
            1.0,
        )
    )

    return projections_normalized, projections


def remove_incomplete_projections(
    filepaths: Sequence[PathLike], filesize_threshold: int = 512 * 1024
):
    filepaths = [Path(f) for f in filepaths]
    cleaned_filepaths = []
    for filepath in [f for f in filepaths if "Proj" in f.name]:
        if os.path.getsize(filepath) < filesize_threshold:
            os.remove(filepath)
        else:
            cleaned_filepaths.append(filepath)

    return cleaned_filepaths


def resample_projections(
    projections: sitk.Image,
    binning: Tuple[PositiveNumber, PositiveNumber] = (2, 2),
    resampler=sitk.sitkLinear,
) -> sitk.Image:
    original_spacing = projections.GetSpacing()
    spacing_factor = (*binning, 1.0)
    new_spacing = tuple(s * f for (s, f) in zip(original_spacing, spacing_factor))
    return resample_itk_image(
        image=projections,
        new_spacing=new_spacing,
        resampler=resampler,
        default_voxel_value=0.0,
    )
