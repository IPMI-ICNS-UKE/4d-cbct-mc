from __future__ import annotations

import logging
import multiprocessing
import re
from functools import partial
from pathlib import Path
from typing import List, Literal, Sequence, Tuple

import numpy as np
import scipy.ndimage as ndi
import SimpleITK as sitk
from ipmi.common.logger import init_fancy_logging

from cbctmc.common_types import PathLike
from cbctmc.defaults import DefaultMCSimulationParameters as MCDefaults

logger = logging.getLogger(__name__)


class MCProjection:
    def __init__(self, data: np.ndarray, detector_pixel_size: Tuple[float, float]):
        self._data = data
        self.detector_pixel_size = detector_pixel_size

        # TODO: check all zero

    @staticmethod
    def _read_itk(filepath: PathLike) -> np.ndarray:
        data = sitk.ReadImage(str(filepath))
        data = sitk.GetArrayFromImage(data)
        data = data.squeeze(axis=0)

        return data

    @staticmethod
    def _read_raw(
        filepath: PathLike,
        n_detector_pixels: Tuple[int, int],
        n_detector_pixels_half_fan: Tuple[int, int] | None,
    ) -> np.ndarray:
        data = np.loadtxt(filepath, dtype=np.float64)
        data = data.astype(np.float32)

        data = data.reshape(*n_detector_pixels[::-1], 4)
        data = np.flip(data, axis=0)

        if n_detector_pixels_half_fan:
            data = data[:, : n_detector_pixels_half_fan[0]]

        return data

    @staticmethod
    def _is_itk_image(filepath: PathLike) -> bool:
        itk_suffixes = (
            ".mha",
            ".nii",
        )
        filepath = Path(filepath)
        for suffix in filepath.suffixes:
            if suffix in itk_suffixes:
                return True

        return False

    @classmethod
    def from_file(
        cls,
        filepath: PathLike,
        n_detector_pixels: Tuple[int, int] = MCDefaults.n_detector_pixels,
        n_detector_pixels_half_fan: Tuple[int, int]
        | None = MCDefaults.n_detector_pixels_half_fan,
        detector_pixel_size: Tuple[float, float] = (0.776, 0.776),
    ) -> "MCProjection":
        logger.info(f"Load projection {filepath}")

        if MCProjection._is_itk_image(filepath):
            data = MCProjection._read_itk(filepath)
        else:
            data = MCProjection._read_raw(
                filepath,
                n_detector_pixels=n_detector_pixels,
                n_detector_pixels_half_fan=n_detector_pixels_half_fan,
            )

        return cls(data=data, detector_pixel_size=detector_pixel_size)

    def __getitem__(self, item):
        return self._data[item]

    def __array__(self):
        return self._data


def projections_to_numpy(projections: Sequence[MCProjection]) -> np.ndarray:
    projections = np.concatenate([p[np.newaxis] for p in projections], axis=0)

    return projections


def normalize_projections(
    projections: np.ndarray,
    air_projection: np.ndarray,
    clip_to_air: bool = False,
    denoise_kernel_size: Tuple[int, int] | None = None,
) -> np.ndarray:
    if denoise_kernel_size:
        # denoise air projection using median filter of given kernel size
        air_projection = ndi.gaussian_filter(air_projection, sigma=denoise_kernel_size)
        logger.debug(
            f"Denoised air projection using gaussian filter of size {denoise_kernel_size}"
        )

    if clip_to_air:
        # clip  max of projections to air projection
        projections = np.where(
            projections > air_projection, air_projection, projections
        )
    # normalize projections according to Beer–Lambert law
    projections = np.log(air_projection / projections)

    return projections


def projections_to_itk(
    projections: Sequence[MCProjection],
    air_projection: MCProjection | None = None,
    air_projection_denoise_kernel_size: Tuple[int, int] | None = None,
    mode: Literal["total", "unscattered", "scattered"] = "total",
):
    detector_pixel_size = projections[0].detector_pixel_size

    do_air_normalization = air_projection is not None and mode == "total"
    projections = projections_to_numpy(projections)

    if projections.ndim == 4:
        if mode == "total":
            projections = projections.sum(axis=-1)
        elif mode == "unscattered":
            projections = projections[..., 0]
        elif mode == "scattered":
            projections = projections[..., 1:].sum(axis=-1)

    min_non_zero = projections[projections > 0.0].min()

    projections = np.where(projections == 0, min_non_zero, projections)
    if do_air_normalization:
        air_projection = np.asarray(air_projection)
        if air_projection.ndim == 3:
            air_projection = air_projection.sum(-1)

        # normalize projections according to Beer–Lambert law
        projections = normalize_projections(
            projections,
            air_projection,
            denoise_kernel_size=air_projection_denoise_kernel_size,
        )

    projections = sitk.GetImageFromArray(projections)
    projections.SetSpacing((detector_pixel_size[0], detector_pixel_size[1], 1))
    projections.SetOrigin(
        (
            -projections.GetSize()[0] * projections.GetSpacing()[0] / 2,
            -projections.GetSize()[1] * projections.GetSpacing()[1] / 2,
            0,
        )
    )

    return projections


def get_projections_from_folder(
    folder: PathLike,
    regex_pattern: str = r"^projection_\d{3}\.\d{6}deg$",
    n_detector_pixels: Tuple[int, int] = MCDefaults.n_detector_pixels,
    n_detector_pixels_half_fan: Tuple[int, int] = MCDefaults.n_detector_pixels_half_fan,
    detector_size: Tuple[float, float] = MCDefaults.detector_size,
) -> List[MCProjection]:
    folder = Path(folder)

    projection_filepaths = [
        p for p in sorted(folder.glob("*")) if re.match(regex_pattern, p.name)
    ]

    detector_pixel_size = (
        detector_size[0] / n_detector_pixels[0],
        detector_size[1] / n_detector_pixels[1],
    )

    with multiprocessing.Pool() as pool:
        projections = pool.map(
            partial(
                MCProjection.from_file,
                n_detector_pixels=n_detector_pixels,
                n_detector_pixels_half_fan=n_detector_pixels_half_fan,
                detector_pixel_size=detector_pixel_size,
            ),
            projection_filepaths,
        )

    return projections


if __name__ == "__main__":
    logging.getLogger("cbctmc").setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    init_fancy_logging()

    # import itk
    # import matplotlib.pyplot as plt
    #
    # from cbctmc.forward_projection import (
    #     create_geometry,
    #     prepare_image_for_rtk,
    #     project_forward,
    # )
    # from cbctmc.mc.geometry import MCGeometry
    #
    # p = MCProjection.from_file("/datalake_fast/mc_test/output_air/projection")
    #
    # fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    # ax[0].imshow(p[..., 0], clim=(0, 35))
    #
    # mc_geometry = MCGeometry.load(
    #     "/datalake_fast/4d_ct_lung_uke_artifact_free/"
    #     "022_4DCT_Lunge_amplitudebased_complete/phase_00_geometry.pkl.gz"
    # )
    #
    # n_projections = 1
    # geometry = create_geometry(start_angle=0, n_projections=n_projections)
    #
    # image = itk.imread(
    #     "/datalake_fast/4d_ct_lung_uke_artifact_free/"
    #     "022_4DCT_Lunge_amplitudebased_complete/phase_00.nii"
    # )
    #
    # image = prepare_image_for_rtk(
    #     image=mc_geometry.densities,
    #     image_spacing=mc_geometry.image_spacing,
    #     input_value_range=None,
    #     output_value_range=None,
    # )
    # forward_projection = project_forward(image, geometry=geometry)
    # # itk.imwrite(
    # #     forward_projection, "/datalake/4d_cbct_mc/CatPhantom/scan_2/catphan_fp.mha"
    # # )
    #
    # import matplotlib.pyplot as plt
    #
    # out = itk.GetArrayFromImage(forward_projection)
    # out = np.swapaxes(out, 0, 2)
    #
    # out = np.swapaxes(out, 0, 1)
    # out = np.flip(out, axis=0)
    #
    # ax[1].imshow(out[..., 0])
