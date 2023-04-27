from __future__ import annotations

import logging
import multiprocessing
import re
from functools import partial
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
from ipmi.common.logger import init_fancy_logging

from cbctmc.common_types import PathLike
from cbctmc.defaults import DefaultMCSimulationParameters as MCDefaults

logger = logging.getLogger(__name__)


class MCProjection:
    def __init__(self, data: np.ndarray, detector_pixel_size: Tuple[float, float]):
        self._data = data
        self.detector_pixel_size = detector_pixel_size

    @classmethod
    def from_file(
        cls,
        filepath: PathLike,
        n_detector_pixels: Tuple[int, int] = MCDefaults.n_detector_pixels,
        n_detector_pixels_half_fan: Tuple[
            int, int
        ] = MCDefaults.n_detector_pixels_half_fan,
        detector_pixel_size: Tuple[float, float] = (0.776, 0.776),
    ) -> "MCProjection":
        logger.info(f"Load projection {filepath}")
        projection = np.loadtxt(filepath, dtype=np.float64)
        projection = projection.astype(np.float32)

        projection = projection.reshape(*n_detector_pixels[::-1], 4)
        projection = np.flip(projection, axis=0)

        projection = projection[:, : n_detector_pixels_half_fan[0]]

        return cls(data=projection, detector_pixel_size=detector_pixel_size)

    def __getitem__(self, item):
        return self._data[item]

    def __array__(self):
        return self._data


def projections_to_numpy(projections: Sequence[MCProjection]) -> np.ndarray:
    projections = np.concatenate([p[np.newaxis] for p in projections], axis=0)

    return projections


def projections_to_itk(
    projections: Sequence[MCProjection], air_projection: MCProjection | None = None
):
    detector_pixel_size = projections[0].detector_pixel_size

    has_air_projection = air_projection is not None
    projections = projections_to_numpy(projections)
    # sum over all photons (non-scattered and scattered)
    if projections.ndim == 4:
        projections = projections.sum(axis=-1)

    min_non_zero = projections[projections > 0.0].min()

    projections = np.where(projections == 0, min_non_zero, projections)
    if has_air_projection:
        air_projection = np.asarray(air_projection)
        if air_projection.ndim == 3:
            air_projection = air_projection.sum(-1)

        projections = np.where(
            projections > air_projection, air_projection, projections
        )
        # normalize projections according to Beer–Lambert law
        projections = np.log(air_projection / projections)

    projections = sitk.GetImageFromArray(projections)
    projections.SetSpacing((detector_pixel_size[0], detector_pixel_size[1], 1))
    projections.SetOrigin(
        (
            int(-projections.GetSize()[0] * projections.GetSpacing()[0] / 2),
            int(-projections.GetSize()[1] * projections.GetSpacing()[1] / 2),
            1,
        )
    )

    return projections


def get_projections_from_folder(
    folder: PathLike,
    regex_pattern: str = r"^projection(_\d{4})?$",
    n_detector_pixels: Tuple[int, int] = MCDefaults.n_detector_pixels,
    n_detector_pixels_half_fan: Tuple[int, int] = MCDefaults.n_detector_pixels_half_fan,
    detector_pixel_size: Tuple[float, float] = (0.776, 0.776),
) -> List[MCProjection]:
    folder = Path(folder)
    projections = []

    projection_filepaths = [
        p for p in sorted(folder.glob("*")) if re.match(regex_pattern, p.name)
    ]

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
