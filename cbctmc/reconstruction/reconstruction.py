from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence, Tuple

import yaml
from ipmi.common.logger import init_fancy_logging
from ipmi.reconstruction.cbct.reconstructors import FDKReconstructor

from cbctmc.common_types import PathLike

logger = logging.getLogger(__name__)
init_fancy_logging()

logging.getLogger(__name__).setLevel(logging.DEBUG)
logging.getLogger("ipmi").setLevel(logging.DEBUG)


def reconstruct_3d(
    projections_filepath: PathLike,
    geometry_filepath: PathLike,
    output_folder: PathLike | None = None,
    output_filename: PathLike | None = None,
    dimension: Tuple[int, int, int] = (464, 250, 464),
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    pad: float = 1.0,
    hann: float = 1.0,
    hann_y: float = 1.0,
    water_pre_correction: Sequence[float] | None = None,
    gpu_id: int = 0,
    **kwargs,
):
    method = "fdk3d"

    projections_filepath = Path(projections_filepath)
    geometry_filepath = Path(geometry_filepath)
    output_folder = Path(output_folder)

    if not output_folder:
        output_folder = projections_filepath.parent / "reconstruction"
    if not output_filename:
        output_filename = f"recon_{method}.mha"

    output_folder.mkdir(parents=True, exist_ok=True)

    reconstructor = FDKReconstructor(use_docker=True, gpu_id=gpu_id)

    fdk_params = dict(
        path=projections_filepath.parent,
        regexp=projections_filepath.name,
        geometry=geometry_filepath,
        hardware="cuda",
        pad=pad,
        hann=hann,
        hannY=hann_y,
        dimension=dimension,
        spacing=spacing,
        wpc=water_pre_correction,
        short=360,
        output_filepath=output_folder / output_filename,
        **kwargs,
    )
    reconstructor.reconstruct(**fdk_params)

    with open((output_folder / output_filename).with_suffix(".yaml"), "w") as f:
        yaml.dump(fdk_params, f)
