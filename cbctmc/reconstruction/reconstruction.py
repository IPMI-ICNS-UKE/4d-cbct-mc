from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple

import click
import numpy as np
import yaml

from cbctmc.common_types import PathLike
from cbctmc.defaults import DefaultReconstructionParameters as ReconDefaults
from cbctmc.logger import init_fancy_logging
from cbctmc.reconstruction.reconstructors import (
    FDKReconstructor,
    ROOSTER4DReconstructor,
)

logger = logging.getLogger(__name__)


def reconstruct_3d(
    projections_filepath: PathLike,
    geometry_filepath: PathLike,
    output_folder: PathLike | None = None,
    output_filename: str | None = None,
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

    if not output_folder:
        output_folder = projections_filepath.parent / "reconstructions"
    if not output_filename:
        output_filename = f"recon_{method}.mha"

    output_folder = Path(output_folder)
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


def reconstruct_4d(
    projections_filepath: PathLike,
    geometry_filepath: PathLike,
    output_folder: PathLike | None = None,
    output_filename: str | None = None,
    dimension: Tuple[int, int, int] = (464, 250, 464),
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    amplitude_signal: Optional[np.ndarray] | None = None,
    phase_signal: Optional[np.ndarray] | None = None,
    water_pre_correction: Sequence[float] | None = None,
    gpu_id: int = 0,
    **kwargs,
):
    method = "rooster4d"

    projections_filepath = Path(projections_filepath)
    geometry_filepath = Path(geometry_filepath)

    if not output_folder:
        output_folder = projections_filepath.parent / "reconstructions"
    if not output_filename:
        output_filename = f"recon_{method}.mha"

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    reconstructor = ROOSTER4DReconstructor(
        amplitude_signal=amplitude_signal,
        phase_signal=phase_signal,
        use_docker=True,
        gpu_id=gpu_id,
    )

    recon_params = dict(
        path=projections_filepath.parent,
        regexp=projections_filepath.name,
        geometry=geometry_filepath,
        fp="CudaRayCast",
        bp="CudaVoxelBased",
        niter=10,
        cgiter=4,
        tviter=10,
        gamma_time=0.0002,
        gamma_space=0.00007,
        dimension=dimension,
        spacing=spacing,
        wpc=water_pre_correction,
        output_filepath=output_folder / output_filename,
        **kwargs,
    )
    reconstructor.reconstruct(**recon_params)

    with open((output_folder / output_filename).with_suffix(".yaml"), "w") as f:
        yaml.dump(recon_params, f)


@click.command()
@click.option(
    "--projections-filepath",
    help="Filepath to normalized projections file",
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    show_default=True,
)
@click.option(
    "--geometry-filepath",
    help="Filepath to geometry file",
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    show_default=True,
)
@click.option(
    "--dimension",
    help="Image dimension of the reconstruction",
    type=click.Tuple([int, int, int]),
    default=(464, 250, 464),
    show_default=True,
)
@click.option(
    "--spacing",
    help="Image spacing of the reconstruction",
    type=click.Tuple([float, float, float]),
    default=(1.0, 1.0, 1.0),
    show_default=True,
)
@click.option(
    "--output-folder",
    help="Output folder for the reconstruction results",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    required=False,
    default=None,
    show_default=True,
)
@click.option(
    "--output-filename",
    help="Output filename for the reconstruction results",
    type=click.STRING,
    required=False,
    default="fdk3d_wpc.mha",
    show_default=True,
)
@click.option(
    "--gpu",
    help="GPU PCI bus ID to use for reconstruction",
    type=int,
    default=0,
    show_default=True,
)
@click.option(
    "--loglevel",
    type=click.Choice(["debug", "info", "warning", "error", "critical"]),
    default="debug",
    show_default=True,
)
def _cli(
    projections_filepath: Path,
    geometry_filepath: Path,
    dimension: Tuple[int, int, int],
    spacing: Tuple[float, float, float],
    output_folder: Path,
    output_filename: str,
    gpu: int,
    loglevel: str,
):
    # set up logging
    loglevel = getattr(logging, loglevel.upper())
    logging.getLogger("cbctmc").setLevel(loglevel)
    logger.setLevel(loglevel)
    init_fancy_logging()

    reconstruct_3d(
        projections_filepath=projections_filepath,
        geometry_filepath=geometry_filepath,
        output_folder=output_folder,
        output_filename=output_filename,
        spacing=spacing,
        dimension=dimension,
        water_pre_correction=ReconDefaults.wpc_catphan604,
        gpu_id=gpu,
    )


if __name__ == "__main__":
    # _cli()

    amplitude_signal = np.loadtxt(
        "/mnt/nas_io/anarchy/4d_cbct_mc/4d/R2017025/mc/ct/phase_02/reference/signal.txt"
    )[:, 0]

    reconstruct_4d(
        projections_filepath=Path(
            "/mnt/nas_io/anarchy/4d_cbct_mc/4d/R2017025/mc/ct/phase_02/reference/projections_total_normalized.mha"
        ),
        geometry_filepath=Path(
            "/mnt/nas_io/anarchy/4d_cbct_mc/4d/R2017025/mc/ct/phase_02/geometry.xml"
        ),
        output_folder=Path(
            "/mnt/nas_io/anarchy/4d_cbct_mc/4d/R2017025/mc/ct/phase_02/reference/reconstructions"
        ),
        output_filename="rooster4d_wpc_00007.mha",
        dimension=(464, 250, 464),
        spacing=(1.0, 1.0, 1.0),
        amplitude_signal=amplitude_signal,
        water_pre_correction=ReconDefaults.wpc_catphan604,
        gpu_id=0,
    )
