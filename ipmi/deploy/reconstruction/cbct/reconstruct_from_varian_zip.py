import logging
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional
from zipfile import ZipFile

import click
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import yaml

import ipmi.reconstruction.respiratory as resp
from ipmi.common.decorators import convert
from ipmi.common.logger import init_fancy_logging
from ipmi.fused_types import PathLike
from ipmi.reconstruction.cbct.binning import (
    interpolate_nan_phases,
    read_curve,
    save_curve,
)
from ipmi.reconstruction.cbct.geometry import generate_geometry
from ipmi.reconstruction.cbct.projections import (
    convert_xim,
    remove_incomplete_projections,
)
from ipmi.reconstruction.cbct.reconstructors import (
    FDKReconstructor,
    ROOSTER4DReconstructor,
)

logger = logging.getLogger(__name__)
init_fancy_logging()

logging.getLogger(__name__).setLevel(logging.DEBUG)
logging.getLogger("ipmi").setLevel(logging.DEBUG)


def _extract_file_and_move(
    zip_file,
    compressed_file: str,
    temp_dir: Path,
    output_folder: Path,
    sub_folder: Path,
):
    print("extract", compressed_file)
    zip_file.extract(compressed_file, temp_dir)
    extracted_filepath = temp_dir / compressed_file

    destination_filepath = output_folder / sub_folder / extracted_filepath.name
    destination_filepath.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(extracted_filepath, destination_filepath)

    return destination_filepath


# def _save_filepaths(filepaths: dict, output_filepath: Path):
#     def _cast_to_str(d: dict):
#         for key, value in d.items():
#             if isinstance(value, dict):
#
#             if isinstance(value, Path):
#
#     filepaths = filepaths.copy()
#
#     with open(output_filepath, "w") as f:
#         yaml.dump(filepaths, f)


# flake8: noqa: C901
@convert("filepath", converter=Path)
@convert("output_folder", converter=Path)
def extract_data_from_zip(
    filepath: PathLike,
    output_folder: PathLike,
    air_scan: PathLike = "AIR-Half-Bowtie-125KV/Current/FilterBowtie.xim",
    clean_projections: bool = True,
) -> dict:
    # overwrite converted types
    filepath: Path
    output_folder: Path

    projection_filepaths = []
    calibration_filepaths = []
    with ZipFile(filepath, "r") as zip_file, TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        compressed_files = zip_file.namelist()
        for compressed_file in compressed_files:
            if compressed_file.endswith(".xim") and "Acquisitions" in compressed_file:
                sub_folder = Path("projections")

            elif compressed_file.endswith(".xim") and "Calibrations" in compressed_file:
                print("extract calibration", compressed_file)
                sub_folder = Path("calibrations") / "/".join(
                    Path(compressed_file).parts[-3:-1]
                )

            elif compressed_file.endswith("Scan.xml"):
                print("extract scan definition", compressed_file)
                sub_folder = Path("meta")

            elif compressed_file.endswith("ImgParameters.h5"):
                print("extract breathing curve", compressed_file)
                sub_folder = Path("meta")
            else:
                continue

            destination_filepath = _extract_file_and_move(
                zip_file=zip_file,
                compressed_file=compressed_file,
                temp_dir=temp_dir,
                output_folder=output_folder,
                sub_folder=sub_folder,
            )
            if sub_folder.parts[0] == "projections":
                projection_filepaths.append(destination_filepath)
            elif sub_folder.parts[0] == "calibrations":
                calibration_filepaths.append(destination_filepath)

    # find right air scan/calibration
    try:
        air_scan_filepath = next(
            c for c in calibration_filepaths if str(c).endswith(air_scan)
        )
    except StopIteration:
        raise FileNotFoundError(f"Air scan {air_scan} not found")

    if clean_projections:
        projection_filepaths = remove_incomplete_projections(projection_filepaths)

    filepaths = {
        "projections": projection_filepaths,
        "air_scan": air_scan_filepath,
        "scan_config": output_folder / "meta/Scan.xml",
        "projections_config": output_folder / "meta/ImgParameters.h5",
    }
    return filepaths


@convert("zip_filepath", converter=Path)
@convert("output_folder", converter=Path)
def reconstruct(
    zip_filepath: Path,
    output_folder: Optional[Path] = None,
    method: str = "fdk3d",
    gpu_id: int = 0,
    **kwargs,
):
    if not output_folder:
        # name of zip file as folder
        output_folder = Path(os.path.splitext(zip_filepath)[0])

    output_folder.mkdir(parents=True, exist_ok=True)

    normalized_projections_filepath = output_folder / "normalized_projections.mha"
    projections_filepath = output_folder / "projections.mha"
    geometry_filepath = output_folder / "meta" / "geometry.xml"
    files_filepath = output_folder / "files.yaml"

    if not files_filepath.exists():
        filepaths = extract_data_from_zip(
            zip_filepath,
            output_folder=output_folder,
        )

        with open(files_filepath, "w") as f:
            yaml.dump(filepaths, f)
    else:
        with open(files_filepath, "r") as f:
            filepaths = yaml.load(f, Loader=yaml.Loader)

    if not normalized_projections_filepath.exists():
        projections_normalized, projections = convert_xim(
            xim_files=filepaths["projections"],
            air_scan_filepath=filepaths["air_scan"],
            detector_spacing=(0.388, 0.388),
            show_progress=True,
        )
        sitk.WriteImage(projections_normalized, str(normalized_projections_filepath))
        sitk.WriteImage(projections, str(projections_filepath))

    if not geometry_filepath.exists():
        # create scan geometry as XML file
        generate_geometry(
            scan_xml_filepath=filepaths["scan_config"],
            projection_folder=filepaths["projections"][0].parent,
            output_filepath=geometry_filepath,
            use_docker=True,
        )

    if method == "rooster4d":
        binning = kwargs["binning"]
        n_bins = kwargs["n_bins"]
        save_binning = kwargs["save_binning"]

        amplitude, phase = read_curve(filepaths["projections_config"])

        if kwargs["phase_overwrite"]:
            overwritten_phase = np.loadtxt(kwargs["phase_overwrite"])
            if len(phase) != len(overwritten_phase):
                raise RuntimeError(
                    f"Length mismatch: {len(phase)} vs. {len(overwritten_phase)}"
                )
            phase = overwritten_phase
            logging.info(f"Overwritten phase with {kwargs['phase_overwrite']}")

        if kwargs["amplitude_overwrite"]:
            overwritten_amplitude = np.loadtxt(kwargs["amplitude_overwrite"])

            if len(amplitude) != len(overwritten_amplitude):
                raise RuntimeError(
                    f"Length mismatch: {len(amplitude)} vs. {len(overwritten_amplitude)}"
                )
            amplitude = overwritten_amplitude
            logging.info(f"Overwritten amplitude with {kwargs['amplitude_overwrite']}")

        if binning == "amplitude":
            amplitude_bins = resp.calculate_amplitude_bins_as_phase_bins(
                breathing_curve=amplitude, n_bins=n_bins
            )

            amplitude_bins = amplitude_bins / n_bins
            phase_signal = amplitude_bins
        elif binning == "phase_varian":
            phase = interpolate_nan_phases(phase)
            phase_signal = phase / 360.0
        elif binning == "phase":
            # phase_bins = resp.calculate_phase_bins(breathing_curve=amplitude,
            #                                        n_bins=n_bins)
            # phase_bins = phase_bins / n_bins

            # calculate phase, no binning. RTK uses continuous phase values, not bins
            phase = resp.calculate_phase(amplitude, phase_range=(0, 1))
            phase = np.hstack(phase)
            phase_signal = phase
        elif binning == "pseudoaverage":
            pseudo_average_phase = resp.calculate_pseudo_average_phase(
                amplitude, phase_range=(0, 1)
            )
            pseudo_average_phase = np.hstack(pseudo_average_phase)
            phase_signal = pseudo_average_phase
        else:
            raise RuntimeError(f"Unknown binning {binning}")

        if kwargs["plot"]:
            fig, ax = plt.subplots(2, sharex=True)
            ax[0].plot(amplitude)
            ax[1].plot(phase)

            for _ax in ax:
                _ax.grid(True)

            plt.savefig(output_folder / "respiratory_curve.png")

        if save_binning:
            signal_filepath = output_folder / f"signal_{binning}.txt"
            save_curve(phase_signal, filepath=signal_filepath)

        reconstructor = ROOSTER4DReconstructor(
            phase_signal=phase_signal, use_docker=True, gpu_id=gpu_id
        )
        rooster_params = dict(
            path=normalized_projections_filepath.parent,
            regexp=normalized_projections_filepath.name,
            geometry=geometry_filepath,
            fp="CudaRayCast",
            bp="CudaVoxelBased",
            dimension=kwargs["dimension"],
            spacing=kwargs["spacing"],
            niter=kwargs["n_iter"],
            cgiter=kwargs["cg_iter"],
            tviter=kwargs["tv_iter"],
            gamma_time=kwargs["gamma_time"],
            gamma_space=kwargs["gamma_space"],
            output_filepath=output_folder / f"4d_rooster_{binning}_binning.mha",
        )
        reconstructor.reconstruct(**rooster_params)

        with open(output_folder / "4d_rooster_params.yaml", "w") as f:
            yaml.dump(rooster_params, f)

    elif method == "fdk3d":
        reconstructor = FDKReconstructor(use_docker=True, gpu_id=gpu_id)
        fdk_params = dict(
            path=normalized_projections_filepath.parent,
            regexp=normalized_projections_filepath.name,
            geometry=geometry_filepath,
            hardware="cuda",
            pad=kwargs["pad"],
            hann=kwargs["hann"],
            hannY=kwargs["hann_y"],
            dimension=kwargs["dimension"],
            spacing=kwargs["spacing"],
            output_filepath=output_folder / "3d_fdk.mha",
        )
        reconstructor.reconstruct(**fdk_params)

        with open(output_folder / "3d_fdk_params.yaml", "w") as f:
            yaml.dump(fdk_params, f)


@click.group()
def cli():
    pass


@click.command()
@click.argument(
    "zip-filepath",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--output-folder",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Output folder for extracted raw data and reconstructions.",
    show_default=True,
)
@click.option(
    "--dimension",
    type=click.Tuple([int, int, int]),
    default=(464, 250, 464),
    help="Image dimension of the reconstruction",
    show_default=True,
)
@click.option(
    "--spacing",
    type=click.Tuple([float, float, float]),
    default=(1.0, 1.0, 1.0),
    help="Image spacing of the reconstruction",
    show_default=True,
)
@click.option(
    "--pad",
    type=click.FLOAT,
    default=1.0,
    help="Data padding parameter to correct for truncation",
    show_default=True,
)
@click.option(
    "--hann",
    type=click.FLOAT,
    default=1.0,
    help="Cut frequency for hann window in ]0,1] (0.0 disables it)",
    show_default=True,
)
@click.option(
    "--hann-y",
    type=click.FLOAT,
    default=1.0,
    help="Cut frequency for hann window in ]0,1] (0.0 disables it)",
    show_default=True,
)
def fdk3d(*args, **kwargs):
    reconstruct(*args, method="fdk3d", **kwargs)


@click.command()
@click.argument(
    "zip-filepath",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--output-folder",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Output folder for extracted raw data and reconstructions.",
    show_default=True,
)
@click.option(
    "--dimension",
    type=click.Tuple([int, int, int]),
    default=(464, 250, 464),
    help="Image dimension of the reconstruction",
    show_default=True,
)
@click.option(
    "--spacing",
    type=click.Tuple([float, float, float]),
    default=(1.0, 1.0, 1.0),
    help="Image spacing of the reconstruction",
    show_default=True,
)
@click.option(
    "--binning",
    type=click.Choice(["amplitude", "phase", "phase_varian", "pseudoaverage"]),
    default="amplitude",
    help="Methods used for projection binning",
    show_default=True,
)
@click.option(
    "--phase_overwrite",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Overwrite phase",
    show_default=True,
)
@click.option(
    "--amplitude_overwrite",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Overwrite amplitude",
    show_default=True,
)
@click.option(
    "--n-bins",
    type=click.INT,
    default=10,
    help="Number of respiratory bins for 4D reconstruction",
    show_default=True,
)
@click.option(
    "--n-iter",
    type=click.INT,
    default=10,
    help="Number of main loop iterations",
    show_default=True,
)
@click.option(
    "--cg-iter",
    type=click.INT,
    default=4,
    help="Number of conjugate gradient nested iterations",
    show_default=True,
)
@click.option(
    "--tv-iter",
    type=click.INT,
    default=10,
    help="Total variation (spatial, temporal and nuclear) regularization: number of iterations",
    show_default=True,
)
@click.option(
    "--gamma-time",
    type=click.FLOAT,
    default=0.0002,
    help="Total variation temporal regularization parameter. The larger, the smoother",
    show_default=True,
)
@click.option(
    "--gamma-space",
    type=click.FLOAT,
    default=0.00005,
    help="Total variation spatial regularization parameter. The larger, the smoother",
    show_default=True,
)
@click.option(
    "--save-binning",
    is_flag=True,
    default=False,
    help="Save the used binning values to a *.txt file",
    show_default=True,
)
@click.option(
    "--plot",
    is_flag=True,
    default=False,
    help="Plot amplitude and phase",
    show_default=True,
)
@click.option(
    "--gpu-id",
    type=click.INT,
    default=0,
    help="ID of GPU to use for reconstruction",
    show_default=True,
)
def rooster4d(*args, **kwargs):
    reconstruct(*args, method="rooster4d", **kwargs)


cli.add_command(fdk3d)
cli.add_command(rooster4d)

if __name__ == "__main__":
    cli()

    # cli(
    #     [
    #         "rooster4d",
    #         "/datalake/recon_laura/306/session1/2020-07-13_093733.zip",
    #         "--output-folder",
    #         "/datalake/recon_laura/306/session1/recon_fm",
    #         "--binning",
    #         "phase_varian",
    #         "--plot",
    #     ]
    # )

    # cli(
    #     [
    #         "rooster4d",
    #         "/datalake/recon_laura/319/session1/2020-09-28_094710.zip",
    #         "--output-folder",
    #         "/datalake/recon_laura/319/session1/recon_fm",
    #         "--binning",
    #         "phase_varian",
    #         "--plot"
    #     ]
    # )
