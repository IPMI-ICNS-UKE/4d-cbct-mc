import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Sequence

import click
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import SimpleITK as sitk
from ipmi.common.logger import init_fancy_logging

from cbctmc.defaults import DefaultMCSimulationParameters as MCDefaults
from cbctmc.defaults import DefaultReconstructionParameters as ReconDefaults
from cbctmc.defaults import DefaultVarianScanParameters
from cbctmc.forward_projection import create_geometry, save_geometry
from cbctmc.mc.geometry import MCCatPhan604Geometry
from cbctmc.mc.reference import REFERENCE_ROI_STATS_CATPHAN604_VARIAN
from cbctmc.mc.simulation import MCSimulation
from cbctmc.reconstruction.reconstruction import reconstruct_3d

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--output-folder",
    help="Output folder for (intermediate) results",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=Path("."),
    show_default=True,
)
@click.option(
    "--gpu",
    help="GPU PCI bus ID to use for simulation",
    type=int,
    default=0,
    show_default=True,
)
@click.option(
    "--n-projections", type=int, default=DefaultVarianScanParameters.n_projections
)
@click.option(
    "--material",
    type=str,
    multiple=True,
    default=(
        "air_1",
        "air_2",
        "pmp",
        "ldpe",
        "polystyrene",
        "bone_020",
        "acrylic",
        "bone_050",
        "delrin",
        "teflon",
        "water",
    ),
    show_default=True,
)
@click.option(
    "--initial-n-histories",
    help="Initial number of histories",
    type=int,
    default=2e9,
    show_default=True,
)
@click.option(
    "--lower-boundary",
    help="Lower boundary for number of histories",
    default=1e8,
    show_default=True,
)
@click.option(
    "--upper-boundary",
    help="Upper boundary for number of histories",
    default=3e9,
    show_default=True,
)
@click.option(
    "--n-runs",
    help="Number of runs to average over for each simulation configuration",
    type=int,
    default=3,
    show_default=True,
)
@click.option(
    "--loglevel",
    type=click.Choice(["debug", "info", "warning", "error", "critical"]),
    default="debug",
    show_default=True,
)
def run(
    output_folder: Path,
    gpu: int,
    n_projections: int,
    material: Sequence[str],
    initial_n_histories: int,
    lower_boundary: int,
    upper_boundary: int,
    n_runs: int,
    loglevel: str,
):
    # set up logging
    loglevel = getattr(logging, loglevel.upper())
    logging.getLogger("cbctmc").setLevel(loglevel)
    logger.setLevel(loglevel)
    init_fancy_logging()

    logger.info(f"Starting noise optimization using meterials {material}")
    function = lambda x: calculate_variance_deviation(
        n_histories=int(x),
        materials=material,
        output_folder=output_folder,
        gpu=gpu,
        n_projections=n_projections,
        number_runs=n_runs,
    )
    res = opt.minimize(
        function,
        x0=np.array(initial_n_histories),
        method="Nelder-Mead",
        bounds=[(lower_boundary, upper_boundary)],
    )

    logger.info(f"Optimization finished with following result for n_histories: {res.x}")


def calculate_variance_deviation(
    n_histories: int,
    materials: Sequence[str],
    output_folder: Path,
    gpu: int,
    n_projections: int,
    number_runs: int,
):
    if not output_folder.exists():
        # create output folder
        output_folder.mkdir(parents=True, exist_ok=True)

    mean_roi_stats = np.zeros(11)

    for i in range(number_runs):
        simulation_config = {
            "n_histories": n_histories,
            "n_projections": n_projections,
            "angle_between_projections": 360.0 / n_projections,
            "random_seed": datetime.now().microsecond,
        }

        output_folder.mkdir(parents=True, exist_ok=True)
        run_folder = f"run_{n_histories}_run_{i:02d}"

        (output_folder / run_folder).mkdir(exist_ok=True)

        # # MC simulate Cat Phan 604
        phantom = MCCatPhan604Geometry(shape=(464, 464, 250))
        if not any((output_folder / run_folder).iterdir()):
            phantom.save_material_segmentation(
                output_folder / run_folder / "catphan_604_materials.nii.gz"
            )
            phantom.save_density_image(
                output_folder / run_folder / "catphan_604_densities.nii.gz"
            )

            fp_geometry = create_geometry(start_angle=90, n_projections=n_projections)
            save_geometry(fp_geometry, output_folder / run_folder / "geometry.xml")

            simulation = MCSimulation(geometry=phantom, **simulation_config)
            simulation.run_simulation(
                output_folder / run_folder,
                run_air_simulation=True,
                clean=True,
                gpu_id=gpu,
                **MCDefaults().geometrical_corrections,
                force_rerun=True,
            )

        reconstruct_3d(
            projections_filepath=output_folder
            / run_folder
            / "projections_total_normalized.mha",
            geometry_filepath=output_folder / run_folder / "geometry.xml",
            output_folder=output_folder / run_folder / "reconstructions",
            output_filename="fdk3d_wpc.mha",
            dimension=(464, 250, 464),
            water_pre_correction=ReconDefaults.wpc_catphan604,
            gpu_id=gpu,
        )

        mc_recon = sitk.ReadImage(
            str(output_folder / run_folder / "reconstructions" / "fdk3d_wpc.mha")
        )
        mc_recon = sitk.GetArrayFromImage(mc_recon)

        mc_recon = np.moveaxis(mc_recon, 1, -1)
        mc_recon = np.rot90(mc_recon, k=-1, axes=(0, 1))

        mid_z_slice = phantom.image_shape[2] // 2
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        ax[0].imshow(phantom.mus[..., mid_z_slice], clim=(0, 0.04))
        ax[1].imshow(mc_recon[..., mid_z_slice], clim=(0, 0.04))
        ax[1].set_title("MC fdk3d_wpc")

        mc_roi_stats = MCCatPhan604Geometry.calculate_roi_statistics(
            mc_recon,
            height_margin=2,
            radius_margin=2,
        )

        with open(
            str(output_folder / run_folder / "reconstructions" / f"roi_stats.json"),
            "wt",
        ) as file:
            json.dump(mc_roi_stats, file, indent=4)
        mean_roi_stats += np.array(
            [mc_roi_stats[material_name]["std"] for material_name in materials]
        )
    mean_roi_stats = mean_roi_stats / number_runs
    reference_roi_stats = REFERENCE_ROI_STATS_CATPHAN604_VARIAN
    reference_roi_stats = np.array(
        [reference_roi_stats[material_name]["std"] for material_name in materials]
    )
    rel_dev = (mean_roi_stats - reference_roi_stats) / reference_roi_stats
    mean_rel_dev = np.mean(rel_dev)

    logger.info(f"Current deviation: {mean_rel_dev} for {n_histories} histories")

    return np.abs(mean_rel_dev)


if __name__ == "__main__":
    run()
