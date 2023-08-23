import logging
from datetime import datetime
from pathlib import Path
from pprint import pprint
import json

import click
import matplotlib.pyplot as plt
import numpy as np
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


@click.command()
@click.option(
    "--output-folder",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=Path("."),
)
@click.option(
    "--gpu",
    type=int,
    default=0,
)
@click.option("--n-projections", default=DefaultVarianScanParameters.n_projections)
@click.option(
    "--initial-lower-guess",
    type=int,
    default=1e8,
)
@click.option(
    "--initial-upper-guess",
    type=int,
    default=int(3e9),
)
@click.option(
    "--relative-noise-deviation-threshold",
    type=float,
    default=1,
    help="relative deviation of reference and simulation in percentile which ends the loop"
)
@click.option(
    "--loglevel",
    type=click.Choice(["debug", "info", "warning", "error", "critical"]),
    default="debug",
)
def run(
        output_folder: Path,
        gpu: int,
        n_projections: int,
        lower_boundary: int,
        upper_boundary: int,
        threshold: float,
        loglevel: str
    ):

    # set up logging
    loglevel = getattr(logging, loglevel.upper())
    logging.getLogger("cbctmc").setLevel(loglevel)
    logger = logging.getLogger(__name__)
    logger.setLevel(loglevel)
    init_fancy_logging()

    if not output_folder.exists():
        # create output folder
        output_folder.mkdir(parents=True, exist_ok=True)

    dev_under_treshold = False
    runs = 0
    n_histories = 0
    n_histories_arr = []
    mean_rel_dev_arr = []
    while not dev_under_treshold:
        if runs == 0:
            n_histories = lower_boundary
        if runs == 1:
            n_histories = upper_boundary

        simulation_config = {
            "n_histories": n_histories,
            "n_projections": n_projections,
            "angle_between_projections": 360.0 / n_projections,
        }

        output_folder.mkdir(parents=True, exist_ok=True)
        run_folder = f"run_{n_histories}"
        if not run_folder.exists():
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

            # reconstruct MC simulation
            # reconstruct_3d(
            #     projections_filepath=output_folder
            #     / run_folder
            #     / "projections_total_normalized.mha",
            #     geometry_filepath=output_folder / run_folder / "geometry.xml",
            #     output_folder=output_folder / run_folder / "reconstructions",
            #     output_filename="fdk3d.mha",
            #     dimension=(464, 250, 464),
            #     water_pre_correction=None,
            # )
            reconstruct_3d(
                projections_filepath=output_folder
                / run_folder
                / "projections_total_normalized.mha",
                geometry_filepath=output_folder / run_folder / "geometry.xml",
                output_folder=output_folder / run_folder / "reconstructions",
                output_filename="fdk3d_wpc.mha",
                dimension=(464, 250, 464),
                water_pre_correction=ReconDefaults.wpc_catphan604,
            )

        materials = (
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
            mc_recon, height_margin=10, radius_margin=3
        )

        print("MC fdk3d_wpc")
        pprint(mc_roi_stats)
        out_file = open(str(output_folder / run_folder / "reconstructions"
                            / f"mc_roi_stats_{str(n_histories)}.json"), "w")
        json.dump(mc_roi_stats, out_file, indent=6)

        reference_roi_stats = REFERENCE_ROI_STATS_CATPHAN604_VARIAN

        std_reference = [
            reference_roi_stats[material_name]["std"] for material_name in materials
        ]
        std_mc = [
            mc_roi_stats[material_name]["std"] for material_name in materials
        ]
        rel_dev = [
            (mc_roi_stats[material_name]["std"] - reference_roi_stats[material_name]["std"])
            /mc_roi_stats[material_name]["std"]
            for material_name in materials
        ]
        fig, ax = plt.subplots()
        ax.scatter(materials, std_reference, label="reference")
        ax.scatter(materials, std_mc, label="mc")
        mean_rel_dev = np.mean(rel_dev)
        n_histories_arr.append(n_histories)
        mean_rel_dev_arr.append(mean_rel_dev)

        # calculate new n_histories
        if runs >= 1:
            n_histories = (n_histories_arr[0] + mean_rel_dev_arr[0]*(n_histories_arr[0] - n_histories[-1]) /
                           (mean_rel_dev_arr[-1] - mean_rel_dev_arr[0]))

        if mean_rel_dev < threshold:
            dev_under_treshold = True
            pprint("Final number histories: " + str(n_histories))
        runs += 1
    mean_rel_dev_arr = np.array(mean_rel_dev_arr)
    n_histories_arr = np.array(n_histories_arr)
    fig, ax = plt.subplots()
    ax.scatter(n_histories_arr, mean_rel_dev_arr)
    with open(output_folder + "n_histories.npy", 'wb') as file:
        np.save(file, n_histories_arr)
    with open(output_folder + "mean_relative_deviation_of_variance.npy", 'wb') as file:
        np.save(file, mean_rel_dev_arr)


if __name__ == "__main__":
    run()
