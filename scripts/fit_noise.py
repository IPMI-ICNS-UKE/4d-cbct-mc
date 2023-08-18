import logging
from datetime import datetime
from pathlib import Path
from pprint import pprint

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
    "--loglevel",
    type=click.Choice(["debug", "info", "warning", "error", "critical"]),
    default="debug",
)
def run(output_folder: Path, n_projections: int, gpu: int, loglevel: str):
    # set up logging
    loglevel = getattr(logging, loglevel.upper())
    logging.getLogger("cbctmc").setLevel(loglevel)
    logger = logging.getLogger(__name__)
    logger.setLevel(loglevel)
    init_fancy_logging()

    if not output_folder.exists():
        # create output folder
        output_folder.mkdir(parents=True, exist_ok=True)

    for n_histories in [int(2.4e8)]:
        simulation_config = {
            "n_histories": n_histories,
            "n_projections": n_projections,
            "angle_between_projections": 360.0 / n_projections,
        }

        output_folder.mkdir(parents=True, exist_ok=True)
        run_folder = f"run_{datetime.now().isoformat()}"

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
        reconstruct_3d(
            projections_filepath=output_folder
            / run_folder
            / "projections_total_normalized.mha",
            geometry_filepath=output_folder / run_folder / "geometry.xml",
            output_folder=output_folder / run_folder / "reconstructions",
            output_filename="fdk3d.mha",
            dimension=(464, 250, 464),
            water_pre_correction=None,
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

    reference_roi_stats = REFERENCE_ROI_STATS_CATPHAN604_VARIAN

    mean_mu_reference = [
        reference_roi_stats[material_name]["mean"] for material_name in materials
    ]

    mu_fig, mu_ax = plt.subplots()
    mu_ax.scatter(materials, mean_mu_reference, label="reference")

    for recon_name in ("fdk3d", "fdk3d_wpc"):
        mc_recon = sitk.ReadImage(
            str(output_folder / run_folder / "reconstructions" / f"{recon_name}.mha")
        )
        mc_recon = sitk.GetArrayFromImage(mc_recon)

        mc_recon = np.moveaxis(mc_recon, 1, -1)
        mc_recon = np.rot90(mc_recon, k=-1, axes=(0, 1))

        mid_z_slice = phantom.image_shape[2] // 2
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        ax[0].imshow(phantom.mus[..., mid_z_slice], clim=(0, 0.04))
        ax[1].imshow(mc_recon[..., mid_z_slice], clim=(0, 0.04))
        ax[1].set_title(f"MC {recon_name}")

        mc_roi_stats = MCCatPhan604Geometry.calculate_roi_statistics(
            mc_recon, height_margin=10, radius_margin=3
        )

        print(f"MC {recon_name}")
        pprint(mc_roi_stats)


if __name__ == "__main__":
    run()
