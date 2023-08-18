import logging
from datetime import datetime
from pathlib import Path
from pprint import pprint

import itk
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from ipmi.common.logger import init_fancy_logging

from cbctmc.defaults import DefaultMCSimulationParameters as MCDefaults
from cbctmc.defaults import DefaultReconstructionParameters as ReconDefaults
from cbctmc.defaults import DefaultVarianScanParameters
from cbctmc.forward_projection import (
    create_geometry,
    prepare_image_for_rtk,
    project_forward,
    save_geometry,
)
from cbctmc.mc.geometry import MCCatPhan604Geometry
from cbctmc.mc.reference import REFERENCE_MU
from cbctmc.mc.simulation import MCSimulation
from cbctmc.reconstruction.reconstruction import reconstruct_3d

if __name__ == "__main__":
    logging.getLogger("cbctmc").setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    init_fancy_logging()

    N_PROJECTIONS = DefaultVarianScanParameters.n_projections
    FORCE_RERUN: bool = True

    for n_histories in [int(2.4e9)]:
        CONFIGS = {
            "high": {
                "n_histories": n_histories,
                "n_projections": N_PROJECTIONS,
                "angle_between_projections": 360.0 / N_PROJECTIONS,
            },
        }

        # device ID: runs
        RUNS = {0: "high"}

        GPU = 0
        run = RUNS[GPU]

        output_folder = Path("/datalake_fast/mc_test/mc_output/fit_noise")

        output_folder.mkdir(parents=True, exist_ok=True)
        run_folder = f"run_{datetime.now().isoformat()}"
        run_folder = "run_2023-07-13T00:40:34.399836"
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

            image = prepare_image_for_rtk(
                image=phantom.densities,
                image_spacing=phantom.image_spacing,
                input_value_range=None,
                output_value_range=None,
            )

            fp_geometry = create_geometry(start_angle=90, n_projections=N_PROJECTIONS)
            forward_projection = project_forward(
                image,
                geometry=fp_geometry,
            )
            save_geometry(fp_geometry, output_folder / run_folder / "geometry.xml")

            itk.imwrite(
                forward_projection,
                str(output_folder / run_folder / "density_fp.mha"),
            )

            simulation_config = CONFIGS[run]

            simulation = MCSimulation(geometry=phantom, **simulation_config)
            simulation.run_simulation(
                output_folder / run_folder,
                run_air_simulation=True,
                clean=True,
                gpu_id=GPU,
                **MCDefaults.geometrical_corrections,
                force_rerun=True,
            )

        # reconstruct MC simulation
        if (
            not (output_folder / run_folder / "reconstructions" / "fdk3d.mha").exists()
            or FORCE_RERUN
        ):
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
            water_pre_correction=[
                3.655898840079317,
                1.3968539125926327,
                -0.9713710009818897,
                0.6286639358149841,
                -0.16873359741293825,
                0.016437618457075587,
            ],
        )

    reference_recon = sitk.ReadImage(
        "/datalake/4d_cbct_mc/CatPhantom/raw_data/2022-12-01_142914/catphan604_varian_registered.mha"
    )
    reference_recon = sitk.GetArrayFromImage(reference_recon)
    reference_recon = np.moveaxis(reference_recon, 1, -1)
    reference_recon = np.rot90(reference_recon, k=-1, axes=(0, 1))

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

    reference_roi_stats = MCCatPhan604Geometry.calculate_roi_statistics(
        reference_recon, height_margin=3, radius_margin=3
    )

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
        fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
        ax[0].imshow(phantom.mus[..., mid_z_slice], clim=(0, 0.04))
        ax[1].imshow(mc_recon[..., mid_z_slice], clim=(0, 0.04))
        ax[2].imshow(reference_recon[..., mid_z_slice], clim=(0, 0.04))
        ax[1].set_title(f"MC {recon_name}")
        ax[2].set_title(f"reference")

        mc_roi_stats = MCCatPhan604Geometry.calculate_roi_statistics(
            mc_recon, height_margin=10, radius_margin=3
        )

        print(f"MC {recon_name}")
        pprint(mc_roi_stats)

        mean_mu_mc = [
            mc_roi_stats[material_name]["mean"] for material_name in materials
        ]

        mu_ax.scatter(materials, mean_mu_mc, label=f"MC {recon_name}")

    pysical_reference = dict(REFERENCE_MU)
    pysical_reference["air_1"] = pysical_reference["air"]
    pysical_reference["air_2"] = pysical_reference["air"]
    del pysical_reference["air"]
    # mu_ax.scatter(pysical_reference.keys(), pysical_reference.values(), label=f"phys ref")

    mu_ax.legend()
    # fig, ax = plt.subplots()
    # ax.plot(reference_recon[..., 87:107].mean(-1)[232], label="reference")
    # ax.plot(mc_recon[..., 87:107].mean(-1)[232], label="simulation")
    # ax.legend()
