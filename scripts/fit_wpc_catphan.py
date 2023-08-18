import logging
from datetime import datetime
from pathlib import Path
from pprint import pprint

import itk
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import xraydb
from ipmi.common.logger import init_fancy_logging
from scipy.ndimage import binary_erosion

from cbctmc.defaults import DefaultMCSimulationParameters
from cbctmc.forward_projection import (
    create_geometry,
    prepare_image_for_rtk,
    project_forward,
    save_geometry,
)
from cbctmc.mc.geometry import MCCatPhan604Geometry, MCWaterPhantomGeometry
from cbctmc.mc.materials import MATERIALS_125KEV
from cbctmc.mc.reference import REFERENCE_MU, REFERENCE_MU_VARIAN
from cbctmc.mc.simulation import MCSimulation
from cbctmc.reconstruction.reconstruction import reconstruct_3d

"""
The following script perform an end-to-end water precorrection (WPC) [1]
for the Monte Carlo scan geometry and X-ray parameters.

References:
[1] Sourbelle et al., Empirical water precorrection for cone-beam computed tomography (2005)
"""

if __name__ == "__main__":
    logging.getLogger("cbctmc").setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    init_fancy_logging()

    N_PROJECTIONS = DefaultMCSimulationParameters.n_projections
    N_AVERAGE_SLICES: int = 20
    IMAGE_SHAPE = (464, 464, 250)
    EDGE_EROSION_KERNEL_SIZE: int = 0
    HIGHEST_ORDER = 5
    IGNORE_AIR = True
    PLOT_DEBUG: bool = True
    FORCE_RERUN: bool = False

    # cf. Chantler: https://www.nist.gov/pml/x-ray-form-factor-attenuation-and-scattering-tables
    # mean energy of used spectrum is 63.140 keV
    # mu (1/mm) water @ 63.140 keV = 0.02011970928851904
    # mu (1/mm) air   @ 63.140 keV = 2.2416145024763944e-05
    MEAN_SPECTRUM_ENERGY = 63.140 * 10**3
    MU_WATER = xraydb.material_mu("water", MEAN_SPECTRUM_ENERGY) / 10.0
    MU_AIR = xraydb.material_mu("air", MEAN_SPECTRUM_ENERGY) / 10.0

    CONFIGS = {
        "high": {
            "n_histories": int(2.4e9),
            "n_projections": N_PROJECTIONS,
            "angle_between_projections": 360.0 / N_PROJECTIONS,
        },
    }

    # device ID: runs
    RUNS = {0: "high"}

    GPU = 0
    run = RUNS[GPU]

    output_folder = Path("/datalake_fast/mc_test/mc_output/fit_wpc_catphan")

    output_folder.mkdir(parents=True, exist_ok=True)
    # run_folder = f"run_{datetime.now().isoformat()}"

    run_folder = "run_2023-07-12T19:44:28.433817"
    (output_folder / run_folder).mkdir(exist_ok=True)

    # MC simulate Cat Phan 604 if not already simulated
    phantom = MCCatPhan604Geometry(shape=IMAGE_SHAPE, reference_mu=REFERENCE_MU_VARIAN)

    if not any((output_folder / run_folder).iterdir()):
        phantom.save_material_segmentation(
            output_folder / run_folder / "water_phantom_materials.nii.gz"
        )
        phantom.save_density_image(
            output_folder / run_folder / "water_phantom_densities.nii.gz"
        )

        image = prepare_image_for_rtk(
            image=phantom.densities,
            image_spacing=phantom.image_spacing,
            input_value_range=None,
            output_value_range=None,
        )

        # fp_geometry = create_geometry(start_angle=90, n_projections=N_PROJECTIONS)
        # forward_projection = project_forward(
        #     image,
        #     geometry=fp_geometry,
        # )
        # save_geometry(fp_geometry, output_folder / run_folder / "geometry.xml")
        #
        # itk.imwrite(
        #     forward_projection,
        #     str(output_folder / run_folder / "density_fp.mha"),
        # )

        simulation_config = CONFIGS[run]

        simulation = MCSimulation(geometry=phantom, **simulation_config)
        simulation.run_simulation(
            output_folder / run_folder,
            run_air_simulation=True,
            clean=True,
            gpu_id=GPU,
            **DefaultMCSimulationParameters().geometrical_corrections,
            force_rerun=True,
        )

    for HIGHEST_ORDER in range(HIGHEST_ORDER, HIGHEST_ORDER + 1):
        # create f_n images, i.e. recon of q^n
        f_n_images = []

        for n in range(HIGHEST_ORDER + 1):
            # calculate q^n projections and reconstruct f_n images (if not exist)
            if (
                not (
                    output_folder / run_folder / "reconstructions" / f"recon_f_{n}.mha"
                ).exists()
                or FORCE_RERUN
            ):
                normalized_projections = sitk.ReadImage(
                    str(output_folder / run_folder / "projections_total_normalized.mha")
                )
                normalized_projections = normalized_projections**n
                projections_filepaths = (
                    output_folder / run_folder / f"projections_total_normalized_{n}.mha"
                )
                sitk.WriteImage(
                    normalized_projections,
                    str(projections_filepaths),
                )

                # reconstruct f_n images
                reconstruct_3d(
                    projections_filepath=projections_filepaths,
                    geometry_filepath=output_folder / run_folder / "geometry.xml",
                    output_folder=output_folder / run_folder / "reconstructions",
                    output_filename=f"recon_f_{n}.mha",
                    dimension=(IMAGE_SHAPE[0], IMAGE_SHAPE[2], IMAGE_SHAPE[1]),
                )
            reconstruction = sitk.ReadImage(
                str(output_folder / run_folder / "reconstructions" / f"recon_f_{n}.mha")
            )
            reconstruction = sitk.GetArrayFromImage(reconstruction)
            reconstruction = np.moveaxis(reconstruction, 1, -1)
            reconstruction = np.rot90(reconstruction, k=-1, axes=(0, 1))
            f_n_images.append(reconstruction)

        water = MATERIALS_125KEV["h2o"]
        air = MATERIALS_125KEV["air"]
        water_mask = phantom.materials == water.number
        air_mask = phantom.materials == air.number
        n_air_voxels = air_mask.sum()
        n_water_voxels = water_mask.sum()

        weight_image = np.ones_like(phantom.densities, dtype=np.float32)

        # if EDGE_EROSION_KERNEL_SIZE:
        #     # binary erosion to exclude edge effects
        #     weight_image = binary_erosion(
        #         weight_image, structure=np.ones((EDGE_EROSION_KERNEL_SIZE,) * 3)
        #     ).astype(weight_image.dtype)

        # finally restrict weight image to FOV
        # (circular FOV in x-y and central z-range used for averaging)
        fov = MCCatPhan604Geometry.cylindrical_mask(
            shape=weight_image.shape,
            center=tuple(s / 2 for s in IMAGE_SHAPE),
            radius=IMAGE_SHAPE[0] / 2,
            height=N_AVERAGE_SLICES,
        )
        weight_image = fov * weight_image
        # same weighting for every material ROI
        materials = (
            "air",
            "h2o",
            "pmp",
            "ldpe",
            "polystyrene",
            "bone_020",
            "acrylic",
            "bone_050",
            "delrin",
            "teflon",
        )
        for material in materials:
            mask = phantom.materials == MATERIALS_125KEV[material].number
            mask &= fov

            if IGNORE_AIR and material == "air":
                # ignore air
                weight_image[mask] = 0.0
            else:
                weight_image[mask] = 1.0 / mask.sum()

        sitk.WriteImage(
            phantom._array_to_itk(weight_image),
            str(output_folder / run_folder / "weight_image.nii.gz"),
        )

        # create the template image, i.e. phantom with reference mu values
        template_image = phantom.mus
        mid_z_slice = IMAGE_SHAPE[-1] // 2

        if PLOT_DEBUG:
            fig, ax = plt.subplots(
                3, HIGHEST_ORDER + 1, sharex=True, sharey=True, squeeze=False
            )

        # variable naming (a, c, B f_n) according to the paper
        a = np.zeros(HIGHEST_ORDER + 1)
        B = np.zeros((HIGHEST_ORDER + 1,) * 2)
        # we average along the cylinder (z-axis) to reduce the image noise, as this
        # improves the WPC result as described in the paper
        average_slicing = np.index_exp[
            ...,
            mid_z_slice - N_AVERAGE_SLICES // 2 : mid_z_slice + N_AVERAGE_SLICES // 2,
        ]
        average_template_image = template_image[average_slicing].mean(-1)
        average_weight_image = weight_image[average_slicing].mean(-1)
        for order_i in range(HIGHEST_ORDER + 1):
            average_f_i = f_n_images[order_i][average_slicing].mean(-1)
            for order_j in range(HIGHEST_ORDER + 1):
                average_f_j = f_n_images[order_j][average_slicing].mean(-1)
                B[order_i, order_j] = np.sum(
                    average_weight_image * average_f_i * average_f_j
                )
            a[order_i] = np.sum(
                average_weight_image * average_f_i * average_template_image
            )

            if PLOT_DEBUG:
                ax[0, order_i].imshow(phantom.densities[..., mid_z_slice])
                ax[1, order_i].imshow(average_weight_image)
                ax[2, order_i].imshow(average_f_i)

        min_order = 0
        B_inv = np.linalg.inv(B[min_order:, min_order:])
        c = B_inv.dot(a[min_order:])

        wpc_image = np.zeros(IMAGE_SHAPE, dtype=np.float32)
        for order in range(min_order, HIGHEST_ORDER + 1):
            wpc_image += c[order - min_order] * f_n_images[order]

        # erode water mask to exclude edge effects
        evaluation_mask = water_mask
        evaluation_mask = binary_erosion(
            evaluation_mask, structure=np.ones((10, 10, 10))
        ).astype(water_mask.dtype)

        rel_diff_before = (f_n_images[1][evaluation_mask].mean() - MU_WATER) / MU_WATER
        rel_diff_after = (wpc_image[evaluation_mask].mean() - MU_WATER) / MU_WATER

        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        fig.suptitle(
            f"WPC ({HIGHEST_ORDER=}, {EDGE_EROSION_KERNEL_SIZE=})\n"
            f"{rel_diff_before=:.4f} / {rel_diff_after=:.4f}"
        )
        ax[0].imshow(f_n_images[1][..., 125], clim=(0, 0.025))
        ax[1].imshow(wpc_image[..., 125], clim=(0, 0.025))

        print("Water precorrection (WPC) report:")
        print(f"{HIGHEST_ORDER=}, {EDGE_EROSION_KERNEL_SIZE=}")
        print(f"rel. difference for water before WPC: {rel_diff_before}")
        print(f"rel. difference for water after WPC: {rel_diff_after}")
        print(f"WPC coefficients: {c.tolist()}")

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
            water_pre_correction=c.tolist(),
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
        reference_recon, height_margin=1, radius_margin=1
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
        ax[2].set_title(f"reference (Varian)")

        mc_roi_stats = MCCatPhan604Geometry.calculate_roi_statistics(
            mc_recon, height_margin=1, radius_margin=1
        )

        print(f"MC {recon_name}")
        pprint(mc_roi_stats)

        mean_mu_mc = [
            mc_roi_stats[material_name]["mean"] for material_name in materials
        ]

        mu_ax.scatter(materials, mean_mu_mc, label=f"MC {recon_name}")

    mu_ax.legend()

    pysical_reference = dict(REFERENCE_MU)
    pysical_reference["air_1"] = pysical_reference["air"]
    pysical_reference["air_2"] = pysical_reference["air"]
    del pysical_reference["air"]

    # reference_recon = sitk.ReadImage(
    #     "/datalake/4d_cbct_mc/CatPhantom/raw_data/2022-12-01_142914/catphan604_varian_registered.mha"
    # )
    # reference_recon = sitk.GetArrayFromImage(reference_recon)
    # reference_recon = np.moveaxis(reference_recon, 1, -1)
    # reference_recon = np.rot90(reference_recon, k=-1, axes=(0, 1))
    #
    # materials = (
    #     "air_1",
    #     "air_2",
    #     "pmp",
    #     "ldpe",
    #     "polystyrene",
    #     "bone_020",
    #     "acrylic",
    #     "bone_050",
    #     "delrin",
    #     "teflon",
    # )
    #
    # reference_roi_stats = MCCatPhan604Geometry.calculate_roi_statistics(
    #     reference_recon, height_margin=10, radius_margin=3
    # )
    #
    # mc_roi_stats = MCCatPhan604Geometry.calculate_roi_statistics(
    #     f_n_images[1], height_margin=10, radius_margin=3
    # )
    #
    # mc_wpc_roi_stats = MCCatPhan604Geometry.calculate_roi_statistics(
    #     wpc_image, height_margin=10, radius_margin=3
    # )
