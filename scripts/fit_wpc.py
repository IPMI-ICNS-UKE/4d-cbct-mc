import logging
from datetime import datetime
from pathlib import Path

import itk
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import xraydb
from ipmi.common.logger import init_fancy_logging
from scipy.ndimage import binary_erosion

from cbctmc.forward_projection import (
    create_geometry,
    prepare_image_for_rtk,
    project_forward,
    save_geometry,
)
from cbctmc.mc.geometry import MCCatPhan604Geometry, MCWaterPhantomGeometry
from cbctmc.mc.materials import MATERIALS_125KEV
from cbctmc.mc.simulation import MCSimulation
from cbctmc.reconstruction.reconstruction import reconstruct_3d

if __name__ == "__main__":
    logging.getLogger("cbctmc").setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    init_fancy_logging()

    N_PROJECTIONS = 45
    N_AVERAGE_SLICES = 70
    IMAGE_SHAPE = (464, 464, 250)

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

    calibrations = {
        "offset_x": 0.45619947910024583,
        "offset_y": -3.9406363975565473,
        "offset_z": -0.3298962266563392,
        "source_to_detector_distance_offset": 1.4269519273107474,
        "source_to_isocenter_distance_offset": 4.3951556461792665,
    }

    GPU = 0
    run = RUNS[GPU]

    output_folder = Path("/datalake_fast/mc_test/mc_output/fit_wpc")

    output_folder.mkdir(parents=True, exist_ok=True)
    run_folder = f"run_{datetime.now().isoformat()}"
    run_folder = "run_2023-07-06T17:10:25.950188"
    (output_folder / run_folder).mkdir(exist_ok=True)

    # MC simulate Cat Phan 604 if not already simulated
    phantom = MCWaterPhantomGeometry(shape=IMAGE_SHAPE)
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
            source_position_offset=(
                calibrations["offset_x"],
                calibrations["offset_y"],
                calibrations["offset_z"],
            ),
            source_to_isocenter_distance_offset=calibrations[
                "source_to_isocenter_distance_offset"
            ],
            source_to_detector_distance_offset=calibrations[
                "source_to_detector_distance_offset"
            ],
            force_rerun=True,
        )

    # create f_n images, i.e. recon of q^n
    f_n_images = []
    highest_order = 5
    for n in range(highest_order + 1):
        # calculate q^n projections
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
        reconstruction = np.swapaxes(reconstruction, 0, 2)
        reconstruction = np.moveaxis(reconstruction, 1, -1)
        f_n_images.append(reconstruction)

    water = MATERIALS_125KEV["h2o"]
    air = MATERIALS_125KEV["air"]
    weight_image = np.zeros_like(phantom.densities, dtype=np.uint8)
    weight_image[phantom.materials == water.number] = 1
    # binary erosion to exclude edge effects
    weight_image = binary_erosion(weight_image, structure=np.ones((10, 10, 10))).astype(
        weight_image.dtype
    )
    weight_image[phantom.materials == air.number] = 1
    # finally restrict weight image to FOV
    fov = MCCatPhan604Geometry.cylindrical_mask(
        shape=weight_image.shape,
        center=tuple(s / 2 for s in IMAGE_SHAPE),
        radius=IMAGE_SHAPE[0] / 2,
        height=20,
    )
    weight_image = fov * weight_image

    sitk.WriteImage(
        phantom._array_to_itk(weight_image),
        str(output_folder / run_folder / "weight_image.nii.gz"),
    )

    template_image = np.zeros_like(phantom.densities, dtype=np.float32)
    water_mask = phantom.materials == water.number
    air_mask = phantom.materials == air.number
    template_image[water_mask] = MU_WATER
    template_image[air_mask] = MU_AIR

    mid_z_slice = IMAGE_SHAPE[-1] // 2

    fig, ax = plt.subplots(
        3, highest_order + 1, sharex=True, sharey=True, squeeze=False
    )

    a = np.zeros(highest_order + 1)
    B = np.zeros((highest_order + 1,) * 2)
    # we average along the cylinder (z-axis) to reduce the image noise, as this
    # improves the WPC result as described in the paper
    average_slicing = np.index_exp[
        ..., mid_z_slice - N_AVERAGE_SLICES // 2 : mid_z_slice + N_AVERAGE_SLICES // 2
    ]
    average_template_image = template_image[average_slicing].mean(-1)
    average_weight_image = weight_image[average_slicing].mean(-1)
    for order_i in range(highest_order + 1):
        average_f_i = f_n_images[order_i][average_slicing].mean(-1)
        for order_j in range(highest_order + 1):
            average_f_j = f_n_images[order_j][average_slicing].mean(-1)
            B[order_i, order_j] = np.sum(
                average_weight_image * average_f_i * average_f_j
            )
        a[order_i] = np.sum(average_weight_image * average_f_i * average_template_image)

        ax[0, order_i].imshow(phantom.densities[..., mid_z_slice])
        ax[1, order_i].imshow(average_weight_image)
        ax[2, order_i].imshow(average_f_i)

    min_order = 0
    B_inv = np.linalg.inv(B[min_order:, min_order:])
    c = B_inv.dot(a[min_order:])

    wpc_image = np.zeros(IMAGE_SHAPE, dtype=np.float32)
    for order in range(min_order, highest_order + 1):
        wpc_image += c[order - min_order] * f_n_images[order]

    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    ax[0].imshow(f_n_images[1][..., 125], clim=(0, 0.025))
    ax[1].imshow(wpc_image[..., 125], clim=(0, 0.025))

    # erode water mask to exclude edge effects
    eroded_water_mask = binary_erosion(
        water_mask, structure=np.ones((10, 10, 10))
    ).astype(water_mask.dtype)

    re_diff_before = (f_n_images[1][eroded_water_mask].mean() - MU_WATER) / MU_WATER
    rel_diff_after = (wpc_image[eroded_water_mask].mean() - MU_WATER) / MU_WATER
    print("Water precorrection (WPC) report:")
    print(f"rel. difference for water before WPC: {re_diff_before}")
    print(f"rel. difference for water after WPC: {rel_diff_after}")
    print(f"WPC coefficients: {c.tolist()}")
