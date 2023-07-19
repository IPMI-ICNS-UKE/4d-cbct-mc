import logging
from datetime import datetime
from pathlib import Path

import itk
from ipmi.common.logger import init_fancy_logging

from cbctmc.defaults import DefaultReconstructionParameters as ReconDefaults
from cbctmc.forward_projection import (
    create_geometry,
    prepare_image_for_rtk,
    project_forward,
    save_geometry,
)
from cbctmc.mc.geometry import MCCatPhan604Geometry
from cbctmc.mc.simulation import MCSimulation
from cbctmc.reconstruction.reconstruction import reconstruct_3d

if __name__ == "__main__":
    logging.getLogger("cbctmc").setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    init_fancy_logging()

    N_PROJECTIONS = 16  # ScanDefaults.n_projections

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

    output_folder = Path("/datalake_fast/mc_test/mc_output/fit_noise")

    output_folder.mkdir(parents=True, exist_ok=True)
    run_folder = f"run_{datetime.now().isoformat()}"
    (output_folder / run_folder).mkdir(exist_ok=True)

    # MC simulate Cat Phan 604
    phantom = MCCatPhan604Geometry(shape=(464, 464, 250))
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

    # reconstruct MC simulation
    reconstruct_3d(
        projections_filepath=output_folder
        / run_folder
        / "projections_total_normalized.mha",
        geometry_filepath=output_folder / run_folder / "geometry.xml",
        output_folder=output_folder / run_folder / "reconstructions",
        dimension=(250, 150, 250),
        water_pre_correction=ReconDefaults.water_pre_correction,
    )
