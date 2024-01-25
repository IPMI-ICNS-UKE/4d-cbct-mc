import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from ipmi.common.logger import init_fancy_logging

from cbctmc.forward_projection import (
    create_geometry,
    prepare_image_for_rtk,
    project_forward,
    save_geometry,
)
from cbctmc.mc.geometry import MCGeometry
from cbctmc.mc.materials import MATERIALS_125KEV
from cbctmc.mc.simulation import MCSimulation

if __name__ == "__main__":
    import itk

    logging.getLogger("cbctmc").setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    init_fancy_logging()

    N_PROJECTIONS = 24

    CONFIGS = {
        "high": {
            "n_histories": int(2.4e9),
            "n_projections": N_PROJECTIONS,
            "angle_between_projections": 360.0 / N_PROJECTIONS,
        },
    }

    # device ID: runs
    RUNS = {
        1: ("high",),
    }

    GPU = 1

    output_folder = Path("/datalake_fast/mc_test/mc_output/geometry_test")

    output_folder.mkdir(parents=True, exist_ok=True)

    air_material = MATERIALS_125KEV["air"]
    water_material = MATERIALS_125KEV["h2o"]

    scale = 1.0

    calibrations = {
        "offset_x": 0.45167524990641345,
        "offset_y": -2.91421985678384,
        "offset_z": -0.2671142029953507,
        "source_to_detector_distance_offset": 2.8987834963872467,
        "source_to_isocenter_distance_offset": 3.391417693032777,
    }

    # calibration = {'offset_x': 0.420692765743637, 'offset_y': -3.30682810190411, 'offset_z': -0.2901534866601099, 'source_to_detector_distance_offset': 3.8702133127249128, 'source_to_isocenter_distance_offset': 3.47298445537025}

    shape = (int(500 * scale), int(500 * scale), int(300 * scale))

    densities = np.full(shape, fill_value=1e-6, dtype=np.float32)
    materials = np.full(shape, fill_value=air_material.number, dtype=np.uint8)

    n_boxes = 10
    box_size = int(10 * scale)
    step = tuple(s // n_boxes for s in shape)
    for i_box in range(n_boxes):
        densities[
            i_box * step[0]
            + step[0] // 2
            - box_size // 2 : i_box * step[0]
            + step[0] // 2
            + box_size // 2,
            i_box * step[1]
            + step[1] // 2
            - box_size // 2 : i_box * step[1]
            + step[1] // 2
            + box_size // 2,
            i_box * step[2]
            + step[2] // 2
            - box_size // 2 : i_box * step[2]
            + step[2] // 2
            + box_size // 2,
        ] = water_material.density
        materials[
            i_box * step[0]
            + step[0] // 2
            - box_size // 2 : i_box * step[0]
            + step[0] // 2
            + box_size // 2,
            i_box * step[1]
            + step[1] // 2
            - box_size // 2 : i_box * step[1]
            + step[1] // 2
            + box_size // 2,
            i_box * step[2]
            + step[2] // 2
            - box_size // 2 : i_box * step[2]
            + step[2] // 2
            + box_size // 2,
        ] = water_material.number

    geometry = MCGeometry(
        materials=materials,
        densities=densities,
        image_spacing=(1.0 / scale, 1.0 / scale, 1.0 / scale),
        image_direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        image_origin=(0.0, 0.0, 0.0),
    )

    run_folder = f"run_{datetime.now().isoformat()}"
    (output_folder / run_folder).mkdir(exist_ok=True)

    geometry.save_material_segmentation(
        output_folder / run_folder / "geometry_materials.nii.gz"
    )

    geometry.save_density_image(
        output_folder / run_folder / "geometry_densities.nii.gz"
    )

    image = prepare_image_for_rtk(
        image=geometry.densities,
        image_spacing=geometry.image_spacing,
        input_value_range=None,
        output_value_range=None,
    )
    # itk.imwrite(image, str(output_folder / run_folder / "geometry_densities.mha"))

    fp_geometry = create_geometry(start_angle=270, n_projections=N_PROJECTIONS)
    forward_projection = project_forward(
        image,
        geometry=fp_geometry,
    )
    save_geometry(fp_geometry, output_folder / "geometry.xml")

    itk.imwrite(
        forward_projection,
        str(output_folder / run_folder / "density_fp.mha"),
    )

    for run in RUNS[GPU]:
        simulation_config = CONFIGS[run]

        simulation = MCSimulation(geometry=geometry, **simulation_config)
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
