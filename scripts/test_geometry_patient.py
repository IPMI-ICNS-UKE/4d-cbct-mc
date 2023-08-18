import logging
from datetime import datetime
from pathlib import Path

from ipmi.common.logger import init_fancy_logging

from cbctmc.forward_projection import (
    create_geometry,
    prepare_image_for_rtk,
    project_forward,
    save_geometry,
)
from cbctmc.mc.geometry import MCGeometry
from cbctmc.mc.simulation import MCSimulation

if __name__ == "__main__":
    import itk

    logging.getLogger("cbctmc").setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    init_fancy_logging()

    N_PROJECTIONS = 32

    CONFIGS = {
        "high": {
            "n_histories": int(2.4e9),
            "n_projections": N_PROJECTIONS,
            "angle_between_projections": 360.0 / N_PROJECTIONS,
        },
    }

    # device ID: runs
    RUNS = {
        0: ("high",),
    }

    GPU = 0

    output_folder = Path("/datalake_fast/mc_test/mc_output/geometry_test")

    output_folder.mkdir(parents=True, exist_ok=True)

    # calibrations = {
    #     "offset_x": 0.45167524990641345,
    #     "offset_y": -2.91421985678384,
    #     "offset_z": -0.2671142029953507,
    #     "source_to_detector_distance_offset": 2.8987834963872467,
    #     "source_to_isocenter_distance_offset": 3.391417693032777,
    # }

    # calibrations = {
    #     "offset_x": 0.502102109728928,
    #     "offset_y": -3.742534490604073,
    #     "offset_z": -0.264611430752569,
    #     "source_to_detector_distance_offset": 1.6565254185658564,
    #     "source_to_isocenter_distance_offset": 4.209057564721922,
    # }

    # calibrations = {
    #     "offset_x": 0.45619947910024583,
    #     "offset_y": -3.9406363975565473,
    #     "offset_z": -0.3525574933952014,
    #     "source_to_detector_distance_offset": 1.4269519273107474,
    #     "source_to_isocenter_distance_offset": 4.3951556461792665,
    # }

    # calibrations = {
    #     "offset_x": 0.45619947910024583,
    #     "offset_y": -3.9406363975565473,
    #     "offset_z": -0.3298962266563392,
    #     "source_to_detector_distance_offset": 1.4269519273107474,
    #     "source_to_isocenter_distance_offset": 4.3951556461792665,
    # }

    calibrations = {
        "offset_x": -0.5030858965528291,
        "offset_y": -3.749082176733503,
        "offset_z": -0.29206039325204886,
        "source_to_detector_distance_offset": 0.13054052787167872,
        "source_to_isocenter_distance_offset": 3.2595168038949205,
    }
    # calibrations = {
    #     "offset_x": -0.5,
    #     "offset_y": -0.5,
    #     "offset_z": -0.5,
    #     "source_to_detector_distance_offset": 0,
    #     "source_to_isocenter_distance_offset": 0,
    # }
    patient_folder = Path(
        "/datalake_fast/4d_ct_lung_uke_artifact_free/022_4DCT_Lunge_amplitudebased_complete"
    )
    geometry = MCGeometry.from_image(
        image_filepath=patient_folder / "phase_00.nii",
        body_segmentation_filepath=patient_folder
        / "segmentations/phase_00/body.nii.gz",
        bone_segmentation_filepath=patient_folder
        / "segmentations/phase_00/upper_body_bones.nii.gz",
        muscle_segmentation_filepath=patient_folder
        / "segmentations/phase_00/upper_body_muscles.nii.gz",
        fat_segmentation_filepath=patient_folder
        / "segmentations/phase_00/upper_body_fat.nii.gz",
        liver_segmentation_filepath=patient_folder
        / "segmentations/phase_00/liver.nii.gz",
        stomach_segmentation_filepath=patient_folder
        / "segmentations/phase_00/stomach.nii.gz",
        lung_segmentation_filepath=patient_folder
        / "segmentations/phase_00/lung.nii.gz",
        lung_vessel_segmentation_filepath=patient_folder
        / "segmentations/phase_00/lung_vessels.nii.gz",
        image_spacing=(1.0, 1.0, 1.0),
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
