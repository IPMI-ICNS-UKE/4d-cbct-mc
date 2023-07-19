import logging
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
from cbctmc.metrics import normalized_cross_correlation

if __name__ == "__main__":
    import itk
    import nevergrad as ng
    import SimpleITK as sitk

    logging.getLogger("cbctmc").setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    init_fancy_logging()

    N_PROJECTIONS = 8

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

    output_folder = Path(
        "/datalake_fast/mc_test/mc_output/geometry_test_brute_force_1x1x1_patient"
    )

    output_folder.mkdir(parents=True, exist_ok=True)

    # air_material = MATERIALS_125KEV["air"]
    # water_material = MATERIALS_125KEV["h2o"]
    #
    # shape = (500, 500, 320)
    #
    # densities = np.full(shape, fill_value=1e-6, dtype=np.float32)
    # materials = np.full(shape, fill_value=air_material.number, dtype=np.uint8)
    #
    # n_boxes = 20
    # box_size = 5
    # step = tuple(s // n_boxes for s in shape)
    # for i_box in range(n_boxes):
    #     densities[
    #         i_box * step[0]
    #         + step[0] // 2
    #         - box_size // 2 : i_box * step[0]
    #         + step[0] // 2
    #         + box_size // 2,
    #         i_box * step[1]
    #         + step[1] // 2
    #         - box_size // 2 : i_box * step[1]
    #         + step[1] // 2
    #         + box_size // 2,
    #         i_box * step[2]
    #         + step[2] // 2
    #         - box_size // 2 : i_box * step[2]
    #         + step[2] // 2
    #         + box_size // 2,
    #     ] = water_material.density
    #     materials[
    #         i_box * step[0]
    #         + step[0] // 2
    #         - box_size // 2 : i_box * step[0]
    #         + step[0] // 2
    #         + box_size // 2,
    #         i_box * step[1]
    #         + step[1] // 2
    #         - box_size // 2 : i_box * step[1]
    #         + step[1] // 2
    #         + box_size // 2,
    #         i_box * step[2]
    #         + step[2] // 2
    #         - box_size // 2 : i_box * step[2]
    #         + step[2] // 2
    #         + box_size // 2,
    #     ] = water_material.number
    #
    # geometry = MCGeometry(
    #     materials=materials,
    #     densities=densities,
    #     image_spacing=(2.0, 2.0, 2.0),
    #     image_direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    #     image_origin=(0.0, 0.0, 0.0),
    # )

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

    geometry.save_material_segmentation(output_folder / "geometry_materials.nii.gz")

    geometry.save_density_image(output_folder / "geometry_densities.nii.gz")

    image = prepare_image_for_rtk(
        image=geometry.densities,
        image_spacing=geometry.image_spacing,
        input_value_range=None,
        output_value_range=None,
    )
    itk.imwrite(image, str(output_folder / "geometry_densities.mha"))

    fp_geometry = create_geometry(start_angle=90, n_projections=N_PROJECTIONS)
    forward_projection = project_forward(
        image,
        geometry=fp_geometry,
    )
    save_geometry(fp_geometry, output_folder / "geometry.xml")
    itk.imwrite(
        forward_projection,
        str(output_folder / "density_fp.mha"),
    )

    forward_projection = itk.array_from_image(forward_projection)

    parametrization = ng.p.Instrumentation(
        offset_x=ng.p.Scalar(lower=-4, upper=4),
        offset_y=ng.p.Scalar(lower=-4, upper=4),
        offset_z=ng.p.Scalar(lower=-4, upper=4),
        source_to_detector_distance_offset=ng.p.Scalar(lower=-10, upper=10),
        source_to_isocenter_distance_offset=ng.p.Scalar(lower=-10, upper=10),
    )

    optimizer_class = ng.optimizers.registry["TwoPointsDE"]
    optimizer = optimizer_class(parametrization=parametrization)

    # optimizer = optimizer_class.load(
    #     "/home/fmadesta/research/4d-cbct-mc/optimizer_brute_force_1x1x1_patient.pkl"
    # )

    optimizer.suggest(
        offset_x=0.0,
        offset_y=0.0,
        offset_z=0.0,
        source_to_detector_distance_offset=0.0,
        source_to_isocenter_distance_offset=0.0,
    )

    for run in RUNS[GPU]:
        simulation_config = CONFIGS[run]

        run_folder = f"run"

        simulation = MCSimulation(geometry=geometry, **simulation_config)

        i_simulation = 0
        while True:
            i_simulation += 1
            params = optimizer.ask()

            logger.info(f"Simulation {i_simulation}: {params.kwargs}")
            logger.info(f"Start simulation {i_simulation}")
            simulation.run_simulation(
                output_folder / run_folder,
                run_air_simulation=True,
                clean=True,
                gpu_id=GPU,
                force_rerun=True,
                source_position_offset=(
                    params.kwargs["offset_x"],
                    params.kwargs["offset_y"],
                    params.kwargs["offset_z"],
                ),
                source_to_detector_distance_offset=params.kwargs[
                    "source_to_detector_distance_offset"
                ],
                source_to_isocenter_distance_offset=params.kwargs[
                    "source_to_isocenter_distance_offset"
                ],
            )

            mc_projections = sitk.ReadImage(
                str(output_folder / run_folder / "projections_total_normalized.mha")
            )
            mc_projections = sitk.GetArrayFromImage(mc_projections)

            ncc = normalized_cross_correlation(mc_projections, forward_projection)

            optimizer.tell(params, -ncc)

            logger.info(f"Result for {params.kwargs}: {ncc=}")
            logger.info(f"Current recommendation: {optimizer.recommend()}")
