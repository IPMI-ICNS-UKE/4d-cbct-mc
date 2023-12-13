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

    N_PROJECTIONS = 16

    CONFIG = {
        "n_histories": int(2.4e10),
        "n_projections": N_PROJECTIONS,
        "angle_between_projections": 360.0 / N_PROJECTIONS,
    }
    GPU = 0

    output_folder = Path(
        "/datalake_fast/mc_test/mc_output/geometry_test_brute_force_2x2x2_patient_fp"
    )

    output_folder.mkdir(parents=True, exist_ok=True)

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
        image_spacing=(2.0, 2.0, 2.0),
    )

    geometry.save_material_segmentation(output_folder / "geometry_materials.nii.gz")
    geometry.save_density_image(output_folder / "geometry_densities.nii.gz")

    run_folder = f"run"
    simulation = MCSimulation(geometry=geometry, **CONFIG)
    simulation.run_simulation(
        output_folder / run_folder,
        run_air_simulation=True,
        clean=True,
        gpu_ids=GPU,
        force_rerun=False,
    )

    mc_projections = sitk.ReadImage(
        str(output_folder / run_folder / "projections_total_normalized.mha")
    )
    mc_projections = sitk.GetArrayFromImage(mc_projections)

    parametrization = ng.p.Instrumentation(
        offset_x=ng.p.Scalar(lower=-5, upper=5),
        offset_y=ng.p.Scalar(lower=-5, upper=5),
        offset_z=ng.p.Scalar(lower=-5, upper=5),
    )

    optimizer_class = ng.optimizers.registry["TwoPointsDE"]
    optimizer = optimizer_class(parametrization=parametrization)

    optimizer.suggest(
        offset_x=0.0,
        offset_y=0.0,
        offset_z=0.0,
    )

    i_simulation = 0
    while True:
        i_simulation += 1
        params = optimizer.ask()

        logger.info(f"Forward projection {i_simulation}: {params.kwargs}")

        image = prepare_image_for_rtk(
            image=geometry.densities,
            image_spacing=geometry.image_spacing,
            input_value_range=None,
            output_value_range=None,
            origin_offset=(
                params.kwargs["offset_x"],
                params.kwargs["offset_y"],
                params.kwargs["offset_z"],
            ),
        )
        itk.imwrite(image, str(output_folder / "geometry_densities.mha"))

        fp_geometry = create_geometry(start_angle=90, n_projections=N_PROJECTIONS)
        forward_projection = project_forward(
            image,
            geometry=fp_geometry,
        )

        forward_projection = itk.array_from_image(forward_projection)

        ncc = normalized_cross_correlation(mc_projections, forward_projection)

        optimizer.tell(params, -ncc)

        logger.info(f"Result for {params.kwargs}: {ncc=}")
        recommendation = optimizer.recommend()
        logger.info(
            f"Current recommendation: {recommendation.kwargs}, "
            f"ncc={-recommendation.loss}",
        )

        # optimizer.dump("/home/fmadesta/research/4d-cbct-mc/optimizer_1x1x1_pat_90deg.pkl")
