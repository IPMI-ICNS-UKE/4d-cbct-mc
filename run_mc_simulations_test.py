import logging
from pathlib import Path

from ipmi.common.logger import init_fancy_logging

from cbctmc.defaults import DefaultMCSimulationParameters as MCDefaults
from cbctmc.forward_projection import (
    create_geometry,
    prepare_image_for_rtk,
    project_forward,
    save_geometry,
)
from cbctmc.mc.geometry import MCGeometry
from cbctmc.segmentation.utils import (
    merge_upper_body_bone_segmentations,
    merge_upper_body_fat_segmentations,
    merge_upper_body_muscle_segmentations,
)

if __name__ == "__main__":
    import itk

    logging.getLogger("cbctmc").setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    init_fancy_logging()

    CONFIGS = {
        "high": {"n_histories": int(2.4e9)},
    }

    # device ID: runs
    RUNS = {
        1: ("high",),
    }

    GPU = 1

    for patient_folder in [
        Path("/datalake_fast/mc_test/022_4DCT_Lunge_amplitudebased_complete")
    ]:
        output_folder = (
            Path("/datalake_fast/mc_test/mc_output") / patient_folder.name / "phase_00"
        )

        output_folder.mkdir(parents=True, exist_ok=True)

        if not (patient_folder / "segmentations/phase_00/body.nii.gz").exists():
            merge_upper_body_bone_segmentations(
                patient_folder / "segmentations/phase_00"
            )
            merge_upper_body_muscle_segmentations(
                patient_folder / "segmentations/phase_00"
            )
            merge_upper_body_fat_segmentations(
                patient_folder / "segmentations/phase_00"
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
        )
        geometry.densities = geometry.densities

        geometry.save_material_segmentation(output_folder / "geometry_materials.nii.gz")

        geometry.save_density_image(output_folder / "geometry_densities.nii.gz")

        image = prepare_image_for_rtk(
            image=geometry.densities,
            image_spacing=geometry.image_spacing,
            input_value_range=None,
            output_value_range=None,
        )
        itk.imwrite(image, str(output_folder / "geometry_densities.mha"))

        n_projections = MCDefaults.n_projections
        if True or not (output_folder / "density_fp.mha").exists():
            fp_geometry = create_geometry(start_angle=270, n_projections=894)
            forward_projection = project_forward(
                image,
                geometry=fp_geometry,
            )
            save_geometry(fp_geometry, output_folder / "geometry.xml")
            itk.imwrite(
                forward_projection,
                str(output_folder / "density_fp.mha"),
            )
        #
        # for run in RUNS[GPU]:
        #     simulation_config = CONFIGS[run]
        #
        #     simulation = MCSimulation(geometry=geometry, **simulation_config)
        #     simulation.run_simulation(
        #         output_folder / run, run_air_simulation=True, clean=True, gpu_id=GPU
        #     )
