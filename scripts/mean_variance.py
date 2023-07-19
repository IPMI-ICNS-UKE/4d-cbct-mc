import logging
from pathlib import Path

from ipmi.common.logger import init_fancy_logging

from cbctmc.defaults import DefaultMCSimulationParameters as MCDefaults
from cbctmc.forward_projection import prepare_image_for_rtk
from cbctmc.mc.geometry import MCGeometry
from cbctmc.mc.simulation import MCSimulation
from cbctmc.segmentation import (
    merge_upper_body_bone_segmentations,
    merge_upper_body_fat_segmentations,
    merge_upper_body_muscle_segmentations,
)

if __name__ == "__main__":
    logging.getLogger("cbctmc").setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    init_fancy_logging()

    for patient_folder in Path("/datalake_fast/4d_ct_lung_uke_artifact_free").iterdir():
        if not (
            patient_folder / "segmentations/phase_00/upper_body_bones.nii.gz"
        ).exists():
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

        output_folder = Path("/datalake2/mc_output") / patient_folder.name / "phase_00"

        output_folder.mkdir(parents=True, exist_ok=True)

        geometry.save_material_segmentation(output_folder / "geometry_materials.nii.gz")

        geometry.save_density_image(output_folder / "geometry_densities.nii.gz")

        image = prepare_image_for_rtk(
            image=geometry.densities,
            image_spacing=geometry.image_spacing,
            input_value_range=None,
            output_value_range=None,
        )

        n_projections = MCDefaults.n_projections
        simulation = MCSimulation(
            geometry=geometry,
            n_histories=int(2.4e9),
            n_projections=1000,
            angle_between_projections=0.0,
        )
        simulation.run_simulation(
            output_folder / "high_mean_var",
            run_air_simulation=True,
            clean=False,
            gpu_id=1,
        )

        break
