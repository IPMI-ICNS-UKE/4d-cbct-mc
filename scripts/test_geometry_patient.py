import logging
from datetime import datetime
from pathlib import Path

from ipmi.common.logger import init_fancy_logging

from cbctmc.defaults import DefaultMCSimulationParameters
from cbctmc.defaults import DefaultReconstructionParameters as ReconDefaults
from cbctmc.forward_projection import (
    create_geometry,
    prepare_image_for_rtk,
    project_forward,
    save_geometry,
)
from cbctmc.mc.geometry import MCGeometry
from cbctmc.mc.simulation import MCSimulation
from cbctmc.reconstruction.reconstruction import reconstruct_3d

if __name__ == "__main__":
    import itk

    logging.getLogger("cbctmc").setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    init_fancy_logging()

    mc_defaults = DefaultMCSimulationParameters()

    N_PROJECTIONS = 16

    CONFIG = {
        "n_projections": N_PROJECTIONS,
        "angle_between_projections": 360.0 / N_PROJECTIONS,
    }

    GPUS = (0, 1)

    output_folder = Path("/datalake_fast/mc_test/mc_output/geometry_test_new")

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
        image_spacing=(1.0, 1.0, 1.0),
    )

    run_folder = f"run_{datetime.now().isoformat()}"
    run_folder = "run_2023-12-11T13:58:42.081127"
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

    fp_geometry = create_geometry(
        start_angle=90,
        n_projections=N_PROJECTIONS,
        source_to_isocenter=1000.0,
        source_to_detector=1500.0,
        detector_offset_x=mc_defaults.detector_lateral_displacement,
        detector_offset_y=0.0,
    )
    forward_projection = project_forward(
        image,
        geometry=fp_geometry,
    )
    save_geometry(fp_geometry, output_folder / run_folder / "geometry.xml")

    itk.imwrite(
        forward_projection,
        str(output_folder / run_folder / "density_fp.mha"),
    )

    simulation = MCSimulation(geometry=geometry, **CONFIG)
    simulation.run_simulation(
        output_folder / run_folder,
        run_air_simulation=True,
        clean=True,
        gpu_ids=GPUS,
        force_rerun=False,
    )

    reconstruct_3d(
        projections_filepath=(
            output_folder / run_folder / "projections_total_normalized.mha"
        ),
        geometry_filepath=output_folder / run_folder / "geometry.xml",
        output_folder=output_folder / run_folder / "reconstructions",
        output_filename="fdk3d_wpc.mha",
        spacing=(1.0, 1.0, 1.0),
        dimension=(464, 250, 464),
        water_pre_correction=ReconDefaults.wpc_catphan604,
        gpu_id=GPUS[0],
    )
