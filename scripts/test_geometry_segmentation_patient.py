import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from ipmi.common.logger import init_fancy_logging

from cbctmc.mc.geometry import MCGeometry
from cbctmc.segmentation.labels import LABELS
from cbctmc.segmentation.segmenter import MCSegmenter
from cbctmc.speedup.models import FlexUNet

if __name__ == "__main__":
    logging.getLogger("cbctmc").setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    init_fancy_logging()

    output_folder = Path("/datalake_fast/mc_test/mc_output/geometry_segmentation_test")

    output_folder.mkdir(parents=True, exist_ok=True)

    patient_folder = Path(
        "/datalake_fast/4d_ct_lung_uke_artifact_free/024_4DCT_Lunge_amplitudebased_complete"
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

    enc_filters = [32, 32, 32, 32]
    dec_filters = [32, 32, 32, 32]

    model = FlexUNet(
        n_channels=1,
        n_classes=len(LABELS),
        n_levels=4,
        n_filters=[32, *enc_filters, *dec_filters, 32],
        convolution_layer=nn.Conv3d,
        downsampling_layer=nn.MaxPool3d,
        upsampling_layer=nn.Upsample,
        norm_layer=nn.InstanceNorm3d,
        skip_connections=True,
        convolution_kwargs=None,
        downsampling_kwargs=None,
        upsampling_kwargs=None,
        return_bottleneck=False,
    )
    state = torch.load(
        # "/datalake2/runs/mc_segmentation/models_9566ce013dfd4797a8b6a0a6/validation/step_155000.pth"
        # "/datalake2/runs/mc_material_segmentation/5147b8f71bb14732b310ec72/models/validation/step_19000.pth"
        "/datalake2/runs/mc_material_segmentation_totalsegmentator/2023-09-20T12:07:30.455221_run_cc0235df0c3a41bb9830774c/models/training/step_75000.pth"
        # "/datalake2/runs/mc_segmentation/"
        # "models_0961491db6c842c3958ffb1d/validation/step_72000.pth",
        # "/datalake2/runs/mc_segmentation/models_3f44803be7e542dba54e8ebd/validation/step_15000.pth"
    )
    model.load_state_dict(state["model"])

    geometry = MCGeometry.from_image(
        image_filepath=patient_folder / "phase_00.nii",
        segmenter=MCSegmenter(
            model=model,
            device="cuda:0",
            patch_shape=(128, 128, 128),
            patch_overlap=0.75,
        ),
        image_spacing=(1.0, 1.0, 1.0),
    )

    geometry.save_material_segmentation(
        output_folder / run_folder / "geometry_materials_segmenter.nii.gz"
    )

    geometry.save_density_image(
        output_folder / run_folder / "geometry_densities_segmenter.nii.gz"
    )
