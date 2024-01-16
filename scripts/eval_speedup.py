import os
from pathlib import Path

import torch

# order GPU ID by PCI bus ID
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import logging

import numpy as np
import SimpleITK as sitk
from sklearn.model_selection import train_test_split

from cbctmc.defaults import DefaultReconstructionParameters as ReconDefaults
from cbctmc.logger import init_fancy_logging
from cbctmc.reconstruction.reconstruction import reconstruct_3d
from cbctmc.speedup.inference import MCSpeedup
from cbctmc.speedup.metrics import psnr


def calculate_metrics(
    low_photon_projections: np.ndarray,
    high_photon_projections: np.ndarray,
    speedup_projections: np.ndarray,
):
    metrics = {}
    # calculate PSNR
    metrics["psnr_low_photon"] = psnr(
        image=low_photon_projections, reference_image=high_photon_projections
    )
    metrics["psnr_speedup"] = psnr(
        image=speedup_projections, reference_image=high_photon_projections
    )

    return metrics


if __name__ == "__main__":
    init_fancy_logging()

    logger = logging.getLogger(__name__)

    logger.setLevel(logging.DEBUG)
    logging.getLogger("cbctmc").setLevel(logging.DEBUG)

    DATA_FOLDER = "/datalake3/speedup_dataset"
    RUN_FOLDER = "/datalake3/speedup_runs"

    SPEEDUP_MODES = [
        # "speedup_2.00x",
        # "speedup_5.00x",
        # "speedup_10.00x",
        # "speedup_20.00x",
        "speedup_50.00x",
    ]

    MODELS = {
        # "speedup_2.00x": "/datalake3/speedup_runs_speedup_2.00x/2024-01-14T22:26:49.468697_run_7ae9ad896cd64e829fb5a2e1/models/validation/step_20000.pth",
        # "speedup_5.00x": "/datalake3/speedup_runs_speedup_5.00x/2024-01-15T03:54:22.889148_run_e8fe55c6308d4a17abe8e358/models/validation/step_20000.pth",
        #
        # "speedup_10.00x": "/datalake3/speedup_runs_speedup_5.00x/2024-01-15T03:54:22.889148_run_e8fe55c6308d4a17abe8e358/models/validation/step_20000.pth",
        # "speedup_20.00x": "/datalake3/speedup_runs_speedup_5.00x/2024-01-15T03:54:22.889148_run_e8fe55c6308d4a17abe8e358/models/validation/step_20000.pth",
        "speedup_50.00x": "/datalake3/speedup_runs_speedup_5.00x/2024-01-15T03:54:22.889148_run_e8fe55c6308d4a17abe8e358/models/validation/step_20000.pth",
    }

    USE_FORWARD_PROJECTION = True
    GPU_ID = 1
    DEVICE = f"cuda:{GPU_ID}"

    patient_ids = [
        22,
        24,
        32,
        33,
        68,
        69,
        74,
        78,
        91,
        92,
        104,
        106,
        109,
        115,
        116,
        121,
        124,
        132,
        142,
        145,
        146,
    ]
    train_patients, test_patients = train_test_split(
        patient_ids, train_size=0.75, random_state=42
    )
    test_patients = sorted(test_patients)
    logger.info(f"Test patients ({len(test_patients)}): {test_patients}")

    test_patients = [91]

    metrics = {}
    for speedup_mode in SPEEDUP_MODES:
        metrics[speedup_mode] = {}
        speedup = MCSpeedup.from_filepath(
            model_filepath=MODELS[speedup_mode],
            device=DEVICE,
        )

        for patient in test_patients:
            logger.info(f"{patient=}, {speedup_mode=}")
            patient_folder = Path(
                f"/datalake3/nas_io/anarchy/4d_cbct_mc/speedup/{patient:03d}_4DCT_Lunge_amplitudebased_complete/phase_00"
            )
            # load projections
            low_photon_projections_filepath = (
                patient_folder / f"{speedup_mode}/projections_total_normalized.mha"
            )
            logger.info(f"Load low projections from {low_photon_projections_filepath}")
            low_photon_projections = sitk.ReadImage(
                str(low_photon_projections_filepath)
            )
            spacing = low_photon_projections.GetSpacing()
            origin = low_photon_projections.GetOrigin()
            direction = low_photon_projections.GetDirection()

            high_photon_projections_filepath = (
                patient_folder / f"reference/projections_total_normalized.mha"
            )
            logger.info(
                f"Load high projections from {high_photon_projections_filepath}"
            )
            high_photon_projections = sitk.ReadImage(
                str(high_photon_projections_filepath)
            )
            forward_projection_filepath = patient_folder / f"density_fp.mha"
            logger.info(f"Load forward projection from {forward_projection_filepath}")
            forward_projection = sitk.ReadImage(str(forward_projection_filepath))

            low_photon_projections = sitk.GetArrayFromImage(low_photon_projections)
            high_photon_projections = sitk.GetArrayFromImage(high_photon_projections)
            forward_projection = sitk.GetArrayFromImage(forward_projection)

            (
                speedup_projections_mean,
                speedup_projections_variance,
                speedup_projections,
            ) = speedup.execute(
                low_photon=low_photon_projections,
                forward_projection=forward_projection,
                batch_size=8,
            )

            metrics[speedup_mode][patient] = calculate_metrics(
                low_photon_projections=low_photon_projections,
                high_photon_projections=high_photon_projections,
                speedup_projections=speedup_projections,
            )

            speedup_projections = sitk.GetImageFromArray(speedup_projections)
            speedup_projections.SetSpacing(spacing)
            speedup_projections.SetOrigin(origin)
            speedup_projections.SetDirection(direction)

            sitk.WriteImage(
                speedup_projections,
                str(
                    patient_folder
                    / speedup_mode
                    / f"projections_total_normalized_speedup.mha"
                ),
            )

            logger.info("Reconstruct simulation")
            reconstruct_3d(
                projections_filepath=(
                    patient_folder
                    / speedup_mode
                    / "projections_total_normalized_speedup.mha"
                ),
                geometry_filepath=patient_folder / "geometry.xml",
                output_folder=(patient_folder / speedup_mode / "reconstructions"),
                output_filename="fdk3d_wpc_speedup.mha",
                dimension=(464, 250, 464),
                water_pre_correction=ReconDefaults.wpc_catphan604,
                gpu_id=GPU_ID,
            )
