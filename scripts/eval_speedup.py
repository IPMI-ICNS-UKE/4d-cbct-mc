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
from cbctmc.reconstruction.reconstruction import reconstruct_3d, reconstruct_4d
from cbctmc.speedup.inference import MCSpeedup
from cbctmc.speedup.metrics import psnr

if __name__ == "__main__":
    init_fancy_logging()

    logger = logging.getLogger(__name__)

    logger.setLevel(logging.DEBUG)
    logging.getLogger("cbctmc").setLevel(logging.DEBUG)

    SPEEDUP_MODES = [
        # "speedup_10.00x",
        "speedup_20.00x",
        # "speedup_50.00x",
    ]

    # mean model + var model more training (works)
    MODEL = "/datalake3/speedup_runs_all_speedups/2024-01-17T18:17:39.555467_run_f28354c2c6484ce5a7b5a783/models/training/step_20000.pth"

    USE_FORWARD_PROJECTION = True
    GPU_ID = 1
    DEVICE = f"cuda:{GPU_ID}"

    speedup = MCSpeedup.from_filepath(
        model_filepath=MODEL,
        device=DEVICE,
    )

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

    test_patients = [24]

    metrics = {}
    for speedup_mode in SPEEDUP_MODES:
        metrics[speedup_mode] = {}

        for patient in test_patients:
            logger.info(f"{patient=}, {speedup_mode=}")
            # patient_folder = Path(
            #     f"/mnt/nas_io/anarchy/4d_cbct_mc/speedup/{patient:03d}_4DCT_Lunge_amplitudebased_complete/phase_00"
            # )

            patient_folder = Path(
                f"/mnt/nas_io/anarchy/4d_cbct_mc/4d/R2017025/mc/ct_rai/phase_02"
            )

            # patient_folder = Path("/mnt/nas_io/anarchy/4d_cbct_mc/4d/024_4DCT_Lunge_amplitudebased_complete/phase_02")

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

            if not (
                forward_projection_filepath := patient_folder
                / speedup_mode
                / f"density_fp_4d.mha"
            ).exists():
                forward_projection_filepath = patient_folder / f"density_fp.mha"

            logger.info(f"Load forward projection from {forward_projection_filepath}")
            forward_projection = sitk.ReadImage(str(forward_projection_filepath))

            low_photon_projections = sitk.GetArrayFromImage(low_photon_projections)
            forward_projection = sitk.GetArrayFromImage(forward_projection)

            (
                speedup_projections_mean,
                speedup_projections_variance,
                speedup_projections,
            ) = speedup.execute(
                low_photon=low_photon_projections,
                forward_projection=forward_projection,
                batch_size=1,
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

            signal = np.loadtxt(patient_folder / speedup_mode / "signal.txt")
            reconstruct_4d(
                amplitude_signal=signal[:, 0],
                projections_filepath=(
                    patient_folder
                    / speedup_mode
                    / "projections_total_normalized_speedup.mha"
                ),
                geometry_filepath=patient_folder / "geometry.xml",
                output_folder=(patient_folder / speedup_mode / "reconstructions"),
                output_filename="rooster4d_wpc_speedup.mha",
                dimension=(464, 250, 464),
                water_pre_correction=ReconDefaults.wpc_catphan604,
                gpu_id=GPU_ID,
            )
