import os
from pathlib import Path
from pprint import pprint

import torch

from cbctmc.metrics import normalized_cross_correlation

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
        "speedup_2.00x",
        "speedup_5.00x",
        "speedup_10.00x",
        "speedup_20.00x",
        "speedup_50.00x",
    ]

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

    metrics = {}

    for patient in patient_ids:
        logger.info(f"{patient=}")
        metrics[patient] = {}
        patient_folder = Path(
            f"/datalake3/nas_io/anarchy/4d_cbct_mc/speedup/{patient:03d}_4DCT_Lunge_amplitudebased_complete/phase_00"
        )

        high_photon_projections = sitk.ReadImage(
            str(patient_folder / f"reference/projections_total_normalized.mha")
        )
        forward_projection = sitk.ReadImage(str(patient_folder / f"density_fp.mha"))
        high_photon_projections = sitk.GetArrayFromImage(high_photon_projections)
        forward_projection = sitk.GetArrayFromImage(forward_projection)

        metrics[patient]["reference"] = normalized_cross_correlation(
            image=high_photon_projections,
            reference_image=forward_projection,
        )

        for speedup_mode in SPEEDUP_MODES:
            logger.info(f"{patient=}, {speedup_mode=}")
            metrics[speedup_mode] = {}
            # load projections
            low_photon_projections = sitk.ReadImage(
                str(patient_folder / f"{speedup_mode}/projections_total_normalized.mha")
            )
            spacing = low_photon_projections.GetSpacing()
            origin = low_photon_projections.GetOrigin()
            direction = low_photon_projections.GetDirection()

            low_photon_projections = sitk.GetArrayFromImage(low_photon_projections)

            metrics[patient][speedup_mode] = normalized_cross_correlation(
                image=low_photon_projections,
                reference_image=forward_projection,
            )

        pprint(metrics[patient], sort_dicts=False)
