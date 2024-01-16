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
        "speedup_10.00x",
        # "speedup_20.00x",
        # "speedup_50.00x",
    ]

    MODELS = {
        "speedup_2.00x": "/datalake3/speedup_runs_speedup_2.00x/2024-01-14T22:26:49.468697_run_7ae9ad896cd64e829fb5a2e1/models/validation/step_20000.pth",
        "speedup_5.00x": "/datalake3/speedup_runs_speedup_5.00x/2024-01-15T03:54:22.889148_run_e8fe55c6308d4a17abe8e358/models/validation/step_20000.pth",
        "speedup_10.00x": "/datalake3/speedup_runs_speedup_5.00x/2024-01-15T03:54:22.889148_run_e8fe55c6308d4a17abe8e358/models/validation/step_20000.pth",
        "speedup_20.00x": "/datalake3/speedup_runs_speedup_5.00x/2024-01-15T03:54:22.889148_run_e8fe55c6308d4a17abe8e358/models/validation/step_20000.pth",
        "speedup_50.00x": "/datalake3/speedup_runs_speedup_5.00x/2024-01-15T03:54:22.889148_run_e8fe55c6308d4a17abe8e358/models/validation/step_20000.pth",
    }

    USE_FORWARD_PROJECTION = True
    GPU_ID = 0
    DEVICE = f"cuda:{GPU_ID}"

    gaps = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.50, 4.00]

    metrics = {}
    for speedup_mode in SPEEDUP_MODES:
        metrics[speedup_mode] = {}
        speedup = MCSpeedup.from_filepath(
            model_filepath=MODELS[speedup_mode],
            device=DEVICE,
        )

        for gap in gaps:
            logger.info(f"{gap=}, {speedup_mode=}")
            folder = Path(
                f"/mnt/nas_io/anarchy/4d_cbct_mc/mc_mtf_final/lp_{gap:.2f}gap"
            )
            # load projections
            low_photon_projections = sitk.ReadImage(
                str(folder / f"{speedup_mode}/projections_total_normalized.mha")
            )
            spacing = low_photon_projections.GetSpacing()
            origin = low_photon_projections.GetOrigin()
            direction = low_photon_projections.GetDirection()

            high_photon_projections = sitk.ReadImage(
                str(folder / f"reference/projections_total_normalized.mha")
            )
            forward_projection = sitk.ReadImage(str(folder / f"density_fp.mha"))
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
                batch_size=1,
            )

            metrics[speedup_mode][gap] = calculate_metrics(
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
                    folder / speedup_mode / f"projections_total_normalized_speedup.mha"
                ),
            )

            logger.info("Reconstruct simulation")
            reconstruct_3d(
                projections_filepath=(
                    folder / speedup_mode / "projections_total_normalized_speedup.mha"
                ),
                geometry_filepath=folder / "geometry.xml",
                output_folder=(folder / speedup_mode / "reconstructions"),
                output_filename="fdk3d_wpc_speedup.mha",
                dimension=(464, 250, 464),
                water_pre_correction=ReconDefaults.wpc_catphan604,
                gpu_id=GPU_ID,
            )
