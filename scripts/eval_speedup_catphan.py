import os
from pathlib import Path

import torch
import yaml
from tabulate import tabulate

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
    bbox: tuple,
):
    metrics = {}
    # calculate PSNR

    psnr_low_photon = [
        psnr(image=p_low[bbox], reference_image=p_high[bbox])
        for (p_low, p_high) in zip(low_photon_projections, high_photon_projections)
    ]
    psnr_speedup = [
        psnr(image=p_speedup[bbox], reference_image=p_high[bbox])
        for (p_speedup, p_high) in zip(speedup_projections, high_photon_projections)
    ]

    metrics["psnr_low_photon"] = psnr_low_photon
    metrics["psnr_speedup"] = psnr_speedup

    return metrics


if __name__ == "__main__":
    init_fancy_logging()

    logger = logging.getLogger(__name__)

    logger.setLevel(logging.DEBUG)
    logging.getLogger("cbctmc").setLevel(logging.DEBUG)

    SPEEDUP_FACTORS = [10, 20, 30, 40, 50, 100]

    # METRIC_ROI = np.index_exp[321:451, 664:1024]  # just ROI around inserts
    METRIC_ROI = np.index_exp[220:550, 664:1024]  # ROI around inserts + full length

    SPEEDUP_MODES = [
        f"speedup_{speedup_factor:.2f}x" for speedup_factor in SPEEDUP_FACTORS
    ]

    # mean model + var model more training (works)
    MODEL = "/datalake3/speedup_runs_all_speedups/2024-01-17T18:17:39.555467_run_f28354c2c6484ce5a7b5a783/models/training/step_20000.pth"

    USE_FORWARD_PROJECTION = True
    GPU_ID = 1
    DEVICE = f"cuda:{GPU_ID}"
    RERUN = True

    speedup = MCSpeedup.from_filepath(
        model_filepath=MODEL,
        device=DEVICE,
    )

    metrics = {}

    patient_folder = Path(f"/datalake_fast/catphan/geometries/phase_00")

    # load projections
    high_photon_projections_filepath = (
        patient_folder / f"reference/projections_total_normalized.mha"
    )
    logger.info(f"Load high projections from {high_photon_projections_filepath}")
    high_photon_projections = sitk.ReadImage(str(high_photon_projections_filepath))

    high_photon_rerun_projections_filepath = (
        patient_folder / f"reference_rerun/projections_total_normalized.mha"
    )
    high_photon_rerun_projections = sitk.ReadImage(
        str(high_photon_rerun_projections_filepath)
    )

    forward_projection_filepath = patient_folder / f"density_fp.mha"
    logger.info(f"Load forward projection from {forward_projection_filepath}")
    forward_projection = sitk.ReadImage(str(forward_projection_filepath))

    high_photon_projections = sitk.GetArrayFromImage(high_photon_projections)
    high_photon_rerun_projections = sitk.GetArrayFromImage(
        high_photon_rerun_projections
    )
    forward_projection = sitk.GetArrayFromImage(forward_projection)

    psnr_reference_rerun = [
        psnr(image=p_high_rerun[METRIC_ROI], reference_image=p_high[METRIC_ROI])
        for (p_high_rerun, p_high) in zip(
            high_photon_rerun_projections, high_photon_projections
        )
    ]

    metrics["psnr_reference_rerun"] = psnr_reference_rerun

    for speedup_mode in SPEEDUP_MODES:
        low_photon_projections_filepath = (
            patient_folder / f"{speedup_mode}/projections_total_normalized.mha"
        )
        logger.info(f"Load low projections from {low_photon_projections_filepath}")
        low_photon_projections = sitk.ReadImage(str(low_photon_projections_filepath))
        spacing = low_photon_projections.GetSpacing()
        origin = low_photon_projections.GetOrigin()
        direction = low_photon_projections.GetDirection()
        low_photon_projections = sitk.GetArrayFromImage(low_photon_projections)

        if (
            RERUN
            or not (
                patient_folder
                / speedup_mode
                / "projections_total_normalized_speedup.mha"
            ).exists()
        ):
            logger.info(f"{speedup_mode=}")
            (
                speedup_projections_mean,
                speedup_projections_variance,
                speedup_projections,
            ) = speedup.execute(
                low_photon=low_photon_projections,
                forward_projection=forward_projection,
                batch_size=1,
            )

            speedup_projections_itk = sitk.GetImageFromArray(speedup_projections)
            speedup_projections_itk.SetSpacing(spacing)
            speedup_projections_itk.SetOrigin(origin)
            speedup_projections_itk.SetDirection(direction)

            sitk.WriteImage(
                speedup_projections_itk,
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
        else:
            logger.info(
                f"Load speedup projections from {(patient_folder / speedup_mode / 'projections_total_normalized_speedup.mha')}"
            )
            speedup_projections = sitk.ReadImage(
                str(
                    patient_folder
                    / speedup_mode
                    / "projections_total_normalized_speedup.mha"
                )
            )
            speedup_projections = sitk.GetArrayFromImage(speedup_projections)

        metrics[speedup_mode] = {}
        metrics[speedup_mode] = calculate_metrics(
            low_photon_projections=low_photon_projections,
            high_photon_projections=high_photon_projections,
            speedup_projections=speedup_projections,
            bbox=METRIC_ROI,
        )
        print(metrics[speedup_mode])

    table = []

    metrics["psnr_reference_rerun_mean"] = np.mean(metrics["psnr_reference_rerun"])
    metrics["psnr_reference_rerun_std"] = np.std(metrics["psnr_reference_rerun"])

    for speedup_factor in SPEEDUP_FACTORS:
        speedup_mode = f"speedup_{speedup_factor:.2f}x"
        for mode in ("psnr_low_photon", "psnr_speedup"):
            metrics[speedup_mode][f"{mode}_mean"] = np.mean(metrics[speedup_mode][mode])
            metrics[speedup_mode][f"{mode}_std"] = np.std(metrics[speedup_mode][mode])

        table.append(
            {
                "speedup_factor": speedup_factor,
                "psnr_low_photon_mean": metrics[speedup_mode]["psnr_low_photon_mean"],
                "psnr_low_photon_std": metrics[speedup_mode]["psnr_low_photon_std"],
                "psnr_speedup_mean": metrics[speedup_mode]["psnr_speedup_mean"],
                "psnr_speedup_std": metrics[speedup_mode]["psnr_speedup_std"],
            }
        )

    with open(patient_folder / "metrics.yaml", "wt") as f:
        yaml.dump(metrics, f)

    print(
        f"# PSNR reference rerun: {metrics['psnr_reference_rerun_mean']:.6f} "
        f"+/- {metrics['psnr_reference_rerun_std']:.6f}"
    )
    print(tabulate(table, headers="keys", tablefmt="plain"))
