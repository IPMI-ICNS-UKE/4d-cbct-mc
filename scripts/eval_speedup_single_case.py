import os
from pathlib import Path

# order GPU ID by PCI bus ID
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import logging

import itk
import SimpleITK as sitk

from cbctmc.logger import init_fancy_logging
from cbctmc.speedup.inference import MCSpeedup

if __name__ == "__main__":
    init_fancy_logging()

    logger = logging.getLogger(__name__)

    logger.setLevel(logging.DEBUG)
    logging.getLogger("cbctmc").setLevel(logging.DEBUG)

    # mean model + var model more training (works)
    MODEL = "/datalake3/speedup_runs_all_speedups/2024-01-17T18:17:39.555467_run_f28354c2c6484ce5a7b5a783/models/training/step_20000.pth"
    MODEL = (
        "/home/fmadesta/research/4d-cbct-mc/cbctmc/assets/weights/speedup/default.pth"
    )

    PATIENT_FOLDER = Path("/mnt/nas_io/anarchy/4d_cbct_mc/4d/debug_4d/ct_rai_bin_02")
    SPEEDUP_MODE = "speedup_20.00x"
    FORWARD_PROJECTIONS_FILEPATH = PATIENT_FOLDER / SPEEDUP_MODE / "density_fp_4d.mha"
    LOW_PHOTON_PROJECTIONS_FILEPATH = (
        PATIENT_FOLDER / SPEEDUP_MODE / "projections_total_normalized.mha"
    )

    USE_FORWARD_PROJECTION = True
    GPU_ID = 1
    DEVICE = f"cuda:{GPU_ID}"

    speedup = MCSpeedup.from_filepath(
        model_filepath=MODEL,
        device=DEVICE,
    )

    metrics = {}

    # load projections
    low_photon_projections = sitk.ReadImage(str(LOW_PHOTON_PROJECTIONS_FILEPATH))
    spacing = low_photon_projections.GetSpacing()
    origin = low_photon_projections.GetOrigin()
    direction = low_photon_projections.GetDirection()

    if not FORWARD_PROJECTIONS_FILEPATH.exists():
        FORWARD_PROJECTIONS_FILEPATH = PATIENT_FOLDER / "density_fp.mha"

    logger.info(f"Load forward projection from {FORWARD_PROJECTIONS_FILEPATH}")
    forward_projection = sitk.ReadImage(str(FORWARD_PROJECTIONS_FILEPATH))

    low_photon_projections = sitk.GetArrayFromImage(low_photon_projections)
    forward_projection = sitk.GetArrayFromImage(forward_projection)

    speedup_projections = speedup.execute(
        low_photon=LOW_PHOTON_PROJECTIONS_FILEPATH,
        forward_projection=FORWARD_PROJECTIONS_FILEPATH,
        batch_size=1,
    )

    itk.imwrite(
        speedup_projections,
        PATIENT_FOLDER / SPEEDUP_MODE / "projections_total_normalized_speedup.mha",
    )

    # logger.info("Reconstruct simulation")
    # reconstruct_3d(
    #     projections_filepath=(
    #         patient_folder
    #         / speedup_mode
    #         / "projections_total_normalized_speedup.mha"
    #     ),
    #     geometry_filepath=patient_folder / "geometry.xml",
    #     output_folder=(patient_folder / speedup_mode / "reconstructions"),
    #     output_filename="fdk3d_wpc_speedup.mha",
    #     dimension=(464, 250, 464),
    #     water_pre_correction=ReconDefaults.wpc_catphan604,
    #     gpu_id=GPU_ID,
    # )
    #
    # signal = np.loadtxt(patient_folder / speedup_mode / "signal.txt")
    # reconstruct_4d(
    #     amplitude_signal=signal[:, 0],
    #     projections_filepath=(
    #         patient_folder
    #         / speedup_mode
    #         / "projections_total_normalized_speedup.mha"
    #     ),
    #     geometry_filepath=patient_folder / "geometry.xml",
    #     output_folder=(patient_folder / speedup_mode / "reconstructions"),
    #     output_filename="rooster4d_wpc_speedup.mha",
    #     dimension=(464, 250, 464),
    #     water_pre_correction=ReconDefaults.wpc_catphan604,
    #     gpu_id=GPU_ID,
    # )
