import os
from pathlib import Path

# order GPU ID by PCI bus ID
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import logging

import SimpleITK as sitk

from cbctmc.defaults import DefaultReconstructionParameters as ReconDefaults
from cbctmc.logger import init_fancy_logging
from cbctmc.reconstruction.reconstruction import reconstruct_3d
from cbctmc.speedup.inference import MCSpeedup

if __name__ == "__main__":
    init_fancy_logging()

    logger = logging.getLogger(__name__)

    logger.setLevel(logging.DEBUG)
    logging.getLogger("cbctmc").setLevel(logging.DEBUG)

    SPEEDUP_MODES = [
        "speedup_10.00x",
        # "speedup_20.00x",
        # "speedup_50.00x",
    ]

    # mean model + var model more training (works)
    MODEL = "/datalake3/speedup_runs_all_speedups/2024-01-17T18:17:39.555467_run_f28354c2c6484ce5a7b5a783/models/training/step_40000.pth"

    USE_FORWARD_PROJECTION = True
    GPU_ID = 0
    DEVICE = f"cuda:{GPU_ID}"

    speedup = MCSpeedup.from_filepath(
        model_filepath=MODEL,
        device=DEVICE,
    )

    gaps = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.50, 4.00]
    gaps = [1.00]

    for speedup_mode in SPEEDUP_MODES:
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

            forward_projection = sitk.ReadImage(str(folder / f"density_fp.mha"))
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
