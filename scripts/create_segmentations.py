import logging
from pathlib import Path

from ipmi.common.logger import init_fancy_logging

from cbctmc.segmentation import create_ct_segmentations

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    init_fancy_logging()

    logger.setLevel(logging.INFO)

    INPUT_FOLDER = Path("/datalake_fast/4d_ct_lung_uke_artifact_free")

    patients = sorted(INPUT_FOLDER.glob("*"))

    for patient in patients:
        for i_phase in range(10):
            image_filepath = patient / f"phase_{i_phase:02d}.nii"
            output_folder = patient / "segmentations" / f"phase_{i_phase:02d}"
            output_folder.mkdir(parents=True, exist_ok=True)

            create_ct_segmentations(
                image_filepath=image_filepath,
                output_folder=output_folder,
                gpu_id=1,
            )
