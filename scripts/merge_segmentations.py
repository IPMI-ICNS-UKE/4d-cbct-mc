import logging
from pathlib import Path

from cbctmc.segmentation.utils import merge_segmentations_of_folders

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    from ipmi.common.logger import init_fancy_logging

    init_fancy_logging()

    logger.setLevel(logging.INFO)

    image_filepaths = sorted(Path("/datalake2/luna16/images_nii").glob("*.nii"))

    folders = []
    for image_filepath in image_filepaths:
        output_folder = (
            image_filepath.parent / "predicted_segmentations" / image_filepath.stem
        )
        folders.append(output_folder)

    merge_segmentations_of_folders(folders, n_processes=16)
