import logging
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from ipmi.common.logger import init_fancy_logging

from cbctmc.segmentation.utils import (
    _merge_segmentations,
    create_ct_segmentations,
    merge_upper_body_segmentations,
)
from cbctmc.utils import get_robust_bounding_box_3d

init_fancy_logging()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("ipmi").setLevel(logging.INFO)

OUTPUT_FOLDER = Path("/datalake/totalsegmentator_mc")
OUTPUT_FOLDER.mkdir(exist_ok=True)

GPU_PCI_ID = 0

LUNG_SEGMENTATION_NAMES = (
    "lung_lower_lobe_left.nii.gz",
    "lung_lower_lobe_right.nii.gz",
    "lung_middle_lobe_right.nii.gz",
    "lung_upper_lobe_left.nii.gz",
    "lung_upper_lobe_right.nii.gz",
)


def create_z_cropper(mask: sitk.Image):
    mask = sitk.GetArrayFromImage(mask)
    mask = np.swapaxes(mask, 0, 2)
    bbox = get_robust_bounding_box_3d(mask)

    z_start = bbox[2].start
    z_stop = bbox[2].stop

    lower_crop_size = (0, 0, int(z_start))
    upper_crop_size = (0, 0, int(mask.shape[2] - z_stop))

    crop_filter = sitk.CropImageFilter()
    crop_filter.SetLowerBoundaryCropSize(lower_crop_size)
    crop_filter.SetUpperBoundaryCropSize(upper_crop_size)

    return crop_filter


def preprocess(folder: Path, output_folder: Path):
    segmentation_folder = folder / "segmentations"
    case_name = folder.name
    case_output_folder = output_folder / case_name
    output_segmentation_folder = case_output_folder / "segmentations"
    output_segmentation_folder.mkdir(parents=True, exist_ok=True)

    if any(output_segmentation_folder.iterdir()):
        # aleady preprocessed: skip folder
        logger.info(f"Skip {folder}")
        return

    bounding_bones = _merge_segmentations(
        folder=segmentation_folder,
        glob_patterns=(
            "rib_*",
            "clavicula_*",
            "scapula_*",
            "humerus_*",
            "sternum*",
        ),
        output_filename=None,
    )

    cropper = create_z_cropper(bounding_bones)

    image = sitk.ReadImage(str(folder / "ct.nii.gz"))
    image = cropper.Execute(image)

    sitk.WriteImage(image, str(case_output_folder / "ct.nii.gz"))

    for segmentation_filepath in segmentation_folder.glob("*.nii.gz"):
        logger.info(f"Crop {segmentation_filepath}")
        segmentation = sitk.ReadImage(str(segmentation_filepath))
        segmentation = cropper.Execute(segmentation)
        sitk.WriteImage(
            segmentation, str(output_segmentation_folder / segmentation_filepath.name)
        )


def create_additional_segmentations(folder: Path, force: bool = False):
    if not force and (folder / "segmentations" / "skin.nii.gz").exists():
        logger.info(f"Skipping {folder}")
        return
    create_ct_segmentations(
        image_filepath=folder / "ct.nii.gz",
        output_folder=folder / "segmentations",
        models=(
            "body",
            "lung_vessels",
            "bones_tissue",
        ),
        gpu_id=GPU_PCI_ID,
    )


if __name__ == "__main__":
    # this is the unprocessed/original Total Segmentator data set
    ROOT_PATH = Path("/datalake/totalsegmentator")

    # get cases with lung
    lung_cases = []
    for folder in sorted(Path(ROOT_PATH).glob("s*")):
        logger.info(f"Processing {folder}")
        include = True
        for lung_segmentation_name in LUNG_SEGMENTATION_NAMES:
            lung_segmentation = sitk.ReadImage(
                str(folder / "segmentations" / lung_segmentation_name)
            )
            lung_segmentation = sitk.GetArrayFromImage(lung_segmentation)
            if not lung_segmentation.any():
                include = False
                break

        if include:
            lung_cases.append(folder)

    logger.info(f"lung_cases: {lung_cases}")

    for lung_case in lung_cases:
        preprocess(lung_case, output_folder=OUTPUT_FOLDER)

    preprocessed_lung_cases = sorted(Path(OUTPUT_FOLDER).glob("s*"))

    for preprocessed_lung_case in preprocessed_lung_cases:
        logger.info(f"Processing {preprocessed_lung_case}")
        create_additional_segmentations(preprocessed_lung_case)
        merge_upper_body_segmentations(
            preprocessed_lung_case / "segmentations", overwrite=True
        )
