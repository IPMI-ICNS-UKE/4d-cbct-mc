import logging
from pathlib import Path
from typing import Sequence

import SimpleITK as sitk
from ipmi.common.dataio.segmentation import merge_segmentations
from ipmi.common.shell import create_cli_command, execute_in_docker
from ipmi.defaults import DOCKER_PATH_PREFIX

from cbctmc.common_types import PathLike

logger = logging.getLogger(__name__)


def create_ct_segmentations(
    image_filepath: PathLike,
    output_folder: PathLike,
    models: Sequence[str] = ("total", "body", "lung_vessels", "bones_tissue"),
    gpu_id: int = 0,
):
    for model in models:
        # rename it for TotalSegmentator
        if model == "bones_tissue":
            models = "bones_tissue_test"
        logger.info(
            f"Start segmentation of {image_filepath!s} "
            f"using {model=} on GPU (PCI {gpu_id})"
        )
        command = create_cli_command(
            "TotalSegmentator",
            ta=model,
            i=Path(image_filepath),
            o=Path(output_folder),
            s=True,  # calculate segmentation statistics
            path_prefix=DOCKER_PATH_PREFIX,
            prefix="-",
        )
        execute_in_docker(command, gpus=[gpu_id])


def _merge_segmentations(
    folder: PathLike, glob_patterns: Sequence[str], output_filename: str
):
    folder = Path(folder)
    filepaths = []
    for glob_pattern in glob_patterns:
        filepaths += list(folder.glob(glob_pattern))
    if not filepaths:
        raise RuntimeError(
            f"No segmentations found in {folder=!s} for {glob_patterns=}"
        )
    merged, _ = merge_segmentations(filepaths, multi_label=False)
    merged = merged > 0
    sitk.WriteImage(merged, str(folder / output_filename))


def merge_upper_body_bone_segmentations(folder: PathLike):
    glob_patterns = (
        "rib_*",
        "vertebrae_*",
        "clavicula_*",
        "scapula_*",
        "humerus_*",
        "sternum*",
    )

    _merge_segmentations(
        folder=folder,
        glob_patterns=glob_patterns,
        output_filename="upper_body_bones.nii.gz",
    )


def merge_upper_body_muscle_segmentations(folder: PathLike):
    glob_patterns = ("autochthon_*", "iliopsoas_*" "skeletal_muscle*")

    _merge_segmentations(
        folder=folder,
        glob_patterns=glob_patterns,
        output_filename="upper_body_muscles.nii.gz",
    )


def merge_upper_body_fat_segmentations(folder: PathLike):
    glob_patterns = ("torso_fat*", "subcutaneous_fat*")

    _merge_segmentations(
        folder=folder,
        glob_patterns=glob_patterns,
        output_filename="upper_body_fat.nii.gz",
    )
