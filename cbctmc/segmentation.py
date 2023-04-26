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
            model = "bones_tissue_test"
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


if __name__ == "__main__":
    from ipmi.common.logger import init_fancy_logging

    init_fancy_logging()

    logger.setLevel(logging.INFO)

    image_filepaths = sorted(
        Path("/datalake2/Totalsegmentator_dataset").glob("*/ct.nii.gz")
    )

    image_filepaths = sorted(Path("/datalake/mega/luna16/images").glob("*mhd"))

    for image_filepath in image_filepaths:
        print(image_filepath)
        image = sitk.ReadImage(str(image_filepath))
        sitk.WriteImage(
            image,
            f"/datalake2/luna16/images_nii/{image_filepath.with_suffix('.nii').name}",
        )

    image_filepaths = sorted(Path("/datalake2/luna16/images_nii").glob("*"))

    for image_filepath in image_filepaths:
        output_folder = (
            image_filepath.parent / "predicted_segmentations" / image_filepath.stem
        )
        if (output_folder / "subcutaneous_fat.nii.gz").exists():
            logger.info(f"Skipping {image_filepath}")
            continue
        output_folder.mkdir(parents=True, exist_ok=True)

        try:
            create_ct_segmentations(
                image_filepath=image_filepath,
                output_folder=output_folder,
                gpu_id=1,
                models=("total", "body", "lung_vessels", "bones_tissue"),
            )
        except Exception as e:
            print(e)
