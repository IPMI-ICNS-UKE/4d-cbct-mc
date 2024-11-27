from __future__ import annotations

import logging
import multiprocessing
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
    folder: PathLike,
    glob_patterns: Sequence[str],
    output_filename: str | None = None,
    overwrite: bool = False,
):
    folder = Path(folder)
    if not overwrite and output_filename and (folder / output_filename).exists():
        return

    filepaths = []
    for glob_pattern in glob_patterns:
        filepaths += list(folder.glob(glob_pattern))
    if not filepaths:
        raise RuntimeError(
            f"No segmentations found in {folder=!s} for {glob_patterns=}"
        )
    merged, _ = merge_segmentations(filepaths, multi_label=False)
    merged = merged > 0
    if output_filename:
        sitk.WriteImage(merged, str(folder / output_filename))

    return merged


def merge_upper_body_bone_segmentations(
    folder: PathLike, save: bool = True, overwrite: bool = False
) -> sitk.Image:
    glob_patterns = (
        "rib_*",
        "vertebrae_*",
        "clavicula_*",
        "scapula_*",
        "humerus_*",
        "sternum*",
    )

    return _merge_segmentations(
        folder=folder,
        glob_patterns=glob_patterns,
        output_filename="upper_body_bones.nii.gz" if save else None,
        overwrite=overwrite,
    )


def merge_rib_body_bone_segmentations(
    folder: PathLike, save: bool = True, overwrite: bool = False
) -> sitk.Image:
    glob_patterns = ("rib_*",)

    return _merge_segmentations(
        folder=folder,
        glob_patterns=glob_patterns,
        output_filename="ribs.nii.gz" if save else None,
        overwrite=overwrite,
    )


def merge_upper_body_muscle_segmentations(
    folder: PathLike, save: bool = True, overwrite: bool = False
):
    glob_patterns = ("autochthon_*", "iliopsoas_*", "skeletal_muscle*")

    return _merge_segmentations(
        folder=folder,
        glob_patterns=glob_patterns,
        output_filename="upper_body_muscles.nii.gz" if save else None,
        overwrite=overwrite,
    )


def merge_upper_body_fat_segmentations(
    folder: PathLike, save: bool = True, overwrite: bool = False
):
    glob_patterns = ("torso_fat*", "subcutaneous_fat*")

    return _merge_segmentations(
        folder=folder,
        glob_patterns=glob_patterns,
        output_filename="upper_body_fat.nii.gz" if save else None,
        overwrite=overwrite,
    )


def merge_upper_body_segmentations(folder: Path, overwrite: bool = True):
    logger.info(f"Merging {folder}")
    merge_upper_body_bone_segmentations(folder, overwrite=overwrite)
    merge_upper_body_muscle_segmentations(folder, overwrite=overwrite)
    merge_upper_body_fat_segmentations(folder, overwrite=overwrite)


def merge_segmentations_of_folders(
    folders: Sequence[Path], n_processes: int | None = None
):
    with multiprocessing.Pool(n_processes) as pool:
        pool.map(merge_upper_body_segmentations, folders)


# if __name__ == "__main__":
#     from ipmi.common.logger import init_fancy_logging
#
#     init_fancy_logging()
#
#     logger.setLevel(logging.INFO)

# image_filepaths = sorted(
#     Path("/datalake2/Totalsegmentator_dataset").glob("*/ct.nii.gz")
# )

# image_filepaths = sorted(Path("/datalake/mega/luna16/images").glob("*mhd"))
#
# for image_filepath in image_filepaths:
#     print(image_filepath)
#     image = sitk.ReadImage(str(image_filepath))
#     sitk.WriteImage(
#         image,
#         f"/datalake2/luna16/images_nii/{image_filepath.with_suffix('.nii').name}",
#     )

# image_filepaths = sorted(Path("/datalake2/luna16/images_nii").glob("*"))
#
# for image_filepath in image_filepaths:
#
#     output_folder = (
#         image_filepath.parent / "predicted_segmentations" / image_filepath.stem
#     )
#
#     if not (output_folder / "subcutaneous_fat.nii.gz").exists():
#         logger.info(f"Skipping {image_filepath}")
#         continue
#
#     output_folder.mkdir(parents=True, exist_ok=True)
#
#     try:
#         create_ct_segmentations(
#             image_filepath=image_filepath,
#             output_folder=output_folder,
#             gpu_id=1,
#             models=("total", "body", "lung_vessels", "bones_tissue"),
#         )
#     except Exception as e:
#         print(e)
