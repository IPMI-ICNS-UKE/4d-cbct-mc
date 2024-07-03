import logging
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Literal, Sequence

import SimpleITK as sitk

from ipmi.common.shell import create_cli_command, execute_in_docker
from ipmi.defaults import DOCKER_PATH_PREFIX
from ipmi.fused_types import PathLike

logger = logging.getLogger(__name__)


def create_ct_segmentations(
    image_filepath: PathLike,
    output_folder: PathLike,
    models: Sequence[Literal["total", "body", "lung_vessels", "bones_tissue"]] = (
        "total",
        "body",
        "lung_vessels",
        "bones_tissue",
    ),
    gpu_id: int = 0,
):
    """Segments a CT scan using the TotalSegmentator [1]. You need the in-house
    imaging docker to run this function [2].

    If no segmentations for given image_filepath exist, models should start with "total"
    as obtained segmentations are used as prior-knowlegede for other
    models/segementations.

    The models lung_vessels and bones_tissue are licensed for academic use only
    (restrictions: max. 1000 cases and no clinical studies).

    References:
    [1] https://arxiv.org/abs/2208.05868
    [2] https://github.com/IPMI-ICNS-UKE/imaging-docker
    """
    image_filepath = Path(image_filepath)
    image_suffix = ".nii"
    maybe_tmp = (
        tempfile.NamedTemporaryFile(suffix=image_suffix)
        if image_filepath.suffix != image_suffix
        else nullcontext()
    )

    with maybe_tmp as tmp:
        if tmp:
            logger.warning(
                f"{image_filepath=} is not a nifti file! "
                f"Will be converted and then passed to TotalSegmentator"
            )
            img = sitk.ReadImage(image_filepath.as_posix())
            image_filepath = image_filepath.parent / tmp.name
            sitk.WriteImage(img, image_filepath.as_posix())
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
