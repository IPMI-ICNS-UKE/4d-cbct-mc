from __future__ import annotations

from pathlib import Path

from ipmi import defaults
from ipmi.common.decorators import convert
from ipmi.common.docker.helper import requires_docker_image
from ipmi.common.shell import create_cli_command, execute_in_docker
from ipmi.fused_types import PathLike


@requires_docker_image("imaging")
@convert("image_dicom_folder", converter=Path)
@convert("image_output_filepath", converter=Path)
@convert("rtstruct_folder", converter=Path)
@convert("segmentation_output_folder", converter=Path)
def convert_dicom(
    image_dicom_folder: PathLike,
    image_output_filepath: PathLike | None = None,
    rtstruct_folder: PathLike | None = None,
    segmentation_output_folder: PathLike | None = None,
    image_output_dtype: str = "short",
    segmentation_output_format: str = "nii.gz",
):
    if image_output_filepath:
        command = create_cli_command(
            "plastimatch",
            "convert",
            input=image_dicom_folder,
            output_img=image_output_filepath,
            output_type=image_output_dtype,
            path_prefix=defaults.DOCKER_PATH_PREFIX,
        )
        execute_in_docker(command)

    if rtstruct_folder:
        if not segmentation_output_folder:
            raise RuntimeError("Please specify a segmentation output folder")

        command = create_cli_command(
            "plastimatch",
            "convert",
            input=rtstruct_folder,
            referenced_ct=image_dicom_folder,
            output_prefix=segmentation_output_folder,
            prefix_format=segmentation_output_format,
            path_prefix=defaults.DOCKER_PATH_PREFIX,
        )
        execute_in_docker(command)
