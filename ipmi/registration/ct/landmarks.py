from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import SimpleITK as sitk

import ipmi.defaults as defaults
from ipmi.common.decorators import convert
from ipmi.common.docker.helper import requires_docker_image
from ipmi.common.shell import create_cli_command, execute_in_docker
from ipmi.fused_types import PathLike


@convert("filepath", converter=Path)
def parse_lung_landmarks(filepath: PathLike) -> dict:
    filepath: Path
    _landmarks = np.fromfile(filepath, dtype=np.float32)
    # reshape to (n_landmarks, {x, y, z}_fixed + {x, y, z}_moving)
    _landmarks = _landmarks.reshape((-1, 6))

    landmarks = []
    for landmark in _landmarks:
        landmarks.append(
            {
                "fixed_image": tuple(landmark[:3].tolist()),
                "moving_image": tuple(landmark[3:].tolist()),
            }
        )

    result = {
        "header": {
            "n_landmarks": len(landmarks),
            "landmark_notation": "(x, y, z) voxel indices",
        },
        "landmarks": landmarks,
    }

    return result


@requires_docker_image("imaging")
def extract_lung_landmarks(
    moving_image: sitk.Image,
    fixed_image: sitk.Image,
    fixed_mask: sitk.Image,
    alpha: float = 1.0,
    max_search_radius: str = "16x8",
    cube_length: str = "6x3",
    step_size_quantizazion: str = "2x1",
    patch_similarity_radius: str = "3x2",
    patch_similarity_skipping: str = "2x1",
    foerster_sigma: float = 1.4,
):
    with TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        moving_image_filepath = temp_dir / "moving_image.nii.gz"
        fixed_image_filepath = temp_dir / "fixed_image.nii.gz"
        fixed_mask_filepath = temp_dir / "fixed_mask.nii.gz"
        output_filepath = temp_dir / "landmarks.dat"

        # save images to .nii.gz as required by corrField
        sitk.WriteImage(moving_image, str(moving_image_filepath))
        sitk.WriteImage(fixed_image, str(fixed_image_filepath))
        sitk.WriteImage(fixed_mask, str(fixed_mask_filepath))

        command = create_cli_command(
            executable="corrField",
            prefix="-",
            path_prefix=defaults.DOCKER_PATH_PREFIX,
            F=fixed_image_filepath,
            M=moving_image_filepath,
            m=fixed_mask_filepath,
            O=output_filepath,
            # optional parameters
            a=alpha,
            L=max_search_radius,
            N=cube_length,
            Q=step_size_quantizazion,
            R=patch_similarity_radius,
            S=patch_similarity_skipping,
            s=foerster_sigma,
        )

        execute_in_docker(command)

        landmarks = parse_lung_landmarks(output_filepath)
        landmarks["header"]["moving_image_shape"] = moving_image.GetSize()
        landmarks["header"]["fixed_image_shape"] = fixed_image.GetSize()
        landmarks["header"]["extraction_parameters"] = " ".join(command)

    return landmarks
