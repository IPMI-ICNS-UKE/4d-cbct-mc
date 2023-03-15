from typing import Tuple

import numpy as np
import torch

from cbctmc.common_types import ArrayOrTensor


def iec61217_to_rsp(image):
    size = image.GetSize()
    spacing = image.GetSpacing()
    dimension = image.GetDimension()

    if dimension == 3:
        image.SetDirection((1, 0, 0, 0, 0, -1, 0, -1, 0))
        origin = np.subtract(
            image.GetOrigin(),
            image.TransformContinuousIndexToPhysicalPoint(
                (size[0] / 2, size[1] / 2, size[2] / 2)
            ),
        )
        origin = np.add(origin, (spacing[0] / 2, -spacing[1] / 2, -spacing[2] / 2))
    elif dimension == 4:
        image.SetDirection((1, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 1))
        origin = np.subtract(
            image.GetOrigin(),
            image.TransformContinuousIndexToPhysicalPoint(
                (size[0] / 2, size[1] / 2, size[2] / 2, size[3] / 2)
            ),
        )
        origin = np.add(
            origin, (spacing[0] / 2, -spacing[1] / 2, -spacing[2] / 2, spacing[0] / 2)
        )

    image.SetOrigin(origin)

    return image


def rescale_range(
    values: ArrayOrTensor, input_range: Tuple, output_range: Tuple, clip: bool = True
):
    if input_range and output_range and (input_range != output_range):
        is_tensor = isinstance(values, torch.Tensor)
        in_min, in_max = input_range
        out_min, out_max = output_range
        values = (
            ((values - in_min) * (out_max - out_min)) / (in_max - in_min)
        ) + out_min
        if clip:
            clip_func = torch.clip if is_tensor else np.clip
            values = clip_func(values, out_min, out_max)

    return values
