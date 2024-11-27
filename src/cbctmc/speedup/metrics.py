import numpy as np
import torch

from cbctmc.speedup import constants


def psnr_np(
    image, reference_image, max_pixel_value: float = constants.global_max_pixel_value
) -> float:
    mse = np.mean((image - reference_image) ** 2)

    return 20 * np.log10(max_pixel_value / np.sqrt(mse))


def psnr_torch(
    image, reference_image, max_pixel_value: float = constants.global_max_pixel_value
) -> float:
    mse = torch.mean((image - reference_image) ** 2)

    return 20 * torch.log10(max_pixel_value / torch.sqrt(mse))


def psnr(
    image, reference_image, max_pixel_value: float = constants.global_max_pixel_value
) -> float:
    if isinstance(image, np.ndarray):
        func = psnr_np
    else:
        func = psnr_torch

    return func(
        image=image, reference_image=reference_image, max_pixel_value=max_pixel_value
    )
