from __future__ import annotations

import logging
from math import sqrt
from typing import Optional, Tuple

import numpy as np
import scipy
import torch
from torch.nn import functional as F

from vroc.decorators import timing
from vroc.interpolation import resize_spacing

logger = logging.getLogger(__name__)


def _apply_mask(
    image: np.ndarray, reference_image: np.ndarray, mask: Optional[np.ndarray] = None
):
    if mask is not None:
        assert image.shape == reference_image.shape == mask.shape, "Dimension mismach"
        if mask.dtype is not np.bool:
            mask = mask.astype(np.bool)
        image = image[mask]
        reference_image = reference_image[mask]
    return image, reference_image


def mean_squared_error(
    image: np.ndarray, reference_image: np.ndarray, mask: Optional[np.ndarray] = None
) -> float:
    assert image.shape == reference_image.shape, "Dimension mismach"
    image, reference_image = _apply_mask(image, reference_image, mask)

    return float(((image - reference_image) ** 2).mean())


def root_mean_squared_error(
    image: np.ndarray, reference_image: np.ndarray, mask: Optional[np.ndarray] = None
) -> float:
    return sqrt(
        mean_squared_error(image=image, reference_image=reference_image, mask=mask)
    )


def mse_improvement(before: float, after: float) -> float:
    return (after - before) / before


def patch_mean(images, patch_shape):

    channels, *patch_size = patch_shape
    dimensions = len(patch_size)
    padding = tuple(side // 2 for side in patch_size)

    conv = (F.conv1d, F.conv2d, F.conv3d)[dimensions - 1]

    # Convolution with these weights will effectively compute the channel-wise means
    patch_elements = torch.Tensor(patch_size).prod().item()
    weights = torch.full(
        (channels, channels, *patch_size), fill_value=1 / patch_elements
    )
    weights = weights.to(images.device)

    # Make convolution operate on single channels
    channel_selector = torch.eye(channels).byte()
    weights[1 - channel_selector] = 0

    result = conv(images, weights, padding=padding, bias=None)

    return result


def patch_std(image, patch_shape):

    return (
        patch_mean(image**2, patch_shape) - patch_mean(image, patch_shape) ** 2
    ).sqrt()


def channel_normalize(template):
    """Z-normalize image channels independently."""
    reshaped_template = template.clone().view(template.shape[0], -1)
    reshaped_template.sub_(reshaped_template.mean(dim=-1, keepdim=True))
    reshaped_template.div_(reshaped_template.std(dim=-1, keepdim=True, unbiased=False))

    return reshaped_template.view_as(template)


class NCC(torch.nn.Module):
    def __init__(self, template, keep_channels=False):
        super().__init__()

        self.keep_channels = keep_channels

        channels, *template_shape = template.shape
        dimensions = len(template_shape)
        self.padding = tuple(side // 2 for side in template_shape)

        self.conv_f = (F.conv1d, F.conv2d, F.conv3d)[dimensions - 1]
        self.normalized_template = channel_normalize(template)
        ones = template.dim() * (1,)
        self.normalized_template = self.normalized_template.repeat(channels, *ones)
        # Make convolution operate on single channels
        channel_selector = torch.eye(channels).byte()
        self.normalized_template[1 - channel_selector] = 0
        # Reweight so that output is averaged
        patch_elements = torch.Tensor(template_shape).prod().item()
        self.normalized_template.div_(patch_elements)

    def forward(self, image):
        result = self.conv_f(
            image, self.normalized_template, padding=self.padding, bias=None
        )
        std = patch_std(image, self.normalized_template.shape[1:])
        result.div_(std)
        if not self.keep_channels:
            result = result.mean(dim=1)

        return result


@timing()
def dice_coefficient(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    image_spacing: Tuple[int, ...] | None = None,
    label: int = 1,
):
    logger.debug(f"Calculate dice score on segmentations with shape {prediction.shape}")
    prediction = np.asarray(prediction)
    ground_truth = np.asarray(ground_truth)
    if image_spacing:
        isotropic_spacing = (1.0,) * len(image_spacing)

        if isotropic_spacing != image_spacing:
            prediction = resize_spacing(
                prediction,
                input_image_spacing=image_spacing,
                output_image_spacing=isotropic_spacing,
                order=0,
            )
            ground_truth = resize_spacing(
                ground_truth,
                input_image_spacing=image_spacing,
                output_image_spacing=isotropic_spacing,
                order=0,
            )
            logger.debug(
                f"Resampled to isotropic image spacing, new shape is {prediction.shape}"
            )

    if prediction.shape != ground_truth.shape:
        raise ValueError("Shape mismatch")

    ground_truth = ground_truth == label
    prediction = prediction == label

    ground_truth_sum = ground_truth.sum()
    prediction_sum = prediction.sum()

    if ground_truth_sum == 0 and prediction_sum == 0:
        dice = 1.0
    elif (ground_truth_sum + prediction_sum) != 0:
        dice = (
            2 * (prediction & ground_truth).sum() / (prediction_sum + ground_truth_sum)
        )
    else:
        dice = 0.0

    return dice


def jacobian_determinant(disp):
    _, _, H, W, D = disp.shape

    gradx = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
    grady = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
    gradz = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)

    gradx_disp = np.stack(
        [
            scipy.ndimage.correlate(
                disp[:, 0, :, :, :], gradx, mode="constant", cval=0.0
            ),
            scipy.ndimage.correlate(
                disp[:, 1, :, :, :], gradx, mode="constant", cval=0.0
            ),
            scipy.ndimage.correlate(
                disp[:, 2, :, :, :], gradx, mode="constant", cval=0.0
            ),
        ],
        axis=1,
    )

    grady_disp = np.stack(
        [
            scipy.ndimage.correlate(
                disp[:, 0, :, :, :], grady, mode="constant", cval=0.0
            ),
            scipy.ndimage.correlate(
                disp[:, 1, :, :, :], grady, mode="constant", cval=0.0
            ),
            scipy.ndimage.correlate(
                disp[:, 2, :, :, :], grady, mode="constant", cval=0.0
            ),
        ],
        axis=1,
    )

    gradz_disp = np.stack(
        [
            scipy.ndimage.correlate(
                disp[:, 0, :, :, :], gradz, mode="constant", cval=0.0
            ),
            scipy.ndimage.correlate(
                disp[:, 1, :, :, :], gradz, mode="constant", cval=0.0
            ),
            scipy.ndimage.correlate(
                disp[:, 2, :, :, :], gradz, mode="constant", cval=0.0
            ),
        ],
        axis=1,
    )

    grad_disp = np.concatenate([gradx_disp, grady_disp, gradz_disp], 0)

    jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = (
        jacobian[0, 0, :, :, :]
        * (
            jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :]
            - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]
        )
        - jacobian[1, 0, :, :, :]
        * (
            jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :]
            - jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :]
        )
        + jacobian[2, 0, :, :, :]
        * (
            jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :]
            - jacobian[0, 2, :, :, :] * jacobian[1, 1, :, :, :]
        )
    )

    return jacdet
