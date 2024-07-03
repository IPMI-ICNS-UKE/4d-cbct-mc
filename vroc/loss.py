from __future__ import annotations

import warnings
from typing import Callable, List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from vroc.blocks import SpatialTransformer
from vroc.common_types import FloatTuple3D, IntTuple3D
from vroc.helper import to_one_hot


class TRELoss(nn.Module):
    def __init__(
        self,
        apply_sqrt: bool = False,
        reduction: Literal["mean", "sum", "none", "quantile_0.95"] | None = "mean",
    ):
        super().__init__()
        self.apply_sqrt = apply_sqrt
        # convert PyTorch's unpythonic string "none"
        self.reduction = reduction if reduction != "none" else None

        if self.reduction and self.reduction.startswith("quantile"):
            self.reduction, self.quantile = self.reduction.split("_")
            self.quantile = float(self.quantile)

    @staticmethod
    def _warped_fixed_landmarks(
        vector_field: torch.Tensor,
        fixed_landmarks: torch.Tensor,
    ) -> torch.Tensor:
        # vector_field: shape of (1, 3, x_dim, y_dim, z_dim), values are in voxel
        # displacement (i.e. not torch grid_sample convention [-1, 1])
        # {moving,fixed}_landmarks: shape of (1, n_landmarks, 3)

        # currently only implemented for batch size of 1
        # get rid of batch dimension
        vector_field = torch.as_tensor(vector_field[0], dtype=torch.float32)

        # round fixed_landmarks coordinates if dtype is float,
        # i.e. NN interpolation of vector_field
        if torch.is_floating_point(fixed_landmarks):
            fixed_landmarks = fixed_landmarks.round().to(torch.long)
        # get voxel of vector_field for each fixed landmark
        x_coordinates = fixed_landmarks[..., 0]
        y_coordinates = fixed_landmarks[..., 1]
        z_coordinates = fixed_landmarks[..., 2]
        # displacement is of shape (3, n_landmarks) after transposing
        displacements = vector_field[:, x_coordinates, y_coordinates, z_coordinates].T

        return fixed_landmarks + displacements

    def forward(
        self,
        vector_field: torch.Tensor | None,
        moving_landmarks: torch.Tensor,
        fixed_landmarks: torch.Tensor,
        image_spacing: torch.Tensor | FloatTuple3D,
    ):
        fixed_landmarks = fixed_landmarks[0]
        moving_landmarks = moving_landmarks[0]
        # warp fixed_landmarks and compare to moving_landmarks (euclidean distance)
        # distances will be float32 as displacements is float32

        if vector_field is not None:
            warped_fixed_landmarks = TRELoss._warped_fixed_landmarks(
                fixed_landmarks=fixed_landmarks, vector_field=vector_field
            )
        else:
            warped_fixed_landmarks = fixed_landmarks

        distances = warped_fixed_landmarks - moving_landmarks
        # scale x, x, z distance component with image spacing
        image_spacing = torch.as_tensor(
            image_spacing, dtype=torch.float32, device=distances.device
        )
        distances = distances * image_spacing
        distances = distances.pow(2).sum(dim=-1)

        if self.apply_sqrt:
            distances = distances.sqrt()

        if self.reduction == "mean":
            distances = distances.mean()
        elif self.reduction == "median":
            distances = torch.median(distances)
        elif self.reduction == "sum":
            distances = distances.sum()
        elif self.reduction == "quantile":
            distances = torch.quantile(distances, q=self.quantile)

        elif not self.reduction:
            # do nothing; this also covers falsy values like None, False, 0
            pass
        else:
            raise RuntimeError(f"Unsupported reduction {self._reduction}")

        return distances


def calculate_gradient_l2(image: torch.tensor, eps: float = 1e-6) -> torch.tensor:
    x_grad, y_grad, z_grad = torch.gradient(image, dim=(2, 3, 4))
    l2_grad = torch.sqrt((eps + x_grad**2 + y_grad**2 + z_grad**2))

    return l2_grad


class WarpedMSELoss(nn.Module):
    def __init__(self, shape: IntTuple3D | None = None, edge_weighting: float = 0.0):
        super().__init__()
        self.spatial_transformer = SpatialTransformer(shape=shape)
        self.edge_weighting = edge_weighting

    def forward(
        self,
        moving_image: torch.Tensor,
        vector_field: torch.Tensor,
        fixed_image: torch.Tensor,
        fixed_mask: torch.Tensor,
    ) -> torch.Tensor:
        warped_image = self.spatial_transformer(
            image=moving_image, transformation=vector_field, mode="bilinear"
        )

        loss = F.mse_loss(warped_image[fixed_mask], fixed_image[fixed_mask])

        if self.edge_weighting > 0.0:
            warped_image_l2_grad = calculate_gradient_l2(warped_image)[fixed_mask]
            fixed_image_l2_grad = calculate_gradient_l2(fixed_image)[fixed_mask]

            loss = loss + self.edge_weighting * F.l1_loss(
                warped_image_l2_grad, fixed_image_l2_grad
            )

        return loss


def mse_loss(
    moving_image: torch.Tensor, fixed_image: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    return F.mse_loss(moving_image[mask], fixed_image[mask])


def mirrored_mse_loss(
    moving_image: torch.Tensor, fixed_image: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    return (
        F.mse_loss(moving_image[mask], torch.flip(moving_image, dims=(4,))[mask])
        + F.mse_loss(moving_image[mask], torch.flip(moving_image, dims=(3,))[mask])
        + F.mse_loss(moving_image[mask], torch.flip(moving_image, dims=(2,))[mask])
    )


def ncc_loss(
    moving_image: torch.Tensor, fixed_image: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    moving_image = torch.masked_select(moving_image, mask)
    fixed_image = torch.masked_select(fixed_image, mask)
    value = (
        -1.0
        * torch.sum(
            (fixed_image - torch.mean(fixed_image))
            * (moving_image - torch.mean(moving_image))
        )
        / torch.sqrt(
            torch.sum((fixed_image - torch.mean(fixed_image)) ** 2)
            * torch.sum((moving_image - torch.mean(moving_image)) ** 2)
            + 1e-10
        )
    )

    return value


def ngf_loss(
    moving_image: torch.Tensor,
    fixed_image: torch.Tensor,
    mask: torch.tensor,
    epsilon=1e-5,
) -> torch.Tensor:
    dx_f = fixed_image[..., 1:, 1:, 1:] - fixed_image[..., :-1, 1:, 1:]
    dy_f = fixed_image[..., 1:, 1:, 1:] - fixed_image[..., 1:, :-1, 1:]
    dz_f = fixed_image[..., 1:, 1:, 1:] - fixed_image[..., 1:, 1:, :-1]

    if epsilon is None:
        with torch.no_grad():
            epsilon = torch.mean(torch.abs(dx_f) + torch.abs(dy_f) + torch.abs(dz_f))

    norm = torch.sqrt(dx_f.pow(2) + dy_f.pow(2) + dz_f.pow(2) + epsilon**2)

    ngf_fixed_image = F.pad(
        torch.cat((dx_f, dy_f, dz_f), dim=1) / norm, (0, 1, 0, 1, 0, 1)
    )

    dx_m = moving_image[..., 1:, 1:, 1:] - moving_image[..., :-1, 1:, 1:]
    dy_m = moving_image[..., 1:, 1:, 1:] - moving_image[..., 1:, :-1, 1:]
    dz_m = moving_image[..., 1:, 1:, 1:] - moving_image[..., 1:, 1:, :-1]

    norm = torch.sqrt(dx_m.pow(2) + dy_m.pow(2) + dz_m.pow(2) + epsilon**2)

    ngf_moving_image = F.pad(
        torch.cat((dx_m, dy_m, dz_m), dim=1) / norm, (0, 1, 0, 1, 0, 1)
    )

    value = 0
    for dim in range(3):
        value = value + ngf_moving_image[:, dim, ...] * ngf_fixed_image[:, dim, ...]

    value = 0.5 * torch.masked_select(-value.pow(2), mask)

    return value.mean()


def jacobian_determinant(vector_field: torch.Tensor) -> torch.Tensor:
    # vector field has shape (1, 3, x, y, z)
    dx, dy, dz = torch.gradient(vector_field, dim=(2, 3, 4))

    # add identity matrix: det(dT/dx) = det(I + du/dx)
    dx[:, 0] = dx[:, 0] + 1
    dy[:, 1] = dy[:, 1] + 1
    dz[:, 2] = dz[:, 2] + 1

    # Straightforward application of rule of sarrus yields the following lines.

    # sarrus_plus_1 = dx[:, 0] * dy[:, 1] * dz[:, 2]
    # sarrus_plus_2 = dy[:, 0] * dz[:, 1] * dx[:, 2]
    # sarrus_plus_3 = dz[:, 0] * dx[:, 1] * dy[:, 2]
    #
    # sarrus_minus_1 = dx[:, 2] * dy[:, 1] * dz[:, 0]
    # sarrus_minus_2 = dy[:, 2] * dz[:, 1] * dx[:, 0]
    # sarrus_minus_3 = dz[:, 2] * dx[:, 1] * dy[:, 0]
    #
    # det_j = (sarrus_plus_1 + sarrus_plus_2 + sarrus_plus_3) - (
    #     sarrus_minus_1 + sarrus_minus_2 + sarrus_minus_3
    # )

    # However, we factor out ∂VFx/∂x, ∂VFx/∂y, ∂VFx/∂z to save a few FLOPS:

    det_j = (
        dx[:, 0] * (dy[:, 1] * dz[:, 2] - dy[:, 2] * dz[:, 1])
        + dy[:, 0] * (dz[:, 1] * dx[:, 2] - dz[:, 2] * dx[:, 1])
        + dz[:, 0] * (dx[:, 1] * dy[:, 2] - dx[:, 2] * dy[:, 1])
    )

    return det_j[:, None]


def smooth_vector_field_loss(
    vector_field: torch.Tensor, mask: torch.Tensor, l2r_variant: bool = False
) -> torch.Tensor:
    det_j = jacobian_determinant(vector_field)

    if l2r_variant:
        det_j = det_j + 3
        det_j = torch.clip(det_j, 1e-9, 1e9)
        det_j = torch.log(det_j)

    return det_j[mask].std()


class DiceLoss(nn.Module):
    """Compute average Dice loss between two tensors. It can support both
    multi-classes and multi-labels tasks. The data `input` (BNHW[D] where N is
    number of classes) is compared with ground truth `target` (BNHW[D]).

    Note that axis N of `input` is expected to be logits or probabilities for each class, if passing logits as input,
    must set `sigmoid=True` or `softmax=True`, or specifying `other_act`. And the same axis of `target`
    can be 1 or N (one-hot format).

    The `smooth_nr` and `smooth_dr` parameters are values added to the intersection and union components of
    the inter-over-union calculation to smooth results respectively, these values should be small.

    The original paper: Milletari, F. et. al. (2016) V-Net: Fully Convolutional Neural Networks forVolumetric
    Medical Image Segmentation, 3DV, 2016.
    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str | None = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
    ) -> None:
        """
        Args:
            include_background: if False, channel index 0 (background category) is excluded from the calculation.
                if the non-background segmentations are small compared to the total image size they can get overwhelmed
                by the signal from the background so excluding it in such cases helps convergence.
            to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction.
            softmax: if True, apply a softmax function to the prediction.
            other_act: if don't want to use `sigmoid` or `softmax`, use other callable function to execute
                other activation layers, Defaults to ``None``. for example:
                `other_act = torch.tanh`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.

        Raises:
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.

        """
        super().__init__()
        if other_act is not None and not callable(other_act):
            raise TypeError(
                f"other_act must be None or callable but is {type(other_act).__name__}."
            )
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError(
                "Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None]."
            )
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.squared_pred = squared_pred
        self.jaccard = jaccard
        self.reduction = reduction
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.batch = batch

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
            target: the shape should be BNH[WD] or B1H[WD], where N is the number of classes.

        Raises:
            AssertionError: When input and target (after one hot transform if set)
                have different shapes.
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        Example:
            >>> from monai.losses.dice import *  # NOQA
            >>> import torch
            >>> from monai.losses.dice import DiceLoss
            >>> B, C, H, W = 7, 5, 3, 2
            >>> input = torch.rand(B, C, H, W)
            >>> target_idx = torch.randint(low=0, high=C - 1, size=(B, H, W)).long()
            >>> target = one_hot(target_idx[:, None, ...], num_classes=C)
            >>> self = DiceLoss(reduction='none')
            >>> loss = self(input, target)
            >>> assert np.broadcast_shapes(loss.shape, input.shape) == input.shape
        """
        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = to_one_hot(target, n_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn(
                    "single channel prediction, `include_background=False` ignored."
                )
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise AssertionError(
                f"ground truth has different shape ({target.shape}) from input ({input.shape})"
            )

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis: List[int] = torch.arange(2, len(input.shape)).tolist()
        if self.batch:
            # reducing spatial dimensions and batch
            reduce_axis = [0] + reduce_axis

        intersection = torch.sum(target * input, dim=reduce_axis)

        if self.squared_pred:
            target = torch.pow(target, 2)
            input = torch.pow(input, 2)

        ground_o = torch.sum(target, dim=reduce_axis)
        pred_o = torch.sum(input, dim=reduce_axis)

        denominator = ground_o + pred_o

        if self.jaccard:
            denominator = 2.0 * (denominator - intersection)

        f: torch.Tensor = 1.0 - (2.0 * intersection + self.smooth_nr) / (
            denominator + self.smooth_dr
        )

        if self.reduction == "mean":
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == "sum":
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction in {"none", None}:
            # If we are not computing voxelwise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            broadcast_shape = list(f.shape[0:2]) + [1] * (len(input.shape) - 2)
            f = f.view(broadcast_shape)
        else:
            raise ValueError(
                f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].'
            )

        return f
