from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generator, List, Literal, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vroc.blocks import (
    ConvBlock,
    DecoderBlock,
    DemonForces,
    DownBlock,
    EncoderBlock,
    GaussianSmoothing2d,
    GaussianSmoothing3d,
    NCCForces,
    NGFForces,
    NormedConv3d,
    SpatialTransformer,
    TrainableDemonForces,
    UpBlock,
    separable_normed_conv_3d,
)
from vroc.checks import are_of_same_length, is_tuple, is_tuple_of_tuples
from vroc.common_types import (
    FloatTuple2D,
    FloatTuple3D,
    IntTuple2D,
    IntTuple3D,
    MaybeSequence,
    Number,
)
from vroc.decay import exponential_decay
from vroc.decorators import timing
from vroc.helper import (
    get_bounding_box,
    get_mode_from_alternation_scheme,
    write_landmarks,
)
from vroc.interpolation import match_vector_field, rescale, resize
from vroc.logger import LoggerMixin
from vroc.loss import TRELoss


# TODO: channel bug with skip_connections=False
class FlexUNet(nn.Module):
    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 1,
        n_levels: int = 6,
        filter_base: int | None = None,
        n_filters: Sequence[int] | None = None,
        convolution_layer=nn.Conv3d,
        downsampling_layer=nn.MaxPool3d,
        upsampling_layer=nn.Upsample,
        norm_layer=nn.BatchNorm3d,
        skip_connections=False,
        convolution_kwargs=None,
        downsampling_kwargs=None,
        upsampling_kwargs=None,
        return_bottleneck: bool = True,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_levels = n_levels

        # either filter_base or n_filters must be set
        self.filter_base = filter_base
        self.n_filters = n_filters

        if not any((filter_base, n_filters)) or all((filter_base, n_filters)):
            raise ValueError("Please set either filter_base or n_filters")

        self.convolution_layer = convolution_layer
        self.downsampling_layer = downsampling_layer
        self.upsampling_layer = upsampling_layer
        self.norm_layer = norm_layer
        self.skip_connections = skip_connections

        self.convolution_kwargs = convolution_kwargs or {
            "kernel_size": 3,
            "padding": "same",
            "bias": True,
        }
        self.downsampling_kwargs = downsampling_kwargs or {"kernel_size": 2}
        self.upsampling_kwargs = upsampling_kwargs or {"scale_factor": 2}

        self.return_bottleneck = return_bottleneck

        self._build_layers()

    @property
    def encoder_block(self):
        return EncoderBlock

    @property
    def decoder_block(self):
        return DecoderBlock

    def _build_layers(self):
        if self.filter_base:
            n_filters = {
                "init": self.filter_base,
                "enc": [
                    self.filter_base * 2**i_level for i_level in range(self.n_levels)
                ],
                "dec": [
                    self.filter_base * 2**i_level
                    for i_level in reversed(range(self.n_levels))
                ],
                "final": self.filter_base,
            }
        else:
            n_filters = {
                "init": self.n_filters[0],
                "enc": self.n_filters[1 : self.n_levels + 1],
                "dec": self.n_filters[self.n_levels + 1 : -1],
                "final": self.n_filters[-1],
            }

        enc_out_channels = []

        self.init_conv = self.convolution_layer(
            in_channels=self.n_channels,
            out_channels=n_filters["init"],
            **self.convolution_kwargs,
        )

        self.final_conv = self.convolution_layer(
            in_channels=n_filters["final"],
            out_channels=self.n_classes,
            **self.convolution_kwargs,
        )

        enc_out_channels.append(n_filters["init"])
        previous_out_channels = n_filters["init"]

        for i_level in range(self.n_levels):
            out_channels = n_filters["enc"][i_level]
            enc_out_channels.append(out_channels)
            self.add_module(
                f"enc_{i_level}",
                self.encoder_block(
                    in_channels=previous_out_channels,
                    out_channels=out_channels,
                    n_convolutions=2,
                    convolution_layer=self.convolution_layer,
                    downsampling_layer=self.downsampling_layer,
                    norm_layer=self.norm_layer,
                    convolution_kwargs=self.convolution_kwargs,
                    downsampling_kwargs=self.downsampling_kwargs,
                ),
            )
            previous_out_channels = out_channels

        for i, i_level in enumerate(reversed(range(self.n_levels))):
            out_channels = n_filters["dec"][i]

            if i_level > 0:  # deeper levels
                if self.skip_connections:
                    in_channels = previous_out_channels + enc_out_channels[i_level]
                else:
                    in_channels = previous_out_channels
            else:
                if self.skip_connections:
                    in_channels = previous_out_channels + n_filters["init"]
                else:
                    in_channels = previous_out_channels

            self.add_module(
                f"dec_{i_level}",
                self.decoder_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    n_convolutions=2,
                    convolution_layer=self.convolution_layer,
                    upsampling_layer=self.upsampling_layer,
                    norm_layer=self.norm_layer,
                    convolution_kwargs=self.convolution_kwargs,
                    upsampling_kwargs=self.upsampling_kwargs,
                ),
            )
            previous_out_channels = out_channels

    def forward(self, *inputs, **kwargs):
        outputs = []
        inputs = self.init_conv(*inputs)
        outputs.append(inputs)
        for i_level in range(self.n_levels):
            inputs = self.get_submodule(f"enc_{i_level}")(inputs)
            outputs.append(inputs)

        for i_level in reversed(range(self.n_levels)):
            inputs = self.get_submodule(f"dec_{i_level}")(inputs, outputs[i_level])

        inputs = self.final_conv(inputs)

        if self.return_bottleneck:
            return inputs, outputs[-1]
        else:
            return inputs


class Unet3d(FlexUNet):
    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 1,
        n_levels: int = 6,
        filter_base: int | None = None,
        n_filters: Sequence[int] | None = None,
    ):
        super().__init__(
            n_channels=n_channels,
            n_classes=n_classes,
            n_levels=n_levels,
            filter_base=filter_base,
            n_filters=n_filters,
            convolution_layer=nn.Conv3d,
            downsampling_layer=nn.MaxPool3d,
            upsampling_layer=nn.Upsample,
            norm_layer=nn.InstanceNorm3d,
            skip_connections=True,
            convolution_kwargs=None,
            downsampling_kwargs=None,
            upsampling_kwargs=None,
        )

    def forward(self, *inputs, **kwargs):
        prediction, _ = super().forward(*inputs, **kwargs)

        return prediction


class AutoEncoder(FlexUNet):
    def forward(self, *inputs, **kwargs):
        outputs = []
        inputs = self.init_conv(*inputs)
        outputs.append(inputs)
        for i_level in range(self.n_levels):
            inputs = self.get_submodule(f"enc_{i_level}")(inputs)
            outputs.append(inputs)

        encoded_size = outputs[-1].size()
        embedded = F.avg_pool3d(outputs[-1], kernel_size=encoded_size[2:]).view(
            encoded_size[0], -1
        )
        inputs = embedded[(...,) + (None,) * len(encoded_size[2:])].repeat(
            (1, 1) + encoded_size[2:]
        )

        for i_level in reversed(range(self.n_levels)):
            inputs = self.get_submodule(f"dec_{i_level}")(inputs, None)

        inputs = self.final_conv(inputs)

        return inputs, embedded


class UNet(nn.Module):
    def __init__(self, n_channels: int = 1, n_classes: int = 1, bilinear: bool = True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        self.inc = ConvBlock(n_channels, 64)
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        self.down4 = DownBlock(512, 1024 // factor)

        self.up1 = UpBlock(1024, 512 // factor, bilinear)
        self.up2 = UpBlock(512, 256 // factor, bilinear)
        self.up3 = UpBlock(256, 128 // factor, bilinear)
        self.up4 = UpBlock(128, 64, bilinear)
        self.out = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.out(x)


class SameSizeConvBlock(nn.Module, LoggerMixin):
    def __init__(
        self,
        in_channels,
        out_channels,
        activation: nn.Module = nn.GELU(),
        kernel_size: int = 3,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding="same",
            padding_mode="circular",
            bias=False,
        )
        self.activation = activation

    def forward(self, x: torch.Tensor):
        self.logger.debug(f"{x.shape=}")
        x = self.conv(x)
        x = self.activation(x)
        return x


class SameSizeConvAutoEncoder(nn.Module, LoggerMixin):
    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: int,
        layers: list[int],
        activation: nn.Module = nn.GELU(),
        kernel_size: int = 3,
        normalize_bottleneck: bool = False,
        add_noise: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.bottleneck_channels = bottleneck_channels
        self.activation = activation
        self.normalize_bottleneck = normalize_bottleneck
        self.add_noise = add_noise
        self.down_layers = []
        self.up_layers = []

        layers = [in_channels] + layers + [bottleneck_channels]
        for i in range(1, len(layers)):
            activation = nn.Identity() if i == len(layers) - 1 else self.activation
            self.down_layers.append(
                SameSizeConvBlock(layers[i - 1], layers[i], activation, kernel_size),
            )
        for i in range(len(layers) - 1, 0, -1):
            activation = nn.Identity() if i == 1 else self.activation
            self.up_layers.append(
                SameSizeConvBlock(layers[i], layers[i - 1], activation, kernel_size)
            )
        self.down_layers = nn.Sequential(*self.down_layers)
        self.up_layers = nn.Sequential(*self.up_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.logger.debug(f"{x.shape=}")
        x_bottleneck = self.down_layers(x)
        # normalize each sample to sum = 1 and min >= 0 if enabled
        if self.normalize_bottleneck:
            x_bottleneck = x_bottleneck - x_bottleneck.amin(dim=(1, 2, 3), keepdim=True)

            sample_sum = x_bottleneck.sum(dim=(1, 2, 3), keepdim=True)
            x_bottleneck = x_bottleneck / sample_sum

        self.logger.debug(f"{x_bottleneck.shape=}")
        x_recon = x_bottleneck
        if self.add_noise and self.training:
            x_recon = x_recon * (
                1 + (torch.rand_like(x_recon) * 2 * self.add_noise - self.add_noise)
            )
        x_recon = self.up_layers(x_recon)
        self.logger.debug(f"{x_recon.shape=}")
        return x_recon, x_bottleneck


class SameSizeConvGAPAutoEncoder(nn.Module, LoggerMixin):
    def __init__(
        self,
        in_channels: int,
        shape: tuple[int, int, int],
        bottleneck_channels: int,
        layers: list[int],
        activation: nn.Module = nn.GELU(),
        kernel_size: int = 3,
        normalize_bottleneck: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.bottleneck_channels = bottleneck_channels
        self.activation = activation
        self.normalize_bottleneck = normalize_bottleneck
        self.down_layers = []
        self.up_layers = []
        self.shape = shape

        layers = [in_channels] + layers + [bottleneck_channels]
        for i in range(1, len(layers)):
            activation = nn.Identity() if i == len(layers) - 1 else self.activation
            self.down_layers.append(
                SameSizeConvBlock(layers[i - 1], layers[i], activation, kernel_size),
            )
        for i in range(len(layers) - 1, 0, -1):
            activation = nn.Identity() if i == 1 else self.activation
            self.up_layers.append(
                SameSizeConvBlock(layers[i], layers[i - 1], activation, kernel_size)
            )
        self.down_layers = nn.Sequential(*self.down_layers)
        self.up_layers = nn.Sequential(*self.up_layers)
        self.mlp = nn.Sequential(
            nn.Linear(
                self.bottleneck_channels,
                self.bottleneck_channels * self.shape[1] * self.shape[2],
            ),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.logger.debug(f"{x.shape=}")
        x_bottleneck = self.down_layers(x)

        # apply global average pooling over channels
        # bottleneck is now n_samples x bottleneck_channels x 1 x 1
        shape = x_bottleneck.shape
        x_bottleneck = x_bottleneck.mean(dim=(2, 3), keepdim=True)

        self.logger.debug(f"{x_bottleneck.shape=}")
        # x_bottleneck = torch.sigmoid(x_bottleneck)
        x_bottleneck_flat = x_bottleneck.flatten(start_dim=1)
        max_noise = 0.5
        # x_bottleneck_flat = x_bottleneck_flat + (torch.rand_like(x_bottleneck_flat) * 2 * max_noise - max_noise)
        # x_bottleneck_flat = torch.clip(x_bottleneck_flat, 0, 1)
        x_bottleneck_flat = x_bottleneck_flat * (
            1 + (torch.rand_like(x_bottleneck_flat) * 2 * max_noise - max_noise)
        )
        x_recon = self.mlp(x_bottleneck_flat)

        # back to image format
        x_recon = x_recon.reshape(shape)

        # add noise

        x_recon = self.up_layers(x_recon)
        self.logger.debug(f"{x_recon.shape=}")
        return x_recon, x_bottleneck


class SameSizeConvVariationalAutoEncoder(nn.Module, LoggerMixin):
    def __init__(
        self,
        in_channels: int,
        shape: tuple[int, int, int],
        bottleneck_channels: int,
        layers: list[int],
        activation: nn.Module = nn.GELU(),
        kernel_size: int = 3,
        normalize_bottleneck: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.bottleneck_channels = bottleneck_channels
        self.activation = activation
        self.normalize_bottleneck = normalize_bottleneck
        self.down_layers = []
        self.up_layers = []
        self.shape = shape

        layers = [in_channels] + layers + [bottleneck_channels]
        for i in range(1, len(layers)):
            activation = nn.Identity() if i == len(layers) - 1 else self.activation
            self.down_layers.append(
                SameSizeConvBlock(layers[i - 1], layers[i], activation, kernel_size),
            )
        for i in range(len(layers) - 1, 0, -1):
            activation = nn.Identity() if i == 1 else self.activation
            self.up_layers.append(
                SameSizeConvBlock(layers[i], layers[i - 1], activation, kernel_size)
            )
        self.down_layers = nn.Sequential(*self.down_layers)
        self.up_layers = nn.Sequential(*self.up_layers)
        self.fc_mu = nn.Linear(self.bottleneck_channels, self.bottleneck_channels)
        self.fc_var = nn.Linear(self.bottleneck_channels, self.bottleneck_channels)
        self.mlp = nn.Sequential(
            nn.Linear(
                self.bottleneck_channels,
                self.bottleneck_channels * self.shape[1] * self.shape[2],
            ),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.logger.debug(f"{x.shape=}")
        x_bottleneck = self.down_layers(x)

        # apply global average pooling over channels
        # bottleneck is now n_samples x bottleneck_channels x 1 x 1
        shape = x_bottleneck.shape
        x_bottleneck = x_bottleneck.mean(dim=(2, 3), keepdim=True)
        self.logger.debug(f"{x_bottleneck.shape=}")

        x_bottleneck_flat = x_bottleneck.flatten(start_dim=1)
        mu = self.fc_mu(x_bottleneck_flat)
        log_var = self.fc_var(x_bottleneck_flat)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps * std + mu

        x_recon = self.mlp(z)

        # back to image format
        x_recon = x_recon.reshape(shape)

        x_recon = self.up_layers(x_recon)
        self.logger.debug(f"{x_recon.shape=}")
        return x_recon, z, mu, log_var

    def loss_function(self, recons, inputs, mu, log_var) -> dict:
        recons_loss = F.mse_loss(recons, inputs)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )
        kld_weight = 100.0

        loss = recons_loss + kld_weight * kld_loss
        return loss


class ParamNet(nn.Module):
    def __init__(
        self,
        params: dict,
        n_channels: int = 3,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.params = params
        self.n_params = len(self.params)

        self.conv_1 = DownBlock(
            in_channels=self.n_channels,
            out_channels=8,
            dimensions=1,
            norm_type="InstanceNorm",
        )
        self.conv_2 = DownBlock(
            in_channels=8, out_channels=16, dimensions=1, norm_type="InstanceNorm"
        )
        self.conv_3 = DownBlock(
            in_channels=16, out_channels=8, dimensions=1, norm_type="InstanceNorm"
        )
        self.conv_4 = DownBlock(
            in_channels=8, out_channels=4, dimensions=1, norm_type="InstanceNorm"
        )
        self.conv_5 = DownBlock(
            in_channels=4, out_channels=self.n_params, dimensions=1, norm_type=None
        )

    def forward(self, features):
        out = self.conv_1(features)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        out = self.conv_5(out)
        out = torch.sigmoid(out)
        out = torch.squeeze(out)

        out_dict = {}
        # scale params according to min/max range
        for i_param, param_name in enumerate(self.params.keys()):
            out_min, out_max = (
                self.params[param_name]["min"],
                self.params[param_name]["max"],
            )
            out_dict[param_name] = (
                out[i_param] * (out_max - out_min)
            ) + out_min  # .to(self.params[param_name]['dtype'])

        return out_dict


class BaseIterativeRegistration(ABC, nn.Module, LoggerMixin):
    def _create_spatial_transformers(self, image_shape: Tuple[int, ...], device):
        if not image_shape == self._image_shape:
            self._image_shape = image_shape
            self._full_size_spatial_transformer = SpatialTransformer(
                shape=image_shape, default_value=self.default_voxel_value
            ).to(device)

            for i_level, scale_factor in enumerate(self.scale_factors):
                scaled_image_shape = tuple(
                    int(round(s * scale_factor)) for s in image_shape
                )

                try:
                    module = getattr(self, f"spatial_transformer_level_{i_level}")
                    del module
                except AttributeError:
                    pass
                self.add_module(
                    name=f"spatial_transformer_level_{i_level}",
                    module=SpatialTransformer(
                        shape=scaled_image_shape, default_value=self.default_voxel_value
                    ).to(device),
                )

    def _perform_scaling(
        self, *images, scale_factor: float = 1.0
    ) -> List[torch.Tensor]:
        scaled = []
        for image in images:
            if image is not None:
                is_mask = image.dtype == torch.bool
                order = 0 if is_mask else 1

                image = rescale(image, factor=scale_factor, order=order)

            scaled.append(image)

        return scaled

    def _calculate_metric(
        self,
        moving_image: torch.Tensor,
        fixed_image: torch.Tensor,
        fixed_mask: torch.Tensor | None = None,
    ) -> float:
        if fixed_mask is not None:
            fixed_image = fixed_image[fixed_mask]
            moving_image = moving_image[fixed_mask]

        return float(F.mse_loss(fixed_image, moving_image))

    def _check_early_stopping(self, metrics: List[float], i_level: int) -> bool:
        early_stop = False
        if (
            self.early_stopping_delta[i_level]
            and self.early_stopping_window[i_level]
            and len(metrics) >= self.early_stopping_window[i_level]
        ):
            window = np.array(metrics[-self.early_stopping_window[i_level] :])
            window_rel_changes = 1 - window[1:] / window[:-1]

            mean_rel_change = window_rel_changes.mean()

            self.logger.debug(
                f"Mean relative metric change over window of "
                f"{self.early_stopping_window[i_level]} steps is {mean_rel_change:.6f}"
            )

            if mean_rel_change < self.early_stopping_delta[i_level]:
                early_stop = True
                self.logger.debug(
                    f"Early stopping triggered for level {i_level} after iteration "
                    f"{len(metrics)}: {mean_rel_change:.6f} < "
                    f"{self.early_stopping_delta[i_level]:.6f}"
                )

        return early_stop

    @staticmethod
    def _expand_to_level_tuple(
        value: Any, n_levels: int, is_tuple: bool = False, expand_none: bool = False
    ) -> Optional[Tuple]:
        if not expand_none and value is None:
            return value
        else:
            if not is_tuple:
                return (value,) * n_levels
            elif is_tuple and not is_tuple_of_tuples(value):
                return (value,) * n_levels
            else:
                return value

    def _update_step(
        self,
        level: int,
        iteration: int,
        scale_factor: float,
        vector_field: torch.Tensor,
        moving_image: torch.Tensor,
        warped_image: torch.Tensor,
        fixed_image: torch.Tensor,
        moving_mask: torch.Tensor | None = None,
        fixed_mask: torch.Tensor | None = None,
        original_image_spacing: FloatTuple3D = (1.0, 1.0, 1.0),
        initial_vector_field: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def run_registration(
        self,
        moving_image: torch.Tensor,
        fixed_image: torch.Tensor,
        moving_mask: torch.Tensor | None = None,
        fixed_mask: torch.Tensor | None = None,
        original_image_spacing: FloatTuple3D = (1.0, 1.0, 1.0),
        initial_vector_field: torch.Tensor | None = None,
    ):
        raise NotImplementedError

    @timing()
    def forward(
        self,
        moving_image: torch.Tensor,
        fixed_image: torch.Tensor,
        moving_mask: torch.Tensor | None = None,
        fixed_mask: torch.Tensor | None = None,
        original_image_spacing: FloatTuple3D = (1.0, 1.0, 1.0),
        initial_vector_field: torch.Tensor | None = None,
    ):
        return self.run_registration(
            moving_image=moving_image,
            fixed_image=fixed_image,
            moving_mask=moving_mask,
            fixed_mask=fixed_mask,
            original_image_spacing=original_image_spacing,
            initial_vector_field=initial_vector_field,
        )


class VariationalRegistration(BaseIterativeRegistration):
    _GAUSSIAN_SMOOTHING = {2: GaussianSmoothing2d, 3: GaussianSmoothing3d}

    def __init__(
        self,
        scale_factors: MaybeSequence[float] = (1.0,),
        use_masks: MaybeSequence[bool] = True,
        iterations: MaybeSequence[int] = 100,
        tau: MaybeSequence[float] = 1.0,
        tau_level_decay: float = 0.0,
        tau_iteration_decay: float = 0.0,
        force_type: Literal["demons", "ncc", "ngf"] = "demons",
        gradient_type: Literal["active", "passive", "dual"] = "dual",
        regularization_sigma: MaybeSequence[FloatTuple2D | FloatTuple3D] = (
            1.0,
            1.0,
            1.0,
        ),
        regularization_radius: MaybeSequence[IntTuple2D | IntTuple3D] | None = None,
        sigma_level_decay: float = 0.0,
        sigma_iteration_decay: float = 0.0,
        original_image_spacing: FloatTuple2D | FloatTuple3D = (1.0, 1.0, 1.0),
        use_image_spacing: bool = False,
        restrict_to_mask_bbox: bool = False,
        restrict_to_mask_bbox_mode: Literal["moving", "fixed", "union"] = "union",
        mask_bbox_padding: IntTuple2D | IntTuple3D | int = 5,
        early_stopping_delta: float = 0.0,
        early_stopping_window: int | None = 20,
        boosting_model: nn.Module | None = None,
        default_voxel_value: Number = 0.0,
        debug: bool = False,
        debug_output_folder: Path | None = None,
        debug_step_size: int = 10,
    ):
        super().__init__()

        n_spatial_dims = len(original_image_spacing)

        if not is_tuple(scale_factors, min_length=1):
            scale_factors = (scale_factors,)
        self.scale_factors = scale_factors

        self.scale_factors = scale_factors  # this also defines "n_levels"
        self.use_masks = VariationalRegistration._expand_to_level_tuple(
            use_masks, n_levels=self.n_levels
        )
        self.iterations = VariationalRegistration._expand_to_level_tuple(
            iterations, n_levels=self.n_levels
        )

        self.tau = VariationalRegistration._expand_to_level_tuple(
            tau, n_levels=self.n_levels
        )
        self.tau_level_decay = tau_level_decay
        self.tau_iteration_decay = tau_iteration_decay

        self.regularization_sigma = VariationalRegistration._expand_to_level_tuple(
            regularization_sigma, n_levels=self.n_levels, is_tuple=True
        )
        self.regularization_radius = VariationalRegistration._expand_to_level_tuple(
            regularization_radius, n_levels=self.n_levels, is_tuple=True
        )
        self.sigma_level_decay = sigma_level_decay
        self.sigma_iteration_decay = sigma_iteration_decay
        self.original_image_spacing = original_image_spacing
        self.use_image_spacing = use_image_spacing

        self.forces = gradient_type

        self.restrict_to_mask_bbox = restrict_to_mask_bbox
        self.restrict_to_mask_bbox_mode = restrict_to_mask_bbox_mode
        self.mask_bbox_padding = mask_bbox_padding
        self.early_stopping_delta = VariationalRegistration._expand_to_level_tuple(
            early_stopping_delta, n_levels=self.n_levels
        )
        self.early_stopping_window = VariationalRegistration._expand_to_level_tuple(
            early_stopping_window, n_levels=self.n_levels, expand_none=True
        )
        self.boosting_model = boosting_model
        self.default_voxel_value = default_voxel_value
        self.debug = debug
        self.debug_output_folder = debug_output_folder
        self.debug_step_size = debug_step_size

        if self.debug:
            from vroc.plot import RegistrationProgressPlotter

            self._plotter = RegistrationProgressPlotter(
                output_folder=self.debug_output_folder
            )
        else:
            self._plotter = None

        # check if params are passed with/converted to consistent length
        # (== self.n_levels)
        if not are_of_same_length(
            self.scale_factors,
            self.iterations,
            self.tau,
            self.regularization_sigma,
        ):
            raise ValueError("Inconsistent lengths of passed parameters")

        self._metrics = []
        self._counter = 0

        self._image_shape = None
        self._full_size_spatial_transformer = None

        if force_type == "demons":
            self._forces_layer = DemonForces(method=self.forces)
        elif force_type == "ncc":
            self._forces_layer = NCCForces(method=self.forces)
        elif force_type == "ngf":
            self._forces_layer = NGFForces()
        else:
            raise NotImplementedError(
                f"Registration variant {force_type} is not implemented"
            )

    @property
    def n_levels(self):
        return len(self.scale_factors)

    @property
    def config(self):
        return dict(
            iterations=self.iterations,
            tau=self.tau,
            regularization_sigma=self.regularization_sigma,
        )

    def _update_step(
        self,
        level: int,
        iteration: int,
        scale_factor: float,
        vector_field: torch.Tensor,
        moving_image: torch.Tensor,
        warped_image: torch.Tensor,
        fixed_image: torch.Tensor,
        moving_mask: torch.Tensor | None = None,
        fixed_mask: torch.Tensor | None = None,
        original_image_spacing: FloatTuple3D = (1.0, 1.0, 1.0),
        initial_vector_field: torch.Tensor | None = None,
    ) -> torch.Tensor:
        forces = self._forces_layer(
            warped_image,
            fixed_image,
            moving_mask,
            fixed_mask,
            original_image_spacing,
            use_masks=self.use_masks[level],
        )
        decayed_tau = exponential_decay(
            initial_value=self.tau[level],
            i_level=level,
            i_iteration=iteration,
            level_lambda=self.tau_level_decay,
            iteration_lambda=self.tau_iteration_decay,
        )

        vector_field = vector_field + decayed_tau * forces

        decayed_sigma = tuple(
            exponential_decay(
                initial_value=s,
                i_level=level,
                i_iteration=iteration,
                level_lambda=self.sigma_level_decay,
                iteration_lambda=self.sigma_iteration_decay,
            )
            for s in self.regularization_sigma[level]
        )
        n_spatial_dims = vector_field.shape[1]
        sigma_cutoff = (2.0, 2.0, 2.0)[:n_spatial_dims]
        gaussian_smoothing = self._GAUSSIAN_SMOOTHING[n_spatial_dims]
        _regularization_layer = gaussian_smoothing(
            sigma=decayed_sigma,
            sigma_cutoff=sigma_cutoff,
            force_same_size=True,
            spacing=self.original_image_spacing,
            use_image_spacing=self.use_image_spacing,
        ).to(vector_field)

        vector_field = _regularization_layer(vector_field)

        return vector_field

    def _run_registration(
        self,
        moving_image: torch.Tensor,
        fixed_image: torch.Tensor,
        moving_mask: torch.Tensor | None = None,
        fixed_mask: torch.Tensor | None = None,
        moving_landmarks: torch.Tensor | None = None,
        fixed_landmarks: torch.Tensor | None = None,
        original_image_spacing: FloatTuple3D = (1.0, 1.0, 1.0),
        initial_vector_field: torch.Tensor | None = None,
        yield_each_step: bool = False,
    ) -> Generator[dict[str]]:
        device = moving_image.device
        if moving_image.ndim != fixed_image.ndim:
            raise RuntimeError("Dimension mismatch betwen moving and fixed image")
        # define dimensionalities and shapes
        n_image_dimensions = moving_image.ndim
        n_spatial_dims = n_image_dimensions - 2  # -1 batch dim, -1 color dim

        full_uncropped_shape = tuple(fixed_image.shape)

        original_moving_image = moving_image
        original_moving_mask = moving_mask

        has_initial_vector_field = initial_vector_field is not None
        original_initial_vector_field = initial_vector_field

        if self.restrict_to_mask_bbox and (
            moving_mask is not None or fixed_mask is not None
        ):
            if self.restrict_to_mask_bbox_mode == "union":
                masks = [m for m in (moving_mask, fixed_mask) if m is not None]
                if len(masks) == 2:
                    # we compute the union of both masks to get the overall bounding box
                    bbox_mask = torch.logical_or(*masks)
                else:
                    bbox_mask = masks[0]
            elif self.restrict_to_mask_bbox_mode == "moving":
                bbox_mask = moving_mask
            elif self.restrict_to_mask_bbox_mode == "fixed":
                bbox_mask = fixed_mask
            else:
                raise ValueError(
                    f"Unknown restrict_to_mask_bbox_mode "
                    f"{self.restrict_to_mask_bbox_mode}"
                )

            bbox = get_bounding_box(bbox_mask, padding=self.mask_bbox_padding)
            self.logger.info(
                f"Restricting registration to bounding box {bbox} of moving/fixed mask "
                f"union + padding of {self.mask_bbox_padding}"
            )

            moving_image = moving_image[bbox]
            fixed_image = fixed_image[bbox]
            if moving_mask is not None:
                moving_mask = moving_mask[bbox]
            if fixed_mask is not None:
                fixed_mask = fixed_mask[bbox]
            if has_initial_vector_field:
                initial_vector_field = initial_vector_field[(..., *bbox[2:])]
        else:
            bbox = ...

        if moving_mask is not None:
            moving_mask = torch.as_tensor(moving_mask, dtype=torch.bool)
        if fixed_mask is not None:
            fixed_mask = torch.as_tensor(fixed_mask, dtype=torch.bool)

        # create new spatial transformers if needed (skip batch and color dimension)
        self._create_spatial_transformers(
            fixed_image.shape[2:], device=fixed_image.device
        )

        full_size_moving = moving_image
        full_cropped_shape = tuple(fixed_image.shape)
        full_cropped_spatial_shape = full_cropped_shape[2:]

        # for tracking level-wise metrics (used for early stopping, if enabled)
        metrics = []
        vector_field = None
        warped_fixed_landmarks = None

        metric_before = self._calculate_metric(
            moving_image=moving_image, fixed_image=fixed_image, fixed_mask=fixed_mask
        )

        for i_level, (scale_factor, iterations) in enumerate(
            zip(self.scale_factors, self.iterations)
        ):
            self.logger.info(
                f"Start level {i_level + 1}/{self.n_levels} with {scale_factor=} and {iterations=}"
            )
            self._counter = 0

            (
                scaled_moving_image,
                scaled_fixed_image,
                scaled_moving_mask,
                scaled_fixed_mask,
            ) = self._perform_scaling(
                moving_image,
                fixed_image,
                moving_mask,
                fixed_mask,
                scale_factor=scale_factor,
            )

            if vector_field is None:
                # create an empty (all zero) vector field
                vector_field = torch.zeros(
                    scaled_fixed_image.shape[:1]
                    + (n_spatial_dims,)
                    + scaled_fixed_image.shape[2:],
                    device=moving_image.device,
                )
            else:
                # this will also scale the initial (full scale) vector field in the
                # first iteration
                vector_field = match_vector_field(vector_field, scaled_fixed_image)

            if has_initial_vector_field:
                scaled_initial_vector_field = match_vector_field(
                    initial_vector_field, scaled_fixed_image
                )
            else:
                scaled_initial_vector_field = None

            spatial_transformer = self.get_submodule(
                f"spatial_transformer_level_{i_level}",
            )

            level_metrics = []

            for i_iteration in range(iterations):
                t_step_start = time.time()
                if has_initial_vector_field:
                    # if we have an initial vector field: compose both vector fields.
                    # Here: resolution defined by given registration level
                    composed_vector_field = vector_field + spatial_transformer(
                        scaled_initial_vector_field, vector_field
                    )
                else:
                    composed_vector_field = vector_field

                warped_moving = spatial_transformer(
                    scaled_moving_image,
                    composed_vector_field,
                )
                warped_scaled_moving_mask = spatial_transformer(
                    scaled_moving_mask, composed_vector_field, default_value=0
                )
                if self.debug:
                    if fixed_landmarks is not None:
                        cropped_fixed_landmarks = fixed_landmarks - [
                            bbox[i].start for i in range(2, n_image_dimensions)
                        ]
                        exact_scale_factor = match_vector_field(
                            vector_field=composed_vector_field,
                            image=fixed_image,
                            return_scale_factor=True,
                        )
                        scaled_fixed_landmarks = torch.as_tensor(
                            cropped_fixed_landmarks / exact_scale_factor
                        ).to(device)
                        warped_fixed_landmarks = TRELoss._warped_fixed_landmarks(
                            fixed_landmarks=scaled_fixed_landmarks,
                            vector_field=composed_vector_field,
                        )
                        warped_fixed_landmarks = (
                            warped_fixed_landmarks.detach().cpu() * exact_scale_factor
                        ).numpy()
                        warped_fixed_landmarks = warped_fixed_landmarks + [
                            bbox[i].start for i in range(2, n_image_dimensions)
                        ]

                level_metrics.append(
                    self._calculate_metric(
                        moving_image=warped_moving,
                        fixed_image=scaled_fixed_image,
                        fixed_mask=scaled_fixed_mask,
                    )
                )

                # one vector field update step
                if yield_each_step:
                    vector_field_before = vector_field.clone()

                vector_field = self._update_step(
                    level=i_level,
                    iteration=i_iteration,
                    scale_factor=scale_factor,
                    vector_field=vector_field,
                    moving_image=scaled_moving_image,
                    warped_image=warped_moving,
                    fixed_image=scaled_fixed_image,
                    moving_mask=warped_scaled_moving_mask,
                    fixed_mask=scaled_fixed_mask,
                    original_image_spacing=original_image_spacing,
                    initial_vector_field=scaled_initial_vector_field,
                )

                if yield_each_step:
                    yield dict(
                        type="intermediate",
                        level=i_level,
                        iteration=i_iteration,
                        moving_image=scaled_moving_image,
                        fixed_image=scaled_fixed_image,
                        warped_image=warped_moving,
                        moving_mask=scaled_moving_mask,
                        fixed_mask=scaled_fixed_mask,
                        warped_mask=warped_scaled_moving_mask,
                        vector_field_before=vector_field_before,
                        vector_field_after=vector_field,
                    )

                # check early stopping
                if self._check_early_stopping(metrics=level_metrics, i_level=i_level):
                    break

                log = {
                    "level": i_level,
                    "iteration": i_iteration,
                    # "tau": decayed_tau,
                    "metric": level_metrics[-1],
                }

                t_step_end = time.time()
                log["step_runtime"] = t_step_end - t_step_start
                self.logger.debug(log)

                # here we do the debug stuff
                if self.debug and (
                    i_iteration % self.debug_step_size == 0 or i_iteration < 10
                ):
                    debug_metrics = {
                        "metric": level_metrics[-1],
                        "level_image_shape": tuple(scaled_moving_image.shape),
                        "vector_field": {},
                    }

                    dim_names = ("x", "y", "z")
                    for i_dim in range(n_spatial_dims):
                        debug_metrics["vector_field"][dim_names[i_dim]] = {
                            "min": float(torch.min(vector_field[:, i_dim])),
                            # "Q0.05": float(
                            #     torch.quantile(vector_field[:, i_dim], 0.05)
                            # ),
                            "mean": float(torch.mean(vector_field[:, i_dim])),
                            # "Q0.95": float(
                            #     torch.quantile(vector_field[:, i_dim], 0.95)
                            # ),
                            "max": float(torch.max(vector_field[:, i_dim])),
                        }
                    self._plotter.save_snapshot(
                        moving_image=scaled_moving_image,
                        fixed_image=scaled_fixed_image,
                        warped_image=warped_moving,
                        forces=None,
                        vector_field=vector_field,
                        moving_mask=scaled_moving_mask,
                        fixed_mask=scaled_fixed_mask,
                        warped_mask=warped_scaled_moving_mask,
                        full_spatial_shape=full_cropped_spatial_shape,
                        stage="vroc",
                        level=i_level,
                        scale_factor=scale_factor,
                        iteration=i_iteration,
                        metrics=debug_metrics,
                        output_folder=self.debug_output_folder,
                    )
                    if fixed_landmarks is not None:
                        write_landmarks(
                            landmarks=fixed_landmarks,
                            filepath=self.debug_output_folder
                            / f"warped_landmarks_initial.csv",
                        )
                        write_landmarks(
                            landmarks=warped_fixed_landmarks,
                            filepath=self.debug_output_folder
                            / f"warped_landmarks_level_{i_level:02d}_iteration_{i_iteration:04d}.csv",
                        )

                    self.logger.debug(
                        f"Created snapshot of registration at level={i_level}, iteration={i_iteration}"
                    )

            metrics.append(level_metrics)

        vector_field = match_vector_field(vector_field, full_size_moving)

        if self.restrict_to_mask_bbox:
            # undo restriction to mask, i.e. insert results into full size data
            _vector_field = torch.zeros(
                vector_field.shape[:2] + original_moving_image.shape[2:],
                device=vector_field.device,
            )
            _vector_field[(...,) + bbox[2:]] = vector_field
            vector_field = _vector_field

        spatial_transformer = SpatialTransformer(
            shape=full_uncropped_shape[2:], default_value=self.default_voxel_value
        ).to(fixed_image.device)

        result = {"type": "final"}

        if initial_vector_field is not None:
            # if we have an initial vector field: compose both vector fields.
            # Here: at full resolution without cropping/bbox
            composed_vector_field = vector_field + spatial_transformer(
                original_initial_vector_field, vector_field
            )
            result["vector_fields"] = [original_initial_vector_field, vector_field]
        else:
            composed_vector_field = vector_field
            result["vector_fields"] = [vector_field]

        result["composed_vector_field"] = composed_vector_field

        warped_moving_image = spatial_transformer(
            original_moving_image, composed_vector_field
        )
        if original_moving_mask is not None:
            warped_moving_mask = spatial_transformer(
                original_moving_mask, composed_vector_field, default_value=0
            )
        else:
            warped_moving_mask = None

        if initial_vector_field is not None:
            warped_affine_moving_image = spatial_transformer(
                original_moving_image, original_initial_vector_field
            )

            if original_moving_mask is not None:
                warped_affine_moving_mask = spatial_transformer(
                    original_moving_mask, original_initial_vector_field, default_value=0
                )
            else:
                warped_affine_moving_mask = None
        else:
            warped_affine_moving_image = None
            warped_affine_moving_mask = None

        metric_after = self._calculate_metric(
            moving_image=warped_moving_image[bbox],
            fixed_image=fixed_image,
            fixed_mask=fixed_mask,
        )

        if self.debug:
            debug_info = {
                "metric_before": metric_before,
                "metric_after": metric_after,
                "level_metrics": metrics,
            }

        else:
            debug_info = None
        result["debug_info"] = debug_info
        result["warped_moving_image"] = warped_moving_image
        result["warped_moving_mask"] = warped_moving_mask
        result["warped_affine_moving_mask"] = warped_affine_moving_mask
        result["warped_affine_moving_image"] = warped_affine_moving_image

        yield result

    def run_registration(
        self,
        moving_image: torch.Tensor,
        fixed_image: torch.Tensor,
        moving_mask: torch.Tensor | None = None,
        fixed_mask: torch.Tensor | None = None,
        moving_landmarks: torch.Tensor | None = None,
        fixed_landmarks: torch.Tensor | None = None,
        original_image_spacing: FloatTuple3D = (1.0, 1.0, 1.0),
        initial_vector_field: torch.Tensor | None = None,
        yield_each_step: bool = False,
    ):
        # registration_generator is either a generator yielding each registration step
        # or just a generator of length 1 yielding the final result
        registration_generator = self._run_registration(
            moving_image=moving_image,
            fixed_image=fixed_image,
            moving_mask=moving_mask,
            fixed_mask=fixed_mask,
            moving_landmarks=moving_landmarks,
            fixed_landmarks=fixed_landmarks,
            original_image_spacing=original_image_spacing,
            initial_vector_field=initial_vector_field,
            yield_each_step=yield_each_step,
        )

        if yield_each_step:

            def yield_registration_steps():
                yield from registration_generator

            return yield_registration_steps()
        else:
            return next(registration_generator)

    @timing()
    def forward(
        self,
        moving_image: torch.Tensor,
        fixed_image: torch.Tensor,
        moving_mask: torch.Tensor,
        fixed_mask: torch.Tensor,
        original_image_spacing: FloatTuple3D,
    ):
        return self.run_registration(
            moving_image=moving_image,
            fixed_image=fixed_image,
            moving_mask=moving_mask,
            fixed_mask=fixed_mask,
            original_image_spacing=original_image_spacing,
        )


class ModelBasedVariationalRegistration(VariationalRegistration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        models = {}
        for i_level in range(3):
            # state = torch.load(
            #     f"/datalake/dirlab_4dct/output/models/f303bb5a2cde4442bd76ef69/level_{i_level}_step_041.pth",
            # )
            #
            # model = DemonsVectorFieldBoosterStable()

            state = torch.load(
                f"/datalake/dirlab_4dct/output/models/37a25ccdc294401593b0f7e4/level_{i_level}_step_5379.pth",
            )

            model = DemonsForceModulator()
            model.load_state_dict(state["model"])
            model.eval()
            models[f"level_{i_level}"] = model

        self.models = nn.ModuleDict(models)

        self.alternation_scheme = {"standard": 0, "model": 1}

    def _update_step(
        self,
        level: int,
        iteration: int,
        scale_factor: float,
        vector_field: torch.Tensor,
        moving_image: torch.Tensor,
        warped_image: torch.Tensor,
        fixed_image: torch.Tensor,
        moving_mask: torch.Tensor | None = None,
        fixed_mask: torch.Tensor | None = None,
        original_image_spacing: FloatTuple3D = (1.0, 1.0, 1.0),
        initial_vector_field: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mode = get_mode_from_alternation_scheme(
            alternation_scheme=self.alternation_scheme, iteration=iteration
        )
        if level > 0:
            mode = "standard"

        if mode == "standard":
            vector_field = super()._update_step(
                level=level,
                iteration=iteration,
                scale_factor=scale_factor,
                vector_field=vector_field,
                moving_image=moving_image,
                warped_image=warped_image,
                fixed_image=fixed_image,
                moving_mask=moving_mask,
                fixed_mask=fixed_mask,
                original_image_spacing=original_image_spacing,
                initial_vector_field=initial_vector_field,
            )

        elif mode == "model":
            model = self.models[f"level_{level}"]

            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=True
            ), torch.inference_mode():
                vector_field = model(
                    moving_image=moving_image,
                    warped_image=warped_image,
                    fixed_image=fixed_image,
                    moving_mask=moving_mask,
                    fixed_mask=fixed_mask,
                    vector_field=vector_field,
                    image_spacing=None,
                )
        else:
            raise ValueError(f"Unknown {mode=}")

        self.logger.debug(f"Using {mode} vector field update step")

        return vector_field


class VariationalRegistrationWithForceEstimation(VariationalRegistration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        model = TrainableDemonForces()
        model.load_state_dict(
            torch.load(
                "/datalake/learn2reg/2023/37db95e6-19e4-4e0a-a177-4b12df6ae7b5/data/force_estimation/model.pth"
            )["model"]
        )

        self.model_level_0 = model
        self.model_level_1 = model
        self.model_level_2 = model

        self.models = {
            "level_0": self.model_level_0,
            "level_1": self.model_level_1,
            "level_2": self.model_level_2,
        }

    def _update_step(
        self,
        level: int,
        iteration: int,
        scale_factor: float,
        vector_field: torch.Tensor,
        moving_image: torch.Tensor,
        warped_image: torch.Tensor,
        fixed_image: torch.Tensor,
        moving_mask: torch.Tensor | None = None,
        fixed_mask: torch.Tensor | None = None,
        original_image_spacing: FloatTuple3D = (1.0, 1.0, 1.0),
        initial_vector_field: torch.Tensor | None = None,
    ) -> torch.Tensor:
        level_model = self.models[f"level_{level}"]

        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=True):
            estimated_forces, active_forces, passive_forces = level_model(
                moving_image=warped_image,
                fixed_image=fixed_image,
                moving_mask=moving_mask,
                fixed_mask=fixed_mask,
            )

        decayed_tau = exponential_decay(
            initial_value=self.tau[level],
            i_level=level,
            i_iteration=iteration,
            level_lambda=self.tau_level_decay,
            iteration_lambda=self.tau_iteration_decay,
        )

        if iteration % 5 == 0:
            forces = estimated_forces
        else:
            forces = active_forces

        forces = active_forces

        vector_field = vector_field + decayed_tau * forces

        decayed_sigma = tuple(
            exponential_decay(
                initial_value=s,
                i_level=level,
                i_iteration=iteration,
                level_lambda=self.sigma_level_decay,
                iteration_lambda=self.sigma_iteration_decay,
            )
            for s in self.regularization_sigma[level]
        )
        n_spatial_dims = vector_field.shape[1]
        sigma_cutoff = (2.0, 2.0, 2.0)[:n_spatial_dims]
        gaussian_smoothing = self._GAUSSIAN_SMOOTHING[n_spatial_dims]
        _regularization_layer = gaussian_smoothing(
            sigma=decayed_sigma,
            sigma_cutoff=sigma_cutoff,
            force_same_size=True,
            spacing=self.original_image_spacing,
            use_image_spacing=self.use_image_spacing,
        ).to(vector_field)

        vector_field = _regularization_layer(vector_field)

        return vector_field


class DemonsVectorFieldBooster(nn.Module, LoggerMixin):
    def __init__(
        self,
        n_iterations: int = 10,
        gradient_type: Literal["active", "passive", "dual"] = "dual",
        use_masks: bool = True,
    ):
        super().__init__()

        self.n_iterations = n_iterations
        self.forces = DemonForces(method=gradient_type)
        self.use_masks = use_masks

        regularization_1 = [
            nn.Conv3d(
                in_channels=4,
                out_channels=16,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
        ]
        regularization_2 = [
            nn.Conv3d(
                in_channels=4,
                out_channels=16,
                kernel_size=3,
                dilation=2,
                padding="same",
                bias=True,
            ),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
        ]

        fuse = [
            nn.Conv3d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(
                in_channels=16,
                out_channels=3,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
        ]

        self.regularization_1 = nn.Sequential(*regularization_1)
        self.regularization_2 = nn.Sequential(*regularization_2)
        self.fuse = nn.Sequential(*fuse)
        self.spatial_transformer = SpatialTransformer()

    def forward(
        self,
        moving_image: torch.Tensor,
        fixed_image: torch.Tensor,
        moving_mask: torch.Tensor,
        fixed_mask: torch.Tensor,
        vector_field: torch.Tensor,
        image_spacing: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        spatial_image_shape = moving_image.shape[2:]
        vector_field_boost = torch.zeros(
            (1, 3) + spatial_image_shape, device=moving_image.device
        )

        for _ in range(self.n_iterations):
            composed_vector_field = vector_field_boost + self.spatial_transformer(
                vector_field, vector_field_boost
            )
            warped_moving_image = self.spatial_transformer(
                moving_image, composed_vector_field
            )

            diff = (warped_moving_image - fixed_image) / (fixed_image + 1e-6)
            diff = F.softsign(diff)

            forces = self.forces(
                moving_image,
                fixed_image,
                moving_mask,
                fixed_mask,
                image_spacing,
            )

            updated_vector_field_boost = vector_field_boost + forces

            updated_vector_field_boost_1 = self.regularization_1(
                torch.concat((updated_vector_field_boost, diff), dim=1)
            )
            updated_vector_field_boost_2 = self.regularization_2(
                torch.concat((updated_vector_field_boost, diff), dim=1)
            )
            vector_field_boost = vector_field_boost + self.fuse(
                torch.concat(
                    (updated_vector_field_boost_1, updated_vector_field_boost_2),
                    dim=1,
                )
            )

        if self.use_masks:
            union_mask = torch.logical_or(moving_mask, fixed_mask).to(torch.float32)
            return vector_field_boost * union_mask
        return vector_field_boost


class DemonsVectorFieldBooster2(nn.Module, LoggerMixin):
    def __init__(
        self,
        n_iterations: int = 10,
        gradient_type: Literal["active", "passive", "dual"] = "dual",
    ):
        super().__init__()

        self.n_iterations = n_iterations
        self.forces = DemonForces(method=gradient_type)

        regularization_1 = [
            nn.Conv3d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
        ]
        regularization_2 = [
            nn.Conv3d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                dilation=2,
                padding="same",
                bias=True,
            ),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
        ]

        fuse = [
            nn.Conv3d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(
                in_channels=16,
                out_channels=3,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
        ]

        self.regularization_1 = nn.Sequential(*regularization_1)
        self.regularization_2 = nn.Sequential(*regularization_2)
        self.fuse = nn.Sequential(*fuse)
        self.spatial_transformer = SpatialTransformer()

    def forward(
        self,
        moving_image: torch.Tensor,
        fixed_image: torch.Tensor,
        moving_mask: torch.Tensor,
        fixed_mask: torch.Tensor,
        vector_field: torch.Tensor,
        image_spacing: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        spatial_image_shape = moving_image.shape[2:]
        vector_field_boost = torch.zeros(
            (1, 3) + spatial_image_shape, device=moving_image.device
        )

        for _ in range(self.n_iterations):
            composed_vector_field = vector_field_boost + self.spatial_transformer(
                vector_field, vector_field_boost
            )
            warped_moving_image = self.spatial_transformer(
                moving_image, composed_vector_field
            )

            # diff = (warped_moving_image - fixed_image) / (fixed_image + 1e-6)
            # diff = F.softsign(diff)

            forces = self.forces(
                warped_moving_image,
                fixed_image,
                moving_mask,
                fixed_mask,
                image_spacing,
            )

            updated_vector_field_boost = vector_field_boost + forces

            updated_vector_field_boost_1 = self.regularization_1(
                updated_vector_field_boost
            )
            updated_vector_field_boost_2 = self.regularization_2(
                updated_vector_field_boost
            )

            vector_field_boost = vector_field_boost + self.fuse(
                torch.concat(
                    (updated_vector_field_boost_1, updated_vector_field_boost_2),
                    dim=1,
                )
            )

        union_mask = torch.logical_or(moving_mask, fixed_mask).to(torch.float32)

        return vector_field_boost * union_mask


class DemonsVectorFieldBooster3(nn.Module, LoggerMixin):
    def __init__(
        self,
        n_iterations: int = 10,
        gradient_type: Literal["active", "passive", "dual"] = "dual",
    ):
        super().__init__()

        self.n_iterations = n_iterations
        self.forces = DemonForces(method=gradient_type)

        regularization_1 = [
            nn.Conv3d(
                in_channels=3,
                out_channels=3,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(
                in_channels=3,
                out_channels=3,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
        ]
        regularization_2 = [
            nn.Conv3d(
                in_channels=3,
                out_channels=3,
                kernel_size=3,
                dilation=2,
                padding="same",
                bias=True,
            ),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(
                in_channels=3,
                out_channels=3,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
        ]

        fuse = [
            nn.Conv3d(
                in_channels=6,
                out_channels=3,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
        ]

        self.regularization_1 = nn.Sequential(*regularization_1)
        self.regularization_2 = nn.Sequential(*regularization_2)
        self.fuse = nn.Sequential(*fuse)
        self.spatial_transformer = SpatialTransformer()

    def forward(
        self,
        moving_image: torch.Tensor,
        fixed_image: torch.Tensor,
        moving_mask: torch.Tensor,
        fixed_mask: torch.Tensor,
        vector_field: torch.Tensor,
        image_spacing: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        spatial_image_shape = moving_image.shape[2:]
        vector_field_boost = torch.zeros(
            (1, 3) + spatial_image_shape, device=moving_image.device
        )

        for _ in range(self.n_iterations):
            composed_vector_field = vector_field_boost + self.spatial_transformer(
                vector_field, vector_field_boost
            )
            warped_moving_image = self.spatial_transformer(
                moving_image, composed_vector_field
            )

            # diff = (warped_moving_image - fixed_image) / (fixed_image + 1e-6)
            # diff = F.softsign(diff)

            forces = self.forces(
                warped_moving_image,
                fixed_image,
                moving_mask,
                fixed_mask,
                image_spacing,
            )

            updated_vector_field_boost = vector_field_boost + forces

            updated_vector_field_boost_1 = self.regularization_1(
                updated_vector_field_boost
            )
            updated_vector_field_boost_2 = self.regularization_2(
                updated_vector_field_boost
            )

            vector_field_boost = vector_field_boost + self.fuse(
                torch.concat(
                    (updated_vector_field_boost_1, updated_vector_field_boost_2),
                    dim=1,
                )
            )

        union_mask = torch.logical_or(moving_mask, fixed_mask).to(torch.float32)

        return vector_field_boost * union_mask


class DemonsVectorFieldBooster4(nn.Module, LoggerMixin):
    def __init__(
        self,
        n_iterations: int = 10,
        gradient_type: Literal["active", "passive", "dual"] = "dual",
    ):
        super().__init__()

        self.n_iterations = n_iterations
        self.forces = DemonForces(method=gradient_type)

        regularization_1 = [
            nn.Conv3d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
        ]
        regularization_2 = [
            nn.Conv3d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                dilation=2,
                padding="same",
                bias=True,
            ),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
        ]

        fuse = [
            nn.Conv3d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(
                in_channels=16,
                out_channels=3,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
        ]

        self.regularization_1 = nn.Sequential(*regularization_1)
        self.regularization_2 = nn.Sequential(*regularization_2)
        self.fuse = nn.Sequential(*fuse)
        self.spatial_transformer = SpatialTransformer()

    def forward(
        self,
        moving_image: torch.Tensor,
        fixed_image: torch.Tensor,
        moving_mask: torch.Tensor,
        fixed_mask: torch.Tensor,
        vector_field: torch.Tensor,
        image_spacing: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        spatial_image_shape = moving_image.shape[2:]
        vector_field_boost = torch.zeros(
            (1, 3) + spatial_image_shape, device=moving_image.device
        )

        for _ in range(self.n_iterations):
            composed_vector_field = vector_field_boost + self.spatial_transformer(
                vector_field, vector_field_boost
            )
            warped_moving_image = self.spatial_transformer(
                moving_image, composed_vector_field
            )

            diff = (warped_moving_image - fixed_image) / (fixed_image + 1e-6)
            diff = F.softsign(diff)

            forces = self.forces(
                moving_image,
                fixed_image,
                moving_mask,
                fixed_mask,
                image_spacing,
            )

            updated_vector_field_boost_1 = self.regularization_1(diff)
            updated_vector_field_boost_2 = self.regularization_2(diff)
            vector_field_boost = vector_field_boost + self.fuse(
                torch.concat(
                    (updated_vector_field_boost_1, updated_vector_field_boost_2),
                    dim=1,
                )
            )

        union_mask = torch.logical_or(moving_mask, fixed_mask).to(torch.float32)

        return vector_field_boost * union_mask


class DemonsVectorFieldBooster5(nn.Module, LoggerMixin):
    def __init__(
        self,
        n_iterations: int = 1,
        gradient_type: Literal["active", "passive", "dual"] = "dual",
        regularization_channels: int = 16,
    ):
        super().__init__()

        self.n_iterations = n_iterations
        self.forces = DemonForces(method=gradient_type)

        tau_1 = [
            nn.Conv3d(
                in_channels=3,
                out_channels=regularization_channels,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(
                in_channels=regularization_channels,
                out_channels=3,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
        ]

        tau_2 = [
            nn.Conv3d(
                in_channels=3,
                out_channels=regularization_channels,
                kernel_size=3,
                dilation=2,
                padding="same",
                bias=True,
            ),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(
                in_channels=regularization_channels,
                out_channels=3,
                kernel_size=3,
                dilation=2,
                padding="same",
                bias=True,
            ),
        ]

        regularization_1 = [
            nn.Conv3d(
                in_channels=3,
                out_channels=regularization_channels,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(
                in_channels=regularization_channels,
                out_channels=3,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
        ]
        regularization_2 = [
            nn.Conv3d(
                in_channels=3,
                out_channels=regularization_channels,
                kernel_size=3,
                dilation=2,
                padding="same",
                bias=True,
            ),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(
                in_channels=regularization_channels,
                out_channels=3,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
        ]

        fuse = [
            nn.Conv3d(
                in_channels=6,
                out_channels=32,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(
                in_channels=16,
                out_channels=3,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
        ]

        self.tau_1 = nn.Sequential(*tau_1)
        self.tau_2 = nn.Sequential(*tau_2)
        self.regularization_1 = nn.Sequential(*regularization_1)
        self.regularization_2 = nn.Sequential(*regularization_2)
        self.fuse = nn.Sequential(*fuse)
        self.spatial_transformer = SpatialTransformer()

    def forward(
        self,
        moving_image: torch.Tensor,
        fixed_image: torch.Tensor,
        moving_mask: torch.Tensor,
        fixed_mask: torch.Tensor,
        vector_field: torch.Tensor,
        image_spacing: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        spatial_image_shape = moving_image.shape[2:]
        vector_field_boost = torch.zeros(
            (1, 3) + spatial_image_shape, device=moving_image.device
        )

        for _ in range(self.n_iterations):
            composed_vector_field = vector_field_boost + self.spatial_transformer(
                vector_field, vector_field_boost
            )
            warped_moving_image = self.spatial_transformer(
                moving_image, composed_vector_field
            )

            forces = self.forces(
                warped_moving_image,
                fixed_image,
                moving_mask,
                fixed_mask,
                image_spacing,
            )

            forces = 0.5 * self.tau_1(forces) + 0.5 * self.tau_2(forces)

            regularized_forces_1 = self.regularization_1(forces)
            regularized_forces_2 = self.regularization_2(forces)

            vector_field_boost = vector_field_boost + self.fuse(
                torch.concat((regularized_forces_1, regularized_forces_2), dim=1)
            )

        # union_mask = torch.logical_or(moving_mask, fixed_mask).to(torch.float32)

        return vector_field_boost


class DemonsVectorFieldBoosterStable(nn.Module, LoggerMixin):
    def __init__(
        self,
        gradient_type: Literal["active", "passive", "dual"] = "dual",
    ):
        super().__init__()

        self.forces = DemonForces(method=gradient_type)

        kernel_guesser_1 = [
            nn.Conv3d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
            nn.Mish(),
            nn.Conv3d(
                in_channels=32,
                out_channels=9 * 7,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
        ]

        kernel_guesser_2 = [
            nn.Conv3d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
            nn.Mish(),
            nn.Conv3d(
                in_channels=32,
                out_channels=9 * 7,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
        ]

        force_weighter = [
            nn.Conv3d(
                in_channels=2,
                out_channels=16,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
            nn.Mish(),
            nn.Conv3d(
                in_channels=16,
                out_channels=3,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
        ]

        self.kernel_guesser_1 = nn.Sequential(*kernel_guesser_1)
        self.kernel_guesser_2 = nn.Sequential(*kernel_guesser_2)
        self.force_weighter = nn.Sequential(*force_weighter)

        self.spatial_transformer = SpatialTransformer()

    def forward(
        self,
        moving_image: torch.Tensor,
        warped_image: torch.Tensor,
        fixed_image: torch.Tensor,
        moving_mask: torch.Tensor,
        fixed_mask: torch.Tensor,
        vector_field: torch.Tensor,
        image_spacing: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # warped_moving_image = self.spatial_transformer(moving_image, vector_field)

        forces = self.forces(
            warped_image,
            fixed_image,
            moving_mask,
            fixed_mask,
            image_spacing,
        )

        full_shape = forces.shape[2:]

        kernel_1 = self.kernel_guesser_1(forces)
        kernel_1 = torch.mean(kernel_1, dim=(-3, -2, -1))
        kernel_1 = kernel_1.reshape((3, 3, 7))
        kernel_1 = torch.softmax(kernel_1, dim=-1)

        # kernel_2 = self.kernel_guesser_2(forces)
        # kernel_2 = torch.mean(kernel_2, dim=(-3, -2, -1))
        # kernel_2 = kernel_2.reshape((3, 3, 7))
        # kernel_2 = torch.softmax(kernel_2, dim=-1)

        regularized_forces_1 = torch.concat(
            (
                separable_normed_conv_3d(
                    forces[:, 0:1],
                    kernel_x=kernel_1[0, 0],
                    kernel_y=kernel_1[0, 1],
                    kernel_z=kernel_1[0, 2],
                ),
                separable_normed_conv_3d(
                    forces[:, 1:2],
                    kernel_x=kernel_1[1, 0],
                    kernel_y=kernel_1[1, 1],
                    kernel_z=kernel_1[1, 2],
                ),
                separable_normed_conv_3d(
                    forces[:, 2:3],
                    kernel_x=kernel_1[2, 0],
                    kernel_y=kernel_1[2, 1],
                    kernel_z=kernel_1[2, 2],
                ),
            ),
            dim=1,
        )

        # regularized_forces_2 = torch.concat(
        #     (
        #         separable_normed_conv_3d(
        #             forces[:, 0:1],
        #             kernel_x=kernel_2[0, 0],
        #             kernel_y=kernel_2[0, 1],
        #             kernel_z=kernel_2[0, 2],
        #         ),
        #         separable_normed_conv_3d(
        #             forces[:, 1:2],
        #             kernel_x=kernel_2[1, 0],
        #             kernel_y=kernel_2[1, 1],
        #             kernel_z=kernel_2[1, 2],
        #         ),
        #         separable_normed_conv_3d(
        #             forces[:, 2:3],
        #             kernel_x=kernel_2[2, 0],
        #             kernel_y=kernel_2[2, 1],
        #             kernel_z=kernel_2[2, 2],
        #         ),
        #     ),
        #     dim=1,
        # )

        # global_min = min(warped_moving_image.min(), fixed_image.min())
        # global_max = min(warped_moving_image.max(), fixed_image.max())
        #
        # warped_moving_image = (warped_moving_image - global_min) / (
        #     global_max - global_min
        # )
        # fixed_image = (fixed_image - global_min) / (global_max - global_min)
        #
        # force_weighting = self.force_weighter(
        #     torch.concat((warped_moving_image, fixed_image), dim=1)
        # )
        # force_weighting = torch.softmax(force_weighting, dim=1)
        #
        # regularized_forces = (
        #     torch.zeros_like(regularized_forces_1) * force_weighting[:, 0:1]
        #     + regularized_forces_1 * force_weighting[:, 1:2]
        #     # + regularized_forces_2 * force_weighting[:, 2:3]
        # )

        tau = 2.25
        vector_field = vector_field + tau * regularized_forces_1

        return vector_field


class DemonsVectorFieldBoosterForceLearning(nn.Module, LoggerMixin):
    def __init__(self):
        super().__init__()

        force_estimator = [
            nn.Conv3d(
                in_channels=2,
                out_channels=16,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
            nn.Mish(),
            nn.Conv3d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
            nn.Mish(),
            nn.Conv3d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
            nn.Mish(),
            nn.Conv3d(
                in_channels=16,
                out_channels=3,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
        ]

        self.force_estimator = nn.Sequential(*force_estimator)
        self.spatial_transformer = SpatialTransformer()

    def forward(
        self,
        moving_image: torch.Tensor,
        fixed_image: torch.Tensor,
        moving_mask: torch.Tensor,
        fixed_mask: torch.Tensor,
        vector_field: torch.Tensor,
        image_spacing: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        warped_moving_image = self.spatial_transformer(moving_image, vector_field)

        global_min = min(warped_moving_image.min(), fixed_image.min())
        global_max = min(warped_moving_image.max(), fixed_image.max())

        warped_moving_image = (warped_moving_image - global_min) / (
            global_max - global_min
        )
        fixed_image = (fixed_image - global_min) / (global_max - global_min)

        forces = self.force_estimator(
            torch.cat((warped_moving_image, fixed_image), dim=1)
        )
        forces = F.softsign(forces)
        vector_field = vector_field + forces

        return vector_field


class DemonsForceCorrector(nn.Module, LoggerMixin):
    def __init__(
        self,
        gradient_type: Literal["active", "passive", "dual"] = "dual",
    ):
        super().__init__()

        self.gradient_type = gradient_type

        force_corrector = [
            nn.Conv3d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=False,
            ),
            nn.Mish(inplace=True),
            nn.Conv3d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=False,
            ),
            nn.Mish(inplace=True),
            nn.Conv3d(
                in_channels=32,
                out_channels=3,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=False,
            ),
        ]

        self.demon_forces = DemonForces(method=gradient_type)
        self.force_corrector = nn.Sequential(*force_corrector)
        self.spatial_transformer = SpatialTransformer()

    def forward(
        self,
        moving_image: torch.Tensor,
        warped_image: torch.Tensor,
        fixed_image: torch.Tensor,
        moving_mask: torch.Tensor,
        fixed_mask: torch.Tensor,
        vector_field: torch.Tensor,
        image_spacing: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # warped_moving_image = self.spatial_transformer(moving_image, vector_field)

        demon_forces = self.demon_forces(
            warped_image,
            fixed_image,
            moving_mask,
            fixed_mask,
            image_spacing,
        )

        corrected_forces = self.force_corrector(demon_forces)

        corrected_forces = 0.5 * torch.tanh(corrected_forces)

        tau = 2.25
        vector_field = vector_field + tau * corrected_forces

        sigma_cutoff = (2.0, 2.0, 2.0)
        _regularization_layer = GaussianSmoothing3d(
            sigma=(1.25, 1.25, 1.25), sigma_cutoff=sigma_cutoff, force_same_size=True
        ).to(vector_field)

        vector_field = _regularization_layer(vector_field)

        return vector_field


class DemonsForceModulator(nn.Module, LoggerMixin):
    def __init__(
        self,
        gradient_type: Literal["active", "passive", "dual"] = "dual",
    ):
        super().__init__()

        self.gradient_type = gradient_type

        modulator = [
            nn.Conv3d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
            nn.Mish(inplace=True),
            nn.Conv3d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
            nn.Mish(inplace=True),
            nn.Conv3d(
                in_channels=32,
                out_channels=3,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
        ]

        self.demon_forces = DemonForces(method=gradient_type)
        self.modulator = nn.Sequential(*modulator)
        self.spatial_transformer = SpatialTransformer()

    def forward(
        self,
        moving_image: torch.Tensor,
        warped_image: torch.Tensor,
        fixed_image: torch.Tensor,
        moving_mask: torch.Tensor,
        fixed_mask: torch.Tensor,
        vector_field: torch.Tensor,
        image_spacing: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # moving image is warped image
        demon_forces = self.demon_forces(
            warped_image,
            fixed_image,
            moving_mask,
            fixed_mask,
            image_spacing,
        )

        max_modulation = 0.10
        modulation = self.modulator(demon_forces)
        modulation = torch.tanh(modulation)
        modulation = 1.0 + modulation * max_modulation

        modulated_forces = demon_forces * modulation
        modulated_forces = torch.clip(modulated_forces, -0.5, 0.5)

        # fig, ax = plt.subplots(1, 3)
        # i_slice = 90
        # ax[0].imshow(demon_forces.detach().cpu().numpy()[0, 2, :, i_slice, :])
        # ax[1].imshow(modulation.detach().cpu().numpy()[0, 2, :, i_slice, :])
        # ax[2].imshow(modulated_forces.detach().cpu().numpy()[0, 2, :, i_slice, :])
        # plt.show()

        tau = 2.25
        sigma = (1.25, 1.25, 1.25)
        sigma_cutoff = (2.0, 2.0, 2.0)
        vector_field = vector_field + tau * modulated_forces

        _regularization_layer = GaussianSmoothing3d(
            sigma=sigma, sigma_cutoff=sigma_cutoff, force_same_size=True
        ).to(vector_field)

        vector_field = _regularization_layer(vector_field)

        return vector_field


# class DemonsVectorFieldBooster(nn.Module, LoggerMixin):
#     def __init__(
#         self,
#         n_iterations: int = 10,
#         filter_base: int = 16,
#         gradient_type: Literal["active", "passive", "dual"] = "dual",
#     ):
#         super().__init__()
#
#         self.n_iterations = n_iterations
#         self.filter_base = filter_base
#         self.forces = DemonForces(method=gradient_type)
#
#         # self.regularization = TrainableRegularization3d(n_levels=4, filter_base=16)
#         self.regularization = DynamicRegularization3d(filter_base=16)
#         self.spatial_transformer = SpatialTransformer()
#
#
#         self.factors = (0.125, 0.25, 0.5, 1.0)
#         self.n_levels = len(self.factors)
#         self.weighting_net = FlexUNet(
#             n_channels=2, n_levels=4, n_classes=self.n_levels + 3, filter_base=4, norm_layer=nn.InstanceNorm3d, return_bottleneck=False, skip_connections=True
#         )
#
#
#
#     def forward(
#         self,
#         moving_image: torch.Tensor,
#         fixed_image: torch.Tensor,
#         moving_mask: torch.Tensor,
#         fixed_mask: torch.Tensor,
#         vector_field: torch.Tensor,
#         image_spacing: torch.Tensor,
#         n_iterations: int | None = None,
#     ) -> torch.Tensor:
#
#         spatial_image_shape = moving_image.shape[2:]
#         vector_field_boost = torch.zeros(
#             (1, 3) + spatial_image_shape, device=moving_image.device
#         )
#
#         _n_iterations = n_iterations or self.n_iterations
#         for _ in range(_n_iterations):
#             composed_vector_field = self.spatial_transformer.compose_vector_fields(
#                 vector_field, vector_field_boost
#             )
#
#             # warp image with boosted vector field
#             warped_moving_image = self.spatial_transformer(
#                 moving_image, composed_vector_field
#             )
#
#             # diff = (warped_moving_image - fixed_image) / (fixed_image + 1e-6)
#             # diff = F.softsign(diff)
#
#             forces = self.forces(
#                 warped_moving_image,
#                 fixed_image,
#                 moving_mask,
#                 fixed_mask,
#                 image_spacing,
#             )
#
#             images = torch.concat((moving_image, fixed_image), dim=1)
#
#             output = self.weighting_net(images)
#             weights = output[:, :self.n_levels]
#             weights = torch.softmax(weights, dim=1)
#             taus = output[:, self.n_levels:]
#             taus = 5 * torch.sigmoid(taus)
#             print(f'mean tau x/y/z: {taus[:, 0].mean():.2f}, {taus[:, 1].mean():.2f}, {taus[:, 2].mean():.2f}')
#
#             vector_field_boost = vector_field_boost + taus * forces
#             vector_field_boost = self.regularization(
#                 vector_field=vector_field_boost, moving_image=warped_moving_image, fixed_image=fixed_image, weights=weights
#             )
#
#             # plot weights and tau
#             m = warped_moving_image.detach().cpu().numpy()
#             f = fixed_image.detach().cpu().numpy()
#             diff = (warped_moving_image - fixed_image)
#             diff = diff.detach().cpu().numpy()
#             w = weights.detach().cpu().numpy()
#
#             m, f = diff, diff
#             clim = (-1, 1)
#             cmap = 'seismic'
#             mid_slice = w.shape[-2] // 2
#             fig, ax = plt.subplots(1, self.n_levels + 2, sharex=True, sharey=True)
#             ax[0].imshow(f[0, 0, :, mid_slice, :], clim=clim, cmap=cmap)
#             ax[1].imshow(m[0, 0, :, mid_slice, :], clim=clim, cmap=cmap)
#             for i in range(self.n_levels):
#                 ax[i + 2].imshow(w[0, i, :, mid_slice, :])
#
#             t = taus.detach().cpu().numpy()
#             mid_slice = t.shape[-2] // 2
#             fig, ax = plt.subplots(1, 3 + 2, sharex=True, sharey=True)
#             ax[0].imshow(f[0, 0, :, mid_slice, :], clim=clim, cmap=cmap)
#             ax[1].imshow(m[0, 0, :, mid_slice, :], clim=clim, cmap=cmap)
#             for i in range(3):
#                 ax[i + 2].imshow(t[0, i, :, mid_slice, :])
#
#         return vector_field_boost

# def forward(
#     self,
#     moving_image: torch.Tensor,
#     fixed_image: torch.Tensor,
#     moving_mask: torch.Tensor,
#     fixed_mask: torch.Tensor,
#     vector_field: torch.Tensor,
#     image_spacing: torch.Tensor,
#     n_iterations: int | None = None,
# ) -> torch.Tensor:
#
#     spatial_image_shape = moving_image.shape[2:]
#     vector_field_boost = torch.zeros(
#         (1, 3) + spatial_image_shape, device=moving_image.device
#     )
#
#     _n_iterations = n_iterations or self.n_iterations
#     for _ in range(_n_iterations):
#         composed_vector_field = self.spatial_transformer.compose_vector_fields(
#             vector_field, vector_field_boost
#         )
#
#         # warp image with boosted vector field
#         warped_moving_image = self.spatial_transformer(
#             moving_image, composed_vector_field
#         )
#
#         diff = (warped_moving_image - fixed_image) / (fixed_image + 1e-6)
#         diff = F.softsign(diff)
#
#         forces = self.forces(
#             warped_moving_image,
#             fixed_image,
#             moving_mask,
#             fixed_mask,
#             image_spacing,
#         )
#
#         updated_vector_field_boost = vector_field_boost + forces
#
#         updated_vector_field_boost_1 = self.regularization_1(
#             torch.concat((updated_vector_field_boost, diff), dim=1)
#         )
#         updated_vector_field_boost_2 = self.regularization_2(
#             torch.concat((updated_vector_field_boost, diff), dim=1)
#         )
#         vector_field_boost = vector_field_boost + self.fuse(
#             torch.concat(
#                 (updated_vector_field_boost_1, updated_vector_field_boost_2),
#                 dim=1,
#             )
#         )
#
#     return vector_field_boost


class DemonsVectorFieldArtifactBooster(nn.Module):
    def __init__(self, shape: Tuple[int, int, int]):
        super().__init__()
        self.shape = shape

        boost_layers_1 = [
            nn.Conv3d(
                in_channels=4,
                out_channels=16,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
        ]

        boost_layers_2 = [
            nn.Conv3d(
                in_channels=4,
                out_channels=16,
                kernel_size=3,
                dilation=2,
                padding="same",
                bias=True,
            ),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
        ]

        fuse_layers = [
            nn.Conv3d(
                in_channels=2 * 32,
                out_channels=16,
                kernel_size=3,
                dilation=2,
                padding="same",
                bias=True,
            ),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(
                in_channels=16,
                out_channels=3,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
        ]

        self.boost_1 = nn.Sequential(*boost_layers_1)
        self.boost_2 = nn.Sequential(*boost_layers_2)
        self.fuse = nn.Sequential(*fuse_layers)
        self.spatial_transformer = SpatialTransformer(shape=self.shape)

    def forward(
        self,
        moving_image: torch.Tensor,
        fixed_image: torch.Tensor,
        moving_mask: torch.Tensor,
        fixed_mask: torch.Tensor,
        fixed_artifact_mask: torch.Tensor,
        vector_field: torch.Tensor,
        image_spacing: torch.Tensor,
    ) -> torch.Tensor:
        spatial_image_shape = moving_image.shape[-3:]
        vector_field_boost = torch.zeros(
            (1, 3) + spatial_image_shape, device=moving_image.device
        )

        # composed_vector_field = vector_field_boost + self.spatial_transformer(
        #     vector_field, vector_field_boost
        # )
        # moving_image = self.spatial_transformer(moving_image, composed_vector_field)
        #
        # diff = (moving_image - fixed_image) / (fixed_image + 1e-6)
        # diff = F.softsign(diff)

        input_images = torch.concat((vector_field, fixed_artifact_mask), dim=1)

        vector_field_boost_1 = self.boost_1(input_images)
        vector_field_boost_2 = self.boost_2(input_images)
        vector_field_boost = self.fuse(
            torch.concat((vector_field_boost_1, vector_field_boost_2), dim=1)
        )

        return vector_field_boost
