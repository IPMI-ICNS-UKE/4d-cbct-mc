from __future__ import annotations

from typing import Literal, Sequence, Type

import torch
import torch.nn as nn

from cbctmc.speedup import blocks
from cbctmc.speedup.blocks import (
    ConvInstanceNormMish2D,
    DecoderBlock,
    EncoderBlock,
    ResidualDenseBlock2D,
    _ConvNormActivation,
)


class ResidualDenseNet2D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        growth_rate: int = 32,
        n_blocks: int = 2,
        n_block_layers: int = 4,
        convolution_block: Type[_ConvNormActivation] = ConvInstanceNormMish2D,
        local_feature_fusion_channels: int = 32,
        pre_block_channels: int = 32,
        post_block_channels: int = 32,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.growth_rate = growth_rate
        self.n_blocks = n_blocks
        self.n_block_layers = n_block_layers
        self.convolution_block = convolution_block
        self.local_feature_fusion_channels = local_feature_fusion_channels

        self.pre_block_channels = pre_block_channels
        self.post_block_channels = post_block_channels

        self.pre_blocks = nn.Sequential(
            self.convolution_block(
                in_channels=self.in_channels,
                out_channels=self.pre_block_channels,
                kernel_size=(3, 3),
                padding="same",
            ),
            self.convolution_block(
                in_channels=self.pre_block_channels,
                out_channels=self.pre_block_channels,
                kernel_size=(3, 3),
                padding="same",
            ),
        )

        for i_block in range(self.n_blocks):
            self.add_module(
                f"residual_dense_block_{i_block}",
                ResidualDenseBlock2D(
                    in_channels=self.pre_block_channels
                    if i_block == 0
                    else self.local_feature_fusion_channels,
                    out_channels=self.local_feature_fusion_channels,
                    growth_rate=self.growth_rate,
                    n_layers=self.n_block_layers,
                ),
            )

        if self.post_block_channels:
            self.global_feature_fuse = self.convolution_block(
                in_channels=self.pre_block_channels
                + self.n_blocks * self.local_feature_fusion_channels,
                out_channels=self.post_block_channels,
                kernel_size=(1, 1),
            )

            self.post_blocks = nn.Sequential(
                self.convolution_block(
                    in_channels=self.post_block_channels,
                    out_channels=self.post_block_channels,
                    kernel_size=(3, 3),
                    padding="same",
                ),
                nn.Conv2d(
                    in_channels=self.post_block_channels,
                    out_channels=self.out_channels,
                    kernel_size=(3, 3),
                    padding="same",
                ),
                # no activation function here, i.e. linear activation
            )
        else:
            self.global_feature_fuse = self.convolution_block(
                in_channels=self.pre_block_channels
                + self.n_blocks * self.local_feature_fusion_channels,
                out_channels=self.out_channels,
                kernel_size=(1, 1),
            )

            self.post_blocks = nn.Identity()

    def forward(self, x):
        x_out = self.pre_blocks(x)

        block_outputs = [x_out]
        for i_block in range(self.n_blocks):
            residual_dense_block = self.get_submodule(f"residual_dense_block_{i_block}")

            x_out = residual_dense_block(x_out)
            block_outputs.append(x_out)

        x_out = self.global_feature_fuse(torch.cat(block_outputs, dim=1))

        x_out = self.post_blocks(x_out)

        return x_out

    @property
    def config(self):
        return dict(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            growth_rate=self.growth_rate,
            n_blocks=self.n_blocks,
            n_block_layers=self.n_block_layers,
            convolution_block=self.convolution_block.__class__.__name__,
            local_feature_fusion_channels=self.local_feature_fusion_channels,
            pre_block_channels=self.pre_block_channels,
            post_block_channels=self.post_block_channels,
        )


class MCSpeedUpNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        growth_rate: int = 16,
        n_blocks: int = 2,
        n_block_layers: int = 4,
        convolution_block: Type[_ConvNormActivation] = ConvInstanceNormMish2D,
        local_feature_fusion_channels: int = 32,
        pre_block_channels: int | None = 32,
        post_block_channels: int | None = 32,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.growth_rate = growth_rate
        self.n_blocks = n_blocks
        self.n_block_layers = n_block_layers
        self.convolution_block = convolution_block
        self.local_feature_fusion_channels = local_feature_fusion_channels

        self.pre_block_channels = pre_block_channels
        self.post_block_channels = post_block_channels

        self.variance_scale = nn.Parameter(0.0025 * torch.ones(1))
        self.variance_offset = nn.Parameter(torch.zeros(1))

        self.pre_blocks = nn.Sequential(
            self.convolution_block(
                in_channels=self.in_channels,
                out_channels=self.pre_block_channels,
                kernel_size=(3, 3),
                padding="same",
            ),
            self.convolution_block(
                in_channels=self.pre_block_channels,
                out_channels=self.pre_block_channels,
                kernel_size=(3, 3),
                padding="same",
            ),
        )

        for i_block in range(self.n_blocks):
            self.add_module(
                f"residual_dense_block_{i_block}",
                ResidualDenseBlock2D(
                    in_channels=self.pre_block_channels
                    if i_block == 0
                    else self.local_feature_fusion_channels,
                    out_channels=self.local_feature_fusion_channels,
                    growth_rate=self.growth_rate,
                    n_layers=self.n_block_layers,
                ),
            )

        if self.post_block_channels:
            self.global_feature_fuse = self.convolution_block(
                in_channels=self.pre_block_channels
                + self.n_blocks * self.local_feature_fusion_channels,
                out_channels=self.post_block_channels,
                kernel_size=(1, 1),
            )

            self.post_blocks = nn.Sequential(
                self.convolution_block(
                    in_channels=self.post_block_channels,
                    out_channels=self.post_block_channels,
                    kernel_size=(3, 3),
                    padding="same",
                ),
                nn.Conv2d(
                    in_channels=self.post_block_channels,
                    out_channels=self.out_channels,
                    kernel_size=(3, 3),
                    padding_mode="replicate",
                    padding="same",
                ),
                # no activation function here, i.e. linear activation
            )
        else:
            self.global_feature_fuse = self.convolution_block(
                in_channels=self.pre_block_channels
                + self.n_blocks * self.local_feature_fusion_channels,
                out_channels=self.out_channels,
                kernel_size=(1, 1),
                activation=None,
            )

            self.post_blocks = nn.Identity()

    def forward(self, x):
        x_out = self.pre_blocks(x)

        block_outputs = [x_out]
        for i_block in range(self.n_blocks):
            residual_dense_block = self.get_submodule(f"residual_dense_block_{i_block}")

            x_out = residual_dense_block(x_out)
            block_outputs.append(x_out)

        x_out = self.global_feature_fuse(torch.cat(block_outputs, dim=1))

        x_out = self.post_blocks(x_out)

        return x_out

    def train_mean(self, mode: bool = True):
        for param in list(self.parameters())[2:]:
            param.requires_grad = mode

    def train_variance(self, mode: bool = True):
        for param in list(self.parameters())[:2]:
            param.requires_grad = mode

    @property
    def config(self):
        return dict(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            growth_rate=self.growth_rate,
            n_blocks=self.n_blocks,
            n_block_layers=self.n_block_layers,
            convolution_block=self.convolution_block.__class__.__name__,
            local_feature_fusion_channels=self.local_feature_fusion_channels,
            pre_block_channels=self.pre_block_channels,
            post_block_channels=self.post_block_channels,
        )


class MCSpeedUpNetSeparated(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        growth_rate: int = 16,
        n_blocks: int = 2,
        n_block_layers: int = 4,
        convolution_block: Type[_ConvNormActivation] = ConvInstanceNormMish2D,
        local_feature_fusion_channels: int = 32,
        pre_block_channels: int | None = 32,
        post_block_channels: int | None = 32,
        mode: str = "unet",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.growth_rate = growth_rate
        self.n_blocks = n_blocks
        self.n_block_layers = n_block_layers
        self.convolution_block = convolution_block
        self.local_feature_fusion_channels = local_feature_fusion_channels

        self.pre_block_channels = pre_block_channels
        self.post_block_channels = post_block_channels

        self.var_scale = nn.Parameter(0.001 * torch.ones(1), requires_grad=True)

        if mode == "unet":
            self.mean_net = FlexUNet(
                n_channels=in_channels,
                n_classes=1,
                n_levels=4,
                filter_base=32,
                n_filters=None,
                convolution_layer=nn.Conv2d,
                downsampling_layer=nn.MaxPool2d,
                upsampling_layer=nn.Upsample,
                norm_layer=nn.InstanceNorm2d,
                skip_connections=True,
                convolution_kwargs={
                    "kernel_size": 3,
                    "padding": "same",
                    "bias": True,
                    "padding_mode": "replicate",
                },
                downsampling_kwargs=None,
                upsampling_kwargs=None,
                return_bottleneck=False,
            )
            self.var_net = FlexUNet(
                n_channels=in_channels,
                n_classes=1,
                n_levels=4,
                filter_base=32,
                n_filters=None,
                convolution_layer=nn.Conv2d,
                downsampling_layer=nn.MaxPool2d,
                upsampling_layer=nn.Upsample,
                norm_layer=nn.InstanceNorm2d,
                skip_connections=True,
                convolution_kwargs={
                    "kernel_size": 3,
                    "padding": "same",
                    "bias": True,
                    "padding_mode": "replicate",
                },
                downsampling_kwargs=None,
                upsampling_kwargs=None,
                return_bottleneck=False,
            )

        else:
            self.mean_net = MCSpeedUpNet(
                in_channels=in_channels,
                out_channels=out_channels // 2,
                growth_rate=growth_rate,
                n_blocks=n_blocks,
                n_block_layers=n_block_layers,
                convolution_block=convolution_block,
                local_feature_fusion_channels=local_feature_fusion_channels,
                pre_block_channels=pre_block_channels,
                post_block_channels=post_block_channels,
            )
            self.var_net = MCSpeedUpNet(
                in_channels=in_channels,
                out_channels=out_channels // 2,
                growth_rate=growth_rate,
                n_blocks=n_blocks,
                n_block_layers=n_block_layers,
                convolution_block=convolution_block,
                local_feature_fusion_channels=local_feature_fusion_channels,
                pre_block_channels=pre_block_channels,
                post_block_channels=post_block_channels,
            )

    def freeze_mean_net(self, freeze: bool = True):
        for param in self.mean_net.parameters():
            param.requires_grad = not freeze

    def freeze_var_net(self, freeze: bool = True):
        for param in self.var_net.parameters():
            param.requires_grad = not freeze

    # def forward(self, x):
    #     mean = self.mean_net(x)
    #     var = self.var_net(x)
    #
    #     return torch.cat((mean, var), dim=1)

    def forward(self, x):
        # x = (low_photon, forward_projection)
        mean_factor = 10.0

        mean_residual = self.mean_net(x)

        # range (-mean_factor, +mean_factor)
        mean_residual = mean_factor * torch.tanh(mean_residual)
        mean = torch.relu(x[:, 0:1] + mean_residual)

        variance = mean * self.var_scale + 1e-6

        return torch.cat((mean, variance), dim=1)


class MCSpeedUpUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.var_scale = nn.Parameter(0.001 * torch.ones(1))

        self.mean_net = FlexUNet(
            n_channels=in_channels,
            n_classes=1,
            n_levels=4,
            filter_base=64,
            n_filters=None,
            convolution_layer=nn.Conv2d,
            downsampling_layer=nn.MaxPool2d,
            upsampling_layer=nn.Upsample,
            norm_layer=nn.InstanceNorm2d,
            skip_connections=True,
            convolution_kwargs={
                "kernel_size": 3,
                "padding": "same",
                "bias": True,
                "padding_mode": "replicate",
            },
            downsampling_kwargs=None,
            upsampling_kwargs=None,
            return_bottleneck=False,
        )

        self.var_net = FlexUNet(
            n_channels=1,
            n_classes=1,
            n_levels=2,
            filter_base=16,
            n_filters=None,
            convolution_layer=nn.Conv2d,
            downsampling_layer=nn.MaxPool2d,
            upsampling_layer=nn.Upsample,
            norm_layer=nn.InstanceNorm2d,
            skip_connections=True,
            convolution_kwargs={
                "kernel_size": 3,
                "padding": "same",
                "bias": True,
                "padding_mode": "replicate",
            },
            downsampling_kwargs=None,
            upsampling_kwargs=None,
            return_bottleneck=False,
        )

    def freeze_mean_net(self, freeze: bool = True):
        for param in self.mean_net.parameters():
            param.requires_grad = not freeze

    def freeze_var_net(self, freeze: bool = True):
        pass

    def forward(self, x):
        # x = (low_photon, forward_projection)
        mean_factor = 10.0

        mean_residual = self.mean_net(x)

        # range (-mean_factor, +mean_factor)
        mean_residual = mean_factor * torch.tanh(mean_residual)
        mean = torch.relu(x[:, 0:1] + mean_residual)

        # variance = mean * self.var_scale + 1e-6

        var_scale = torch.sigmoid(self.var_net(mean)) * 0.10
        print("mean var_scale: ", var_scale.mean().item())
        variance = mean * var_scale + 1e-6

        return torch.cat((mean, variance), dim=1)


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


class DenseNet(nn.Module):
    _LAYERS = {
        2: {
            "conv": nn.Conv2d,
            "residual_dense_block": blocks.ResidualDenseBlock2D,
            # "convolution_block": blocks.ConvInstanceNormReLU2D,
            "convolution_block": blocks.ConvInstanceNormMish2D,
        },
        3: {
            "conv": nn.Conv3d,
            "residual_dense_block": blocks.ResidualDenseBlock3D,
            # "convolution_block": blocks.ConvInstanceNormReLU3D,
            "convolution_block": blocks.ConvInstanceNormMish3D,
        },
    }

    def __init__(
        self,
        n_dims: Literal[2, 3] = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        growth_rate: int = 32,
        n_blocks: int = 2,
        n_block_layers: int = 4,
        local_feature_fusion_channels: int = 32,
        pre_block_channels: int = 32,
        post_block_channels: int = 32,
    ):
        super().__init__()

        self.n_dims = n_dims

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.growth_rate = growth_rate
        self.n_blocks = n_blocks
        self.n_block_layers = n_block_layers

        self.local_feature_fusion_channels = local_feature_fusion_channels

        self.pre_block_channels = pre_block_channels
        self.post_block_channels = post_block_channels

        # select the rights layers
        self.residual_dense_block = DenseNet._LAYERS[self.n_dims][
            "residual_dense_block"
        ]
        self.convolution_block = DenseNet._LAYERS[self.n_dims]["convolution_block"]
        self.conv_layer = DenseNet._LAYERS[self.n_dims]["conv"]

        self.pre_blocks = nn.Sequential(
            self.convolution_block(
                in_channels=self.in_channels,
                out_channels=self.pre_block_channels,
                kernel_size=(3,) * self.n_dims,
                padding="same",
            ),
            self.convolution_block(
                in_channels=self.pre_block_channels,
                out_channels=self.pre_block_channels,
                kernel_size=(3,) * self.n_dims,
                padding="same",
            ),
        )

        for i_block in range(self.n_blocks):
            self.add_module(
                f"residual_dense_block_{i_block}",
                self.residual_dense_block(
                    in_channels=self.pre_block_channels
                    if i_block == 0
                    else self.local_feature_fusion_channels,
                    out_channels=self.local_feature_fusion_channels,
                    growth_rate=self.growth_rate,
                    n_layers=self.n_block_layers,
                    convolution_block=self.convolution_block,
                ),
            )

        self.global_feature_fuse = self.convolution_block(
            in_channels=self.pre_block_channels
            + self.n_blocks * self.local_feature_fusion_channels,
            out_channels=self.post_block_channels,
            kernel_size=(1,) * self.n_dims,
        )

        self.post_blocks = nn.Sequential(
            self.convolution_block(
                in_channels=self.post_block_channels,
                out_channels=self.post_block_channels,
                kernel_size=(3,) * self.n_dims,
                padding="same",
            ),
            self.conv_layer(
                in_channels=self.post_block_channels,
                out_channels=self.out_channels,
                kernel_size=(3,) * self.n_dims,
                padding="same",
            ),
            # no activation function here, i.e. linear activation
        )

    def forward(self, x):
        x_out = self.pre_blocks(x)

        block_outputs = [x_out]
        for i_block in range(self.n_blocks):
            residual_dense_block = self.get_submodule(f"residual_dense_block_{i_block}")

            x_out = residual_dense_block(x_out)
            block_outputs.append(x_out)

        x_out = self.global_feature_fuse(torch.cat(block_outputs, dim=1))

        x_out = self.post_blocks(x_out)

        return x_out


if __name__ == "__main__":
    model = FlexUNet(
        n_channels=2,
        n_classes=1,
        n_levels=4,
        filter_base=64,
        n_filters=None,
        convolution_layer=nn.Conv2d,
        downsampling_layer=nn.MaxPool2d,
        upsampling_layer=nn.Upsample,
        norm_layer=nn.InstanceNorm2d,
        skip_connections=True,
        convolution_kwargs={
            "kernel_size": 3,
            "padding": "same",
            "bias": True,
            "padding_mode": "replicate",
        },
        downsampling_kwargs=None,
        upsampling_kwargs=None,
        return_bottleneck=True,
    )

    prediction, bottleneck = model(torch.ones((1, 2, 1024, 768)))
