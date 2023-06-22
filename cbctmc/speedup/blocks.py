from typing import Optional, Tuple, Type, Union

import torch
import torch.functional as F
import torch.nn as nn


class _ConvNormActivation(nn.Module):
    def __init__(
        self,
        convolution,
        normalization,
        activation,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        padding="same",
    ):
        super().__init__()
        self.convolution = convolution(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        if normalization:
            self.normalization = normalization(out_channels)
        else:
            self.normalization = None
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolution(x)
        if self.normalization:
            x = self.normalization(x)
        x = self.activation(x)

        return x


class ConvInstanceNormReLU2D(_ConvNormActivation):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        padding="same",
    ):
        super().__init__(
            convolution=nn.Conv2d,
            normalization=nn.InstanceNorm2d,
            activation=nn.ReLU,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )


class ConvInstanceNormMish2D(_ConvNormActivation):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        padding="same",
    ):
        super().__init__(
            convolution=nn.Conv2d,
            normalization=nn.InstanceNorm2d,
            activation=nn.Mish,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )


class ConvReLU2D(_ConvNormActivation):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        padding="same",
    ):
        super().__init__(
            convolution=nn.Conv2d,
            normalization=None,
            activation=nn.ReLU,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )


class ResidualDenseBlock2D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels: int = 32,
        growth_rate: int = 16,
        n_layers: int = 4,
        convolution_block: nn.Module = ConvInstanceNormReLU2D,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.growth_rate = growth_rate
        self.n_layers = n_layers

        for i_layer in range(self.n_layers):
            conv = convolution_block(
                in_channels=in_channels,
                out_channels=self.growth_rate,
                kernel_size=(3, 3),
                padding="same",
            )
            name = f"conv_block_{i_layer}"

            self.add_module(name, conv)

            in_channels = (i_layer + 1) * self.growth_rate + self.in_channels

        self.local_feature_fusion = convolution_block(
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=(1, 1),
            padding="same",
        )

    def forward(self, x):
        outputs = []
        for i_layer in range(self.n_layers):
            layer = self.get_submodule(f"conv_block_{i_layer}")
            stacked = torch.cat((x, *outputs), dim=1)
            x_out = layer(stacked)
            outputs.append(x_out)

        stacked = torch.cat((x, *outputs), dim=1)
        x_out = self.local_feature_fusion(stacked)

        x_out = x + x_out

        return x_out


class EncoderBlock(nn.Module):
    def __init__(
        self,
        convolution_layer: Type[nn.Module],
        downsampling_layer: Type[nn.Module],
        norm_layer: Type[nn.Module],
        in_channels,
        out_channels,
        n_convolutions: int = 1,
        convolution_kwargs: Optional[dict] = None,
        downsampling_kwargs: Optional[dict] = None,
    ):
        super().__init__()

        if not convolution_kwargs:
            convolution_kwargs = {}
        if not downsampling_kwargs:
            downsampling_kwargs = {}

        self.down = downsampling_layer(**downsampling_kwargs)

        layers = []
        for i_conv in range(n_convolutions):
            layers.append(
                convolution_layer(
                    in_channels=in_channels if i_conv == 0 else out_channels,
                    out_channels=out_channels,
                    **convolution_kwargs,
                )
            )
            if norm_layer:
                layers.append(norm_layer(out_channels))
            layers.append(nn.LeakyReLU(inplace=True))
        self.convs = nn.Sequential(*layers)

    def forward(self, *inputs):
        x = self.down(*inputs)
        return self.convs(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        convolution_layer: Type[nn.Module],
        upsampling_layer: Type[nn.Module],
        norm_layer: Type[nn.Module],
        in_channels,
        out_channels,
        n_convolutions: int = 1,
        convolution_kwargs: Optional[dict] = None,
        upsampling_kwargs: Optional[dict] = None,
    ):
        super().__init__()

        if not convolution_kwargs:
            convolution_kwargs = {}
        if not upsampling_kwargs:
            upsampling_kwargs = {}

        self.up = upsampling_layer(**upsampling_kwargs)

        layers = []
        for i_conv in range(n_convolutions):
            layers.append(
                convolution_layer(
                    in_channels=in_channels if i_conv == 0 else out_channels,
                    out_channels=out_channels,
                    **convolution_kwargs,
                )
            )
            if norm_layer:
                layers.append(norm_layer(out_channels))
            layers.append(nn.LeakyReLU(inplace=True))
        self.convs = nn.Sequential(*layers)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if x2 is not None:
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        x = self.convs(x)
        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = None,
        dimensions: int = 2,
        norm_type: Optional[str] = "BatchNorm",
    ):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        conv = getattr(nn, f"Conv{dimensions}d")
        if norm_type:
            norm = getattr(nn, f"{norm_type}{dimensions}d")
            layers = [
                conv(in_channels, mid_channels, kernel_size=3, padding=1),
                norm(mid_channels),
                nn.ReLU(inplace=True),
                conv(mid_channels, out_channels, kernel_size=3, padding=1),
                norm(out_channels),
                nn.ReLU(inplace=True),
            ]
        else:
            layers = [
                conv(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                conv(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ]

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dimensions: int = 2,
        pooling: Union[int, Tuple[int, ...]] = 2,
        norm_type: Optional[str] = "BatchNorm",
    ):
        super().__init__()

        if dimensions == 1:
            pool = nn.MaxPool1d
        elif dimensions == 2:
            pool = nn.MaxPool2d
        elif dimensions == 3:
            pool = nn.MaxPool3d
        else:
            raise ValueError(f"Cannot handle {dimensions=}")

        self.maxpool_conv = nn.Sequential(
            pool(pooling),
            ConvBlock(
                in_channels, out_channels, dimensions=dimensions, norm_type=norm_type
            ),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = ConvBlock(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        if x2 is not None:
            diff_y = x2.size()[2] - x1.size()[2]
            diff_x = x2.size()[3] - x1.size()[3]

            x1 = F.pad(
                x1,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        return self.conv(x)
