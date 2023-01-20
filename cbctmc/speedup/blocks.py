from typing import Union, Tuple

import torch
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
