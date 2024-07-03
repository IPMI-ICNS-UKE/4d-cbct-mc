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
from vroc.keypoint.models import CenterOfMass3d
from vroc.logger import LoggerMixin
from vroc.loss import TRELoss
from vroc.models import FlexUNet


class KeypointMatcherLoss(nn.Module):
    def __init__(self, image_spacing: FloatTuple3D = (1.0, 1.0, 1.0)):
        super().__init__()

        self.tre_loss = TRELoss(apply_sqrt=True)
        self.image_spacing = image_spacing

    def forward(
        self, predicted_keypoints: torch.Tensor, reference_keypoints: torch.Tensor
    ) -> torch.Tensor:
        # calcualte TRE
        tre = self.tre_loss(
            vector_field=None,
            fixed_landmarks=reference_keypoints,
            moving_landmarks=predicted_keypoints,
            image_spacing=self.image_spacing,
        )

        return tre
