import json
import logging
import re
from functools import partial, reduce
from operator import add
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from ipmi.common.logger import init_fancy_logging
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader

from cbctmc.segmentation.dataset import PickleDataset, SegmentationDataset
from cbctmc.segmentation.labels import LABELS
from cbctmc.segmentation.losses import DiceLoss
from cbctmc.segmentation.trainer import CTSegmentationTrainer
from cbctmc.speedup.models import FlexUNet
from cbctmc.utils import dict_collate

logging.getLogger("cbctmc").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

init_fancy_logging()

DEVICE = "cuda:0"


class SegmentationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice_loss_function = DiceLoss(include_background=True, reduction="none")

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input[:, 0:8] = torch.softmax(input[:, 0:8], dim=1)  # softmax group 1
        input[:, 8] = torch.sigmoid(input[:, 8])  # sigmoid for lung vessels

        return self.dice_loss_function(input=input, target=target)


train_filepaths = sorted(
    Path("/datalake2/mc_segmentation_dataset_fixed/train").glob("*")
)
test_filepaths = sorted(Path("/datalake2/mc_segmentation_dataset_fixed/test").glob("*"))

train_dataset = PickleDataset(filepaths=train_filepaths)
test_dataset = PickleDataset(filepaths=test_filepaths)


COLLATE_NOOP_KEYS = (
    "image_spacing",
    "full_image_shape",
    "i_patch",
    "n_patches",
    "patch_slicing",
    "labels",
)

train_data_loader = DataLoader(
    train_dataset,
    batch_size=4,
    collate_fn=partial(dict_collate, noop_keys=COLLATE_NOOP_KEYS),
    shuffle=True,
    pin_memory=True,
    num_workers=4,
    persistent_workers=True,
)
test_data_loader = DataLoader(
    test_dataset,
    batch_size=1,
    collate_fn=partial(dict_collate, noop_keys=COLLATE_NOOP_KEYS),
    shuffle=False,
    pin_memory=True,
    num_workers=4,
    persistent_workers=True,
)

model = FlexUNet(
    n_channels=1,
    n_classes=len(LABELS),
    n_levels=4,
    filter_base=16,
    n_filters=None,
    convolution_layer=nn.Conv3d,
    downsampling_layer=nn.MaxPool3d,
    upsampling_layer=nn.Upsample,
    norm_layer=nn.InstanceNorm3d,
    skip_connections=True,
    convolution_kwargs=None,
    downsampling_kwargs=None,
    upsampling_kwargs=None,
    return_bottleneck=False,
)

# loss_function = nn.BCEWithLogitsLoss(reduction="none")
# loss_function = DiceLoss(include_background=True, sigmoid=True, reduction="none")
loss_function = SegmentationLoss()

optimizer = Adam(params=model.parameters(), lr=1e-4)
trainer = CTSegmentationTrainer(
    model=model,
    loss_function=loss_function,
    optimizer=optimizer,
    train_loader=train_data_loader,
    val_loader=test_data_loader,
    run_folder="/datalake2/runs/mc_segmentation",
    experiment_name="mc_segmentation",
    device=DEVICE,
)

trainer.run(steps=100_000_000, validation_interval=1000, save_interval=5_000)
