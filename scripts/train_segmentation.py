import logging
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from ipmi.common.logger import init_fancy_logging
from sklearn.model_selection import train_test_split
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from cbctmc.segmentation.dataset import SegmentationDataset
from cbctmc.segmentation.labels import LABELS, LABELS_TO_LOAD
from cbctmc.segmentation.losses import DiceLoss
from cbctmc.segmentation.trainer import CTSegmentationTrainer
from cbctmc.speedup.models import FlexUNet
from cbctmc.utils import dict_collate

logging.getLogger("cbctmc").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

init_fancy_logging()


class SegmentationLoss(nn.Module):
    def __init__(self, use_dice: bool = True):
        super().__init__()
        self.use_dice = use_dice
        self.ce_loss = CrossEntropyLoss()
        self.bce_loss = BCEWithLogitsLoss()
        self.dice_loss_function = DiceLoss(include_background=True, reduction="none")

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.use_dice:
            input[:, 0:-1] = torch.softmax(input[:, 0:-1], dim=1)  # softmax group 1
            input[:, -1] = torch.sigmoid(input[:, -1])  # sigmoid for lung vessels
            loss = self.dice_loss_function(input=input, target=target)
        else:
            loss = 8 / 9 * self.ce_loss(
                input[:, 0:-1], target[:, 0:-1]
            ) + 1 / 9 * self.bce_loss(input[:, -1:], target[:, -1:])

        return loss


# train_filepaths = sorted(
#     Path("/datalake2/mc_segmentation_dataset_fixed/train").glob("*")
# )
# test_filepaths = sorted(
#     Path("/datalake2/mc_segmentation_dataset_fixed/test").glob("*")
# )
#
# train_dataset = PickleDataset(filepaths=train_filepaths)
# test_dataset = PickleDataset(filepaths=test_filepaths)


# compile filepaths
# LUNA16
ROOT_DIR_LUNA16 = Path("/datalake2/luna16/images_nii")
IMAGE_FILEPATHS_LUNA16 = sorted(p for p in ROOT_DIR_LUNA16.glob("*.nii"))
SEGMENTATION_FILEPATHS_LUNA16 = [
    {
        segmentation_name: ROOT_DIR_LUNA16
        / "predicted_segmentations"
        / image_filepath.with_suffix("").name
        / f"{segmentation_name}.nii.gz"
        for segmentation_name in LABELS_TO_LOAD
    }
    for image_filepath in IMAGE_FILEPATHS_LUNA16
]

# TOTALSEGMENTATOR
ROOT_DIR_TOTALSEGMENTATOR = Path("/datalake/totalsegmentator_mc")
IMAGE_FILEPATHS_TOTALSEGMENTATOR = sorted(
    p for p in ROOT_DIR_TOTALSEGMENTATOR.glob("*/ct.nii.gz")
)
SEGMENTATION_FILEPATHS_TOTALSEGMENTATOR = [
    {
        segmentation_name: ROOT_DIR_TOTALSEGMENTATOR
        / image_filepath.parent.name
        / "segmentations"
        / f"{segmentation_name}.nii.gz"
        for segmentation_name in LABELS_TO_LOAD
    }
    for image_filepath in IMAGE_FILEPATHS_TOTALSEGMENTATOR
]

IMAGE_FILEPATHS = []
SEGMENTATION_FILEPATHS = []

# IMAGE_FILEPATHS += IMAGE_FILEPATHS_LUNA16
# SEGMENTATION_FILEPATHS += SEGMENTATION_FILEPATHS_LUNA16

IMAGE_FILEPATHS += IMAGE_FILEPATHS_TOTALSEGMENTATOR
SEGMENTATION_FILEPATHS += SEGMENTATION_FILEPATHS_TOTALSEGMENTATOR

(
    train_image_filepaths,
    test_image_filepaths,
    train_segmentation_filepaths,
    test_segmentation_filepaths,
) = train_test_split(
    IMAGE_FILEPATHS, SEGMENTATION_FILEPATHS, train_size=0.90, random_state=1337
)


train_dataset = SegmentationDataset(
    image_filepaths=train_image_filepaths,
    segmentation_filepaths=train_segmentation_filepaths,
    segmentation_merge_function=SegmentationDataset.merge_mc_segmentations,
    patch_shape=(128, 128, 64),
    image_spacing_range=((1.0, 1.0), (1.0, 1.0), (1.0, 1.0)),
    patches_per_image=8 * 4,
    force_non_background=True,
    force_balanced_sampling=False,
    random_rotation=False,
    input_value_range=(-1024, 3071),
    output_value_range=(0, 1),
)
test_dataset = SegmentationDataset(
    image_filepaths=test_image_filepaths,
    segmentation_filepaths=test_segmentation_filepaths,
    segmentation_merge_function=SegmentationDataset.merge_mc_segmentations,
    patch_shape=(128, 128, 64),
    image_spacing_range=((1.0, 1.0), (1.0, 1.0), (1.0, 1.0)),
    patches_per_image=8 * 4,
    force_non_background=True,
    force_balanced_sampling=False,
    random_rotation=False,
    input_value_range=(-1024, 3071),
    output_value_range=(0, 1),
)


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
    batch_size=8,
    collate_fn=partial(dict_collate, noop_keys=COLLATE_NOOP_KEYS),
    # shuffle=True,
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

enc_filters = [32, 32, 32, 32]
dec_filters = [32, 32, 32, 32]

DEVICE = "cuda:0"
model = FlexUNet(
    n_channels=1,
    n_classes=len(LABELS),
    n_levels=4,
    # filter_base=4,
    n_filters=[32, *enc_filters, *dec_filters, 32],
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
state = torch.load(
    "/datalake2/runs/mc_segmentation/"
    "models_0961491db6c842c3958ffb1d/validation/step_72000.pth",
    map_location=DEVICE,
)
model.load_state_dict(state["model"])
# loss_function = nn.BCEWithLogitsLoss(reduction="none")
# loss_function = DiceLoss(include_background=True, sigmoid=True, reduction="none")
loss_function = SegmentationLoss(use_dice=True)
# loss_function = DiceLoss(include_background=True, softmax=True, reduction="none")


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

trainer.run(steps=100_000_000, validation_interval=1000, save_interval=1000)
