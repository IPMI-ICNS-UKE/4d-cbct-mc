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

from cbctmc.segmentation.dataset import PickleDataset, SegmentationDataset
from cbctmc.segmentation.labels import LABELS, LABELS_TO_LOAD
from cbctmc.segmentation.losses import DiceLoss
from cbctmc.segmentation.trainer import CTSegmentationTrainer
from cbctmc.speedup.models import FlexUNet
from cbctmc.utils import dict_collate

logging.getLogger("cbctmc").setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

init_fancy_logging()


class SegmentationLoss(nn.Module):
    def __init__(self, use_dice: bool = True):
        super().__init__()
        self.use_dice = use_dice
        self.ce_loss = CrossEntropyLoss(reduction="none")
        self.bce_loss = BCEWithLogitsLoss(reduction="none")
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
ROOT_DIR_TOTALSEGMENTATOR = Path("/datalake_fast/totalsegmentator_mc")
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

# INHOUSE
patient_ids = [
    22,
    24,
    32,
    33,
    68,
    69,
    74,
    78,
    91,
    92,
    104,
    106,
    109,
    115,
    116,
    121,
    124,
    132,
    142,
    145,
    146,
]
train_patients, test_patients = train_test_split(
    patient_ids, train_size=0.75, random_state=42
)
ROOT_DIR_INHOUSE = Path("/datalake_fast/4d_ct_lung_uke_artifact_free/")

PHASES = [0, 5]

IMAGE_FILEPATHS_INHOUSE_TRAIN = [
    ROOT_DIR_INHOUSE
    / f"{patient_id:03d}_4DCT_Lunge_amplitudebased_complete/phase_{i_phase:02d}.nii"
    for i_phase in PHASES
    for patient_id in sorted(train_patients)
]

IMAGE_FILEPATHS_INHOUSE_TEST = [
    ROOT_DIR_INHOUSE
    / f"{patient_id:03d}_4DCT_Lunge_amplitudebased_complete/phase_{i_phase:02d}.nii"
    for i_phase in PHASES
    for patient_id in sorted(test_patients)
]

SEGMENTATION_FILEPATHS_INHOUSE_TRAIN = [
    {
        segmentation_name: ROOT_DIR_INHOUSE
        / image_filepath.parent.name
        / "segmentations"
        / f"phase_{i_phase:02d}"
        / f"{segmentation_name}.nii.gz"
        for segmentation_name in LABELS_TO_LOAD
    }
    for i_phase in PHASES
    for image_filepath in IMAGE_FILEPATHS_INHOUSE_TRAIN
]
SEGMENTATION_FILEPATHS_INHOUSE_TEST = [
    {
        segmentation_name: ROOT_DIR_INHOUSE
        / image_filepath.parent.name
        / "segmentations"
        / f"phase_{i_phase:02d}"
        / f"{segmentation_name}.nii.gz"
        for segmentation_name in LABELS_TO_LOAD
    }
    for i_phase in PHASES
    for image_filepath in IMAGE_FILEPATHS_INHOUSE_TEST
]

# IMAGE_FILEPATHS = []
# SEGMENTATION_FILEPATHS = []
#
# # IMAGE_FILEPATHS += IMAGE_FILEPATHS_LUNA16
# # SEGMENTATION_FILEPATHS += SEGMENTATION_FILEPATHS_LUNA16
#
# IMAGE_FILEPATHS += IMAGE_FILEPATHS_TOTALSEGMENTATOR
# SEGMENTATION_FILEPATHS += SEGMENTATION_FILEPATHS_TOTALSEGMENTATOR
# (
#     train_image_filepaths,
#     test_image_filepaths,
#     train_segmentation_filepaths,
#     test_segmentation_filepaths,
# ) = train_test_split(
#     IMAGE_FILEPATHS, SEGMENTATION_FILEPATHS, train_size=0.90, random_state=1337
# )

train_image_filepaths = IMAGE_FILEPATHS_INHOUSE_TRAIN
test_image_filepaths = IMAGE_FILEPATHS_INHOUSE_TEST
train_segmentation_filepaths = SEGMENTATION_FILEPATHS_INHOUSE_TRAIN
test_segmentation_filepaths = SEGMENTATION_FILEPATHS_INHOUSE_TEST

train_dataset = SegmentationDataset(
    image_filepaths=train_image_filepaths,
    segmentation_filepaths=train_segmentation_filepaths,
    segmentation_merge_function=SegmentationDataset.merge_mc_segmentations,
    patch_shape=(384, 384, 64),
    image_spacing_range=((1.0, 1.0), (1.0, 1.0), (1.0, 1.0)),
    patches_per_image=16,
    force_non_background=True,
    force_balanced_sampling=True,
    random_rotation=True,
    add_noise=100.0,
    shift_image_values=(0.9, 1.1),
    input_value_range=(-1024, 3071),
    output_value_range=(0, 1),
)
test_dataset = SegmentationDataset(
    image_filepaths=test_image_filepaths,
    segmentation_filepaths=test_segmentation_filepaths,
    segmentation_merge_function=SegmentationDataset.merge_mc_segmentations,
    patch_shape=(384, 384, 64),
    image_spacing_range=((1.0, 1.0), (1.0, 1.0), (1.0, 1.0)),
    patches_per_image=16,
    force_non_background=True,
    force_balanced_sampling=True,
    random_rotation=False,
    add_noise=0.0,
    shift_image_values=None,
    input_value_range=(-1024, 3071),
    output_value_range=(0, 1),
)

# compiled_dataset_folder = Path(
#     "/datalake2/mc_material_segmentation_dataset_inhouse_full_aug_256_256_64"
# )
# train_dataset = PickleDataset(
#     filepaths=list((compiled_dataset_folder / "train").glob("*"))
# )
# test_dataset = PickleDataset(
#     filepaths=list((compiled_dataset_folder / "test").glob("*"))
# )


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
    batch_size=1,
    collate_fn=partial(dict_collate, noop_keys=COLLATE_NOOP_KEYS),
    pin_memory=True,
    # shuffle=True,
    # num_workers=4,
    # persistent_workers=True,
)
test_data_loader = DataLoader(
    test_dataset,
    batch_size=1,
    collate_fn=partial(dict_collate, noop_keys=COLLATE_NOOP_KEYS),
    pin_memory=True,
    # shuffle=True,
    # num_workers=4,
    # persistent_workers=True,
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

optimizer = Adam(params=model.parameters(), lr=1e-4)

state = torch.load(
    "/datalake2/runs/mc_material_segmentation_inhouse/2023-09-21T17:18:03.218908_run_39a7956b4719411f99ddf071/models/validation/step_95000.pth",
    map_location=DEVICE,
)
model.load_state_dict(state["model"])

# loss_function = nn.BCEWithLogitsLoss(reduction="none")
# loss_function = DiceLoss(include_background=True, sigmoid=True, reduction="none")

loss_function = SegmentationLoss(use_dice=True)
# loss_function = DiceLoss(include_background=True, softmax=True, reduction="none")


lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=0.9,
    patience=1000,
    threshold=1e-2,
    threshold_mode="rel",
    cooldown=1000,
    min_lr=1e-6,
)

trainer = CTSegmentationTrainer(
    model=model,
    loss_function=loss_function,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    train_loader=train_data_loader,
    val_loader=test_data_loader,
    run_folder="/datalake2/runs/mc_material_segmentation_inhouse_fine",
    experiment_name="mc_material_segmentation_inhouse_fine",
    device=DEVICE,
)

trainer.run(steps=100_000_000, validation_interval=1000, save_interval=1000)
