import logging
import random
from pathlib import Path

import numpy as np
from ipmi.common.logger import init_fancy_logging
from sklearn.model_selection import train_test_split

from cbctmc.segmentation.dataset import SegmentationDataset
from cbctmc.segmentation.labels import LABELS_TO_LOAD

np.random.seed(1337)
random.seed(1337)

logging.getLogger("cbctmc").setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

init_fancy_logging()

DEVICE = "cuda:1"


OUTPUT_FOLDER = Path("/datalake2/mc_segmentation_dataset_128_128_32")
OUTPUT_FOLDER.mkdir(exist_ok=True)

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
    patch_shape=(128, 128, 128),
    image_spacing_range=((1.0, 1.0), (1.0, 1.0), (1.0, 1.0)),
    patches_per_image=1.0,
    force_non_background=True,
    force_balanced_sampling=True,
    random_rotation=True,
    input_value_range=(-1024, 3071),
    output_value_range=(0, 1),
)
test_dataset = SegmentationDataset(
    image_filepaths=test_image_filepaths,
    segmentation_filepaths=test_segmentation_filepaths,
    segmentation_merge_function=SegmentationDataset.merge_mc_segmentations,
    patch_shape=(128, 128, 128),
    image_spacing_range=((1.0, 1.0), (1.0, 1.0), (1.0, 1.0)),
    patches_per_image=1.0,
    force_non_background=True,
    force_balanced_sampling=True,
    random_rotation=True,
    input_value_range=(-1024, 3071),
    output_value_range=(0, 1),
)

# train_dataset.compile_and_save(OUTPUT_FOLDER / "train")
# test_dataset.compile_and_save(OUTPUT_FOLDER / "test")
