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


OUTPUT_FOLDER = Path(
    "/datalake2/mc_material_segmentation_dataset_inhouse_noise+shift_256_256_64"
)
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

PHASES = [0, 1, 2, 3, 4]

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

# IMAGE_FILEPATHS_INHOUSE_TRAIN = sorted(
#     ROOT_DIR_INHPUSE / f"{patient_id}_4DCT_Lunge_amplitudebased_complete/phase_00.nii"
#     for patient_id in train_patients
# )
# IMAGE_FILEPATHS_INHOUSE_TEST = sorted(
#     ROOT_DIR_INHPUSE / f"{patient_id}_4DCT_Lunge_amplitudebased_complete/phase_00.nii"
#     for patient_id in test_patients
# )


IMAGE_FILEPATHS = []
SEGMENTATION_FILEPATHS = []

# IMAGE_FILEPATHS += IMAGE_FILEPATHS_LUNA16
# SEGMENTATION_FILEPATHS += SEGMENTATION_FILEPATHS_LUNA16

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
    patch_shape=(256, 256, 64),
    image_spacing_range=((1.0, 1.0), (1.0, 1.0), (1.0, 1.0)),
    patches_per_image=32,
    force_non_background=True,
    force_balanced_sampling=True,
    random_rotation=False,
    add_noise=True,
    shift_image_values=True,
    input_value_range=(-1024, 3071),
    output_value_range=(0, 1),
)
test_dataset = SegmentationDataset(
    image_filepaths=test_image_filepaths,
    segmentation_filepaths=test_segmentation_filepaths,
    segmentation_merge_function=SegmentationDataset.merge_mc_segmentations,
    patch_shape=(256, 256, 64),
    image_spacing_range=((1.0, 1.0), (1.0, 1.0), (1.0, 1.0)),
    patches_per_image=32,
    force_non_background=True,
    force_balanced_sampling=True,
    random_rotation=False,
    add_noise=False,
    shift_image_values=False,
    input_value_range=(-1024, 3071),
    output_value_range=(0, 1),
)

train_dataset.compile_and_save(OUTPUT_FOLDER / "train")
test_dataset.compile_and_save(OUTPUT_FOLDER / "test")
