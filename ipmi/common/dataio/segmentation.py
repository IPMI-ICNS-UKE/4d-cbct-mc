from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import SimpleITK as sitk

from ipmi.common.sorting import natural_sort_key


def merge_segmentations(
    segmentation_filepaths: Sequence[Path], sort: bool = True, multi_label: bool = False
) -> Tuple[sitk.Image, dict]:
    """Merges multiple segmentations read from `segmentation_filepaths`. Order
    determines the class label (returned as `classes`)

    :param segmentation_filepaths:
    :type segmentation_filepaths:
    :param sort:
    :type sort:
    :param multi_label:
    :type multi_label:
    :return:
    :rtype:
    """
    if sort:
        segmentation_filepaths = sorted(segmentation_filepaths, key=natural_sort_key)

    segmentations = None
    spacing, origin, direction = None, None, None
    classes = {}
    for i, segmentation_filepath in enumerate(segmentation_filepaths, start=1):
        print(segmentation_filepath)
        segmentation = sitk.ReadImage(str(segmentation_filepath), sitk.sitkUInt8)
        if segmentations is None:
            spatial_shape = segmentation.GetSize()[::-1]
            spacing = segmentation.GetSpacing()
            origin = segmentation.GetOrigin()
            direction = segmentation.GetDirection()

            segmentations = np.zeros(
                (*spatial_shape, len(segmentation_filepaths) + 1),
                dtype=np.uint8,
            )
        segmentation = sitk.GetArrayFromImage(segmentation)

        segmentations[..., i] = segmentation
        classes[i] = segmentation_filepath.name

    # add backrgound class to index 0
    is_background = np.logical_not(segmentations.any(axis=-1))
    segmentations[is_background, 0] = 1

    if not multi_label:
        segmentations = segmentations.argmax(axis=-1).astype(np.uint8)

    segmentations = sitk.GetImageFromArray(segmentations)
    segmentations.SetSpacing(spacing)
    segmentations.SetOrigin(origin)
    segmentations.SetDirection(direction)

    return segmentations, classes


if __name__ == "__main__":
    import json

    from tqdm import tqdm

    segmentation_folders = sorted(
        list(Path("/datalake/totalsegmentator").glob("*/segmentations"))
    )
    for segmentation_folder in tqdm(segmentation_folders):
        segmentation_filepaths = segmentation_folder.glob("*.nii.gz")
        merged, classes = merge_segmentations(segmentation_filepaths)

        sitk.WriteImage(
            merged, str(segmentation_folder.parent / "merged_segmentations.nii.gz")
        )

        with open("/datalake/totalsegmentator/classes.json", "wt") as f:
            json.dump(classes, f, indent=4)
