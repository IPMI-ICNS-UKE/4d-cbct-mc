from typing import Sequence, Tuple

import numpy as np


def michelson_contrast(data: np.ndarray) -> float:
    """Calculate the Michelson contrast of the data.

    Ranges from 0 to 1.
    """
    data_min, data_max = data.min(), data.max()
    return (data_max - data_min) / (data_max + data_min)


def calculate_mtf(
    line_pair_spacings: Sequence[float],
    line_pair_maximums: Sequence[float],
    line_pair_minimums: Sequence[float],
    relative: bool = True,
) -> dict[float, float]:
    """Calculate the MTF from the line pair maximums and minimums.

    :param line_pair_spacings: Line pair spacings in lp/mm
    :param line_pair_maximums: (mean) maximum voxel values of the line pairs
    :param line_pair_minimums: (mean) minimum voxel values of the line pairs
    :return:
    """
    # sort input by line pair spacing descending
    line_pair_spacings, line_pair_maximums, line_pair_minimums = zip(
        *sorted(
            zip(line_pair_spacings, line_pair_maximums, line_pair_minimums),
            reverse=True,
        )
    )

    mtf = {}
    reference_contrast = None
    for spacing, maximum, minimum in zip(
        line_pair_spacings, line_pair_maximums, line_pair_minimums
    ):
        contrast = michelson_contrast(np.array([minimum, maximum]))
        if relative and reference_contrast is None:
            reference_contrast = contrast

        if relative:
            mtf[spacing] = contrast / reference_contrast
        else:
            mtf[spacing] = contrast

    return mtf


def extract_line_pair_profile(
    image: np.ndarray,
    bounding_box: Tuple[slice, slice, slice],
    average_axes: Sequence[int] = (1, 2),
) -> np.ndarray:
    patch = image[bounding_box]
    profile = patch.mean(axis=average_axes)

    return profile


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import SimpleITK as sitk

    line_pair_spacings = [2, 4, 6, 8]
    minimums = []
    maximums = []

    fig, ax = plt.subplots(1, len(line_pair_spacings))

    for i, line_pair_spacing in enumerate(line_pair_spacings):
        gap = line_pair_spacing // 2

        image = sitk.ReadImage(
            f"/datalake2/mc_test/lp_{gap}mm/reference/reconstructions/fdk3d_wpc_0.25mm.mha"
        )
        image_spacing = image.GetSpacing()
        if len(set(image_spacing)) > 1:
            raise ValueError(f"{image_spacing=} is not isotropic")

        image_spacing = image_spacing[0]

        image = sitk.GetArrayFromImage(image)
        image = np.swapaxes(image, 0, 2)

        pattern_length = int(0.50 * line_pair_spacing / image_spacing * 4)
        pattern_depth = int(20 / (2 * image_spacing))
        image_center = (np.array(image.shape) / 2).astype(int)

        bounding_box = np.index_exp[
            image_center[0]
            - pattern_length // 2 : image_center[0]
            + pattern_length // 2,
            image_center[1] - pattern_depth // 2 : image_center[1] + pattern_depth // 2,
            image_center[2] - pattern_depth // 2 : image_center[2] + pattern_depth // 2,
        ]

        profile = extract_line_pair_profile(
            image, bounding_box=bounding_box, average_axes=(1, 2)
        )
        ax[i].plot(profile)

        minimums.append(profile.min())
        maximums.append(profile.max())

    mtf = calculate_mtf(
        line_pair_spacings=line_pair_spacings,
        line_pair_maximums=maximums,
        line_pair_minimums=minimums,
    )
    fig, ax = plt.subplots()
    ax.plot(1 / np.array(list(mtf.keys())), mtf.values())
