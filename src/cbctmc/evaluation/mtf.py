from typing import Sequence, Tuple

import numpy as np

from cbctmc.peaks import find_peaks


def michelson_contrast(data: np.ndarray) -> float:
    """Calculate the Michelson contrast of the data.

    Ranges from 0 to 1.
    """
    data_min, data_max = data.min(), data.max()
    if data_min == data_max:
        return 0.0
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
    # sort input by line pair spacing ascending
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
    min_peak_distance: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    patch = image[bounding_box]
    profile = patch.mean(axis=average_axes)

    # select profile from first to last peak
    maxs = find_peaks(profile)
    profile = profile[maxs[0] : maxs[-1] + 1]

    maxs = find_peaks(profile)
    mins = find_peaks(-profile)

    return profile, maxs, mins
