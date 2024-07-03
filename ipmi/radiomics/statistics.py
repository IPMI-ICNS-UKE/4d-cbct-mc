from collections import namedtuple
from typing import List

import numpy as np


def get_youdens_j_cutoff(fpr, tpr, thresholds):
    operating_point = namedtuple("ROCOperatingPoint", ["fpr", "tpr"])
    j_scores = tpr - fpr
    youdens_index, optimal_threshold, selected_fpr, selected_tpr = sorted(
        zip(j_scores, thresholds, fpr, tpr)
    )[-1]
    return (
        youdens_index,
        optimal_threshold,
        operating_point(fpr=selected_fpr, tpr=selected_tpr),
    )


def calculate_mean_roc_auc(
    false_positive_rates: List[List[float]],
    true_positive_rates: List[List[float]],
    n_interpolation_steps: int = 100,
):
    interpolated_fpr = np.linspace(0, 1, n_interpolation_steps)
    interpolated_tprs = []
    for fpr, tpr in zip(false_positive_rates, true_positive_rates):
        interpolated_tpr = np.interp(interpolated_fpr, xp=fpr, fp=tpr)
        interpolated_tprs.append(interpolated_tpr)

    mean_interpolated_tpr = np.mean(interpolated_tprs, axis=0)
    std_interpolated_tpr = np.std(interpolated_tprs, axis=0)

    return interpolated_fpr, mean_interpolated_tpr, std_interpolated_tpr
