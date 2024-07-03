from enum import Enum, auto
from typing import Sequence

import numpy as np
import scipy.stats as stats

from vroc.helper import rescale_range


class EarlyStopping(Enum):
    NO_IMPROVEMENT = auto()
    LEAST_SQUARES = auto()


def check_early_stopping(metrics: Sequence[float], i_level):

    rel_change = (metrics[-early_stopping_window[i_level]] - metrics[-1]) / (
        metrics[-early_stopping_window[i_level]] + 1e-9
    )

    if rel_change < early_stopping_delta[i_level]:
        return True
    return False


def _check_early_stopping_average_improvement(metrics: Sequence[float], i_level):

    window = np.array(metrics[-early_stopping_window[i_level] :])
    window_rel_changes = 1 - window[1:] / window[:-1]

    if window_rel_changes.mean() < early_stopping_delta[i_level]:
        return True
    return False


def _check_early_stopping_increase_count(metrics: Sequence[float], i_level):
    window = np.array(metrics)
    if np.argmin(window) == (len(window) - 1):
        _counter = 0
    else:
        _counter += 1
    if _counter == early_stopping_delta[i_level]:
        return True
    return False


def _check_early_stopping_lstsq(metrics: Sequence[float], i_level):

    window = np.array(metrics[-early_stopping_window[i_level] :])
    scaled_window = rescale_range(window, (np.min(metrics), np.max(metrics)), (0, 1))
    lstsq_result = stats.linregress(
        np.arange(early_stopping_window[i_level]), scaled_window
    )
    if lstsq_result.slope > -early_stopping_delta[i_level]:
        return True
    return False
