import logging
from typing import List, Tuple

import h5py as h5
import numpy as np

from ipmi.fused_types import PathLike

logger = logging.getLogger(__name__)


def read_curve(image_params_filepath: PathLike) -> Tuple[np.ndarray, np.ndarray]:
    amplitudes = []
    phases = []
    with h5.File(image_params_filepath, "r") as f:
        for p in range(len(f["ImageParameters"])):
            projection_number = str(p).zfill(5)
            current_amplitude = f["ImageParameters"][projection_number].attrs.get(
                "GatingAmplitude", np.nan
            )
            current_phase = f["ImageParameters"][projection_number].attrs.get(
                "GatingPhase", np.nan
            )

            if current_amplitude is np.nan:
                logger.warning(f"Amplitude is NaN for projection {projection_number}")

            if current_phase is np.nan:
                logger.warning(f"Phase is NaN for projection {projection_number}")

            amplitudes.append(current_amplitude)
            phases.append(current_phase)

    return (
        np.array(amplitudes, dtype=np.float32).squeeze(),
        np.array(phases, dtype=np.float32).squeeze(),
    )


def get_nan_sections(curve: np.ndarray) -> List[Tuple[int, int]]:
    nan_idx = np.where(np.isnan(curve))[0]

    first = None
    indices = []
    for idx, difference in zip(nan_idx, np.diff(nan_idx)):
        if first is None:
            first = idx
        if difference > 1:
            last = idx
            indices.append((first, last))
            first = None
    if first and idx:
        indices.append((first, idx + 1))

    return indices


def interpolate_nan_phases(phase: np.ndarray) -> np.ndarray:
    phase = phase.copy()
    nan_sections = get_nan_sections(phase)

    for first, last in nan_sections:
        first_valid_idx = first - 1
        last_valid_idx = last + 1

        n_nans = last - first + 1

        interpolated = np.linspace(
            phase[first_valid_idx], phase[last_valid_idx], num=n_nans + 2
        )
        logger.info(f"Interpolated phase section ({first}, {last})")

        phase[first_valid_idx : last_valid_idx + 1] = interpolated

    return phase


def save_curve(
    curve: np.ndarray,
    filepath: PathLike,
    scaling_factor: float = 1.0,
    format: str = "%.4f",
):
    np.savetxt(filepath, curve * scaling_factor, fmt=format)
