from dataclasses import dataclass
from math import ceil, pi
from typing import List, Tuple

import numpy as np

from cbctmc.peaks import find_peaks


@dataclass
class RespiratoryStatistics:
    mean_cycle_period: float
    median_cycle_period: float
    std_cycle_period: float
    n_complete_cycles: float
    mean_cycle_span: float
    std_cycle_span: float
    total_length_secs: float


def split_into_cycles(curve: np.ndarray, peaks: np.ndarray = None) -> List[np.ndarray]:
    if peaks is None:
        peaks = find_peaks(curve)

    # discard potentially incomplete first and last cycle
    slicing = slice(None)
    if peaks[0] == 0:
        slicing = slice(1, None)
    if peaks[-1] == len(curve) - 1:
        slicing = slice(slicing.start, -1)

    return np.split(curve, peaks[slicing])


def align_cycles(cycles: List[np.ndarray]) -> np.ndarray:
    minimum_indices = [np.argmin(c) for c in cycles]

    lefts = [c[:min_idx] for (c, min_idx) in zip(cycles, minimum_indices)]
    rights = [c[min_idx:] for (c, min_idx) in zip(cycles, minimum_indices)]

    max_left_length = max(len(left) for left in lefts)
    max_right_length = max(len(right) for right in rights)

    for i, (left, right) in enumerate(zip(lefts, rights)):
        lefts[i] = np.pad(
            left,
            (max_left_length - len(left), 0),
            mode="constant",
            constant_values=np.nan,
        )
        rights[i] = np.pad(
            right,
            (0, max_right_length - len(right)),
            mode="constant",
            constant_values=np.nan,
        )

    return np.hstack((lefts, rights))


def calculate_median_cycle_length(curve: np.ndarray) -> int:
    cycles = split_into_cycles(curve)
    return int(np.median([len(c) for c in cycles]))


def calculate_respiratory_statistics(
    amplitudes: np.ndarray, sampling_rate: float = 1.0
) -> RespiratoryStatistics:
    cycles = split_into_cycles(amplitudes)
    cycle_lengths = [len(c) / sampling_rate for c in cycles]
    cycle_spans = [max(c) - min(c) for c in cycles]
    return RespiratoryStatistics(
        mean_cycle_period=float(np.mean(cycle_lengths)),
        median_cycle_period=float(np.median(cycle_lengths)),
        std_cycle_period=float(np.std(cycle_lengths)),
        n_complete_cycles=len(cycle_lengths),
        mean_cycle_span=float(np.mean(cycle_spans)),
        std_cycle_span=float(np.std(cycle_spans)),
        total_length_secs=float(np.sum(cycle_lengths)),
    )


def calculate_median_cycle(curve: np.ndarray) -> np.ndarray:
    cycles = split_into_cycles(curve)
    resp_stats = calculate_respiratory_statistics(curve)

    cycles = [
        c
        for c in cycles
        if resp_stats.median_cycle_period - resp_stats.std_cycle_period
        <= len(c)
        <= resp_stats.median_cycle_period + resp_stats.std_cycle_period
    ]

    # stretch each cycle to median cycle length
    cycles = [
        np.interp(
            x=np.linspace(
                0, len(c) - 1, int(resp_stats.median_cycle_period), endpoint=True
            ),
            xp=np.arange(len(c)),
            fp=c,
        )
        for c in cycles
    ]

    return np.median(cycles, axis=0)


def calculate_amplitude_bins(
    breathing_curve: np.ndarray, n_bins: int = 10
) -> np.ndarray:
    median_cycle = calculate_median_cycle(breathing_curve)
    min_amplitude, max_amplitude = median_cycle.min(), median_cycle.max()

    edges = np.linspace(min_amplitude, max_amplitude, num=n_bins + 1, endpoint=True)

    bins = np.digitize(breathing_curve, edges) - 1

    return bins


def calculate_phase(
    breathing_curve: np.ndarray, phase_range: Tuple[float, float] = (0, 2 * pi)
) -> List[np.ndarray]:
    peaks = list(find_peaks(breathing_curve))

    # skip peaks at start/end of breathing curve since they are not reliable
    if peaks[0] == 0:
        peaks = peaks[1:]
    elif peaks[-1] == len(breathing_curve) - 1:
        peaks = peaks[:-1]

    phase = np.zeros_like(breathing_curve, dtype=np.float32) * np.nan

    for left_peak, right_peak in zip(peaks[:-1], peaks[1:]):
        n_timesteps = right_peak - left_peak
        phase[left_peak:right_peak] = np.linspace(
            phase_range[0], phase_range[1], num=n_timesteps
        )

    # fill incomplete cycles at start/end with phase of median cycle
    median_cycle = calculate_median_cycle(breathing_curve)

    median_cycle_phase = np.linspace(
        phase_range[0], phase_range[1], num=len(median_cycle)
    )
    len_start_part = peaks[0]
    len_end_part = len(breathing_curve) - peaks[-1]

    # if start/end part is longer than median cycle: repeat median cycle
    n_repeats = ceil(max(len_start_part, len_end_part) / len(median_cycle))
    median_cycle_phase = np.tile(median_cycle_phase, reps=n_repeats)

    # do the filling
    phase[:len_start_part] = median_cycle_phase[-len_start_part:]
    phase[-len_end_part:] = median_cycle_phase[:len_end_part]

    return np.split(phase, peaks)


def calculate_pseudo_average_phase(
    breathing_curve: np.ndarray,
    phase_range: Tuple[float, float] = (0, 2 * pi),
    n_bins: int = 10,
) -> List[np.ndarray]:
    phase = calculate_phase(breathing_curve, phase_range=phase_range)

    min_phase, max_phase = phase_range[0], phase_range[1]
    abs_phase_range = max_phase - min_phase

    pseudo_average_phase = []
    for i_cycle, cycle_phase in enumerate(phase):
        shift = (abs_phase_range / n_bins) * (i_cycle % n_bins)

        shifted_cycle_phase = (cycle_phase - shift) % max_phase
        pseudo_average_phase.append(shifted_cycle_phase)

    return pseudo_average_phase


def calculate_phase_bins(breathing_curve: np.ndarray, n_bins: int = 10) -> np.ndarray:
    phase = calculate_phase(breathing_curve)
    edges = np.linspace(0, 2 * pi, num=n_bins + 1, endpoint=True) - PI2 / (2 * n_bins)
    edges[edges < 0.0] = 0.0
    bins = []
    for cycle_phase in phase:
        bins.append(np.digitize(cycle_phase, edges) - 1)

    bins = np.hstack(bins)
    bins[bins == n_bins] = 0

    return bins


def calculate_amplitude_bins_as_phase_bins(
    breathing_curve: np.ndarray, n_bins: int = 10
) -> np.ndarray:
    if n_bins % 2 != 0:
        raise ValueError("n_bins has to be multiple of 2")

    amplitude_bins = calculate_amplitude_bins(breathing_curve, n_bins=n_bins // 2)
    phase_bins = calculate_phase_bins(breathing_curve, n_bins=n_bins)

    peaks = find_peaks(breathing_curve)
    _amplitude = split_into_cycles(breathing_curve, peaks=peaks)
    _amplitude_bins = split_into_cycles(amplitude_bins, peaks=peaks)
    _phase_bins = split_into_cycles(phase_bins, peaks=peaks)

    b = []
    for a, ab, pb in zip(_amplitude, _amplitude_bins, _phase_bins):
        min_idx = np.argmin(a)
        exhale_ab = -(ab[:min_idx] - 5)
        inhale_ab = ab[min_idx:] + 5
        transformed = np.hstack((exhale_ab, inhale_ab))
        b.append(transformed)

    b = np.hstack(b)
    b = np.clip(b, 0, 9)

    return b
