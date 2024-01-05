from __future__ import annotations

import pickle
from math import ceil
from typing import Sequence, Tuple

import numpy as np
from scipy.signal import savgol_filter

from cbctmc.utils import rescale_range


class RespiratorySignal:
    def __init__(
        self,
        signal: np.ndarray,
        dt_signal: float | None = None,
        sampling_frequency: float = 25.0,
    ):
        self.signal = signal
        self.sampling_frequency = sampling_frequency
        self.dt_signal = (
            dt_signal if dt_signal is not None else self._calculate_time_derivative()
        )
        self.time = np.linspace(0, self.total_seconds, len(self.signal), endpoint=False)

    def save(self, filepath: str):
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "signal": self.signal,
                    "dt_signal": self.dt_signal,
                    "sampling_frequency": self.sampling_frequency,
                },
                f,
            )

    @classmethod
    def load(cls, filepath: str) -> RespiratorySignal:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        return cls(**data)

    def resample(self, sampling_frequency: float) -> "RespiratorySignal":
        resampled_time = np.linspace(
            0, self.total_seconds, int(self.total_seconds * sampling_frequency)
        )
        signal = np.interp(resampled_time, self.time, self.signal)
        dt_signal = np.interp(resampled_time, self.time, self.dt_signal)
        sampling_frequency = sampling_frequency

        return RespiratorySignal(
            signal=signal, dt_signal=dt_signal, sampling_frequency=sampling_frequency
        )

    def _calculate_time_derivative(self):
        return np.gradient(self.signal, 1 / self.sampling_frequency)

    @property
    def total_seconds(self):
        return len(self.signal) / self.sampling_frequency

    @staticmethod
    def quantize_signal(signal: np.ndarray, n_bins: int = 20) -> np.ndarray:
        bins = np.linspace(signal.min(), signal.max(), n_bins + 1)
        signal_bins = np.digitize(signal, bins=bins)
        # add half a bin width to get center the bins
        bin_width = bins[1] - bins[0]
        return bins[signal_bins - 1] + 0.5 * bin_width

    def get_signal(self, n_bins: int = 2) -> np.ndarray:
        bins = np.linspace(self.signal.min(), self.signal.max(), n_bins + 1)
        signal_bins = np.digitize(self.signal, bins=bins)
        # add half a bin width to get center the bins
        bin_width = bins[1] - bins[0]
        return bins[signal_bins - 1] + 0.5 * bin_width

    @staticmethod
    def get_unique_signals(
        signal: np.ndarray, dt_signal: np.ndarray
    ) -> dict[tuple[float, float], list[int]]:
        samples = np.stack((signal, dt_signal), axis=-1)
        unique_samples = np.unique(samples, axis=0)

        unique_samples_dict = {}
        for unique_sample in unique_samples:
            # add unique sample to dict with all indices where it occurs
            unique_samples_dict[tuple(unique_sample)] = np.where(
                (samples == unique_sample).all(axis=1)
            )[0].tolist()

        return unique_samples_dict

    @classmethod
    def create_sin4(
        cls,
        total_seconds: float,
        period: float = 5.0,
        amplitude: float = 1.0,
        sampling_frequency: float = 25.0,
    ):
        # sin**4 has double the frequency of sin, thus we need to divide by 2
        frequency = 1 / (2 * period)
        t = np.linspace(0, total_seconds, int(total_seconds * sampling_frequency))
        signal = amplitude * np.sin(2 * np.pi * frequency * t) ** 4

        return cls(signal, sampling_frequency=sampling_frequency)

    @staticmethod
    def _repeat_signal(
        signal: np.ndarray, sampling_frequency: float, total_seconds: float
    ) -> np.ndarray:
        n_repeats = ceil(total_seconds * sampling_frequency / len(signal))
        n_target_samples = int(total_seconds * sampling_frequency)
        return np.tile(signal, n_repeats)[:n_target_samples]

    @classmethod
    def from_masks(
        cls,
        masks: Sequence[np.ndarray],
        timepoints: Sequence[float],
        target_total_seconds: float = 60.0,
        target_sampling_frequency: float = 25.0,
        smooth_window_seconds: float | None = None,
        smooth_order: int | None = 3,
        output_range: Tuple[float, float] = (-1, 1),
    ):
        mask_volumes = []
        for mask in masks:
            mask_volumes.append(np.sum(mask > 0))

        mask_volumes = np.array(mask_volumes)

        # timepoints may not be spaced regularly, thus interpolate to regular grid
        # with target_sampling_frequency
        timepoints = np.array(timepoints)
        timepoints_range = timepoints.max() - timepoints.min()
        regular_timepoints = np.linspace(
            timepoints.min(),
            timepoints.max(),
            int(timepoints_range * target_sampling_frequency),
        )
        mask_volumes = np.interp(regular_timepoints, timepoints, mask_volumes)

        signal = cls._repeat_signal(
            signal=mask_volumes,
            sampling_frequency=target_sampling_frequency,
            total_seconds=target_total_seconds,
        )
        if smooth_window_seconds != 0 and smooth_order is not None:
            if smooth_window_seconds is None:
                smooth_window_seconds = timepoints_range

            window_length = int(smooth_window_seconds * target_sampling_frequency)
            signal = savgol_filter(
                signal,
                window_length=window_length,
                polyorder=smooth_order,
                mode="mirror",
            )

        # normalize signal volumes to given range
        signal = rescale_range(
            signal,
            input_range=(signal.min(), signal.max()),
            output_range=output_range,
        )

        return cls(signal, sampling_frequency=target_sampling_frequency)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    signal = RespiratorySignal.create_sin4(total_seconds=200)
    # plt.plot(signal.time, signal.signal)
    # plt.plot(signal.time, signal.dt_signal)

    resampled_signal = signal.resample(15)
    plt.plot(resampled_signal.time, resampled_signal.signal)
    plt.plot(resampled_signal.time, resampled_signal.dt_signal)

    quantized_signal, quantized_dt_signal = signal.quantize_signal(n_bins=10)
    print(RespiratorySignal.get_unique_signals(quantized_signal, quantized_dt_signal))
    plt.plot(signal.time, quantized_signal)
    plt.plot(signal.time, quantized_dt_signal)
