import numpy as np


class RespiratorySignal:
    def __init__(self, signal: np.ndarray, sampling_frequency: float = 25.0):
        self.signal = signal
        self.sampling_frequency = sampling_frequency

        self.dt_signal = self._calculate_time_derivative()

    def __len__(self):
        return len(self.signal)

    def _calculate_time_derivative(self):
        return np.gradient(self.signal, 1 / self.sampling_frequency)

    @property
    def total_seconds(self):
        return len(self) / self.sampling_frequency

    @property
    def time(self):
        return np.linspace(0, self.total_seconds, len(self), endpoint=False)

    def quantize_amplitude(self, n_bins: int = 2) -> np.ndarray:
        bins = np.linspace(self.signal.min(), self.signal.max(), n_bins + 1)
        signal_bins = np.digitize(self.signal, bins=bins)
        # add half a bin width to get center the bins
        bin_width = bins[1] - bins[0]
        return bins[signal_bins - 1] + 0.5 * bin_width

    @property
    def unique_signals(self):
        return np.unique(self.signal)

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
        return cls(signal, sampling_frequency)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    signal = RespiratorySignal.create_sin4(total_seconds=20)
    plt.plot(signal.time, signal.signal)
    plt.plot(signal.time, signal.dt_signal)

    signal.quantize_amplitude(50)
    plt.plot(signal.time, signal.signal)
