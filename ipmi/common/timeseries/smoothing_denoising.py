from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from scipy import fftpack

logger = logging.getLogger(__name__)


def calc_snr_for_signal(noisy_signal: np.ndarray, smooth_signal: np.ndarray) -> float:
    """Calculates signal-to-noise-ratio (SNR) for a given noisy signal and
    corresponding smooth signal. Same shape and same sampling rate is assumed.

    See https://en.wikipedia.org/wiki/Signal-to-noise_ratio

    :param noisy_signal: 1-D array, signal with noise
    :type noisy_signal:
    :param smooth_signal: 1-D array, signal used to baseline the noise
    :type smooth_signal:
    :return: calculated SNR in dB
    :rtype:
    """

    if noisy_signal.shape != smooth_signal.shape:
        raise ValueError("Shape mismatch")

    power_smooth_signal = smooth_signal**2
    power_noise = (smooth_signal - noisy_signal) ** 2
    snr = power_smooth_signal.mean() / power_noise.mean()
    snr_db = 10 * np.log10(snr)
    return snr_db


def fourier_smoothing(
    time_series: np.ndarray,
    freq_threshold_hz: float,
    sampling_rate: int,
    return_spectrum: bool,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """De-noises a signal using Fourier smoothing.

    Especially at the edges this approach performs poorly.
    For details look at https://en.wikipedia.org/wiki/Gibbs_phenomenon

    :param time_series: 1-D array which is going to be de-noised
    :type time_series:
    :param freq_threshold_hz: cutoff frequency in Hz
    :type freq_threshold_hz:
    :param sampling_rate: samples per second of time_series
    :type sampling_rate:
    :param return_spectrum: if set to True, return smoothed signal, raw spectrum
    (of time_series) and new spec (of smoothed time series)
    :type return_spectrum:
    :return: smoothed signal
    :rtype:
    """

    raw_spec = fftpack.rfft(time_series)
    time_step = 1 / sampling_rate
    freq = fftpack.rfftfreq(len(raw_spec), d=time_step)

    # set all frequencies greater freq_threshold to zero
    new_spec = raw_spec.copy()
    new_spec[freq > freq_threshold_hz] = 0
    # reverse fourier transformation
    smoothed_time_series = fftpack.irfft(new_spec)

    if return_spectrum:
        return smoothed_time_series, raw_spec, new_spec, freq
    return smoothed_time_series


def add_white_noise_to_signal(
    target_snr_db: int, signal: np.ndarray, **kwargs
) -> np.ndarray:
    """
    :param target_snr_db: signal-to-noise-ratio (SNR) in dB
    :type target_snr_db:
    :param signal: 1-D array on which to add noise
    :type signal:
    :return: 1-D array. time_series with SNR target_snr_db
    :rtype:
    """
    seed = kwargs.get("seed", None)
    # Calculate signal power and convert to dB
    signal_power = signal**2
    sig_avg_power = np.mean(signal_power)
    sig_avg_db = 10 * np.log10(sig_avg_power)
    # Calculate noise according to SNR_dB = P_signal_dB - P_noise_dB
    # with P being the average signal power then convert
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_square = 10 ** (noise_avg_db / 10)
    # Generate a sample of white noise
    mean_noise = 0
    if seed:
        np.random.seed(seed)
    noise = np.random.normal(mean_noise, np.sqrt(noise_avg_square), len(signal_power))
    # Noise up the original signal
    noisy_time_series = signal + noise
    return noisy_time_series


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    logger.setLevel(logging.DEBUG)
    sampling_rate = 25
    t = np.arange(0, 61, 1 / sampling_rate)
    a_smooth = np.cos(t) + 10
    added_snr = 20
    a_noisy = add_white_noise_to_signal(target_snr_db=added_snr, signal=a_smooth)
    calc_snr = calc_snr_for_signal(noisy_signal=a_noisy, smooth_signal=a_smooth)

    fig, ax = plt.subplots(1, 1, sharex=True)
    ax.plot(t, a_noisy, label="noisy", c="g")
    ax.plot(t, a_smooth - 0.1, label="target", c="blue")
    ax.legend(loc=1)
    fig.show()
    # ax.plot(t, a_noisy, label="noisy")
    # ax.plot(t, a_re_smoothed + 0.1, label="savgol")

    assert np.allclose(
        added_snr, calc_snr, atol=0.5
    ), f"SNRs do not match; added_snr: {added_snr} vs calc_snr: {calc_snr}"
    # ft_cutoff = 0.3
    # a_smooth_calc, noisy_spec, smooth_spec, freq = fourier_smoothing(
    #     a_noisy,
    #     freq_threshold_hz=ft_cutoff,
    #     sampling_rate=sampling_rate,
    #     return_spectrum=True,
    # )

    # fig, ax = plt.subplots(3, 1)
    # ax[0].plot(t, a_smooth - 0.8, label="smooth signal (-0.8)")
    # ax[0].plot(t, a_noisy, label=f"noisy signal, SNR {added_snr} dB ")
    # ax[0].plot(t, a_smooth_calc, label=f"smooth signal, FT cutoff {ft_cutoff} Hz")
    # ax[0].legend(loc=1)
    # ax[1].plot(
    #     t,
    #     abs(np.subtract(a_smooth, a_smooth_calc)),
    #     label="Diff smooth and denoised signal",
    # )
    # ax[1].legend(loc=1)
    # ax[2].set_title("Spectra")
    # ax[2].plot(
    #     freq, noisy_spec, label=f"Spec of signal SNR {added_snr} dB", ls="dashed"
    # )
    # ax[2].plot(freq, smooth_spec, label="Spec of smooth signal", ls="dotted")
    # ax[2].legend(loc=1)
    # ax[2].set_xlim([0, 10])
    # fig.show()
