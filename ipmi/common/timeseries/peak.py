from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.signal as signal
from scipy import interpolate

from ipmi.fused_types import Number


def resample_time_series(
    signal_time_secs: np.ndarray,
    signal_amplitude: np.ndarray,
    target_samples_per_second: Number,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resamples a given time series to the desired samples per second (target)
    by performing a simple 1D interpolation.

    :param signal_time_secs: 1-D array containing time component
    :type signal_time_secs:
    :param signal_amplitude: 1-D array containing amplitude information
    :type signal_amplitude:
    :param target_samples_per_second:
    :type target_samples_per_second:
    :return: time and corresponding amplitude of resampled signal
    :rtype: tuple(1-D array, 1-D array)
    """
    if signal_amplitude.shape != signal_time_secs.shape:
        raise ValueError("Shape mismatch")

    # define function between time and amplitude
    sampler = interpolate.interp1d(signal_time_secs, signal_amplitude)

    # calc time_new with given samples_per_second
    step = 1 / target_samples_per_second
    t_new = np.arange(
        start=signal_time_secs.min(),
        stop=signal_time_secs.min() + max(signal_time_secs),
        step=step,
    )
    a_new = sampler(t_new)

    return t_new, a_new


def calc_derivative(
    signal: np.ndarray,
    samples_per_second: Number,
    smoothing_kernel_size: int | None = None,
) -> np.ndarray:
    """Calculates the derivative of a given time series signal. If a kernel is
    given, derivative is smoothed.

    :param signal: 1-D array containing time component
    :type signal:
    :param samples_per_second: samples per second of given signal
    :type signal:
    :param smoothing_kernel_size: size of smoothing kernel
    :type smoothing_kernel_size:

    :return: derivative
    :rtype:
    """
    dt = 1 / samples_per_second
    derivative = np.gradient(signal, dt)
    if smoothing_kernel_size:
        if not isinstance(smoothing_kernel_size, int):
            raise ValueError("smoothing_kernel_size has to be integer")
        con_vec = np.ones(smoothing_kernel_size) / smoothing_kernel_size
        derivative = np.convolve(derivative, con_vec, mode="same")
    return derivative


def find_peaks(x: np.ndarray, scale: int = None, debug: bool = False):
    """Find peaks in quasi-periodic noisy signals using AMPD algorithm.
    Extended implementation handles peaks near start/end of the signal.
    Optimized implementation by Igor Gotlibovych, 2018.

    Taken from https://github.com/ig248/pyampd

    Parameters
    ----------
    x : ndarray
        1-D array on which to find peaks
    scale : int, optional
        specify maximum scale window size of (2 * scale + 1)
    debug : bool, optional
        if set to True, return the Local Scalogram Matrix, `LSM`,
        weigted number of maxima, 'G',
        and scale at which G is maximized, `l`,
        together with peak locations
    Returns
    -------
    pks: ndarray
        The ordered array of peak indices found in `x`
    """
    x = signal.detrend(x)
    N = len(x)
    L = N // 2
    if scale:
        L = min(scale, L)

    # create LSM matix
    LSM = np.ones((L, N), dtype=bool)
    for k in np.arange(1, L + 1):
        LSM[k - 1, 0 : N - k] &= x[0 : N - k] > x[k:N]  # compare to right neighbours
        LSM[k - 1, k:N] &= x[k:N] > x[0 : N - k]  # compare to left neighbours

    # Find scale with most maxima
    G = LSM.sum(axis=1)
    G = G * np.arange(
        N // 2, N // 2 - L, -1
    )  # normalize to adjust for new edge regions
    l_scale = np.argmax(G)

    # find peaks that persist on all scales up to l
    pks_logical = np.min(LSM[0:l_scale, :], axis=0)
    pks = np.flatnonzero(pks_logical)
    if debug:
        return pks, LSM, G, l_scale
    return pks
