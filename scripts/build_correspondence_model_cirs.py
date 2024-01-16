import logging
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
from ipmi.common.logger import init_fancy_logging
from scipy.signal import savgol_filter
from vroc.blocks import SpatialTransformer

from cbctmc.mc.respiratory import RespiratorySignal
from cbctmc.registration.correspondence import CorrespondenceModel
from cbctmc.utils import resample_image_spacing


def read_image(filepath) -> np.ndarray:
    image = sitk.ReadImage(str(filepath))
    image = resample_image_spacing(
        image, new_spacing=(1.0, 1.0, 1.0), resampler=sitk.sitkNearestNeighbor
    )
    image = sitk.GetArrayFromImage(image)
    image = np.swapaxes(image, 0, 2)
    return image


if __name__ == "__main__":
    from cbctmc.defaults import DefaultMCSimulationParameters as MCDefaults

    logging.getLogger("cbctmc").setLevel(logging.DEBUG)
    logging.getLogger("vroc").setLevel(logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    init_fancy_logging()

    DEVICE = "cuda:0"

    image_filepaths = [
        Path("/data_l79a/fmadesta/4d_cbct/R4DCIRS/4DCT/phase_based_gt_curve")
        / f"phase_{i:02d}.nii"
        for i in range(10)
    ]

    images = [read_image(image_filepath) for image_filepath in image_filepaths]
    images = np.stack(images, axis=0)

    signal = np.loadtxt("/data_l79a/fmadesta/4d_cbct/R4DCIRS/4DCT/ref_cycle.txt")
    # smooth using savgol filter
    signal = savgol_filter(signal, 15, 3)
    sampling_frequency = len(signal) / 5.0

    # scale signal to [0, 1]
    signal = (signal - signal.min()) / (signal.max() - signal.min())
    # convert signal to be from peak to peak: phase 0 is max inhalation = peak at signal
    signal = np.tile(signal, 2)
    # select peak-to-peak signal
    max_indices = np.where(signal == signal.max())[0]
    signal = signal[max_indices[0] : max_indices[1]]

    signal = RespiratorySignal(
        signal=signal,
        sampling_frequency=sampling_frequency,
    )

    # read Varian CBCT signal
    varian_amplitude = np.loadtxt(
        "/data_l79a/fmadesta/4d_cbct/R4DCIRS/for_mc/4d_cbct_phantom_data/2018_08_09_session_2/recons_custom/amplitude_varian.txt"
    )
    varian_amplitude = savgol_filter(varian_amplitude, 15, 3)
    # scale amplitude to [0, 1]
    varian_amplitude = (varian_amplitude - varian_amplitude.min()) / (
        varian_amplitude.max() - varian_amplitude.min()
    )
    varian_fps = MCDefaults.frame_rate
    varian_signal = RespiratorySignal(
        signal=varian_amplitude,
        sampling_frequency=varian_fps,
    )
    plt.figure()
    plt.plot(varian_signal.time, varian_signal.signal)
    plt.plot(varian_signal.time, varian_signal.dt_signal)
    plt.title("Varian 4D CBCT signal")
    varian_signal.save(
        "/mnt/nas_io/anarchy/4d_cbct_mc/cirs_varian_respiratory_signal.pkl"
    )

    # load signal from real scan
    phase_timepoints = np.arange(0, len(signal.time), len(signal.time) // 10)[:10]

    plt.figure()
    plt.plot(signal.time, signal.signal)
    plt.plot(signal.time, signal.dt_signal)
    plt.scatter(signal.time[phase_timepoints], signal.signal[phase_timepoints])
    plt.title("4D CT respiratory cycle")

    # just select signal at timepoints of phase images
    phase_signals = np.stack(
        (signal.signal[phase_timepoints], signal.dt_signal[phase_timepoints]), axis=-1
    )
    model = CorrespondenceModel.build_default(
        images=images, signals=phase_signals, reference_phase=2, device=DEVICE
    )
    model.save("/mnt/nas_io/anarchy/4d_cbct_mc/cirs_correspondence_model.pkl")

    # predict phases for comparison
    reference_image = images[2]
    fig, ax = plt.subplots()
    ax.plot(phase_signals[:, 0])
    ax.plot(phase_signals[:, 1])

    fig, ax = plt.subplots(3, 10, sharex=True, sharey=True)
    for i_phase, phase_signal in enumerate(phase_signals):
        vector_field = model.predict(phase_signal)
        spatial_transformer = SpatialTransformer().to(DEVICE)
        warped_image = spatial_transformer(
            image=torch.as_tensor(
                reference_image[None, None], dtype=torch.float32, device=DEVICE
            ),
            transformation=torch.as_tensor(
                vector_field[None], dtype=torch.float32, device=DEVICE
            ),
            mode="nearest",
        )

        warped_image = warped_image.cpu().numpy()[0, 0]
        ax[0, i_phase].imshow(images[i_phase][:, 150, :])
        ax[1, i_phase].imshow(warped_image[:, 150, :])
        ax[2, i_phase].imshow(
            vector_field[2, :, 150, :], clim=(-20, 20), cmap="seismic"
        )
