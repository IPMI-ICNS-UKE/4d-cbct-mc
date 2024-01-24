import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
from ipmi.common.logger import init_fancy_logging
from scipy.signal import savgol_filter
from vroc.blocks import SpatialTransformer

from cbctmc.mc.respiratory import RespiratorySignal
from cbctmc.reconstruction.respiratory import calculate_median_cycle
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
    logging.getLogger("vroc").setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    init_fancy_logging()

    DEVICE = "cuda:0"

    folder = Path("/mnt/nas_io/anarchy/4d_cbct_mc/4d/R2017025")

    cbct_amplitude = np.loadtxt(folder / "cbct/amplitude_varian.txt")
    ct_amplitude = np.loadtxt(
        folder / "ct_rai/resp_curve/R2017025_1_1.vxp",
        skiprows=10,
        delimiter=",",
        usecols=0,
    )
    ct_amplitude = -ct_amplitude

    ct_amplitude = (
        RespiratorySignal(signal=ct_amplitude, sampling_frequency=25)
        .resample(MCDefaults.frame_rate)
        .signal
    )

    ct_amplitude = savgol_filter(
        ct_amplitude,
        window_length=15,
        polyorder=3,
        mode="mirror",
    )
    cbct_amplitude = savgol_filter(
        cbct_amplitude,
        window_length=15,
        polyorder=3,
        mode="mirror",
    )

    cbct_median_amplitude = calculate_median_cycle(cbct_amplitude)
    cbct_median_amplitude_offset = cbct_median_amplitude.mean()
    cbct_median_amplitude -= cbct_median_amplitude_offset

    ct_median_amplitude = calculate_median_cycle(ct_amplitude)
    ct_median_amplitude_offset = ct_median_amplitude.mean()
    ct_median_amplitude -= ct_median_amplitude_offset

    plt.plot(ct_median_amplitude)
    plt.plot(cbct_median_amplitude)

    cbct_signal = RespiratorySignal(
        signal=cbct_amplitude - cbct_median_amplitude_offset,
        sampling_frequency=MCDefaults.frame_rate,
    )
    ct_median_signal = RespiratorySignal(
        signal=ct_median_amplitude,
        sampling_frequency=MCDefaults.frame_rate,
    )

    plt.figure()
    plt.plot(cbct_signal.time, cbct_signal.signal)
    plt.plot(cbct_signal.time, cbct_signal.dt_signal)
    plt.title("Varian 4D CBCT signal")
    cbct_signal.save(folder / "cbct/respiratory_signal.pkl")

    # load signal from real scan
    phase_timepoints = np.arange(
        0, len(ct_median_signal.time), len(ct_median_signal.time) // 10
    )[:10]

    plt.figure()
    plt.plot(ct_median_signal.time, ct_median_signal.signal)
    plt.plot(ct_median_signal.time, ct_median_signal.dt_signal)
    plt.scatter(
        ct_median_signal.time[phase_timepoints],
        ct_median_signal.signal[phase_timepoints],
    )
    plt.title("4D CT median respiratory cycle")

    image_filepaths = [folder / f"ct_rai/bin_{i:02d}.nii" for i in range(10)]

    images = [read_image(image_filepath) for image_filepath in image_filepaths]
    images = np.stack(images, axis=0)

    # just select signal at timepoints of phase images
    phase_signals = np.stack(
        (
            ct_median_signal.signal[phase_timepoints],
            ct_median_signal.dt_signal[phase_timepoints],
        ),
        axis=-1,
    )
    model = CorrespondenceModel.build_default(
        images=images, signals=phase_signals, reference_phase=2, device=DEVICE
    )
    model.save(folder / "correspondence_model_rai.pkl")

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
        ax[0, i_phase].imshow(images[i_phase][:, 225, :])
        ax[1, i_phase].imshow(warped_image[:, 225, :])
        ax[2, i_phase].imshow(
            vector_field[2, :, 225, :], clim=(-20, 20), cmap="seismic"
        )
