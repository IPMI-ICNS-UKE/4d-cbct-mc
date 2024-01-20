from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from vroc.blocks import SpatialTransformer

from cbctmc.defaults import DefaultMCSimulationParameters as MCDefaults
from cbctmc.mc.geometry import MCCIRSPhantomGeometry
from cbctmc.mc.respiratory import RespiratorySignal
from cbctmc.registration.correspondence import CorrespondenceModel

if __name__ == "__main__":
    output_folder = Path("/data_l79a/fmadesta/4d_cbct/R4DCIRS/for_mc/4d_cirs_large")
    geometry = MCCIRSPhantomGeometry.from_base_geometry()

    DEVICE = "cuda:0"
    n_phases = 10
    amplitude = 20.0
    period = 5.0

    signal = RespiratorySignal.create_cos4(
        total_seconds=period, amplitude=amplitude, sampling_frequency=25.0
    )
    signal_phases = signal.resample(sampling_frequency=n_phases / period)
    plt.plot(signal.time, signal.signal)
    plt.plot(signal.time, signal.dt_signal)
    plt.plot(signal_phases.time, signal_phases.signal)
    plt.plot(signal_phases.time, signal_phases.dt_signal)
    plt.show()

    signals = np.stack(
        [
            signal_phases.signal,
            signal_phases.dt_signal,
        ],
        axis=-1,
    )

    geometries = []
    for i_phase, shift in enumerate(signal_phases.signal):
        shift -= 10
        geometry_with_insert = geometry.place_insert(shift=(0, 0, -shift))

        geometry_with_insert = geometry_with_insert.pad_to_shape((500, 500, 300))
        geometry_with_insert.save_material_segmentation(
            output_folder / f"cirs_phase_{i_phase:02d}.nii.gz"
        )
        geometry_with_insert.save_density_image(
            output_folder / f"cirs_phase_{i_phase:02d}_density.nii.gz"
        )
        geometry_with_insert.save(output_folder / f"cirs_phase_{i_phase:02d}.pkl")
        geometries.append(geometry_with_insert)

    images = np.stack([g.densities for g in geometries], axis=0)
    model = CorrespondenceModel.build_default(
        images=images, signals=signals, reference_phase=2, device=DEVICE
    )
    model.save(output_folder / "cirs_correspondence_model.pkl")

    # read Varian CBCT signal
    varian_amplitude = np.loadtxt(
        "/data_l79a/fmadesta/4d_cbct/R4DCIRS/for_mc/4d_cbct_phantom_data/2018_08_09_session_2/recons_custom/amplitude_varian.txt"
    )
    varian_amplitude = savgol_filter(varian_amplitude, 15, 3)
    # scale amplitude to [0, 20]
    varian_amplitude = (
        amplitude
        * (varian_amplitude - varian_amplitude.min())
        / (varian_amplitude.max() - varian_amplitude.min())
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
    varian_signal.save(output_folder / "cirs_varian_respiratory_signal.pkl")

    reference_image = images[2]
    fig, ax = plt.subplots(3, 10, sharex=True, sharey=True)
    for i_phase, phase_signal in enumerate(signals):
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
        ax[0, i_phase].imshow(images[i_phase][239, :, :])
        ax[1, i_phase].imshow(warped_image[239, :, :])
        ax[2, i_phase].imshow(
            vector_field[2, 239, :, :], clim=(-20, 20), cmap="seismic"
        )
