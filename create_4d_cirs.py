import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from cbctmc.mc.geometry import MCCIRSPhantomGeometry
from cbctmc.mc.respiratory import RespiratorySignal
def plot_phantom(phantom):
    plt.imshow(phantom.densities[...,71])
    plt.show()

def create_signal(output, period, amplitude, n_phases):
    signal = RespiratorySignal.create_cos4(
        total_seconds=period, amplitude=amplitude, sampling_frequency=25.0
    )
    signal.save(os.path.join(output, "cirs_signal.pkl"))
    signal_phases = signal.resample(sampling_frequency=n_phases / period)
    print(signal_phases.signal)
    plt.plot(signal_phases.time, signal_phases.signal, "*")
    plt.plot(signal.time, signal.signal)
    plt.show()
    signals = np.stack(
        [
            signal_phases.signal,
            signal_phases.dt_signal,
        ],
        axis=-1,
    )
    return signals, signal_phases

def create_4d_cirs(signal_phases, output):
    CIRS = MCCIRSPhantomGeometry.from_base_geometry()
    CIRS = CIRS.place_insert()
    plot_phantom(CIRS)

    geometries = []
    for i_phase, shift in enumerate(signal_phases.signal):
        shift -= 10
        geometry_with_insert = CIRS.place_insert(shift=(0, 0, -shift))
        geometry_with_insert = geometry_with_insert.pad_to_shape((500, 500, 300))
        geometry_with_insert.save_material_segmentation(
            output / f"cirs_phase_{i_phase:02d}.nii.gz"
        )
        geometry_with_insert.save_density_image(
            output / f"cirs_phase_{i_phase:02d}_density.nii.gz"
        )
        geometry_with_insert.save(output / f"cirs_phase_{i_phase:02d}.pkl")
        geometries.append(geometry_with_insert)
    return geometries


def main(output):
    amplitude = 20.0
    period = 5.0
    n_phases = 10
    signals, signal_phases = create_signal(output, period, amplitude, n_phases)
  #  create_4d_cirs(signal_phases, output)
  #  images = np.stack([g.densities for g in geometries], axis=0)
  #  model = CorrespondenceModel.build_default(
  #      images=images, signals=signals, reference_phase=2, device=DEVICE
  #  )

if __name__ == '__main__':
    #create_4d_cirs()
    output = Path("/media/laura/2TB/4dcbct_sim/cirs/4d_cirs")
    main(output)