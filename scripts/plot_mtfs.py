from pathlib import Path

import numpy as np
import tabulate

from cbctmc.evaluation.mtf import calculate_mtf, extract_line_pair_profile

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import SimpleITK as sitk

    gaps = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.50, 4.00]

    # fig, ax = plt.subplots(1, len(line_pair_spacings))
    fig, ax = plt.subplots()
    for mode in (
        "reference",
        "speedup_10.00x",
        # "speedup_20.00x",
        # "speedup_50.00x",
    ):
        for with_speedup_model in (False,):
            minimums = []
            maximums = []
            line_pair_spacings = []
            for i, gap in enumerate(gaps):
                line_pair_spacing = 2 * gap
                line_pair_spacings.append(line_pair_spacing)
                if gap == 0.25:
                    minimums.append(0)
                    maximums.append(0)
                    continue

                filepath = Path(
                    f"/mnt/nas_io/anarchy/4d_cbct_mc/mc_mtf_final_0/lp_{gap:.2f}gap/{mode}/reconstructions/fdk3d_wpc.mha"
                )
                image = sitk.ReadImage(filepath)
                image_spacing = image.GetSpacing()
                if len(set(image_spacing)) > 1:
                    raise ValueError(f"{image_spacing=} is not isotropic")

                image_spacing = image_spacing[0]

                image = sitk.GetArrayFromImage(image)
                image = np.swapaxes(image, 0, 2)

                pattern_length = int(line_pair_spacing / image_spacing * 4)
                pattern_depth = int(20 / (2 * image_spacing))
                # to ignore edge effects
                pattern_length = int(pattern_length * 1.00)

                image_center = (np.array(image.shape) / 2).astype(int)

                bounding_box = np.index_exp[
                    image_center[0]
                    - pattern_length // 2 : image_center[0]
                    + pattern_length // 2,
                    image_center[1]
                    - pattern_depth // 2 : image_center[1]
                    + pattern_depth // 2,
                    image_center[2]
                    - pattern_depth // 2 : image_center[2]
                    + pattern_depth // 2,
                ]

                try:
                    profile, max_indices, min_indices = extract_line_pair_profile(
                        image,
                        bounding_box=bounding_box,
                        average_axes=(1, 2),
                        min_peak_distance=line_pair_spacing / image_spacing,
                    )

                    minimums.append(profile[min_indices].mean())
                    maximums.append(profile[max_indices].mean())
                except ValueError:
                    print(f"Could not find peaks for {filepath=}")
                    minimums.append(0)
                    maximums.append(0)

            mtf = calculate_mtf(
                line_pair_spacings=line_pair_spacings,
                line_pair_maximums=maximums,
                line_pair_minimums=minimums,
                relative=True,
            )

            spatial_frequency = 1 / np.array(list(mtf.keys()))
            mtf = np.array(list(mtf.values()))

            print("***")
            print(mode)
            print(
                tabulate.tabulate(
                    zip(spatial_frequency, mtf),
                    headers=["spatial_frequency", "mtf"],
                    tablefmt="plain",
                )
            )

            ax.plot(
                spatial_frequency,
                mtf,
                label=f"{mode} (speedup model: {with_speedup_model})",
                marker="x",
            )

    ax.set_xlabel("Spatial frequency [lp/mm]")
    ax.set_ylabel("MTF")
    ax.legend()
