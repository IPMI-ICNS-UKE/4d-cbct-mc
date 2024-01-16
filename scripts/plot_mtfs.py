from pathlib import Path

import numpy as np
import tabulate

from cbctmc.evaluation.mtf import calculate_mtf, extract_line_pair_profile

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import SimpleITK as sitk

    line_pair_spacings = np.linspace(0.25, 4.0, 16, endpoint=True)

    # fig, ax = plt.subplots(1, len(line_pair_spacings))
    fig, ax = plt.subplots()
    for mode in (
        "reference",
        # "speedup_2.00x",
        # "speedup_5.00x",
        # "speedup_10.00x",
        # "speedup_20.00x",
        # "speedup_50.00x",
    ):
        minimums = []
        maximums = []
        for i, gap in enumerate(line_pair_spacings):
            if gap == 0.25:
                minimums.append(0)
                maximums.append(0)
                continue

            line_pair_spacing = 2 * gap
            filepath = next(
                p
                for p in (
                    f"/datalake2/mc_mtf_final/lp_{gap:.2f}gap/{mode}/reconstructions/fdk3d_wpc.mha",
                    f"/mnt/nas_io/anarchy/4d_cbct_mc/mc_mtf_final/lp_{gap:.2f}gap/{mode}/reconstructions/fdk3d_wpc.mha",
                )
                if Path(p).exists()
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
                - pattern_depth // 2 : image_center[0]
                + pattern_depth // 2,
                image_center[1]
                - pattern_depth // 2 : image_center[1]
                + pattern_depth // 2,
                image_center[2]
                - pattern_length // 2 : image_center[2]
                + pattern_length // 2,
            ]

            try:
                profile, max_indices, min_indices = extract_line_pair_profile(
                    image,
                    bounding_box=bounding_box,
                    average_axes=(0, 1),
                    min_peak_distance=line_pair_spacing / image_spacing,
                )
                # ax[i].plot(profile)
                # ax[i].scatter(max_indices, profile[max_indices], c="g", marker="x")
                # ax[i].scatter(min_indices, profile[min_indices], c="r", marker="x")

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

        print(
            tabulate.tabulate(
                zip(spatial_frequency, mtf),
                headers=["spatial_frequency", "mtf"],
                tablefmt="plain",
            )
        )

        ax.plot(spatial_frequency, mtf, label=mode, marker="x")

    ax.set_xlabel("Spatial frequency [lp/mm]")
    ax.set_ylabel("MTF")
    ax.legend()
