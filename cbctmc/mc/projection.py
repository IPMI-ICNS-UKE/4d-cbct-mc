from typing import Tuple

import numpy as np

from cbctmc.common_types import PathLike
from cbctmc.defaults import DefaultMCSimulationParameters as MCDefaults


class MCProjection:
    def __init__(self):
        pass

    @classmethod
    def from_file(
        cls,
        filepath: PathLike,
        n_detector_pixels: Tuple[int, int] = MCDefaults.n_detector_pixels,
        n_detector_pixels_half_fan: Tuple[
            int, int
        ] = MCDefaults.n_detector_pixels_half_fan,
    ) -> "MCProjection":
        projection = np.loadtxt(filepath, dtype="float")

        projection = projection.reshape(*n_detector_pixels[::-1], 4)
        projection = np.flip(projection, axis=0)

        projection = projection[:, : n_detector_pixels_half_fan[0]]

        return projection


if __name__ == "__main__":
    import itk
    import matplotlib.pyplot as plt

    from cbctmc.forward_projection import (
        create_geometry,
        prepare_image_for_rtk,
        project_forward,
    )
    from cbctmc.mc.geometry import MCGeometry

    p = MCProjection.from_file("/datalake_fast/mc_test/output_air/projection")

    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    ax[0].imshow(p[..., 0], clim=(0, 35))

    mc_geometry = MCGeometry.load(
        "/datalake_fast/4d_ct_lung_uke_artifact_free/"
        "022_4DCT_Lunge_amplitudebased_complete/phase_00_geometry.pkl.gz"
    )

    n_projections = 1
    geometry = create_geometry(start_angle=0, n_projections=n_projections)

    # image = itk.imread(
    #     "/datalake_fast/4d_ct_lung_uke_artifact_free/022_4DCT_Lunge_amplitudebased_complete/phase_00.nii"
    # )

    image = prepare_image_for_rtk(
        image=mc_geometry.densities,
        image_spacing=mc_geometry.image_spacing,
        input_value_range=None,
        output_value_range=None,
    )
    forward_projection = project_forward(image, geometry=geometry)
    # itk.imwrite(
    #     forward_projection, "/datalake/4d_cbct_mc/CatPhantom/scan_2/catphan_fp.mha"
    # )

    import matplotlib.pyplot as plt

    out = itk.GetArrayFromImage(forward_projection)
    out = np.swapaxes(out, 0, 2)

    out = np.swapaxes(out, 0, 1)
    out = np.flip(out, axis=0)

    ax[1].imshow(out[..., 0])
