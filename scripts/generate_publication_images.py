from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from cbctmc.utils import resample_image_spacing


def read_image(
    filepath, resample_spacing: tuple[float, float, float] | None = (1.0, 1.0, 1.0)
) -> np.ndarray:
    image = sitk.ReadImage(str(filepath))
    image = resample_image_spacing(
        image,
        new_spacing=resample_spacing,
        resampler=sitk.sitkNearestNeighbor,
    )
    image = sitk.GetArrayFromImage(image)
    image = np.swapaxes(image, 0, 2)
    return image


def save_image(image: np.ndarray, cmap: str, clim: tuple, filepath: Path):
    plt.imsave(filepath, image, cmap=cmap, vmin=clim[0], vmax=clim[1])


if __name__ == "__main__":
    OUTPUT_FOLDER = Path("/mnt/nas_io/anarchy/4d_cbct_mc/publication_figures")
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    CT_CLIM = (-1024, 700)

    # workflow figure
    # first row
    ct_image = read_image(
        "/mnt/nas_io/anarchy/4d_cbct_mc/4d_ct_lung_uke_artifact_free/024_4DCT_Lunge_amplitudebased_complete/phase_00.nii"
    )
    density_image = read_image(
        "/mnt/nas_io/anarchy/4d_cbct_mc/speedup/024_4DCT_Lunge_amplitudebased_complete/phase_00/geometry_densities.nii.gz"
    )
    material_image = read_image(
        "/mnt/nas_io/anarchy/4d_cbct_mc/speedup/024_4DCT_Lunge_amplitudebased_complete/phase_00/geometry_materials.nii.gz"
    )

    slicing = np.index_exp[55:440, 140:360, 180]
    save_image(
        ct_image[slicing].T,
        cmap="gray",
        clim=CT_CLIM,
        filepath=OUTPUT_FOLDER / "workflow_row_1_ct.png",
    )
    save_image(
        density_image[slicing].T,
        cmap="gray",
        clim=(0, 2),
        filepath=OUTPUT_FOLDER / "workflow_row_1_density.png",
    )
    save_image(
        material_image[slicing].T,
        cmap="jet",
        clim=(1, 20),
        filepath=OUTPUT_FOLDER / "workflow_row_1_material.png",
    )

    # second row
    low_photon_projections = read_image(
        "/mnt/nas_io/anarchy/4d_cbct_mc/speedup/024_4DCT_Lunge_amplitudebased_complete/phase_00/speedup_50.00x/projections_total_normalized.mha",
        resample_spacing=None,
    )
    forward_projections = read_image(
        "/mnt/nas_io/anarchy/4d_cbct_mc/speedup/024_4DCT_Lunge_amplitudebased_complete/phase_00/density_fp.mha",
        resample_spacing=None,
    )
    speedup_projections = read_image(
        "/mnt/nas_io/anarchy/4d_cbct_mc/speedup/024_4DCT_Lunge_amplitudebased_complete/phase_00/speedup_50.00x/projections_total_normalized_speedup.mha",
        resample_spacing=None,
    )
    slicing = np.index_exp[:, :, 243]
    save_image(
        low_photon_projections[slicing].T,
        cmap="gray",
        clim=(0, 5),
        filepath=OUTPUT_FOLDER / "workflow_row_2_50x_low_photon_projection.png",
    )
    save_image(
        forward_projections[slicing].T,
        cmap="gray",
        clim=(0, 350),
        filepath=OUTPUT_FOLDER / "workflow_row_2_forward_projection.png",
    )
    save_image(
        speedup_projections[slicing].T,
        cmap="gray",
        clim=(0, 5),
        filepath=OUTPUT_FOLDER / "workflow_row_2_speedup_projection.png",
    )

    # third row
    recon = read_image(
        "/mnt/nas_io/anarchy/4d_cbct_mc/speedup/024_4DCT_Lunge_amplitudebased_complete/phase_00/reference/reconstructions/fdk3d_wpc.mha",
    )

    slicing = np.index_exp[35:420, 102, 125:345]
    save_image(
        np.rot90(recon[slicing]),
        cmap="gray",
        clim=(0, 0.03),
        filepath=OUTPUT_FOLDER / "workflow_row_3_recon_3d.png",
    )

    # 4D CIRS
