from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from cbctmc.mc.materials import MATERIALS_125KEV
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

    slicing = np.index_exp[55:440, 140:360, 228]
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
    # print HTML color codes
    for material_number in np.unique(material_image[slicing]):
        cmap = plt.get_cmap("jet")

        rgba = cmap((material_number - 1) / (20 - 1))
        rgb = rgba[:3]
        for material_id, material in MATERIALS_125KEV.items():
            if material.number == material_number:
                print(f"material {material_number} ({material.identifier}): {rgb=}")

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

    slicing = np.index_exp[35:420, 55, 125:345]
    save_image(
        np.rot90(recon[slicing]),
        cmap="gray",
        clim=(0, 0.03),
        filepath=OUTPUT_FOLDER / "workflow_row_3_recon_3d.png",
    )

    # 4D CIRS
    real_clim = (-0.0040, 0.028)
    mc_clim = (-0.00639, 0.02968)
    real_4d_image = read_image(
        "/data_l79a/fmadesta/4d_cbct/R4DCIRS/for_mc/4d_cbct_phantom_data/2018_08_09_session_2/recons_custom/3d_fdk.mha"
    )
    mc_ref_4d_image = read_image(
        "/datalake_fast/mc_output/4d_cirs_20_bins/4d_cirs/phase_02/reference/reconstructions/fdk3d_wpc.mha"
    )
    mc_speedup_50_low_photon_4d_image = read_image(
        "/datalake_fast/mc_output/4d_cirs/4d_cirs_large/phase_02/speedup_50.00x/reconstructions/fdk3d_wpc.mha"
    )
    mc_speedup_50_4d_image = read_image(
        "/datalake_fast/mc_output/4d_cirs/4d_cirs_large/phase_02/speedup_50.00x/reconstructions/fdk3d_wpc_speedup.mha"
    )

    slicing = np.index_exp[72:390, 119, 131:343]
    save_image(
        np.rot90(real_4d_image[slicing]),
        cmap="gray",
        clim=real_clim,
        filepath=OUTPUT_FOLDER / "4d_cirs_real_ref.png",
    )
    save_image(
        np.rot90(mc_ref_4d_image[72 - 2 : 390 - 2, 119, 131 + 2 : 343 + 2]),
        cmap="gray",
        clim=mc_clim,
        filepath=OUTPUT_FOLDER / "4d_cirs_mc_ref.png",
    )
    save_image(
        np.rot90(mc_speedup_50_low_photon_4d_image[slicing]),
        cmap="gray",
        clim=mc_clim,
        filepath=OUTPUT_FOLDER / "4d_cirs_mc_speedup_50_low_photon.png",
    )
    save_image(
        np.rot90(mc_speedup_50_4d_image[slicing]),
        cmap="gray",
        clim=mc_clim,
        filepath=OUTPUT_FOLDER / "4d_cirs_mc_speedup_50.png",
    )
