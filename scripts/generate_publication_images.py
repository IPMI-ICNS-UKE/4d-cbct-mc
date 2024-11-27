import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from vroc.logger import init_fancy_logging
from vroc.registration import VrocRegistration

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
    init_fancy_logging()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logging.getLogger("vroc").setLevel(logging.DEBUG)

    OUTPUT_FOLDER = Path("/mnt/nas_io/anarchy/4d_cbct_mc/publication_figures_phiro_reg")
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    CT_CLIM = (-1024, 700)

    device = "cuda:0"
    registration = VrocRegistration(
        roi_segmenter=None,
        feature_extractor=None,
        parameter_guesser=None,
        device=device,
    )

    # # workflow figure
    # # first row
    # ct_image = read_image(
    #     "/mnt/nas_io/anarchy/4d_cbct_mc/4d/R2017025/ct_rai/bin_02.nii"
    # )
    # density_image = read_image(
    #     "/mnt/nas_io/anarchy/4d_cbct_mc/3d/R2017025/fine_seg5/ct_rai_bin_02/geometry_densities.nii.gz"
    # )
    # material_image = read_image(
    #     "/mnt/nas_io/anarchy/4d_cbct_mc/3d/R2017025/fine_seg5/ct_rai_bin_02/geometry_materials.nii.gz"
    # )
    #
    # slicing = np.index_exp[55+5:440+5, 140-10:360-10, 162]
    # slicing = np.index_exp[58+15:418+15, 127+42:313+42, 162]
    # save_image(
    #     ct_image[slicing].T,
    #     cmap="gray",
    #     clim=CT_CLIM,
    #     filepath=OUTPUT_FOLDER / "workflow_row_1_ct.png",
    # )
    # save_image(
    #     density_image[slicing].T,
    #     cmap="gray",
    #     clim=(0, 2),
    #     filepath=OUTPUT_FOLDER / "workflow_row_1_density.png",
    # )
    # save_image(
    #     material_image[slicing].T,
    #     cmap="jet",
    #     clim=(1, 20),
    #     filepath=OUTPUT_FOLDER / "workflow_row_1_material.png",
    # )
    # # print HTML color codes
    # for material_number in np.unique(material_image[slicing]):
    #     cmap = plt.get_cmap("jet")
    #
    #     rgba = cmap((material_number - 1) / (20 - 1))
    #     rgb = rgba[:3]
    #     for material_id, material in MATERIALS_125KEV.items():
    #         if material.number == material_number:
    #             print(f"material {material_number} ({material.identifier}): {rgb=}")

    # # second row
    # low_photon_projections = read_image(
    #     "/mnt/nas_io/anarchy/4d_cbct_mc/speedup/024_4DCT_Lunge_amplitudebased_complete/phase_00/speedup_50.00x/projections_total_normalized.mha",
    #     resample_spacing=None,
    # )
    # forward_projections = read_image(
    #     "/mnt/nas_io/anarchy/4d_cbct_mc/speedup/024_4DCT_Lunge_amplitudebased_complete/phase_00/density_fp.mha",
    #     resample_spacing=None,
    # )
    # speedup_projections = read_image(
    #     "/mnt/nas_io/anarchy/4d_cbct_mc/speedup/024_4DCT_Lunge_amplitudebased_complete/phase_00/speedup_50.00x/projections_total_normalized_speedup.mha",
    #     resample_spacing=None,
    # )
    # slicing = np.index_exp[:, :, 243]
    # save_image(
    #     low_photon_projections[slicing].T,
    #     cmap="gray",
    #     clim=(0, 5),
    #     filepath=OUTPUT_FOLDER / "workflow_row_2_50x_low_photon_projection.png",
    # )
    # save_image(
    #     forward_projections[slicing].T,
    #     cmap="gray",
    #     clim=(0, 350),
    #     filepath=OUTPUT_FOLDER / "workflow_row_2_forward_projection.png",
    # )
    # save_image(
    #     speedup_projections[slicing].T,
    #     cmap="gray",
    #     clim=(0, 5),
    #     filepath=OUTPUT_FOLDER / "workflow_row_2_speedup_projection.png",
    # )
    #
    # third row
    # recon = read_image(
    #     "/mnt/nas_io/anarchy/4d_cbct_mc/speedup/024_4DCT_Lunge_amplitudebased_complete/phase_00/reference/reconstructions/fdk3d_wpc.mha",
    # )
    #
    # slicing = np.index_exp[35:420, 55, 125:345]
    recon = read_image(
        "/mnt/nas_io/anarchy/4d_cbct_mc/3d/R2017025/ct_rai_bin_02/reference/reconstructions/fdk3d_wpc.mha"
    )
    slicing = np.index_exp[58:418, 105, 127:313]

    save_image(
        np.rot90(recon[slicing]),
        cmap="gray",
        clim=(0, 0.03),
        filepath=OUTPUT_FOLDER / "workflow_row_3_recon_3d.png",
    )

    # 4D CIRS
    real_clim = (-0.0060, 0.0286)
    mc_clim = (-0.00639, 0.02968)
    real_4d_image = read_image(
        "/data_l79a/fmadesta/4d_cbct/R4DCIRS/for_mc/4d_cbct_phantom_data/2018_08_09_session_2/recons_custom/3d_fdk.mha"
    )
    mc_ref_4d_image = read_image(
        "/mnt/nas_io/anarchy/4d_cbct_mc/4d/cirs/output/4d_cirs_large_with_water/phase_02/reference/reconstructions/fdk3d_wpc.mha"
    )
    mc_speedup_50_low_photon_4d_image = read_image(
        "/mnt/nas_io/anarchy/4d_cbct_mc/4d/cirs/output/4d_cirs_large_with_water/phase_02/speedup_50.00x/reconstructions/fdk3d_wpc.mha"
    )
    mc_speedup_50_4d_image = read_image(
        "/mnt/nas_io/anarchy/4d_cbct_mc/4d/cirs/output/4d_cirs_large_with_water/phase_02/speedup_50.00x/reconstructions/fdk3d_wpc_speedup.mha"
    )

    # clipping for metric calculation, >0
    real_4d_image = np.clip(real_4d_image, 1e-6, None)
    mc_ref_4d_image = np.clip(mc_ref_4d_image, 1e-6, None)
    mc_speedup_50_low_photon_4d_image = np.clip(
        mc_speedup_50_low_photon_4d_image, 1e-6, None
    )
    mc_speedup_50_4d_image = np.clip(mc_speedup_50_4d_image, 1e-6, None)

    eval_slicing = ...
    mean_abs_mu_diff = {
        "real/mc_ref": np.abs(
            real_4d_image[eval_slicing] - mc_ref_4d_image[eval_slicing]
        ).mean(),
        "real/mc_speedup_50_low_photon": np.abs(
            real_4d_image[eval_slicing]
            - mc_speedup_50_low_photon_4d_image[eval_slicing]
        ).mean(),
        "real/mc_speedup_50": np.abs(
            real_4d_image[eval_slicing] - mc_speedup_50_4d_image[eval_slicing]
        ).mean(),
    }
    print(f"4D CIRS diffs: {mean_abs_mu_diff=}")

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

    # 4D end-to-en
    # real_clim = (-0.0013, 0.03009)
    # mc_clim = (-0.00236, 0.030)
    # ct_clim = (-1024, 800)

    real_clim = (-0.00691, 0.02903)
    mc_clim = (-0.00319, 0.02905)
    mc_speedup_clim = (-0.00480, 0.02905)
    ct_clim = (-1024, 800)
    real_4d_image = read_image(
        "/mnt/nas_io/anarchy/4d_cbct_mc/4d/R2017025/cbct/3d_fdk_reg_to_mc.mha"
    )
    real_4d_ct_image = read_image(
        "/mnt/nas_io/anarchy/4d_cbct_mc/4d/R2017025/ct_rai/avg_reg_to_mc.nii"
    )
    mc_ref_4d_image = read_image(
        "/mnt/nas_io/anarchy/4d_cbct_mc/4d/R2017025/mc/ct_rai/phase_02/reference/reconstructions/fdk3d_wpc.mha"
    )
    mc_speedup_20_low_photon_4d_image = read_image(
        "/mnt/nas_io/anarchy/4d_cbct_mc/4d/R2017025/mc/ct_rai/phase_02/speedup_20.00x/reconstructions/fdk3d_wpc.mha"
    )
    mc_speedup_20_4d_image = read_image(
        "/mnt/nas_io/anarchy/4d_cbct_mc/4d/R2017025/mc/ct_rai/phase_02/speedup_20.00x/reconstructions/fdk3d_wpc_speedup.mha"
    )

    # # corr: enable for plotting, disable for metric calculation
    # mean_real = real_4d_image[slicing].mean()
    # mean_mc = mc_ref_4d_image[slicing].mean()
    # real_4d_image = real_4d_image - mean_real + mean_mc

    # clipping for metric calculation, >0
    real_4d_image = np.clip(real_4d_image, 1e-6, None)
    mc_ref_4d_image = np.clip(mc_ref_4d_image, 1e-6, None)
    mc_speedup_20_low_photon_4d_image = np.clip(
        mc_speedup_20_low_photon_4d_image, 1e-6, None
    )
    mc_speedup_20_4d_image = np.clip(mc_speedup_20_4d_image, 1e-6, None)

    mean_abs_mu_diff = {
        "real/mc_ref": np.abs(real_4d_image - mc_ref_4d_image).mean(),
        "real/mc_speedup_20_low_photon": np.abs(
            real_4d_image - mc_speedup_20_low_photon_4d_image
        ).mean(),
        "real/mc_speedup_20": np.abs(real_4d_image - mc_speedup_20_4d_image).mean(),
    }
    print(f"real 4D patient diffs: {mean_abs_mu_diff=}")

    # # register mc images to real image
    # reg_result = registration.register(
    #     moving_image=mc_ref_4d_image,
    #     fixed_image=real_4d_image,
    #     image_spacing=(1.0, 1.0, 1.0),
    #     register_affine=False,
    #     affine_loss_function=mse_loss,
    #     affine_step_size=0.01,
    #     affine_iterations=500,
    #     affine_enable_scaling=False,
    #     affine_enable_rotation=True,
    #     affine_enable_shearing=True,
    #     affine_enable_translation=True,
    #     force_type="demons",
    #     gradient_type="dual",
    #     valid_value_range=None,
    #     early_stopping_delta=0.00,
    #     early_stopping_window=100,
    #     default_parameters={
    #         "iterations": 100,
    #         "tau": 2.25,
    #         "sigma_x": 2.0,
    #         "sigma_y": 2.0,
    #         "sigma_z": 2.0,
    #         "n_levels": 1,
    #         "largest_scale_factor": 0.5,
    #     },
    # )
    #
    # transformer = SpatialTransformer()
    # vector_field = torch.as_tensor(
    #     reg_result.composed_vector_field[None], dtype=torch.float32, device=device
    # )
    #
    # mc_ref_4d_image = transformer.forward(
    #     image=torch.as_tensor(mc_ref_4d_image[None, None], dtype=torch.float32,
    #                           device=device),
    #     transformation=vector_field,
    # )
    # mc_ref_4d_image = mc_ref_4d_image.detach().cpu().numpy()[0, 0]
    #
    # mc_speedup_20_low_photon_4d_image = transformer.forward(
    #     image=torch.as_tensor(mc_speedup_20_low_photon_4d_image[None, None], dtype=torch.float32,
    #                             device=device),
    #     transformation=vector_field,
    # )
    # mc_speedup_20_low_photon_4d_image = mc_speedup_20_low_photon_4d_image.detach().cpu().numpy()[0, 0]
    #
    # mc_speedup_20_4d_image = transformer.forward(
    #     image=torch.as_tensor(mc_speedup_20_4d_image[None, None], dtype=torch.float32,
    #                             device=device),
    #     transformation=vector_field,
    # )
    # mc_speedup_20_4d_image = mc_speedup_20_4d_image.detach().cpu().numpy()[0, 0]

    slicing = np.index_exp[58:418, 105, 127:313]
    save_image(
        np.rot90(real_4d_image[slicing]),
        cmap="gray",
        clim=real_clim,
        filepath=OUTPUT_FOLDER / "4d_patient_real_ref.png",
    )
    save_image(
        np.rot90(real_4d_ct_image[slicing]),
        cmap="gray",
        clim=ct_clim,
        filepath=OUTPUT_FOLDER / "4d_patient_real_ct.png",
    )
    save_image(
        np.rot90(mc_ref_4d_image[slicing]),
        cmap="gray",
        clim=mc_clim,
        filepath=OUTPUT_FOLDER / "4d_patient_mc_ref.png",
    )
    save_image(
        np.rot90(mc_speedup_20_low_photon_4d_image[slicing]),
        cmap="gray",
        clim=mc_clim,
        filepath=OUTPUT_FOLDER / "4d_patient_mc_speedup_20_low_photon.png",
    )
    save_image(
        np.rot90(mc_speedup_20_4d_image[slicing]),
        cmap="gray",
        clim=mc_clim,
        filepath=OUTPUT_FOLDER / "4d_patient_mc_speedup_20.png",
    )

    # save diff of MC images to real image
    diff_clim = (-0.025, 0.025)
    diff_mc_ref_4d_image = mc_ref_4d_image - real_4d_image
    diff_mc_speedup_20_low_photon_4d_image = (
        mc_speedup_20_low_photon_4d_image - real_4d_image
    )
    diff_mc_speedup_20_4d_image = mc_speedup_20_4d_image - real_4d_image
    diff_3d_vs_4d_image = mc_ref_4d_image - recon

    diff_low_photon_to_mc_ref = mc_speedup_20_low_photon_4d_image - mc_ref_4d_image
    diff_speedup_to_mc_ref = mc_speedup_20_4d_image - mc_ref_4d_image

    save_image(
        np.rot90(diff_mc_ref_4d_image[slicing]),
        cmap="seismic",
        clim=diff_clim,
        filepath=OUTPUT_FOLDER / "4d_mc_ref_diff_to_real.png",
    )
    save_image(
        np.rot90(diff_mc_speedup_20_low_photon_4d_image[slicing]),
        cmap="seismic",
        clim=diff_clim,
        filepath=OUTPUT_FOLDER / "4d_mc_low_photon_diff_to_real.png",
    )
    save_image(
        np.rot90(diff_mc_speedup_20_4d_image[slicing]),
        cmap="seismic",
        clim=diff_clim,
        filepath=OUTPUT_FOLDER / "4d_mc_speedup_diff_to_real.png",
    )
    save_image(
        np.rot90(diff_3d_vs_4d_image[slicing]),
        cmap="seismic",
        clim=diff_clim,
        filepath=OUTPUT_FOLDER / "3d_vs_4d_mc_diff.png",
    )
    save_image(
        np.rot90(diff_low_photon_to_mc_ref[slicing]),
        cmap="seismic",
        clim=diff_clim,
        filepath=OUTPUT_FOLDER / "4d_mc_low_photon_diff_to_mc_ref.png",
    )
    save_image(
        np.rot90(diff_speedup_to_mc_ref[slicing]),
        cmap="seismic",
        clim=diff_clim,
        filepath=OUTPUT_FOLDER / "4d_mc_speedup_diff_to_mc_ref.png",
    )
