import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from ipmi.common.logger import init_fancy_logging

from cbctmc.cli.run_mc_simulations import _reconstruct_mc_simulation
from cbctmc.utils import resample_image_spacing


def save_image(image: np.ndarray, cmap: str, clim: tuple, filepath: Path):
    plt.imsave(filepath, image, cmap=cmap, vmin=clim[0], vmax=clim[1])


def read_image(filepath) -> np.ndarray:
    if not Path(filepath).exists():
        return np.zeros((464, 250, 464), dtype=np.float32)

    image = sitk.ReadImage(str(filepath))
    image = resample_image_spacing(
        image, new_spacing=(1.0, 1.0, 1.0), resampler=sitk.sitkNearestNeighbor
    )
    image = sitk.GetArrayFromImage(image)
    # image = np.swapaxes(image, 0, 2)
    return image


if __name__ == "__main__":
    from vroc.registration import VrocRegistration

    logging.getLogger("cbctmc").setLevel(logging.DEBUG)
    logging.getLogger("vroc").setLevel(logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    init_fancy_logging()

    DEVICE = "cuda:0"

    simulation_folder = Path(
        "/mnt/nas_io/anarchy/4d_cbct_mc/4d/R2017025/mc_010lung/ct_rai/phase_02"
    )

    # recon
    _reconstruct_mc_simulation(
        simulation_folder=simulation_folder,
        config_name="reference",
        gpu_id=0,
        reconstruct_3d=False,
        reconstruct_4d=True,
        suffix="",
        logger=logger,
    )

    ct_image_filepaths = [
        f"/mnt/nas_io/anarchy/4d_cbct_mc/4d/R2017025/ct_rai/bin_{i:02d}_reg_to_mc.nii"
        for i in range(10)
    ]
    ct_images = [read_image(image_filepath) for image_filepath in ct_image_filepaths]
    ct_images = np.stack(ct_images, axis=0)
    ct_images = np.moveaxis(ct_images, 2, 3)
    ct_images = np.rot90(ct_images, axes=(1, 2), k=-1)
    ct_images = np.flip(ct_images, axis=-1)

    rooster_images = sitk.ReadImage(
        str(simulation_folder / "reference/reconstructions/rooster4d_wpc.mha")
    )
    rooster_images = sitk.GetArrayFromImage(rooster_images)
    rooster_images = np.moveaxis(rooster_images, 2, 3)
    rooster_images = np.rot90(rooster_images, axes=(1, 2), k=-1)
    rooster_images = np.flip(rooster_images, axis=-1)

    cg_images = sitk.ReadImage(
        str(simulation_folder / "reference/reconstructions/cg4d_wpc.mha")
    )
    cg_images = sitk.GetArrayFromImage(cg_images)
    cg_images = np.moveaxis(cg_images, 2, 3)
    cg_images = np.rot90(cg_images, axes=(1, 2), k=-1)
    cg_images = np.flip(cg_images, axis=-1)

    fig, ax = plt.subplots(2, 4, sharex=True, sharey=True)
    ax[1, 0].imshow(ct_images[0][:, ct_images[0].shape[1] // 2, :], cmap="gray")
    ax[1, 1].imshow(
        rooster_images[0][:, rooster_images[0].shape[1] // 2, :], cmap="gray"
    )
    ax[1, 2].imshow(cg_images[0][:, cg_images[0].shape[1] // 2, :], cmap="gray")
    plt.show()

    # plt.imshow(ct_images[0][:, 200])

    registration = VrocRegistration(
        device=DEVICE,
    )

    vector_fields = []
    for images in (ct_images, rooster_images, cg_images):
        images = images.astype(np.float32, order="C")
        registration_result = registration.register(
            moving_image=images[0],
            fixed_image=images[5],
            register_affine=False,
            default_parameters={
                "iterations": 800,
                "tau": 2.25,
                "tau_level_decay": 0.0,
                "tau_iteration_decay": 0.0,
                "sigma_x": 1.25,
                "sigma_y": 1.25,
                "sigma_z": 1.25,
                "sigma_level_decay": 0.0,
                "sigma_iteration_decay": 0.0,
                "n_levels": 3,
                "largest_scale_factor": 1.0,
            },
            valid_value_range=(-1024, 3071),
        )

        vector_fields.append(registration_result.composed_vector_field)

    DVF_CLIM = (-40, 40)
    RECON_CLIM = (0.0, 0.035)
    CT_CLIM = (-1024, 700)

    slicing = np.index_exp[75:-75, 464 // 2, 45:-45]
    fig, ax = plt.subplots(2, 4, sharex=True, sharey=True)
    ax[0, 0].imshow(vector_fields[0][2][slicing], cmap="seismic", clim=DVF_CLIM)
    ax[0, 0].set_title("CT")
    ax[0, 1].imshow(vector_fields[1][2][slicing], cmap="seismic", clim=DVF_CLIM)
    ax[0, 1].set_title("Rooster")
    ax[0, 2].imshow(vector_fields[2][2][slicing], cmap="seismic", clim=DVF_CLIM)
    ax[0, 2].set_title("CG")

    ax[0, 3].imshow(
        vector_fields[1][2][slicing] - vector_fields[2][2][slicing],
        cmap="seismic",
        clim=DVF_CLIM,
    )
    ax[0, 3].set_title("diff(Rooster - CG)")

    ax[1, 0].imshow(ct_images[0][slicing], cmap="gray", clim=CT_CLIM)
    ax[1, 1].imshow(rooster_images[0][slicing], cmap="gray", clim=RECON_CLIM)
    ax[1, 2].imshow(cg_images[0][slicing], cmap="gray", clim=RECON_CLIM)

    OUTPUT_FOLDER = Path("/mnt/nas_io/anarchy/4d_cbct_mc/publication_figures_phiro_reg")
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    save_image(
        np.rot90(ct_images[0][slicing]),
        cmap="gray",
        clim=CT_CLIM,
        filepath=OUTPUT_FOLDER / "4d_ct_phase_0.png",
    )
    save_image(
        np.rot90(rooster_images[0][slicing]),
        cmap="gray",
        clim=RECON_CLIM,
        filepath=OUTPUT_FOLDER / "4d_cbct_mc_rooster_phase_0.png",
    )
    save_image(
        np.rot90(cg_images[0][slicing]),
        cmap="gray",
        clim=RECON_CLIM,
        filepath=OUTPUT_FOLDER / "4d_cbct_mc_cg_phase_0.png",
    )

    save_image(
        np.rot90(vector_fields[0][2][slicing]),
        cmap="seismic",
        clim=DVF_CLIM,
        filepath=OUTPUT_FOLDER / "4d_ct_dvf.png",
    )
    save_image(
        np.rot90(vector_fields[1][2][slicing]),
        cmap="seismic",
        clim=DVF_CLIM,
        filepath=OUTPUT_FOLDER / "4d_cbct_mc_rooster_dvf.png",
    )
    save_image(
        np.rot90(vector_fields[2][2][slicing]),
        cmap="seismic",
        clim=DVF_CLIM,
        filepath=OUTPUT_FOLDER / "4d_cbct_mc_cg_dvf.png",
    )

    save_image(
        np.rot90(vector_fields[1][2][slicing] - vector_fields[2][2][slicing]),
        cmap="seismic",
        clim=DVF_CLIM,
        filepath=OUTPUT_FOLDER / "4d_cbct_mc_diff_dvf.png",
    )
