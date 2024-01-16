from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
from torch import nn

from cbctmc.mc.geometry import MCCIRSPhantomGeometry
from cbctmc.mc.materials import MATERIALS_125KEV
from cbctmc.segmentation.labels import LABELS
from cbctmc.segmentation.segmenter import MCSegmenter
from cbctmc.speedup.models import FlexUNet

if __name__ == "__main__":
    DEVICE = "cuda:0"

    output_folder = Path(
        "/data_l79a/fmadesta/4d_cbct/R4DCIRS/4DCT/phase_based_gt_curve"
    )
    output_filepath = "/data_l79a/fmadesta/4d_cbct/R4DCIRS/4DCT/phase_based_gt_curve/composed_phase_05.nii"

    ct_image = sitk.ReadImage(
        "/data_l79a/fmadesta/4d_cbct/R4DCIRS/4DCT/phase_based_gt_curve/phase_05.nii"
    )
    ct_image = sitk.GetArrayFromImage(ct_image)
    ct_image = np.swapaxes(ct_image, 0, 2).astype(np.float32)

    mirror_at = 260
    spine = sitk.ReadImage(
        "/data_l79a/fmadesta/4d_cbct/R4DCIRS/4DCT/phase_based_gt_curve/spine_area.nii.gz"
    )
    origin = spine.GetOrigin()
    spacing = spine.GetSpacing()
    direction = spine.GetDirection()
    spine = sitk.GetArrayFromImage(spine)
    spine = np.swapaxes(spine, 0, 2).astype(np.float32)
    spine_mask = spine.any(axis=2)
    padded_spine_mask = np.pad(
        spine_mask,
        ((0, 2 * mirror_at - spine_mask.shape[0]), (0, 0)),
        mode="constant",
        constant_values=0,
    )

    insert = sitk.ReadImage(
        "/data_l79a/fmadesta/4d_cbct/R4DCIRS/4DCT/phase_based_gt_curve/insert_phase_05.nii.gz"
    )
    insert = sitk.GetArrayFromImage(insert)
    insert = np.swapaxes(insert, 0, 2).astype(np.float32)

    ct_image_flipped_lr = np.flip(ct_image, axis=0)
    ct_image_flipped_lr = ct_image_flipped_lr.astype(np.float32)

    composed_image = np.concatenate(
        (ct_image[:mirror_at], np.flip(ct_image[:mirror_at], axis=0)), axis=0
    )

    composed_image[padded_spine_mask] = ct_image[spine_mask]

    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
    ax[0].imshow(ct_image[:, :, 35])
    ax[1].imshow(composed_image[:, :, 35])
    ax[2].imshow(spine_mask)

    composed_image = np.swapaxes(composed_image, 0, 2)
    composed_image = sitk.GetImageFromArray(composed_image)
    composed_image.SetOrigin(origin)
    composed_image.SetSpacing(spacing)
    composed_image.SetDirection(direction)
    sitk.WriteImage(composed_image, output_filepath)

    enc_filters = [32, 32, 32, 32]
    dec_filters = [32, 32, 32, 32]

    model = FlexUNet(
        n_channels=1,
        n_classes=len(LABELS),
        n_levels=4,
        n_filters=[32, *enc_filters, *dec_filters, 32],
        convolution_layer=nn.Conv3d,
        downsampling_layer=nn.MaxPool3d,
        upsampling_layer=nn.Upsample,
        norm_layer=nn.InstanceNorm3d,
        skip_connections=True,
        convolution_kwargs=None,
        downsampling_kwargs=None,
        upsampling_kwargs=None,
        return_bottleneck=False,
    )
    state = torch.load(
        "/mnt/nas_io/anarchy/4d_cbct_mc/segmenter/2023-09-21T17:18:03.218908_run_39a7956b4719411f99ddf071__step_95000.pth"
    )
    model.load_state_dict(state["model"])

    segmenter = MCSegmenter(
        model=model,
        device=DEVICE,
        patch_shape=(288, 288, 32),
        patch_overlap=0.25,
    )

    geometry = MCCIRSPhantomGeometry.from_image(
        image_filepath=output_filepath,
        segmenter=segmenter,
        image_spacing=(1.0, 1.0, 1.0),
    )

    def create_spherical_mask(
        radius: float, shape: Tuple[int, int, int], sphere_center: Tuple[int, int, int]
    ):
        x = np.arange(0, shape[0])
        y = np.arange(0, shape[1])
        z = np.arange(0, shape[2])
        x, y, z = np.meshgrid(x, y, z, indexing="ij")
        mask = (x - sphere_center[0]) ** 2 + (y - sphere_center[1]) ** 2 + (
            z - sphere_center[2]
        ) ** 2 <= radius**2

        return mask

    def create_cirs_insert(
        shape: Tuple[int, int, int], insert_center: Tuple[int, int, int]
    ):
        radius = 15.0
        sphere_mask = create_spherical_mask(
            radius=radius, shape=shape, sphere_center=insert_center
        )

        insert_center = np.array(insert_center)

        cylinder_center = insert_center + np.array([0, 0, radius / 2])
        cylinder_radius = 1.5
        cylinder_height = radius
        x = np.arange(0, shape[0])
        y = np.arange(0, shape[1])
        z = np.arange(0, shape[2])
        x, y, z = np.meshgrid(x, y, z, indexing="ij")
        cutout_mask = (
            (
                (x - cylinder_center[0]) ** 2 + (y - cylinder_center[1]) ** 2
                <= cylinder_radius**2
            )
            & (z >= cylinder_center[2] - cylinder_height / 2)
            & (z <= cylinder_center[2] + cylinder_height / 2)
        )
        sphere_mask[cutout_mask] = False

        return sphere_mask

    cirs_insert = create_cirs_insert(
        shape=geometry.image_shape,
        insert_center=(238, 141, 71),
    )

    geometry.save(output_folder / "base_geometry.pkl.gz")
    geometry.save_material_segmentation(
        output_folder / "base_geometry_materials.nii.gz"
    )
    geometry.save_density_image(output_folder / "base_geometry_densities.nii.gz")

    geometry_with_insert = geometry.copy()
    geometry_with_insert.materials[cirs_insert] = MATERIALS_125KEV["soft_tissue"].number
    geometry_with_insert.densities[cirs_insert] = MATERIALS_125KEV[
        "soft_tissue"
    ].density
    geometry_with_insert.save_material_segmentation(
        output_folder / "geometry_with_insert_materials.nii.gz"
    )
    geometry_with_insert.save_density_image(
        output_folder / "geometry_with_insert_densities.nii.gz"
    )
    geometry_with_insert.save(output_folder / "geometry_with_insert.pkl.gz")
