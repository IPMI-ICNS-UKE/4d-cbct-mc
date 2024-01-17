from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Sequence, Tuple

import click
import numpy as np
import SimpleITK
import torch
import yaml
from ipmi.common.logger import init_fancy_logging
from torch import nn

from cbctmc.mc.respiratory import RespiratorySignal
from cbctmc.reconstruction.reconstruction import reconstruct_3d
from cbctmc.registration.correspondence import CorrespondenceModel
from cbctmc.utils import get_asset_filepath, get_folders_by_regex

# order GPU ID by PCI bus ID
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import faulthandler

import itk

from cbctmc.defaults import DefaultMCSimulationParameters as MCDefaults
from cbctmc.defaults import DefaultReconstructionParameters as ReconDefaults
from cbctmc.defaults import DefaultVarianScanParameters as VarianDefaults
from cbctmc.forward_projection import (
    create_geometry,
    prepare_image_for_rtk,
    project_forward,
    save_geometry,
)
from cbctmc.mc.geometry import MCCatPhan604Geometry, MCCIRSPhantomGeometry, MCGeometry
from cbctmc.mc.simulation import MCSimulation, MCSimulation4D
from cbctmc.segmentation.labels import LABELS
from cbctmc.segmentation.segmenter import MCSegmenter
from cbctmc.segmentation.utils import (
    merge_upper_body_bone_segmentations,
    merge_upper_body_fat_segmentations,
    merge_upper_body_muscle_segmentations,
)
from cbctmc.speedup.models import FlexUNet

faulthandler.enable()


@click.command()
@click.option(
    "--data-folder",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
)
@click.option(
    "--regex",
    type=str,
    default=".*",
)
@click.option(
    "--geometry-filepath",
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    default=None,
)
@click.option(
    "--output-folder",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--gpu",
    help="GPU PCI bus ID to use for simulation",
    type=int,
    default=(0,),
    multiple=True,
    show_default=True,
)
@click.option("--reference", is_flag=True)
@click.option(
    "--reference-n-histories",
    type=click.INT,
    default=MCDefaults.n_histories,
)
@click.option(
    "--speedups",
    type=float,
    multiple=True,
    default=[],
)
@click.option(
    "--phases",
    type=int,
    multiple=True,
    default=[0],
)
@click.option(
    "--segmenter-weights",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
@click.option(
    "--segmenter-patch-shape",
    type=click.Tuple([int, int, int]),
    default=(496, 496, 128),
)
@click.option(
    "--segmenter-patch-overlap",
    type=click.FloatRange(0.0, 1.0, min_open=True, max_open=False),
    default=0.25,
)
@click.option(
    "--n-projections",
    type=int,
    default=MCDefaults.n_projections,
)
@click.option("--reconstruct", is_flag=True)
@click.option("--forward-projection", is_flag=True)
@click.option("--no-clean", is_flag=True)
@click.option(
    "--correspondence-model",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
    default=None,
)
@click.option(
    "--respiratory-signal",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
    default=None,
)
@click.option(
    "--respiratory-signal-quantization",
    type=int,
    default=None,
)
@click.option(
    "--respiratory-signal-scaling",
    type=float,
    default=1.0,
)
@click.option(
    "--cirs-phantom",
    is_flag=True,
)
@click.option(
    "--catphan-phantom",
    is_flag=True,
)
@click.option(
    "--dry-run",
    is_flag=True,
)
@click.option(
    "--loglevel",
    type=click.Choice(["debug", "info", "warning", "error", "critical"]),
    default="info",
)
def run(
    data_folder: Path | None,
    regex: str | None,
    geometry_filepath: Path | None,
    output_folder: Path,
    gpu: Sequence[int],
    reference: bool,
    reference_n_histories: int,
    phases: List[int],
    speedups: List[int],
    segmenter_weights: Path,
    segmenter_patch_shape: Tuple[int, int, int],
    segmenter_patch_overlap: float,
    n_projections: int,
    reconstruct: bool,
    forward_projection: bool,
    no_clean: bool,
    correspondence_model: Path | None,
    respiratory_signal: Path | None,
    respiratory_signal_quantization: int | None,
    respiratory_signal_scaling: float,
    cirs_phantom: bool,
    catphan_phantom: bool,
    dry_run: bool,
    loglevel: str,
):
    # set up logging
    loglevel = getattr(logging, loglevel.upper())
    logging.getLogger("cbctmc").setLevel(loglevel)
    logger = logging.getLogger(__name__)
    logger.setLevel(loglevel)
    init_fancy_logging()

    CONFIGS = {}
    if reference:
        CONFIGS["reference"] = {
            "n_histories": reference_n_histories,
            "n_projections": n_projections,
            "angle_between_projections": 360.0 / n_projections,
        }
    CONFIGS.update(
        {
            f"speedup_{s:.2f}x": {
                "n_histories": int(MCDefaults.n_histories / s),
                "n_projections": n_projections,
                "angle_between_projections": 360.0 / n_projections,
            }
            for s in speedups
        }
    )
    if not CONFIGS:
        logger.warning(
            "No simulation configs specified. "
            "Please use --reference and/or --speedups to specify runs"
        )
        return

    is_4d = correspondence_model is not None and respiratory_signal is not None
    if is_4d:
        mc_simulation_class = MCSimulation4D
        logger.info(f"This is a 4D simulation, thus using {mc_simulation_class}")
        logger.info(f"Load correspondence model from {correspondence_model}")
        correspondence_model = CorrespondenceModel.load(correspondence_model)
        logger.info(f"Load respiratory signal from {respiratory_signal}")
        respiratory_signal = RespiratorySignal.load(respiratory_signal)

        if phases != (correspondence_model.reference_phase,):
            logger.warning(
                f"Phases {phases} do not match "
                f"{correspondence_model.reference_phase=}. Overwrite phases to "
                f"{[correspondence_model.reference_phase]}"
            )
            phases = [correspondence_model.reference_phase]

        if respiratory_signal_scaling != 1:
            logger.info(
                f"Scale respiratory signal by factor {respiratory_signal_scaling}"
            )
            respiratory_signal.signal *= respiratory_signal_scaling
            respiratory_signal.dt_signal *= respiratory_signal_scaling

        # add correspondence model to each config entry
        for config in CONFIGS.values():
            config["correspondence_model"] = correspondence_model
    else:
        # 3D MC simulation, i.e. no correspondence model and no respiratory signal
        mc_simulation_class = MCSimulation
        logger.info(f"This is a 3D simulation, thus using {mc_simulation_class}")
        respiratory_signal = None

    logger.info(f"Simulation configs: {CONFIGS}")
    if segmenter_weights:
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
        state = torch.load(segmenter_weights)
        model.load_state_dict(state["model"])

        segmenter = MCSegmenter(
            model=model,
            device=f"cuda:{gpu[0]}",
            patch_shape=segmenter_patch_shape,
            patch_overlap=segmenter_patch_overlap,
        )
        logger.info(f"Segmenter loaded from weights {segmenter_weights}")
    else:
        segmenter = None

    if catphan_phantom:
        geometry_filepath = get_asset_filepath("geometries/catphan604_geometry.pkl.gz")
        geometry_class = MCCatPhan604Geometry
    elif cirs_phantom:
        geometry_class = MCCIRSPhantomGeometry
    else:
        geometry_class = MCGeometry

    if geometry_filepath:
        patients = [Path(geometry_filepath).parent]
    else:
        patients = sorted(get_folders_by_regex(data_folder, regex=regex))

    logger.info(
        f"Found {len(patients)} patients using "
        f"data folder {data_folder} and regex pattern {regex}"
    )

    for patient_folder in patients:
        for phase in phases:
            logger.info(
                f"Prepare simulation for patient {patient_folder.name} and phase {phase}"
            )
            simulation_folder = (
                output_folder / patient_folder.name / f"phase_{phase:02d}"
            )
            simulation_folder.mkdir(parents=True, exist_ok=True)

            geometry = None
            if geometry_filepath is not None:
                logger.info(f"Load geometry from {geometry_filepath}")

                geometry = geometry_class.load(geometry_filepath)
                geometry.save_material_segmentation(
                    simulation_folder / "geometry_materials.nii.gz"
                )
                geometry.save_density_image(
                    simulation_folder / "geometry_densities.nii.gz"
                )
                geometry.save(simulation_folder / "geometry.pkl.gz")
            else:
                possible_extensions = [".nii", ".nii.gz"]
                for extension in possible_extensions:
                    image_filepath = patient_folder / f"phase_{phase:02d}{extension}"
                    if image_filepath.exists():
                        break
                else:
                    raise ValueError(
                        f"Could not find image for phase {phase} in {patient_folder}"
                    )

            geometry_already_prepared = all(
                (
                    (simulation_folder / "geometry_materials.nii.gz").exists(),
                    (simulation_folder / "geometry_densities.nii.gz").exists(),
                    (simulation_folder / "geometry.pkl.gz").exists(),
                )
            )

            logger.info(f"Geometry already prepared: {geometry_already_prepared}")

            if not geometry_already_prepared:
                if segmenter is None:
                    segmentation_folder = (
                        patient_folder / f"segmentations/phase_{phase:02d}"
                    )
                    logger.info("Create geometry using existing segmentations")
                    # use TotalSegmentator segmentations

                    if not (patient_folder / "body.nii.gz").exists():
                        merge_upper_body_bone_segmentations(segmentation_folder)
                        merge_upper_body_muscle_segmentations(segmentation_folder)
                        merge_upper_body_fat_segmentations(segmentation_folder)

                    geometry = geometry_class.from_image(
                        image_filepath=image_filepath,
                        body_segmentation_filepath=segmentation_folder / "body.nii.gz",
                        bone_segmentation_filepath=segmentation_folder
                        / "upper_body_bones.nii.gz",
                        muscle_segmentation_filepath=segmentation_folder
                        / "upper_body_muscles.nii.gz",
                        fat_segmentation_filepath=segmentation_folder
                        / "upper_body_fat.nii.gz",
                        liver_segmentation_filepath=segmentation_folder
                        / "liver.nii.gz",
                        stomach_segmentation_filepath=segmentation_folder
                        / "stomach.nii.gz",
                        lung_segmentation_filepath=segmentation_folder / "lung.nii.gz",
                        lung_vessel_segmentation_filepath=segmentation_folder
                        / "lung_vessels.nii.gz",
                    )
                else:
                    logger.info("Create geometry using segmentator")
                    # use segmetator
                    geometry = geometry_class.from_image(
                        image_filepath=image_filepath,
                        segmenter=segmenter,
                        image_spacing=(1.0, 1.0, 1.0),
                    )

                geometry.save_material_segmentation(
                    simulation_folder / "geometry_materials.nii.gz"
                )
                geometry.save_density_image(
                    simulation_folder / "geometry_densities.nii.gz"
                )
                geometry.save(simulation_folder / "geometry.pkl.gz")
            else:
                if geometry is None:
                    geometry = MCGeometry.load(simulation_folder / "geometry.pkl.gz")

            fp_geometry = create_geometry(start_angle=90, n_projections=n_projections)
            save_geometry(fp_geometry, simulation_folder / "geometry.xml")
            if (
                forward_projection
                and not (simulation_folder / "density_fp.mha").exists()
            ):
                logger.info("Perform forward projection")
                image = prepare_image_for_rtk(
                    image=geometry.densities,
                    image_spacing=geometry.image_spacing,
                    input_value_range=None,
                    output_value_range=None,
                )
                density_forward_projection = project_forward(
                    image,
                    geometry=fp_geometry,
                    detector_size=MCDefaults.n_detector_pixels_half_fan,
                    detector_pixel_spacing=MCDefaults.detector_pixel_size,
                )
                itk.imwrite(
                    density_forward_projection,
                    str(simulation_folder / "density_fp.mha"),
                )

            for config_name, config in CONFIGS.items():
                logger.info(
                    f"Run simulation with config {config_name} "
                    f"for patient {patient_folder.name} and phase {phase}"
                )

                additional_run_kwargs = {}
                if is_4d:
                    additional_run_kwargs["respiratory_signal"] = respiratory_signal
                    additional_run_kwargs[
                        "respiratory_signal_quantization"
                    ] = respiratory_signal_quantization

                simulation = mc_simulation_class(geometry=geometry, **config)
                simulation.run_simulation(
                    output_folder=simulation_folder / config_name,
                    run_air_simulation=True,
                    clean=not no_clean,
                    gpu_ids=gpu,
                    force_rerun=False,
                    dry_run=dry_run,
                    **additional_run_kwargs,
                )

                if not dry_run and is_4d and forward_projection:
                    logger.info("Perform 4D forward projection")

                    with open(
                        simulation_folder / config_name / "projection_geometries.yaml",
                        "rt",
                    ) as f:
                        logger.info("Load used projection geometries")
                        projection_geometries = yaml.safe_load(f)
                    forward_projections = []
                    for projection_angle, meta in projection_geometries.items():
                        # perform 4D forward projection using warped geometries
                        fp_geometry = create_geometry(
                            start_angle=projection_angle - 180.0, n_projections=1
                        )

                        logger.debug(
                            "Perform 4D forward projection for angle "
                            f"{projection_angle} and "
                            f"geometry {meta['geometry_filename']}"
                        )
                        geometry = geometry_class.load(
                            simulation_folder / config_name / meta["geometry_filename"]
                        )
                        image = prepare_image_for_rtk(
                            image=geometry.densities,
                            image_spacing=geometry.image_spacing,
                            input_value_range=None,
                            output_value_range=None,
                        )
                        density_forward_projection = project_forward(
                            image,
                            geometry=fp_geometry,
                            detector_size=MCDefaults.n_detector_pixels_half_fan,
                            detector_pixel_spacing=VarianDefaults.detector_pixel_size,
                        )
                        density_forward_projection = itk.array_from_image(
                            density_forward_projection
                        )
                        forward_projections.append(density_forward_projection[0])

                    forward_projections = np.stack(forward_projections, axis=0)
                    forward_projections = itk.image_from_array(forward_projections)
                    forward_projections.SetSpacing(
                        [
                            MCDefaults.detector_pixel_size[0],
                            MCDefaults.detector_pixel_size[1],
                            1.0,
                        ]
                    )
                    forward_projections.SetOrigin(
                        [
                            -0.5
                            * MCDefaults.n_detector_pixels_half_fan[0]
                            * MCDefaults.detector_pixel_size[0],
                            -0.5
                            * MCDefaults.n_detector_pixels_half_fan[1]
                            * MCDefaults.detector_pixel_size[1],
                            0.0,
                        ]
                    )
                    itk.imwrite(
                        forward_projections,
                        str(simulation_folder / config_name / "density_fp_4d.mha"),
                    )

                if not dry_run and reconstruct:
                    logger.info("Reconstruct simulation")
                    if (
                        simulation_folder
                        / config_name
                        / "reconstructions"
                        / "fdk3d.mha"
                    ).exists():
                        logger.info("Reconstruction already exists, skip")
                    else:
                        reconstruct_3d(
                            projections_filepath=(
                                simulation_folder
                                / config_name
                                / "projections_total_normalized.mha"
                            ),
                            geometry_filepath=simulation_folder / "geometry.xml",
                            output_folder=(
                                simulation_folder / config_name / "reconstructions"
                            ),
                            output_filename="fdk3d_wpc.mha",
                            dimension=(464, 250, 464),
                            water_pre_correction=ReconDefaults.wpc_catphan604,
                            gpu_id=gpu[0],
                        )


if __name__ == "__main__":
    run()

    # # for debugging
    # run(
    #     [
    #         "--data-folder",
    #         "/datalake_fast/4d_ct_lung_uke_artifact_free",
    #         "--output-folder",
    #         "/datalake_fast/mc_output/4d",
    #         "--phases",
    #         "0",
    #         "--gpu",
    #         "0",
    #         "--speedups",
    #         "20.0",
    #         "--regex",
    #         "024.*",
    #         "--segmenter-weights",
    #         "/mnt/nas_io/anarchy/4d_cbct_mc/segmenter/2023-09-21T17:18:03.218908_run_39a7956b4719411f99ddf071__step_95000.pth",
    #         "--segmenter-patch-overlap",
    #         "0.25",
    #         "--segmenter-patch-shape",
    #         "496",
    #         "496",
    #         "32",
    #         "--correspondence-model",
    #         "/mnt/nas_io/anarchy/4d_cbct_mc/024_correspondence_model.pkl",
    #         "--respiratory-signal",
    #         "/mnt/nas_io/anarchy/4d_cbct_mc/024_respiratory_signal.pkl",
    #         "--respiratory-signal-quantization",
    #         "5",
    #         "--reconstruct",
    #         "--no-clean",
    #         "--loglevel",
    #         "debug",
    #         "--n-projections",
    #         "100",
    #     ]
    # )


# --data-folder /mnt/nas_io/anarchy/4d_cbct_mc/4d_ct_lung_uke_artifact_free --output-folder /mnt/nas_io/anarchy/4d_cbct_mc/4d --phases 0 --gpu 0 --gpu 1 --gpu 2 --regex 024.* --segmenter-weights /mnt/nas_io/anarchy/4d_cbct_mc/segmenter/2023-09-21T17:18:03.218908_run_39a7956b4719411f99ddf071__step_95000.pth --segmenter-patch-overlap 0.25 --segmenter-patch-shape 496 496 32 --correspondence-model /mnt/nas_io/anarchy/4d_cbct_mc/024_correspondence_model.pkl --respiratory-signal /mnt/nas_io/anarchy/4d_cbct_mc/024_respiratory_signal.pkl --respiratory-signal-quantization 20 --respiratory-signal-scaling 2 --reconstruct
