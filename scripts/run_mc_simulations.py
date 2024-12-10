from __future__ import annotations

import json
import logging
import os
from itertools import product
from pathlib import Path
from typing import List, Sequence, Tuple

import click
import numpy as np
import pkg_resources
import SimpleITK as sitk
import torch
import yaml
from ipmi.common.logger import init_fancy_logging
from torch import nn

from cbctmc.mc.respiratory import RespiratorySignal
from cbctmc.reconstruction import reconstruction
from cbctmc.registration.correspondence import CorrespondenceModel
from cbctmc.speedup.inference import MCSpeedup
from cbctmc.utils import get_asset_filepath

# order GPU ID by PCI bus ID
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


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
from cbctmc.speedup.models import FlexUNet

_DEFAULT_SEGMENTER_WEIGHTS = Path(
    pkg_resources.resource_filename("cbctmc", f"assets/weights/segmenter/default.pth")
)

_DEFAULT_SPEEDUP_WEIGHTS = Path(
    pkg_resources.resource_filename("cbctmc", f"assets/weights/speedup/default.pth")
)


def _reconstruct_mc_simulation(
    simulation_folder,
    config_name,
    gpu_id: int,
    reconstruct_3d: bool,
    reconstruct_4d: bool,
    suffix: str,
    logger,
):
    """Wrapper function for reconstructing a single MC simulation."""
    if reconstruct_3d:
        logger.info("Reconstruct 3D simulation")
        reconstruction.reconstruct_3d(
            projections_filepath=(
                simulation_folder
                / config_name
                / f"projections_total_normalized{suffix}.mha"
            ),
            geometry_filepath=simulation_folder / "geometry.xml",
            output_folder=(simulation_folder / config_name / "reconstructions"),
            output_filename=f"fdk3d_wpc{suffix}.mha",
            dimension=(464, 250, 464),
            water_pre_correction=ReconDefaults.wpc_catphan604,
            gpu_id=gpu_id,
        )

    if reconstruct_4d:
        logger.info("Reconstruct 4D simulation")

        signal = np.loadtxt(simulation_folder / config_name / "signal.txt")
        reconstruction.reconstruct_4d(
            amplitude_signal=signal[:, 0],
            projections_filepath=(
                simulation_folder
                / config_name
                / f"projections_total_normalized{suffix}.mha"
            ),
            geometry_filepath=simulation_folder / "geometry.xml",
            output_folder=(simulation_folder / config_name / "reconstructions"),
            output_filename=f"rooster4d_wpc{suffix}.mha",
            dimension=(464, 250, 464),
            water_pre_correction=ReconDefaults.wpc_catphan604,
            gpu_id=gpu_id,
        )


@click.command()
@click.option(
    "--image-filepath",
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    show_default=True,
    help="CT image to use for simulation",
)
@click.option(
    "--geometry-filepath",
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    show_default=True,
    help="Geometry to use for simulation. Can be provided instead of CT image.",
)
@click.option(
    "--output-folder",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    show_default=True,
    help="Output folder for simulation results",
)
@click.option(
    "--simulation-name",
    type=str,
    default=None,
    show_default=True,
    help="Name of the simulation. If not provided, the name is derived from the image filepath.",
)
@click.option(
    "--gpu",
    type=int,
    default=(0,),
    multiple=True,
    show_default=True,
    help="GPU PCI bus ID to use for simulation (can be checked with nvidia-smi)",
)
@click.option("--reference", is_flag=True, help="Enable reference simulation")
@click.option(
    "--reference-n-histories",
    type=click.INT,
    default=MCDefaults.n_histories,
    show_default=True,
    help="Number of histories for reference simulation",
)
@click.option(
    "--speedups",
    type=float,
    multiple=True,
    default=[],
    show_default=True,
    help="Speedup factors for simulation",
)
@click.option(
    "--speedup-weights",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
    default=_DEFAULT_SPEEDUP_WEIGHTS,
    show_default=True,
    help="Weights file for speedup model",
)
@click.option(
    "--segmenter-weights",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
    help="Weights file for the segmenter model",
    default=_DEFAULT_SEGMENTER_WEIGHTS,
)
@click.option(
    "--segmenter-patch-shape",
    type=click.Tuple([int, int, int]),
    default=(256, 256, 128),
    show_default=True,
    help="Patch shape for the segmenter model",
)
@click.option(
    "--segmenter-patch-overlap",
    type=click.FloatRange(0.0, 1.0, min_open=True, max_open=False),
    default=0.50,
    show_default=True,
    help="Overlap ratio for patch-based segmentation",
)
@click.option(
    "--n-projections",
    type=int,
    default=MCDefaults.n_projections,
    show_default=True,
    help="Number of projections for simulation",
)
@click.option("--reconstruct-3d", is_flag=True, help="Enable 3D reconstruction")
@click.option("--reconstruct-4d", is_flag=True, help="Enable 4D reconstruction")
@click.option("--forward-projection", is_flag=True, help="Enable forward projection")
@click.option("--no-clean", is_flag=True, help="Disable cleaning of intermediate files")
@click.option(
    "--correspondence-model",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
    default=None,
    show_default=True,
    help="Correspondence model file. Must be provided for 4D simulation.",
)
@click.option(
    "--respiratory-signal",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
    default=None,
    show_default=True,
    help="Respiratory signal file. Must be provided for 4D simulation.",
)
@click.option(
    "--respiratory-signal-quantization",
    type=int,
    default=None,
    show_default=True,
    help="Quantization level for respiratory signal. A lower value means that the respiratory signal is more coarse.",
)
@click.option(
    "--respiratory-signal-scaling",
    type=float,
    default=1.0,
    show_default=True,
    help="Scaling factor for respiratory signal",
)
@click.option(
    "--precompile-geometries",
    is_flag=True,
    help="Precompile geometries for 4D simulation",
)
@click.option("--cirs-phantom", is_flag=True, help="Use CIRS phantom for simulation")
@click.option(
    "--catphan-phantom", is_flag=True, help="Use Catphan604 phantom for simulation"
)
@click.option(
    "--dry-run", is_flag=True, help="Perform a dry run without executing the simulation"
)
@click.option(
    "--random-seed",
    type=int,
    default=42,
    show_default=True,
    help="Random seed for simulation",
)
@click.option(
    "--loglevel",
    type=click.Choice(["debug", "info", "warning", "error", "critical"]),
    default="info",
    show_default=True,
    help="Logging level",
)
def run(
    image_filepath: Path | None,
    geometry_filepath: Path | None,
    output_folder: Path,
    simulation_name: str | None,
    gpu: Sequence[int],
    reference: bool,
    reference_n_histories: int,
    speedups: List[int],
    speedup_weights: Path | None,
    segmenter_weights: Path | None,
    segmenter_patch_shape: Tuple[int, int, int],
    segmenter_patch_overlap: float,
    n_projections: int,
    reconstruct_3d: bool,
    reconstruct_4d: bool,
    forward_projection: bool,
    no_clean: bool,
    correspondence_model: Path | None,
    respiratory_signal: Path | None,
    respiratory_signal_quantization: int | None,
    respiratory_signal_scaling: float,
    precompile_geometries: bool,
    cirs_phantom: bool,
    catphan_phantom: bool,
    dry_run: bool,
    random_seed: int,
    loglevel: str,
):
    # set up logging
    loglevel = getattr(logging, loglevel.upper())
    logging.getLogger("cbctmc").setLevel(loglevel)
    logger = logging.getLogger(__name__)
    logger.setLevel(loglevel)
    init_fancy_logging()

    if speedups:
        if not forward_projection:
            raise ValueError("Speedup requires forward projection")
        if not speedup_weights:
            raise ValueError("Speedup requires weights file")

    CONFIGS = {}
    if reference:
        CONFIGS["reference"] = {
            "n_histories": reference_n_histories,
            "n_projections": n_projections,
            "angle_between_projections": 360.0 / n_projections,
            "random_seed": random_seed,
        }
    CONFIGS.update(
        {
            f"speedup_{s:.2f}x": {
                "n_histories": int(MCDefaults.n_histories / s),
                "n_projections": n_projections,
                "angle_between_projections": 360.0 / n_projections,
                "random_seed": random_seed,
            }
            for s in speedups
        }
    )
    if not CONFIGS:
        raise ValueError(
            "No simulation configs specified. "
            "Please use --reference and/or --speedups to specify runs"
        )

    is_4d = correspondence_model is not None and respiratory_signal is not None
    if is_4d:
        mc_simulation_class = MCSimulation4D
        logger.info(
            f"This is a 4D simulation, thus using {mc_simulation_class.__name__}"
        )
        logger.info(f"Load correspondence model from {correspondence_model}")
        correspondence_model = CorrespondenceModel.load(correspondence_model)
        logger.info(
            f"Correspondece model reference phase is "
            f"{correspondence_model.reference_phase}. "
            f"Please make sure that you load the correct CT phase image or "
            f"corresponding geometry."
        )

        logger.info(f"Load respiratory signal from {respiratory_signal}")
        respiratory_signal = RespiratorySignal.load(respiratory_signal)

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
        logger.info(
            f"This is a 3D simulation, thus using {mc_simulation_class.__name__}"
        )
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

    image_folder = image_filepath.parent

    logger.info(f"Prepare simulation for image {image_filepath}")

    if not simulation_name:
        simulation_name = f"{image_folder.name}_{image_filepath.stem}"
    simulation_folder = output_folder / simulation_name
    simulation_folder.mkdir(parents=True, exist_ok=True)

    geometry = None
    if geometry_filepath is not None:
        logger.info(f"Load geometry from {geometry_filepath}")

        geometry = geometry_class.load(geometry_filepath)
        geometry.save_material_segmentation(
            simulation_folder / "geometry_materials.nii.gz"
        )
        geometry.save_density_image(simulation_folder / "geometry_densities.nii.gz")
        geometry.save(simulation_folder / "geometry.pkl.gz")

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
            raise ValueError(
                "Segmenter is None, thus geometry has to be provided "
                "via --geometry-filepath"
            )
        else:
            logger.info("Create geometry using segmentator")
            # use segmenter
            geometry = geometry_class.from_image(
                image_filepath=image_filepath,
                segmenter=segmenter,
                image_spacing=(1.0, 1.0, 1.0),
            )

        geometry.save_material_segmentation(
            simulation_folder / "geometry_materials.nii.gz"
        )
        geometry.save_density_image(simulation_folder / "geometry_densities.nii.gz")
        geometry.save(simulation_folder / "geometry.pkl.gz")
    elif geometry is None:
        geometry = MCGeometry.load(simulation_folder / "geometry.pkl.gz")

    fp_geometry = create_geometry(start_angle=90, n_projections=n_projections)
    save_geometry(fp_geometry, simulation_folder / "geometry.xml")
    if forward_projection and not (simulation_folder / "density_fp.mha").exists():
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
            f"Run simulation with config {config_name} " f"for image {image_filepath}"
        )

        additional_run_kwargs = {}
        if is_4d:
            additional_run_kwargs["respiratory_signal"] = respiratory_signal
            additional_run_kwargs[
                "respiratory_signal_quantization"
            ] = respiratory_signal_quantization
            additional_run_kwargs["geometry_output_folder"] = (
                simulation_folder
                / f"4d_geometries_{correspondence_model.model_hash[:7]}"
            )
            additional_run_kwargs["precompile_geometries"] = precompile_geometries

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
                    additional_run_kwargs["geometry_output_folder"]
                    / meta["geometry_filename"]
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

        perform_speedup = "speedup" in config_name

        if not dry_run and perform_speedup:
            logger.info(
                f"Perform simulation speedup. Load speedup model from {speedup_weights}"
            )
            speedup = MCSpeedup.from_filepath(
                model_filepath=speedup_weights,
                device=f"cuda:{gpu[0]}",
            )
            low_photon_projections_filepath = (
                simulation_folder / config_name / "projections_total_normalized.mha"
            )
            if is_4d:
                forward_projection_filepath = (
                    simulation_folder / config_name / "density_fp_4d.mha"
                )
            else:
                forward_projection_filepath = simulation_folder / "density_fp.mha"
            speedup_projections = speedup.execute(
                low_photon=low_photon_projections_filepath,
                forward_projection=forward_projection_filepath,
                batch_size=1,
            )
            speedup_projections_filepath = (
                simulation_folder
                / config_name
                / "projections_total_normalized_speedup.mha"
            )
            sitk.WriteImage(speedup_projections, str(speedup_projections_filepath))

        if not dry_run:
            reconstruct_4d = is_4d and reconstruct_4d

            _reconstruct_mc_simulation(
                simulation_folder=simulation_folder,
                config_name=config_name,
                gpu_id=gpu[0],
                reconstruct_3d=reconstruct_3d,
                reconstruct_4d=reconstruct_4d,
                suffix="",
                logger=logger,
            )

            if perform_speedup:
                _reconstruct_mc_simulation(
                    simulation_folder=simulation_folder,
                    config_name=config_name,
                    gpu_id=gpu[0],
                    reconstruct_3d=reconstruct_3d,
                    reconstruct_4d=reconstruct_4d,
                    suffix="_speedup",
                    logger=logger,
                )


if __name__ == "__main__":
    run()
