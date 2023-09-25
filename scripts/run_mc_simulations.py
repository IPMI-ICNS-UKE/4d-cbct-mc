import logging
import os
import re
from pathlib import Path
from typing import List, Tuple

import click
import torch
from ipmi.common.logger import init_fancy_logging
from torch import nn

from cbctmc.utils import get_folders_by_regex

# order GPU ID by PCI bus ID
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import itk

from cbctmc.defaults import DefaultMCSimulationParameters as MCDefaults
from cbctmc.forward_projection import (
    create_geometry,
    prepare_image_for_rtk,
    project_forward,
    save_geometry,
)
from cbctmc.mc.geometry import MCGeometry
from cbctmc.mc.simulation import MCSimulation
from cbctmc.segmentation.labels import LABELS
from cbctmc.segmentation.segmenter import MCSegmenter
from cbctmc.segmentation.utils import (
    merge_upper_body_bone_segmentations,
    merge_upper_body_fat_segmentations,
    merge_upper_body_muscle_segmentations,
)
from cbctmc.speedup.models import FlexUNet


@click.command()
@click.option(
    "--data-folder",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--regex",
    type=str,
    default=re.compile(".*"),
)
@click.option(
    "--output-folder",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--gpu",
    type=int,
    default=0,
)
@click.option(
    "--i-worker",
    type=int,
    default=1,
)
@click.option(
    "--n-workers",
    type=int,
    default=1,
)
@click.option("--reference", is_flag=True)
@click.option(
    "--speedups",
    type=int,
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
@click.option(
    "--loglevel",
    type=click.Choice(["debug", "info", "warning", "error", "critical"]),
    default="info",
)
def run(
    data_folder: Path,
    regex: str,
    output_folder: Path,
    gpu: int,
    i_worker: int,
    n_worker: int,
    reference: bool,
    phases: List[int],
    speedups: List[int],
    segmenter_weights: Path,
    segmenter_patch_shape: Tuple[int, int, int],
    segmenter_patch_overlap: float,
    n_projections: int,
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
        CONFIGS["reference"] = {"n_histories": int(MCDefaults.n_histories)}
    CONFIGS.update(
        {
            f"speedup_{s:02d}x": {
                "n_histories": int(MCDefaults.n_histories / s),
                "n_projections": n_projections,
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
            device=f"cuda:{gpu}",
            patch_shape=segmenter_patch_shape,
            patch_overlap=segmenter_patch_overlap,
        )
        logger.info(f"Segmenter loaded from weights {segmenter_weights}")
    else:
        segmenter = None

    patients = sorted(get_folders_by_regex(data_folder, regex=regex))

    logger.info(
        f"Found {len(patients)} patients using "
        f"data folder {data_folder} and regex pattern {regex}"
    )
    patients = patients[i_worker - 1 :: n_worker]

    logger.info(
        f"Running simulations for {len(patients)} patients "
        f"(worker {i_worker}/{n_worker})"
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

            image_filepath = patient_folder / f"phase_{phase:02d}.nii"

            simulation_already_prepared = all(
                (
                    (simulation_folder / "geometry_materials.nii.gz").exists(),
                    (simulation_folder / "geometry_densities.nii.gz").exists(),
                    (simulation_folder / "density_fp.mha").exists(),
                )
            )

            if not simulation_already_prepared:
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
                    geometry = MCGeometry.from_image(
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
                    geometry = MCGeometry.from_image(
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

                image = prepare_image_for_rtk(
                    image=geometry.densities,
                    image_spacing=geometry.image_spacing,
                    input_value_range=None,
                    output_value_range=None,
                )
                logger.info("Perform forward projection")
                fp_geometry = create_geometry(n_projections=n_projections)
                forward_projection = project_forward(
                    image,
                    geometry=fp_geometry,
                )
                save_geometry(fp_geometry, simulation_folder / "geometry.xml")
                itk.imwrite(
                    forward_projection,
                    str(simulation_folder / "density_fp.mha"),
                )

            else:
                geometry = MCGeometry.load(simulation_folder / "geometry.pkl.gz")

            for config_name, config in CONFIGS.items():
                logger.info(
                    f"Run simulation with config {config_name} "
                    f"for patient {patient_folder.name} and phase {phase}"
                )
                simulation = MCSimulation(geometry=geometry, **config)
                simulation.run_simulation(
                    simulation_folder / config_name,
                    run_air_simulation=True,
                    clean=True,
                    gpu_id=gpu,
                )


if __name__ == "__main__":
    run()
