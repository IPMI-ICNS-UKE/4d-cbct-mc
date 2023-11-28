import logging
import os
import re
from pathlib import Path
from typing import List, Tuple

import click
import torch
from ipmi.common.logger import init_fancy_logging
from torch import nn

from cbctmc.reconstruction.reconstruction import reconstruct_3d
from cbctmc.utils import get_folders_by_regex

# order GPU ID by PCI bus ID
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import itk

from cbctmc.defaults import DefaultMCSimulationParameters
from cbctmc.defaults import DefaultMCSimulationParameters as MCDefaults
from cbctmc.defaults import DefaultReconstructionParameters as ReconDefaults
from cbctmc.forward_projection import (
    create_geometry,
    prepare_image_for_rtk,
    project_forward,
    save_geometry,
)
from cbctmc.mc.geometry import MCGeometry, MCLinePairPhantomGeometry
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
    "--output-folder",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--gpu",
    type=int,
    default=0,
)
@click.option(
    "--line-spacing-factor",
    type=int,
    required=True,
)
@click.option("--reference", is_flag=True)
@click.option(
    "--reference-n-histories",
    type=int,
    default=MCDefaults.n_histories,
)
@click.option(
    "--speedups",
    type=float,
    multiple=True,
    default=[],
)
@click.option(
    "--n-projections",
    type=int,
    default=MCDefaults.n_projections,
)
@click.option("--reconstruct", is_flag=True)
@click.option(
    "--loglevel",
    type=click.Choice(["debug", "info", "warning", "error", "critical"]),
    default="info",
)
def run(
    output_folder: Path,
    gpu: int,
    line_spacing_factor: int,
    reference: bool,
    reference_n_histories: int,
    speedups: List[int],
    n_projections: int,
    reconstruct: bool,
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

    logger.info(f"Simulation configs: {CONFIGS}")

    output_folder.mkdir(parents=True, exist_ok=True)
    simulation_folder = output_folder

    geometry = MCLinePairPhantomGeometry(line_spacing_factor=line_spacing_factor)
    geometry.save_material_segmentation(simulation_folder / "geometry_materials.nii.gz")
    geometry.save_density_image(simulation_folder / "geometry_densities.nii.gz")
    geometry.save(simulation_folder / "geometry.pkl.gz")

    image = prepare_image_for_rtk(
        image=geometry.densities,
        image_spacing=geometry.image_spacing,
        input_value_range=None,
        output_value_range=None,
    )
    logger.info("Perform forward projection")
    fp_geometry = create_geometry(start_angle=90, n_projections=n_projections)
    forward_projection = project_forward(
        image,
        geometry=fp_geometry,
    )
    save_geometry(fp_geometry, simulation_folder / "geometry.xml")
    itk.imwrite(
        forward_projection,
        str(simulation_folder / "density_fp.mha"),
    )

    for config_name, config in CONFIGS.items():
        logger.info(f"Run simulation with config {config_name} ")
        simulation = MCSimulation(geometry=geometry, **config)
        simulation.run_simulation(
            simulation_folder / config_name,
            run_air_simulation=True,
            clean=True,
            gpu_id=gpu,
            **DefaultMCSimulationParameters().geometrical_corrections,
            force_rerun=False,
        )

        if reconstruct:
            logger.info("Reconstruct simulation")
            reconstruct_3d(
                projections_filepath=(
                    simulation_folder / config_name / "projections_total_normalized.mha"
                ),
                geometry_filepath=simulation_folder / "geometry.xml",
                output_folder=(simulation_folder / config_name / "reconstructions"),
                output_filename="fdk3d_wpc.mha",
                dimension=(464, 250, 464),
                water_pre_correction=ReconDefaults.wpc_catphan604,
                gpu_id=gpu,
            )


if __name__ == "__main__":
    run()
