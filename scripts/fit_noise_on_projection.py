import logging
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import scipy.optimize as opt
import SimpleITK as sitk
from ipmi.common.logger import init_fancy_logging

from cbctmc.mc.geometry import MCWaterPhantomGeometry
from cbctmc.mc.simulation import MCSimulation

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--output-folder",
    help="Output folder for (intermediate) results",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=Path("."),
    show_default=True,
)
@click.option(
    "--gpu",
    help="GPU PCI bus ID to use for simulation",
    type=int,
    default=0,
    show_default=True,
)
@click.option(
    "--initial-n-histories",
    help="Initial number of histories",
    type=int,
    default=3.5e10,
    show_default=True,
)
@click.option(
    "--n-runs",
    help="Number of runs to average over for each simulation configuration",
    type=int,
    default=1,
    show_default=True,
)
@click.option(
    "--loglevel",
    type=click.Choice(["debug", "info", "warning", "error", "critical"]),
    default="debug",
    show_default=True,
)
def run(
    output_folder: Path,
    gpu: int,
    initial_n_histories: int,
    n_runs: int,
    loglevel: str,
):
    # set up logging
    loglevel = getattr(logging, loglevel.upper())
    logging.getLogger("cbctmc").setLevel(loglevel)
    logger.setLevel(loglevel)
    init_fancy_logging()

    function = lambda x: calculate_variance_deviation(
        n_histories=int(x),
        output_folder=output_folder,
        gpu=gpu,
        number_runs=n_runs,
    )
    res = opt.minimize(
        function,
        x0=np.array(initial_n_histories),
        method="BFGS",
        options={"eps": initial_n_histories / 20},
    )

    logger.info(f"Optimization finished with following result for n_histories: {res.x}")


def calculate_variance_deviation(
    n_histories: int,
    output_folder: Path,
    gpu: int,
    number_runs: int,
):
    if not output_folder.exists():
        # create output folder
        output_folder.mkdir(parents=True, exist_ok=True)

    n_projections = 1
    for i in range(number_runs):
        simulation_config = {
            "n_histories": n_histories,
            "n_projections": n_projections,
            "angle_between_projections": 360.0 / n_projections,
            "random_seed": datetime.now().microsecond,
        }
        logger.info(
            f"Starting simulation {i+1}/{number_runs} with {simulation_config=}"
        )

        output_folder.mkdir(parents=True, exist_ok=True)
        run_folder = f"run_{n_histories}_run_{i:02d}"

        (output_folder / run_folder).mkdir(exist_ok=True)

        # MC simulate Cat Phan 604
        phantom = MCWaterPhantomGeometry(shape=(250, 250, 150))
        if not any((output_folder / run_folder).iterdir()):
            phantom.save_material_segmentation(
                output_folder / run_folder / "catphan_604_materials.nii.gz"
            )
            phantom.save_density_image(
                output_folder / run_folder / "catphan_604_densities.nii.gz"
            )

        simulation = MCSimulation(geometry=phantom, **simulation_config)
        simulation.run_simulation(
            output_folder / run_folder,
            run_air_simulation=True,
            clean=True,
            gpu_id=gpu,
            force_rerun=True,
        )

        projections = sitk.ReadImage(
            str(output_folder / run_folder / "projections_total_normalized.mha")
        )

    # rel_dev = (mean_roi_stats - reference_roi_stats) / reference_roi_stats
    # rel_dev = np.abs(rel_dev)
    # mean_rel_dev = np.mean(rel_dev)
    #
    # logger.info(f"Current deviation: {mean_rel_dev} for {n_histories} histories")
    #
    # return mean_rel_dev


if __name__ == "__main__":
    run()
