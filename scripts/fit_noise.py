import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Sequence, Tuple

import click
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import SimpleITK as sitk
from ipmi.common.logger import init_fancy_logging
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

from cbctmc.defaults import DefaultMCSimulationParameters as MCDefaults
from cbctmc.defaults import DefaultReconstructionParameters as ReconDefaults
from cbctmc.defaults import DefaultVarianScanParameters
from cbctmc.forward_projection import create_geometry, save_geometry
from cbctmc.mc.geometry import MCCatPhan604Geometry, MCWaterPhantomGeometry
from cbctmc.mc.reference import REFERENCE_ROI_STATS_CATPHAN604_VARIAN
from cbctmc.mc.simulation import MCSimulation
from cbctmc.reconstruction.reconstruction import reconstruct_3d

logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


@cli.command()
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
    default=(0,),
    multiple=True,
    show_default=True,
)
@click.option(
    "--n-projections",
    type=int,
    default=DefaultVarianScanParameters.n_projections,
    show_default=True,
)
@click.option(
    "--material",
    type=str,
    multiple=True,
    default=(
        "air_1",
        "air_2",
        "pmp",
        "ldpe",
        "polystyrene",
        "bone_020",
        "acrylic",
        "bone_050",
        "delrin",
        "teflon",
        "water",
    ),
    show_default=True,
)
@click.option(
    "--optimize",
    help="Optimize number of histories automatically",
    is_flag=True,
    default=False,
    show_default=True,
)
@click.option(
    "--initial-n-histories",
    help="Initial number of histories",
    type=int,
    default=1e10,
    show_default=True,
)
@click.option(
    "--n-histories-range",
    help="Range of number of histories to run simulations over",
    type=int,
    nargs=2,
    default=(5e8, 5e10),
    show_default=True,
)
@click.option(
    "--n-simulations",
    help="Number of simulations to run over the n-histories-range",
    type=int,
    default=10,
    show_default=True,
)
@click.option(
    "--n-runs-per-setting",
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
    gpu: Sequence[int],
    n_projections: int,
    material: Sequence[str],
    optimize: bool,
    initial_n_histories: int,
    n_histories_range: Tuple[int, int],
    n_simulations: int,
    n_runs_per_setting: int,
    loglevel: str,
):
    # set up logging
    loglevel = getattr(logging, loglevel.upper())
    logging.getLogger("cbctmc").setLevel(loglevel)
    logger.setLevel(loglevel)
    init_fancy_logging()
    logger.info(f"Starting noise runs using material {material}")
    if optimize:
        logger.info("Optimizing number of histories")
        function = lambda x: calculate_variance_deviation(
            n_histories=int(x),
            materials=material,
            output_folder=output_folder,
            gpu=gpu,
            n_projections=n_projections,
            number_runs=n_runs_per_setting,
        )
        res = opt.minimize(
            function,
            x0=np.array(initial_n_histories),
            method="Nelder-Mead",
        )
        logger.info(
            f"Optimization finished with following result for n_histories: {res.x}"
        )
    else:
        n_histories = np.linspace(
            n_histories_range[0], n_histories_range[1], n_simulations, dtype=int
        )
        logger.info(f"Running simulations for following n_histories: {n_histories}")
        for _n_histories in n_histories:
            calculate_variance_deviation(
                n_histories=_n_histories,
                materials=material,
                output_folder=output_folder,
                gpu=gpu,
                n_projections=n_projections,
                number_runs=n_runs_per_setting,
            )


def calculate_variance_deviation(
    n_histories: int,
    materials: Sequence[str],
    output_folder: Path,
    gpu: Sequence[int],
    n_projections: int,
    number_runs: int,
):
    if not output_folder.exists():
        # create output folder
        output_folder.mkdir(parents=True, exist_ok=True)

    mean_roi_stats = np.zeros(11)

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

        # # MC simulate Cat Phan 604
        phantom = MCWaterPhantomGeometry(shape=(250, 250, 150))
        if not any((output_folder / run_folder).iterdir()):
            phantom.save_material_segmentation(
                output_folder / run_folder / "catphan_604_materials.nii.gz"
            )
            phantom.save_density_image(
                output_folder / run_folder / "catphan_604_densities.nii.gz"
            )

            fp_geometry = create_geometry(start_angle=90, n_projections=n_projections)
            save_geometry(fp_geometry, output_folder / run_folder / "geometry.xml")

            simulation = MCSimulation(geometry=phantom, **simulation_config)
            simulation.run_simulation(
                output_folder / run_folder,
                run_air_simulation=True,
                clean=True,
                gpu_ids=gpu,
                force_rerun=False,
            )

        reconstruct_3d(
            projections_filepath=output_folder
            / run_folder
            / "projections_total_normalized.mha",
            geometry_filepath=output_folder / run_folder / "geometry.xml",
            output_folder=output_folder / run_folder / "reconstructions",
            output_filename="fdk3d_wpc.mha",
            dimension=(464, 250, 464),
            water_pre_correction=ReconDefaults.wpc_catphan604,
            gpu_id=gpu[0],
        )

        mc_recon = sitk.ReadImage(
            str(output_folder / run_folder / "reconstructions" / "fdk3d_wpc.mha")
        )
        mc_recon = sitk.GetArrayFromImage(mc_recon)

        mc_recon = np.moveaxis(mc_recon, 1, -1)
        mc_recon = np.rot90(mc_recon, k=-1, axes=(0, 1))

        mc_roi_stats = MCWaterPhantomGeometry.calculate_roi_statistics(
            mc_recon,
            height_margin=2,
            radius_margin=2,
        )

        with open(
            str(output_folder / run_folder / "reconstructions" / f"roi_stats.json"),
            "wt",
        ) as file:
            json.dump(mc_roi_stats, file, indent=4)
        mean_roi_stats += np.array(
            [mc_roi_stats[material_name]["std"] for material_name in materials]
        )
    mean_roi_stats = mean_roi_stats / number_runs
    reference_roi_stats = REFERENCE_ROI_STATS_CATPHAN604_VARIAN
    reference_roi_stats = np.array(
        [reference_roi_stats[material_name]["std"] for material_name in materials]
    )
    rel_dev = (mean_roi_stats - reference_roi_stats) / reference_roi_stats
    rel_dev = np.abs(rel_dev)
    mean_rel_dev = np.mean(rel_dev)

    logger.info(f"Current deviation: {mean_rel_dev} for {n_histories} histories")

    return mean_rel_dev


@cli.command()
@click.argument(
    "folder", type=click.Path(file_okay=False, dir_okay=True, path_type=Path)
)
@click.option(
    "--reference",
    type=str,
    default="water",
    show_default=True,
)
def plot(folder: Path, reference: str = "water"):
    n_histories = []
    noises = []
    reference_noise = REFERENCE_ROI_STATS_CATPHAN604_VARIAN[reference]["std"]
    for simulation in folder.glob("run*"):
        # read input.in as text and extract n_histories
        with open(str(simulation / "input.in"), "rt") as f:
            input_content = f.read()
        match = re.search("(\d+)\s+# TOTAL NUMBER OF HISTORIES", input_content)
        n_histories.append(int(match.group(1)))

        with open(
            str(simulation / "reconstructions" / f"roi_stats.json"),
            "rt",
        ) as file:
            roi_stats = json.load(file)
        noises.append(roi_stats[reference]["std"])

    # sort by n_histories
    n_histories, noises = zip(*sorted(zip(n_histories, noises)))
    n_histories = np.array(n_histories)

    noises = np.array(noises)
    best_idx = np.argmin(np.abs(noises - reference_noise))

    def sqrt_func(x, a):
        return a / np.sqrt(x)

    (a,), _ = curve_fit(sqrt_func, xdata=n_histories, ydata=noises)

    n_histories_fit = np.linspace(n_histories[0], n_histories[-1], 1000)
    noises_fit = sqrt_func(n_histories_fit, a)

    r2 = r2_score(noises, sqrt_func(n_histories, a))

    best_n_histories_fit = a**2 / reference_noise**2

    print(f"# n_histories optimization report for {folder}")
    print(f"# optimal sqrt fit: {a} / sqrt(n_histories)")
    print(f"# R² of sqrt fit: {r2}")
    print(f"# reference noise ({reference}): {reference_noise}")
    print(
        f"# best n_histories optimization: {n_histories[best_idx]} (noise: {noises[best_idx]})"
    )
    print(f"# best n_histories fit: {int(best_n_histories_fit):.2e}")
    print("n_histories\tnoise")
    for _n_histories, noise in zip(n_histories, noises):
        print(f"{_n_histories}\t{noise}")

    fig, ax = plt.subplots()
    ax.plot(n_histories, noises, label="simulation")
    ax.plot(
        n_histories_fit,
        noises_fit,
        label=f"{a:.6f}/sqrt(n_histories) fit (R²={r2:.4f})",
    )
    ax.axhline(
        y=reference_noise, color="r", linestyle="-", label="Varian reference noise"
    )
    plt.title(f"n_histories/noise optimization")
    plt.xlabel("n_histories")
    plt.ylabel("noise (sigma) in recon")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    cli()
