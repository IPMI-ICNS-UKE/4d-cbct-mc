from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Sequence, Tuple

import pkg_resources
import SimpleITK as sitk
from ipmi.common.logger import tqdm
from jinja2 import Environment, FileSystemLoader

from cbctmc.common_types import PathLike
from cbctmc.defaults import DefaultMCSimulationParameters as MCDefaults
from cbctmc.docker import execute_in_docker
from cbctmc.mc.dataio import save_text_file
from cbctmc.mc.geometry import MCAirGeometry, MCGeometry
from cbctmc.mc.projection import (
    MCProjection,
    get_projections_from_folder,
    projections_to_itk,
)
from cbctmc.mc.utils import replace_root

logger = logging.getLogger(__name__)


class MCSimulation:
    _AIR_SIMULATION_FOLDER = "air"

    def __init__(
        self,
        geometry: MCGeometry,
        material_filepaths: Sequence[PathLike] = MCDefaults.material_filepaths,
        xray_spectrum_filepath: PathLike = MCDefaults.spectrum_filepath,
        n_histories: int = MCDefaults.n_histories,
        n_projections: int = MCDefaults.n_projections,
        angle_between_projections: float = MCDefaults.angle_between_projections,
        source_direction_cosines: Tuple[
            float, float, float
        ] = MCDefaults.source_direction_cosines,
        source_aperture: Tuple[float, float] = MCDefaults.source_aperture,
        n_detector_pixels: Tuple[int, int] = MCDefaults.n_detector_pixels,
        detector_size: Tuple[float, float] = MCDefaults.detector_size,
        source_to_detector_distance: float = MCDefaults.source_to_detector_distance,
        source_to_isocenter_distance: float = MCDefaults.source_to_isocenter_distance,
        random_seed: int = MCDefaults.random_seed,
    ):
        self.geometry = geometry
        self.material_filepaths = material_filepaths
        self.xray_spectrum_filepath = xray_spectrum_filepath
        self.n_histories = n_histories
        self.n_projections = n_projections
        self.angle_between_projections = angle_between_projections
        self.source_direction_cosines = source_direction_cosines
        self.source_aperture = source_aperture
        self.n_detector_pixels = n_detector_pixels
        self.detector_size = detector_size
        self.source_to_detector_distance = source_to_detector_distance
        self.source_to_isocenter_distance = source_to_isocenter_distance
        self.random_seed = random_seed

    def run_air_simulation(self, output_folder: PathLike):
        logger.info("Run air simulation")
        output_folder = Path(output_folder) / MCSimulation._AIR_SIMULATION_FOLDER
        air_geometry = MCAirGeometry()
        simulation = MCSimulation(
            geometry=air_geometry, n_histories=int(2.4e10), n_projections=1
        )
        simulation.run_simulation(output_folder, run_air_simulation=False)

    def _already_simulated(self, output_folder: PathLike) -> bool:
        output_folder = Path(output_folder)
        # simply check if simulation output_folder is not empty
        return output_folder.is_dir() and any(output_folder.iterdir())

    def run_simulation(
        self,
        output_folder: PathLike,
        run_air_simulation: bool = True,
        clean: bool = True,
        gpu_id: int = 0,
        force_rerun: bool = False,
        force_geometry_recompile: bool = False,
        source_position_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        source_to_detector_distance_offset: float = 0.0,
        source_to_isocenter_distance_offset: float = 0.0,
    ):
        output_folder = Path(output_folder)

        if self._already_simulated(output_folder) and not force_rerun:
            logger.info(
                f"Output folder {output_folder} is not empty, skipping simulation"
            )
            return

        if not output_folder.is_dir():
            output_folder.mkdir(parents=True)

        if run_air_simulation:
            self.run_air_simulation(output_folder)

        input_filepath = output_folder / "input.in"
        geometry_filepath = output_folder / "geometry.vox.gz"

        if not geometry_filepath.exists() or force_geometry_recompile:
            # TODO: check via hash
            self.geometry.save_mcgpu_geometry(geometry_filepath)

        image_size = self.geometry.image_size

        source_position = (
            image_size[0] / 2 + source_position_offset[0],
            image_size[1] / 2
            - self.source_to_isocenter_distance
            + source_position_offset[1],
            image_size[2] / 2 + source_position_offset[2],
        )

        # here we prepend "/host" to all paths for running MC-GPU with docker
        docker_input_filepath = replace_root(input_filepath, new_root="/host")
        docker_output_folder = replace_root(output_folder, new_root="/host")
        docker_geometry_filepath = replace_root(geometry_filepath, new_root="/host")
        docker_xray_spectrum_filepath = replace_root(
            self.xray_spectrum_filepath, new_root="/host"
        )
        docker_material_filepaths = [
            replace_root(m, new_root="/host") for m in self.material_filepaths
        ]

        mcgpu_input = MCSimulation.create_mcgpu_input(
            voxel_geometry_filepath=docker_geometry_filepath,
            material_filepaths=docker_material_filepaths,
            xray_spectrum_filepath=docker_xray_spectrum_filepath,
            source_position=source_position,
            output_folder=docker_output_folder,
            n_histories=self.n_histories,
            n_projections=self.n_projections,
            angle_between_projections=self.angle_between_projections,
            source_direction_cosines=self.source_direction_cosines,
            source_aperture=self.source_aperture,
            n_detector_pixels=self.n_detector_pixels,
            detector_size=self.detector_size,
            source_to_detector_distance=self.source_to_detector_distance
            + source_to_detector_distance_offset,
            source_to_isocenter_distance=self.source_to_isocenter_distance
            + source_to_isocenter_distance_offset,
            random_seed=self.random_seed,
        )

        MCSimulation.save_mcgpu_input(mcgpu_input, output_filepath=input_filepath)

        i_projection = 0
        container = None
        try:
            logger.info("Starting MC simulation in Docker container")
            progress_bar = tqdm(
                desc="Simulating MC projections",
                total=self.n_projections,
                logger=logger,
            )
            container = execute_in_docker(
                cli_command=[
                    "MC-GPU_v1.3.x",
                    str(docker_input_filepath),
                ],
                gpus=(gpu_id,),
            )

            progress_pattern = re.compile(
                r"Simulating Projection "
                r"(?P<i_projection>\d{1,4}) of (?P<n_projections>\d{1,4})"
            )
            error_pattern = re.compile(r"(?i)error")
            with open(output_folder / "run.log", "wt") as f:
                for log_line in container.logs(stream=True):
                    log_line = log_line.decode()
                    f.write(log_line)
                    log_line = log_line.strip()

                    # check for simulation progress
                    if match := progress_pattern.search(log_line):
                        _i_projection = int(match.groupdict()["i_projection"])
                        if _i_projection > i_projection:
                            diff = _i_projection - i_projection
                            progress_bar.update(diff)
                            i_projection = _i_projection

                    # check for errors
                    if error_pattern.search(log_line):
                        logger.error(
                            f"An error occurred while executing the MC simulation. "
                            f"Please check the run.log file: {log_line}"
                        )

        except KeyboardInterrupt:
            # stopping detached container
            if container:
                logger.info("Stopping Docker container")
                container.stop()
                logger.info("Docker container stopped")

        # simulation finished or stopped
        self.postprocess_simulation(
            output_folder, clean=clean, air_normalization=run_air_simulation
        )

    def postprocess_simulation(
        self, folder: PathLike, clean: bool = True, air_normalization: bool = True
    ):
        folder = Path(folder)
        projections = get_projections_from_folder(folder)

        if not projections:
            # nothing to postprocess
            return

        for mode in ("total", "unscattered", "scattered"):
            projections_itk = projections_to_itk(projections, mode=mode)
            output_filepath = folder / f"projections_{mode}.mha"
            logger.info(f"Write projection stack {output_filepath}")
            sitk.WriteImage(projections_itk, str(output_filepath))

        if air_normalization:
            air_projection = MCProjection.from_file(
                folder / MCSimulation._AIR_SIMULATION_FOLDER / "projections_total.mha"
            )
            projections_itk = projections_to_itk(
                projections,
                air_projection=air_projection,
                mode="total",
            )
            output_filepath = folder / "projections_total_normalized.mha"
            logger.info(f"Write projection stack {output_filepath}")
            sitk.WriteImage(projections_itk, str(output_filepath))

        if clean:
            # clean foder
            MCSimulation._clean_simulation_folder(folder)

    @staticmethod
    def _clean_simulation_folder(folder: PathLike):
        folder = Path(folder)
        # delete projection and projection.raw
        pattern = re.compile(r"^projection(_\d{4})?(\.raw)?$")
        for filepath in folder.glob("*"):
            if pattern.match(filepath.name):
                os.remove(filepath)

    @staticmethod
    def create_mcgpu_input(
        voxel_geometry_filepath: PathLike,
        material_filepaths: Sequence[PathLike],
        xray_spectrum_filepath: PathLike,
        source_position: Tuple[float, float, float],
        output_folder: PathLike,
        n_histories: int = MCDefaults.n_histories,
        n_projections: int = MCDefaults.n_projections,
        angle_between_projections: float = MCDefaults.angle_between_projections,
        source_direction_cosines: Tuple[
            float, float, float
        ] = MCDefaults.source_direction_cosines,
        source_aperture: Tuple[float, float] = MCDefaults.source_aperture,
        n_detector_pixels: Tuple[int, int] = MCDefaults.n_detector_pixels,
        detector_size: Tuple[float, float] = MCDefaults.detector_size,
        source_to_detector_distance: float = MCDefaults.source_to_detector_distance,
        source_to_isocenter_distance: float = MCDefaults.source_to_isocenter_distance,
        random_seed: int = MCDefaults.random_seed,
    ) -> str:
        # Note: MC-GPU uses cm instead of mm (thus dividing by 10)
        params = {
            "angle_between_projections": angle_between_projections,
            "detector_size_x": round(detector_size[0] / 10.0, 6),
            "detector_size_y": round(detector_size[1] / 10.0, 6),
            "material_filepaths": [str(path) for path in material_filepaths],
            "n_detector_pixels_x": n_detector_pixels[0],
            "n_detector_pixels_y": n_detector_pixels[1],
            "n_histories": n_histories,
            "n_projections": n_projections,
            "output_folder": str(output_folder),
            "random_seed": random_seed,
            "source_polar_aperture": source_aperture[0],
            "source_azimuthal_aperture": source_aperture[1],
            "source_direction_cosine_u": source_direction_cosines[0],
            "source_direction_cosine_v": source_direction_cosines[1],
            "source_direction_cosine_w": source_direction_cosines[2],
            "source_position_x": round(source_position[0] / 10.0, 6),
            "source_position_y": round(source_position[1] / 10.0, 6),
            "source_position_z": round(source_position[2] / 10.0, 6),
            "source_to_detector_distance": round(source_to_detector_distance / 10.0, 6),
            "source_to_isocenter_distance": round(
                source_to_isocenter_distance / 10.0, 6
            ),
            "voxel_geometry_filepath": str(voxel_geometry_filepath),
            "xray_spectrum_filepath": str(xray_spectrum_filepath),
        }

        logger.info(
            f"Creating MC-GPU input file with the following parameters: {params}"
        )

        assets_folder = pkg_resources.resource_filename("cbctmc", "assets/templates")
        environment = Environment(loader=FileSystemLoader(assets_folder))
        template = environment.get_template("mcgpu_input.jinja2")
        rendered = template.render(params)

        return rendered

    @staticmethod
    def save_mcgpu_input(contents: str, output_filepath: PathLike) -> Path:
        return save_text_file(
            contents,
            output_filepath=output_filepath,
            compress=False,
            content_type="MC-GPU input file",
        )
