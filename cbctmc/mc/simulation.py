from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Sequence, Tuple

import pkg_resources
from ipmi.common.logger import tqdm
from jinja2 import Environment, FileSystemLoader

from cbctmc.common_types import PathLike
from cbctmc.defaults import DefaultMCSimulationParameters as MCDefaults
from cbctmc.docker import execute_in_docker
from cbctmc.mc.dataio import save_text_file
from cbctmc.mc.geometry import MCGeometry
from cbctmc.mc.utils import replace_root

logger = logging.getLogger(__name__)


class MCSimulation:
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

    def run(self, output_folder: PathLike):
        output_folder = Path(output_folder)
        if not output_folder.is_dir():
            output_folder.mkdir(parents=True)

        input_filepath = output_folder / "input.in"
        geometry_filepath = output_folder / "geometry.vox.gz"
        self.geometry.save_mcgpu_geometry(geometry_filepath)

        image_size = self.geometry.image_size

        source_position = (
            image_size[0] / 2,
            image_size[1] / 2 - self.source_to_isocenter_distance,
            image_size[2] / 2,
        )

        # here we append "/host" to all paths for running MC-GPU with docker
        output_folder = replace_root(output_folder, new_root="/host")
        geometry_filepath = replace_root(geometry_filepath, new_root="/host")
        xray_spectrum_filepath = replace_root(
            self.xray_spectrum_filepath, new_root="/host"
        )
        material_filepaths = [
            replace_root(m, new_root="/host") for m in self.material_filepaths
        ]

        mcgpu_input = MCSimulation.create_mcgpu_input(
            voxel_geometry_filepath=geometry_filepath,
            material_filepaths=material_filepaths,
            xray_spectrum_filepath=xray_spectrum_filepath,
            source_position=source_position,
            output_folder=output_folder,
            n_histories=self.n_histories,
            n_projections=self.n_projections,
            angle_between_projections=self.angle_between_projections,
            source_direction_cosines=self.source_direction_cosines,
            source_aperture=self.source_aperture,
            n_detector_pixels=self.n_detector_pixels,
            detector_size=self.detector_size,
            source_to_detector_distance=self.source_to_detector_distance,
            source_to_isocenter_distance=self.source_to_isocenter_distance,
            random_seed=self.random_seed,
        )

        MCSimulation.save_mcgpu_input(mcgpu_input, output_filepath=input_filepath)
        input_filepath = replace_root(input_filepath, new_root="/host")

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
                cli_command=["MC-GPU_v1.3.x", str(input_filepath)]
            )

            pattern = re.compile(
                r"Simulating Projection "
                r"(?P<i_projection>\d{1,4}) of (?P<n_projections>\d{1,4})"
            )
            while container.status == "created":
                for line in container.logs().decode().split("\n\n\n\n")[::-1]:
                    line = line.strip()
                    if match := pattern.search(line):
                        _i_projection = int(match.groupdict()["i_projection"])
                        if _i_projection > i_projection:
                            diff = _i_projection - i_projection
                            progress_bar.update(diff)
                            i_projection = _i_projection

                        break

                time.sleep(1.0)
        except KeyboardInterrupt:
            # stopping detached container
            if container:
                logger.info("Stopping Docker container")
                container.stop()
                logger.info("Docker container stopped")

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
            "detector_size_x": round(detector_size[0] / 10.0, 3),
            "detector_size_y": round(detector_size[1] / 10.0, 3),
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
            "source_position_x": round(source_position[0] / 10.0, 3),
            "source_position_y": round(source_position[1] / 10.0, 3),
            "source_position_z": round(source_position[2] / 10.0, 3),
            "source_to_detector_distance": round(source_to_detector_distance / 10.0, 3),
            "source_to_isocenter_distance": round(
                source_to_isocenter_distance / 10.0, 3
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
