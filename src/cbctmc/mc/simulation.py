from __future__ import annotations

import hashlib
import logging
import os
import re
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pkg_resources
import SimpleITK as sitk
import yaml
from ipmi.common.logger import tqdm
from jinja2 import Environment, FileSystemLoader

from cbctmc.common_types import PathLike
from cbctmc.defaults import DefaultMCSimulationParameters as MCDefaults
from cbctmc.docker import DOCKER_HOST_PATH_PREFIX, execute_in_docker
from cbctmc.mc.dataio import save_text_file
from cbctmc.mc.geometry import MCAirGeometry, MCGeometry
from cbctmc.mc.projection import (
    MCProjection,
    get_projections_from_folder,
    projections_to_itk,
)
from cbctmc.mc.respiratory import RespiratorySignal
from cbctmc.mc.utils import replace_root
from cbctmc.registration.correspondence import CorrespondenceModel

logger = logging.getLogger(__name__)


class BaseMCSimulation:
    _AIR_SIMULATION_FOLDER = "air"

    def __init__(
        self,
        geometry: MCGeometry,
        material_filepaths: Sequence[PathLike] = MCDefaults.material_filepaths,
        xray_spectrum_filepath: PathLike = MCDefaults.spectrum_filepath,
        n_histories: int = MCDefaults.n_histories,
        projection_angles: Sequence[float] = MCDefaults.projection_angles,
        n_projections: int = MCDefaults.n_projections,
        angle_between_projections: float = MCDefaults.angle_between_projections,
        source_direction_cosines: Tuple[
            float, float, float
        ] = MCDefaults.source_direction_cosines,
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
        self.projection_angles = projection_angles
        self.n_projections = len(self.projection_angles) or n_projections
        self.angle_between_projections = angle_between_projections
        self.source_direction_cosines = source_direction_cosines
        self.n_detector_pixels = n_detector_pixels
        self.detector_size = detector_size
        self.source_to_detector_distance = source_to_detector_distance
        self.source_to_isocenter_distance = source_to_isocenter_distance
        self.random_seed = random_seed

    @staticmethod
    def run_air_simulation(
        output_folder: PathLike,
        n_histories: int = int(5e10),
        gpu_ids: Sequence[int] | int = 0,
        dry_run: bool = False,
    ):
        logger.info("Run air simulation")
        output_folder = Path(output_folder) / MCSimulation._AIR_SIMULATION_FOLDER
        air_geometry = MCAirGeometry()
        simulation = MCSimulation(
            geometry=air_geometry, n_histories=n_histories, n_projections=1
        )
        simulation.run_simulation(
            output_folder, gpu_ids=gpu_ids, run_air_simulation=False, dry_run=dry_run
        )

    @staticmethod
    def _already_simulated(output_folder: PathLike) -> bool:
        output_folder = Path(output_folder)
        # check if projections are already present
        return (output_folder / "projections_total.mha").is_file()

    def _prepare_simulation(
        self,
        output_folder: PathLike,
        geometry_output_folder: PathLike,
        output_suffix: str = "",
        gpu_ids: Sequence[int] | int = 0,
        force_rerun: bool = False,
        force_geometry_recompile: bool = False,
    ) -> Path | None:
        output_folder = Path(output_folder)
        geometry_output_folder = Path(geometry_output_folder)
        gpu_ids = (gpu_ids,) if isinstance(gpu_ids, int) else gpu_ids

        logger.debug(f"Running simulation on {len(gpu_ids)} GPUs: {gpu_ids}")

        if not output_folder.is_dir():
            output_folder.mkdir(parents=True)

        input_filepath = output_folder / f"input{output_suffix}.in"
        geometry_filepath = geometry_output_folder / f"geometry{output_suffix}.vox.gz"

        if not geometry_filepath.exists() or force_geometry_recompile:
            logger.info("Compile geometry and safe to folder {geometry_output_folder}")
            self.geometry.save_mcgpu_geometry(geometry_filepath)

            self.geometry.save_material_segmentation(
                geometry_output_folder / f"geometry_materials{output_suffix}.nii.gz"
            )
            self.geometry.save_density_image(
                geometry_output_folder / f"geometry_densities{output_suffix}.nii.gz"
            )
            self.geometry.save(
                geometry_output_folder / f"geometry{output_suffix}.pkl.gz"
            )

        image_size = self.geometry.image_size

        source_position = (
            image_size[0] / 2,
            image_size[1] / 2 - self.source_to_isocenter_distance,
            image_size[2] / 2,
        )

        # here we prepend "/host" to all paths for running MC-GPU with docker
        docker_output_folder = replace_root(
            output_folder, new_root=DOCKER_HOST_PATH_PREFIX
        )
        docker_geometry_filepath = replace_root(
            geometry_filepath, new_root=DOCKER_HOST_PATH_PREFIX
        )
        docker_xray_spectrum_filepath = replace_root(
            self.xray_spectrum_filepath, new_root=DOCKER_HOST_PATH_PREFIX
        )
        docker_material_filepaths = [
            replace_root(m, new_root=DOCKER_HOST_PATH_PREFIX)
            for m in self.material_filepaths
        ]

        mcgpu_input = MCSimulation.create_mcgpu_input(
            voxel_geometry_filepath=docker_geometry_filepath,
            material_filepaths=docker_material_filepaths,
            xray_spectrum_filepath=docker_xray_spectrum_filepath,
            source_position=source_position,
            output_folder=docker_output_folder,
            n_histories=self.n_histories,
            projection_angles=self.projection_angles,
            n_projections=self.n_projections,
            angle_between_projections=self.angle_between_projections,
            source_direction_cosines=self.source_direction_cosines,
            n_detector_pixels=self.n_detector_pixels,
            detector_size=self.detector_size,
            source_to_detector_distance=self.source_to_detector_distance,
            source_to_isocenter_distance=self.source_to_isocenter_distance,
            random_seed=self.random_seed,
            gpu_ids=gpu_ids,
        )

        MCSimulation.save_mcgpu_input(mcgpu_input, output_filepath=input_filepath)

        return input_filepath

    def _run_simulation(
        self,
        input_filepath: PathLike,
        log_output_filepath: PathLike,
        gpu_ids: Sequence[int],
    ):
        container = None
        input_filepath = replace_root(input_filepath, new_root=DOCKER_HOST_PATH_PREFIX)

        try:
            logger.info("Starting MC simulation in Docker container")
            container = execute_in_docker(
                cli_command=[
                    "mpirun",
                    "--tag-output",
                    "-v",
                    "-n",
                    str(len(gpu_ids)),
                    "MC-GPU_v1.3.x",
                    str(input_filepath),
                ],
                gpus=gpu_ids,
            )

            progress_pattern = re.compile(
                r"Simulating Projection "
                r"(?P<i_projection>\d{1,4}) of (?P<n_projections>\d{1,4})"
            )
            error_pattern = re.compile(r"(?i)error")
            progress_bar = tqdm(
                desc="Simulating MC projections",
                total=self.n_projections,
                logger=logger,
            )
            with open(log_output_filepath, "wt") as f:
                for log_line in container.logs(stream=True):
                    log_line = log_line.decode()
                    f.write(log_line)
                    log_line = log_line.strip()

                    # check for simulation progress
                    if match := progress_pattern.search(log_line):
                        _i_projection = int(match.groupdict()["i_projection"])
                        progress_bar.update(1)

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

    @staticmethod
    def postprocess_simulation(
        folder: PathLike,
        clean: bool = True,
        stack_projections: bool = True,
        air_normalization: bool = True,
        air_projection_denoise_kernel_size: Tuple[int, int] | None = (10, 10),
    ):
        logger.info("Start simulation postprocessing")
        if air_normalization and not stack_projections:
            raise ValueError(
                "Cannot perform air normalization without stacking projections"
            )

        if not any((stack_projections, air_normalization, clean)):
            return

        folder = Path(folder)
        projections = get_projections_from_folder(folder)
        if projections and stack_projections:
            for mode in ("total", "unscattered", "scattered"):
                projections_itk = projections_to_itk(projections, mode=mode)
                output_filepath = folder / f"projections_{mode}.mha"
                logger.info(f"Write projection stack {output_filepath}")
                sitk.WriteImage(projections_itk, str(output_filepath))

        if projections and air_normalization:
            air_projection = MCProjection.from_file(
                folder / MCSimulation._AIR_SIMULATION_FOLDER / "projections_total.mha"
            )
            projections_itk = projections_to_itk(
                projections,
                air_projection=air_projection,
                air_projection_denoise_kernel_size=air_projection_denoise_kernel_size,
                mode="total",
            )
            output_filepath = folder / "projections_total_normalized.mha"
            logger.info(f"Write projection stack {output_filepath}")
            sitk.WriteImage(projections_itk, str(output_filepath))

        if clean:
            # clean folder
            MCSimulation._clean_simulation_folder(folder)

    @staticmethod
    def _clean_simulation_folder(folder: PathLike):
        folder = Path(folder)
        # delete projection and projection.raw
        pattern = re.compile(r"^projection_\d{3}\.\d{6}deg$")
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
        projection_angles: Sequence[float] = MCDefaults.projection_angles,
        n_projections: int = MCDefaults.n_projections,
        angle_between_projections: float = MCDefaults.angle_between_projections,
        source_direction_cosines: Tuple[
            float, float, float
        ] = MCDefaults.source_direction_cosines,
        source_polar_aperture: Tuple[float, float] = MCDefaults.source_polar_aperture,
        source_azimuthal_aperture: float = MCDefaults.source_azimuthal_aperture,
        n_detector_pixels: Tuple[int, int] = MCDefaults.n_detector_pixels,
        detector_size: Tuple[float, float] = MCDefaults.detector_size,
        detector_lateral_displacement: float = MCDefaults.detector_lateral_displacement,
        source_to_detector_distance: float = MCDefaults.source_to_detector_distance,
        source_to_isocenter_distance: float = MCDefaults.source_to_isocenter_distance,
        random_seed: int = MCDefaults.random_seed,
        gpu_ids: Sequence[int] = (0,),
    ) -> str:
        # Note: MC-GPU uses cm instead of mm (thus dividing by 10)
        # cf. input template for documentation
        params = {
            "gpu_id": -1 if len(gpu_ids) > 1 else gpu_ids[0],
            "angle_between_projections": angle_between_projections,
            "detector_size_x": round(detector_size[0] / 10.0, 6),
            "detector_size_y": round(detector_size[1] / 10.0, 6),
            "detector_lateral_displacement": round(
                detector_lateral_displacement / 10.0, 6
            ),
            "material_filepaths": [str(path) for path in material_filepaths],
            "n_detector_pixels_x": n_detector_pixels[0],
            "n_detector_pixels_y": n_detector_pixels[1],
            "n_histories": n_histories,
            "specify_projection_angles": "YES" if projection_angles else "NO",
            "projection_angles": projection_angles,
            "n_projections": n_projections,
            "output_folder": str(output_folder),
            "random_seed": random_seed,
            "source_polar_aperture_1": source_polar_aperture[0],
            "source_polar_aperture_2": source_polar_aperture[1],
            "source_azimuthal_aperture": source_azimuthal_aperture,
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

        logger.debug(
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


class MCSimulation(BaseMCSimulation):
    def run_simulation(
        self,
        output_folder: PathLike,
        geometry_output_folder: PathLike | None = None,
        output_suffix: str = "",
        run_air_simulation: bool = True,
        air_projection_denoise_kernel_size: Tuple[int, int] | None = (10, 10),
        clean: bool = True,
        stack_projections: bool = True,
        gpu_ids: Sequence[int] | int = 0,
        force_rerun: bool = False,
        force_geometry_recompile: bool = False,
        dry_run: bool = False,
    ):
        if not geometry_output_folder:
            geometry_output_folder = output_folder
        if dry_run:
            logger.info("Entering dry run mode. Skipping MC simulation.")

        # check if already simulated
        if self._already_simulated(output_folder) and not force_rerun:
            logger.info(
                f"Output folder {output_folder} already contains a "
                f"finished simulation and {force_rerun=}. Skipping."
            )
            return

        if run_air_simulation:
            self.run_air_simulation(output_folder, gpu_ids=gpu_ids, dry_run=dry_run)

        # input_filepath is the filepath of the MCGPU input file inside
        # the docker container
        input_filepath = self._prepare_simulation(
            output_folder,
            geometry_output_folder=geometry_output_folder,
            gpu_ids=gpu_ids,
            force_rerun=force_rerun,
            force_geometry_recompile=force_geometry_recompile,
            output_suffix=output_suffix,
        )
        logger.info("Simulation fully prepared")

        # run simulation
        if not dry_run:
            log_output_filepath = output_folder / f"run{output_suffix}.log"
            self._run_simulation(
                input_filepath=input_filepath,
                log_output_filepath=log_output_filepath,
                gpu_ids=gpu_ids,
            )
            # simulation finished or stopped
            self.postprocess_simulation(
                output_folder,
                clean=clean,
                stack_projections=stack_projections,
                air_normalization=run_air_simulation,
                air_projection_denoise_kernel_size=air_projection_denoise_kernel_size,
            )


class MCSimulation4D:
    def __init__(
        self,
        correspondence_model: CorrespondenceModel,
        geometry: MCGeometry,
        material_filepaths: Sequence[PathLike] = MCDefaults.material_filepaths,
        xray_spectrum_filepath: PathLike = MCDefaults.spectrum_filepath,
        n_histories: int = MCDefaults.n_histories,
        n_projections: int = MCDefaults.n_projections,
        gantry_rotation_speed: float = MCDefaults.gantry_rotation_speed,
        frame_rate: float = MCDefaults.frame_rate,
        angle_between_projections: float = MCDefaults.angle_between_projections,
        source_direction_cosines: Tuple[
            float, float, float
        ] = MCDefaults.source_direction_cosines,
        n_detector_pixels: Tuple[int, int] = MCDefaults.n_detector_pixels,
        detector_size: Tuple[float, float] = MCDefaults.detector_size,
        source_to_detector_distance: float = MCDefaults.source_to_detector_distance,
        source_to_isocenter_distance: float = MCDefaults.source_to_isocenter_distance,
        random_seed: int = MCDefaults.random_seed,
    ):
        self.correspondence_model = correspondence_model
        self.geometry = geometry
        self.material_filepaths = material_filepaths
        self.xray_spectrum_filepath = xray_spectrum_filepath
        self.n_histories = n_histories
        self.n_projections = n_projections
        self.gantry_rotation_speed = gantry_rotation_speed
        self.frame_rate = frame_rate
        self.angle_between_projections = angle_between_projections
        self.source_direction_cosines = source_direction_cosines
        self.n_detector_pixels = n_detector_pixels
        self.detector_size = detector_size
        self.source_to_detector_distance = source_to_detector_distance
        self.source_to_isocenter_distance = source_to_isocenter_distance
        self.random_seed = random_seed

    @staticmethod
    def _already_simulated(output_folder: PathLike) -> bool:
        output_folder = Path(output_folder)
        # check if projections are already present
        return (output_folder / "projections_total.mha").is_file()

    def _warp_geometry(
        self, signal: float, dt_signal: float, device: str = "cpu"
    ) -> MCGeometry:
        logger.debug(f"warp geometry for {signal=} and {dt_signal=}")
        vector_field = self.correspondence_model.predict(np.array([signal, dt_signal]))
        return self.geometry.warp(vector_field=vector_field, device=device)

    def _warp_and_save_geometry(
        self,
        signal: float,
        dt_signal: float,
        output_folder: PathLike,
    ):
        unique_signal_hash = hashlib.sha256(
            np.array([signal, dt_signal], dtype=np.float32).tobytes()
        ).hexdigest()[:7]
        output_suffix = f"_{unique_signal_hash}"
        logger.debug(f"Precompile geometry for {signal=} and {dt_signal=}")
        warped_geometry = self._warp_geometry(signal, dt_signal)

        output_folder = Path(output_folder)
        warped_geometry.save_mcgpu_geometry(
            output_folder / f"geometry{output_suffix}.vox.gz"
        )

        warped_geometry.save_material_segmentation(
            output_folder / f"geometry_materials{output_suffix}.nii.gz"
        )
        warped_geometry.save_density_image(
            output_folder / f"geometry_densities{output_suffix}.nii.gz"
        )
        warped_geometry.save(output_folder / f"geometry{output_suffix}.pkl.gz")

    def _precompile_geometries(
        self,
        unique_signals: Sequence[Tuple[float, float]],
        output_folder: PathLike,
        n_workers: int = 8,
    ):
        output_folder = Path(output_folder)
        results = []
        with ThreadPool(n_workers) as pool:
            for signal, dt_signal in unique_signals:
                result = pool.apply_async(
                    self._warp_and_save_geometry,
                    kwds={
                        "signal": signal,
                        "dt_signal": dt_signal,
                        "output_folder": output_folder,
                    },
                )
                results.append(result)
            [result.get() for result in results]

    def run_simulation(
        self,
        respiratory_signal: RespiratorySignal,
        respiratory_signal_quantization: int | None,
        output_folder: PathLike,
        geometry_output_folder: PathLike | None,
        run_air_simulation: bool = True,
        air_projection_denoise_kernel_size: Tuple[int, int] | None = (10, 10),
        clean: bool = True,
        stack_projections: bool = True,
        gpu_ids: Sequence[int] | int = 0,
        force_rerun: bool = False,
        force_geometry_recompile: bool = False,
        precompile_geometries: bool = False,
        dry_run: bool = False,
    ):
        output_folder = Path(output_folder)
        if not geometry_output_folder:
            geometry_output_folder = output_folder

        # check if already simulated
        if self._already_simulated(output_folder) and not force_rerun:
            logger.info(
                f"Output folder {output_folder} already contains a "
                f"finished simulation and {force_rerun=}. Skipping."
            )
            return

        output_folder.mkdir(parents=True, exist_ok=True)
        geometry_output_folder.mkdir(parents=True, exist_ok=True)
        # resample respiratory signal to match frame rate of CBCT scan
        # then the signal index corresponds to the projection index
        respiratory_signal = respiratory_signal.resample(self.frame_rate)

        # now clip the signal to the number of projections, i.e. signal has
        # the length of the number of projections (one signal value per projection)
        signal = respiratory_signal.signal[: self.n_projections]
        dt_signal = respiratory_signal.dt_signal[: self.n_projections]

        np.savetxt(
            output_folder / "signal.txt",
            np.stack((signal, dt_signal)).T,
            header=(
                "original respiratory signal and its derivative\n"
                "signal quantization: None\n"
                "signal dt_signal"
            ),
            fmt="%.6f",
        )

        # quantize the signal if requested
        if respiratory_signal_quantization:
            signal = RespiratorySignal.quantize_signal(
                signal, n_bins=respiratory_signal_quantization
            )
            dt_signal = RespiratorySignal.quantize_signal(
                dt_signal, n_bins=respiratory_signal_quantization
            )

        np.savetxt(
            output_folder / "signal_quantized.txt",
            np.stack((signal, dt_signal)).T,
            header=(
                "quantized respiratory signal and its derivative\n"
                f"signal quantization: {respiratory_signal_quantization} bins\n"
                "signal dt_signal"
            ),
            fmt="%.6f",
        )

        # get the unique combinations of signal and dt_signal
        unique_signals = RespiratorySignal.get_unique_signals(
            signal=signal, dt_signal=dt_signal
        )
        logger.info(f"Unique signals: {len(unique_signals)}")

        if precompile_geometries:
            self._precompile_geometries(
                unique_signals=unique_signals,
                output_folder=geometry_output_folder,
                n_workers=8,
            )

        # run one air simulation for all following simulations
        MCSimulation.run_air_simulation(output_folder, gpu_ids=gpu_ids, dry_run=dry_run)

        projection_geometries = {}

        start_angle = 270.0
        overall_progress = tqdm(
            desc="Simulating 4D MC projections",
            total=self.n_projections,
            logger=logger,
            log_level=logging.INFO,
        )
        for unique_signal, projection_indices in unique_signals.items():
            # warp the geometry, i.e., materials and densities, according to the
            # correspondence model using the signal and dt_signal
            signal, dt_signal = unique_signal

            unique_signal_hash = hashlib.sha256(
                np.array([signal, dt_signal], dtype=np.float32).tobytes()
            ).hexdigest()[:7]
            output_suffix = f"_{unique_signal_hash}"

            if (
                warped_geometry_filepath := (
                    geometry_output_folder / f"geometry{output_suffix}.pkl.gz"
                )
            ).is_file():
                # load warped geometry from file as it has been already computed
                warped_geometry = MCGeometry.load(warped_geometry_filepath)

            else:
                warped_geometry = self._warp_geometry(signal, dt_signal)

            projection_angles = [
                start_angle + i_projection * self.angle_between_projections
                for i_projection in projection_indices
            ]

            # save used signal/projection/geometry combination to text file
            for projection_angle in projection_angles:
                projection_geometries[projection_angle] = {
                    "signal": float(signal),
                    "dt_signal": float(dt_signal),
                    "signal_quantization": respiratory_signal_quantization,
                    "hash": unique_signal_hash,
                    "geometry_filename": f"geometry{output_suffix}.pkl.gz",
                }

            # temporary bug fix for 0th projection always at 270deg
            # source direction of projection 0 of each simulation is wrong here
            projection_angles = projection_angles[0:1] + projection_angles

            logger.debug(
                f"Selected the {len(projection_angles)} projection angles for "
                f"{signal=} and {dt_signal=}: {projection_angles}"
            )
            simulation = MCSimulation(
                geometry=warped_geometry,
                material_filepaths=self.material_filepaths,
                xray_spectrum_filepath=self.xray_spectrum_filepath,
                n_histories=self.n_histories,
                projection_angles=projection_angles,
                angle_between_projections=self.angle_between_projections,
                source_direction_cosines=self.source_direction_cosines,
                n_detector_pixels=self.n_detector_pixels,
                detector_size=self.detector_size,
                source_to_detector_distance=self.source_to_detector_distance,
                source_to_isocenter_distance=self.source_to_isocenter_distance,
                random_seed=self.random_seed,
            )
            simulation.run_simulation(
                output_folder=output_folder,
                geometry_output_folder=geometry_output_folder,
                run_air_simulation=False,
                air_projection_denoise_kernel_size=air_projection_denoise_kernel_size,
                clean=False,
                stack_projections=False,
                gpu_ids=gpu_ids,
                force_rerun=force_rerun,
                force_geometry_recompile=force_geometry_recompile,
                output_suffix=output_suffix,
                dry_run=dry_run,
            )

            # TODO: remove -1 due to fix (see above)
            overall_progress.update(len(projection_angles) - 1)

        # save used signal/projection/geometry combination to json file
        with open(output_folder / "projection_geometries.yaml", "wt") as f:
            # sort by projection angle
            projection_geometries = dict(sorted(projection_geometries.items()))
            yaml.dump(projection_geometries, f)

        if not dry_run:
            BaseMCSimulation.postprocess_simulation(
                output_folder,
                clean=clean,
                stack_projections=stack_projections,
                air_normalization=run_air_simulation,
                air_projection_denoise_kernel_size=air_projection_denoise_kernel_size,
            )
