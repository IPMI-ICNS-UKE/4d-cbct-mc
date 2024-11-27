from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import numpy as np
import SimpleITK as sitk

import docker
from cbctmc.common_types import PathLike
from cbctmc.docker import (
    DOCKER_HOST_PATH_PREFIX,
    DOCKER_IMAGE,
    check_image_exists,
    execute_in_docker,
)
from cbctmc.logger import LoggerMixin
from cbctmc.reconstruction.binning import save_curve
from cbctmc.reconstruction.respiratory import calculate_phase
from cbctmc.shell import create_cli_command, execute
from cbctmc.utils import iec61217_to_rsp
from docker.errors import ImageNotFound


class Reconstructor(ABC, LoggerMixin):
    def __init__(
        self,
        executable: PathLike,
        detector_binning: int = 1,
        use_docker: bool = True,
        gpu_id: int = 0,
    ):
        self.executable = executable
        self.detector_binning = detector_binning
        self.use_docker = use_docker
        self.gpu_id = gpu_id

        if self.use_docker:
            if not check_image_exists(DOCKER_IMAGE):
                raise ImageNotFound(f"Docker image {DOCKER_IMAGE} not found.")
            self.docker_client = docker.from_env()
        else:
            self.docker_client = None

    @property
    def _execute_function(self):
        execute_func = (
            partial(execute_in_docker, gpus=[self.gpu_id], detach=False)
            if self.use_docker
            else partial(execute, gpus=[self.gpu_id])
        )
        return execute_func

    @property
    def detector_binning(self):
        return self.__detector_binning

    @detector_binning.setter
    def detector_binning(self, value):
        self.__detector_binning = value

    @abstractmethod
    def _preprocessing(self, **kwargs):
        pass

    @abstractmethod
    def _reconstruct(self, output_filepath: PathLike, **kwargs) -> Path:
        pass

    @abstractmethod
    def _postprocessing(self, reconstruction_filepath: PathLike, **kwargs):
        pass

    def reconstruct(
        self, output_filepath: PathLike, post_process: bool = True, **kwargs
    ) -> Path:
        self.logger.debug(f"Start reconstruction with params: {kwargs}")
        self._preprocessing(**kwargs)

        reconstruction_filepath = self._reconstruct(
            output_filepath=output_filepath, **kwargs
        )

        if post_process:
            self._postprocessing(reconstruction_filepath)

        return reconstruction_filepath


class RTKReconstructor(Reconstructor):
    def _preprocessing(self, **kwargs):
        pass

    def _reconstruct(self, output_filepath: PathLike, **kwargs) -> Path:
        bin_call = create_cli_command(
            self.executable,
            output=output_filepath,
            path_prefix=DOCKER_HOST_PATH_PREFIX if self.use_docker else None,
            convert_underscore=None,
            verbose=True,
            **kwargs,
        )
        self.logger.debug(f"Converted to binary call: {bin_call}")
        self._execute_function(bin_call)

        return Path(output_filepath)

    def _postprocessing(self, reconstruction_filepath: PathLike, **kwargs):
        reconstruction_filepath = str(reconstruction_filepath)
        image = sitk.ReadImage(reconstruction_filepath)
        image = iec61217_to_rsp(image)

        sitk.WriteImage(image, reconstruction_filepath)


class _RTK4DReconstructor(RTKReconstructor):
    def __init__(
        self,
        executable: PathLike,
        amplitude_signal: np.ndarray | None = None,
        phase_signal: np.ndarray | None = None,
        detector_binning: int = 1,
        use_docker: bool = True,
        gpu_id: int = 0,
    ):
        super().__init__(
            executable=executable,
            detector_binning=detector_binning,
            use_docker=use_docker,
            gpu_id=gpu_id,
        )

        if amplitude_signal is not None and phase_signal is not None:
            raise ValueError(
                "Only one of amplitude_signal and phase_signal can be set."
            )

        self.amplitude_signal = amplitude_signal
        self.phase_signal = phase_signal

    def _reconstruct(self, output_filepath: PathLike, **kwargs) -> Path:
        if self.phase_signal is None:
            phase_signal = calculate_phase(breathing_curve=self.amplitude_signal)
            phase_signal = np.hstack(phase_signal)
        else:
            phase_signal = self.phase_signal

        # scale phase signal to [0, 1] as requested by RTK
        phase_signal -= phase_signal.min()
        phase_signal /= phase_signal.max()

        with TemporaryDirectory() as temp_dir:
            signal_filepath = Path(temp_dir) / "signal.txt"
            save_curve(phase_signal, filepath=signal_filepath)

            kwargs["signal"] = signal_filepath

            return super()._reconstruct(output_filepath=output_filepath, **kwargs)


class ROOSTER4DReconstructor(_RTK4DReconstructor):
    def __init__(
        self,
        amplitude_signal: Optional[np.ndarray] | None = None,
        phase_signal: Optional[np.ndarray] | None = None,
        detector_binning: int = 1,
        use_docker: bool = True,
        gpu_id: int = 0,
    ):
        super().__init__(
            executable="rtkfourdrooster",
            amplitude_signal=amplitude_signal,
            phase_signal=phase_signal,
            detector_binning=detector_binning,
            use_docker=use_docker,
            gpu_id=gpu_id,
        )


class FDKReconstructor(RTKReconstructor):
    def __init__(
        self,
        executable: PathLike = "rtkfdk",
        detector_binning: int = 1,
        use_docker: bool = True,
        gpu_id: int = 0,
    ):
        super().__init__(
            executable=executable,
            detector_binning=detector_binning,
            use_docker=use_docker,
            gpu_id=gpu_id,
        )


class FDK4DReconstructor(_RTK4DReconstructor):
    def __init__(
        self,
        amplitude_signal: Optional[np.ndarray] | None = None,
        phase_signal: Optional[np.ndarray] | None = None,
        detector_binning: int = 1,
        use_docker: bool = True,
        gpu_id: int = 0,
    ):
        super().__init__(
            executable="rtkfourdfdk",
            amplitude_signal=amplitude_signal,
            phase_signal=phase_signal,
            detector_binning=detector_binning,
            use_docker=use_docker,
            gpu_id=gpu_id,
        )


class CG4DReconstructor(_RTK4DReconstructor):
    def __init__(
        self,
        amplitude_signal: Optional[np.ndarray] | None = None,
        phase_signal: Optional[np.ndarray] | None = None,
        detector_binning: int = 1,
        use_docker: bool = True,
        gpu_id: int = 0,
    ):
        super().__init__(
            executable="rtkfourdconjugategradient",
            amplitude_signal=amplitude_signal,
            phase_signal=phase_signal,
            detector_binning=detector_binning,
            use_docker=use_docker,
            gpu_id=gpu_id,
        )
