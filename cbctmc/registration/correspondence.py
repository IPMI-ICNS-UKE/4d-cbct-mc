from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from vroc.helper import as_registration_to_reference
from vroc.registration import VrocRegistration


class CorrespondenceModel:
    def __init__(self):
        self._coefficients: np.ndarray | None = None
        self._spatial_shape = None

    @property
    def is_fitted(self) -> bool:
        return self._coefficients is not None

    def fit(
        self,
        vector_fields: np.ndarray,
        signals: np.ndarray,
        tikhonov_regularization: float = 0.2,
    ):
        """Fit the correspondence model according to Wilms et al. (2014) using
        multivariate regression solved by ordinary least squares.

        References:
        [1] https://doi.org/10.1088/0031-9155/59/5/1147
        """
        # vector_fields is of shape (timesteps, 3, x, y, z)
        self._spatial_shape = vector_fields.shape[2:]
        timesteps = vector_fields.shape[0]
        # reshape vector_fields to matrix of shape (3+x+y+z, t)
        vector_fields = vector_fields.reshape(timesteps, -1).T
        signals = signals.reshape(timesteps, -1)
        signal_n_dims = signals.shape[1]

        # calculating Moore-Penrose pseudo-inverse of signals matrix
        # Here: avoid non-invertibility by using Tikhonov regularization
        if timesteps >= signal_n_dims:
            signals_pinv = np.linalg.inv(
                signals @ signals.T + tikhonov_regularization * np.eye(timesteps)
            )
        else:
            signals_pinv = (
                np.linalg.inv(
                    signals.T @ signals + tikhonov_regularization * np.eye(timesteps)
                )
                @ signals.T
            )

        # calculating regression coefficients
        self._coefficients = vector_fields @ signals_pinv

    @staticmethod
    def extract_signal_from_vector_fields(vector_fields: np.ndarray) -> np.ndarray:
        # extract 99th percentile of each vector field
        vector_fields = vector_fields.reshape(vector_fields.shape[0], -1)
        amplitude_signal = np.percentile(vector_fields, 99, axis=1)
        velocity_signal = np.gradient(amplitude_signal)

        signal = np.stack([amplitude_signal, velocity_signal], axis=1)

        return signal

    @classmethod
    def build_default(
        cls,
        images: np.ndarray,
        signals: np.ndarray | None = None,
        masks: np.ndarray | None = None,
        device: str = "cuda",
    ):
        """Build a default correspondence model using the images, masks and
        signal."""
        registration = VrocRegistration(
            device=device,
        )

        vector_fields = []

        for data in as_registration_to_reference(
            images, masks=masks, reference_index=2
        ):
            registration_result = registration.register(
                moving_image=data["moving_image"],
                fixed_image=data["fixed_image"],
                moving_mask=data["moving_mask"],
                fixed_mask=data["fixed_mask"],
                register_affine=False,
                default_parameters={
                    "iterations": 800,
                    "tau": 2.25,
                    "tau_level_decay": 0.0,
                    "tau_iteration_decay": 0.0,
                    "sigma_x": 1.25,
                    "sigma_y": 1.25,
                    "sigma_z": 1.25,
                    "sigma_level_decay": 0.0,
                    "sigma_iteration_decay": 0.0,
                    "n_levels": 3,
                    "largest_scale_factor": 1.0,
                },
            )

            vector_fields.append(registration_result.composed_vector_field)

        vector_fields = np.stack(vector_fields, axis=0)

        if signals is None:
            signals = cls.extract_signal_from_vector_fields(vector_fields)

        correspondence_model = cls()
        correspondence_model.fit(vector_fields=vector_fields, signals=signals)

        return correspondence_model

    def predict(self, signal: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Correspondence model is not fitted")


if __name__ == "__main__":
    from pathlib import Path

    import SimpleITK as sitk
    from ipmi.common.logger import init_fancy_logging

    logging.getLogger("cbctmc").setLevel(logging.DEBUG)
    logging.getLogger("vroc").setLevel(logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    init_fancy_logging()

    image_filepaths = [
        Path(
            "/mnt/nas_io/anarchy/4d_cbct_mc/4d_ct_lung_uke_artifact_free/024_4DCT_Lunge_amplitudebased_complete"
        )
        / f"phase_{i:02d}.nii"
        for i in range(10)
    ]

    mask_filepaths = [
        Path(
            "/mnt/nas_io/anarchy/4d_cbct_mc/4d_ct_lung_uke_artifact_free/024_4DCT_Lunge_amplitudebased_complete/masks"
        )
        / f"lung_phase_{i:02d}.nii"
        for i in range(10)
    ]

    def read_image(filepath) -> np.ndarray:
        image = sitk.ReadImage(str(filepath))
        image = sitk.GetArrayFromImage(image)
        image = np.swapaxes(image, 0, 2)
        return image

    images = [read_image(image_filepath) for image_filepath in image_filepaths]
    images = np.stack(images, axis=0)

    masks = [read_image(mask_filepath) for mask_filepath in mask_filepaths]
    masks = np.stack(masks, axis=0)

    model = CorrespondenceModel.build_default(images=images, masks=masks)
