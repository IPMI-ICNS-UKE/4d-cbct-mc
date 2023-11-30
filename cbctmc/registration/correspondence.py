from __future__ import annotations

import logging
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from vroc.helper import as_registration_to_reference
from vroc.models import Unet3d
from vroc.registration import VrocRegistration
from vroc.segmentation.segmentation import Segmenter3d

from cbctmc.segmentation.labels import LABELS, get_label_index
from cbctmc.segmentation.segmenter import MCSegmenter
from cbctmc.speedup.models import FlexUNet


class CorrespondenceModel:
    def __init__(self):
        self._coefficients: np.ndarray | None = None
        self._timesteps: int | None = None
        self._mean_signal: np.ndarray | None = None
        self._signal_n_dims: int | None = None
        self._mean_vector_field: np.ndarray | None = None
        self._spatial_shape = None

    @property
    def is_fitted(self) -> bool:
        return all(
            v is not None
            for v in (self._coefficients, self._mean_signal, self._mean_vector_field)
        )

    def _regularize_matrix(
        self,
        matrix: np.ndarray,
        condition_number_treshold: float = 30.0,
        step_size: float = 1e-3,
    ) -> np.ndarray:
        """Regularize a matrix by adding a small value to the diagonal.

        This is called Tikhonov regularization and is used to avoid non-
        invertibility of the matrix. We regularize the matrix until its
        condition number is below a given threshold. If the condition
        number is already below the threshold, the matrix is not
        regularized.
        """

        tikhonov_regularization = 0.0
        regularizing_matrix = np.eye(matrix.shape[0]) * tikhonov_regularization

        # check if matrix is full rank => can be inverted without regularization
        if np.linalg.matrix_rank(matrix) == min(matrix.shape):
            logger.debug("Matrix is full rank")
            condition_number = np.linalg.cond(matrix)
        else:
            condition_number = float("inf")

        if condition_number > condition_number_treshold:
            logger.info("Iterative Tikhnov regularization started")
            while condition_number > condition_number_treshold:
                # add small value to diagonal
                tikhonov_regularization += step_size
                regularizing_matrix = np.eye(matrix.shape[0]) * tikhonov_regularization
                condition_number = np.linalg.cond(matrix + regularizing_matrix)
                logger.debug(
                    f"Regularized matrix with {tikhonov_regularization=} "
                    f"results in {condition_number=}"
                )
                if tikhonov_regularization > 1.0:
                    raise RuntimeError(
                        "Abort matrix regularization. Tikhnov regularization reached 1.0."
                    )

                if condition_number < condition_number_treshold:
                    logger.info("Iterative Tikhnov regularization finished")
                    break
            logger.info(f"Regularized matrix with {tikhonov_regularization=}")
        else:
            logger.debug(
                f"No regularization needed as "
                f"{condition_number=} <= {condition_number_treshold=}"
            )
        return matrix + regularizing_matrix

    def fit(
        self,
        vector_fields: np.ndarray,
        signals: np.ndarray,
    ):
        """Fit the correspondence model according to Wilms et al. (2014) using
        multivariate regression solved by ordinary least squares.

        References:
        [1] https://doi.org/10.1088/0031-9155/59/5/1147
        """
        # final shapes of used matrices:
        # vector_fields: (3*x*y*z, timesteps)
        # signals: (signal_n_dims, timesteps)
        # coefficients: (3*x*y*z, signal_n_dims)

        # input vector_fields is of shape (timesteps, 3, x, y, z)
        self._spatial_shape = vector_fields.shape[2:]
        self._timesteps = vector_fields.shape[0]
        # reshape vector_fields to matrix of shape (3*x*y*z, t)
        vector_fields = vector_fields.reshape(self._timesteps, -1).T
        # calculate mean along timesteps
        self._mean_vector_field = np.mean(vector_fields, axis=1, keepdims=True)

        signals = signals.reshape(self._timesteps, -1).T
        self._signal_n_dims = signals.shape[0]
        # calculate mean along timesteps
        self._mean_signal = np.mean(signals, axis=1, keepdims=True)

        centered_vector_fields = vector_fields - self._mean_vector_field
        centered_signals = signals - self._mean_signal

        # calculating Moore-Penrose pseudo-inverse of signals matrix
        # Here: avoid non-invertibility by using Tikhonov regularization if needed

        if self._timesteps >= self._signal_n_dims:
            # time steps >= signal dimensions, i.e., n_rows >= n_columns
            #
            logger.info("time steps >= signal dimensions")
            covariance_matrix = centered_signals @ centered_signals.T
            covariance_matrix = self._regularize_matrix(covariance_matrix)
            centered_signals_pinv = centered_signals.T @ np.linalg.inv(
                covariance_matrix
            )
        else:
            logger.info("time steps < signal dimensions")
            covariance_matrix = centered_signals.T @ centered_signals
            covariance_matrix = self._regularize_matrix(covariance_matrix)
            centered_signals_pinv = (
                np.linalg.inv(covariance_matrix) @ centered_signals.T
            )

        self._coefficients = centered_vector_fields @ centered_signals_pinv

    def predict(self, signal: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Correspondence model is not fitted")
        if signal.shape != (self._signal_n_dims,):
            raise ValueError(
                f"Given signal has wrong shape. "
                f"Expected ({self._signal_n_dims},), but got {signal.shape}"
            )

        # input signal is of shape (signal_n_dims,)
        # reshape to (signal_n_dims, timestamps=1)
        signal = signal[:, None]

        # prediction is of shape (3*x*y*z, 1)
        prediction = self._mean_vector_field + self._coefficients @ signal

        # reshape to (3, x, y, z)
        prediction = prediction.reshape(3, *self._spatial_shape)

        return prediction

    @staticmethod
    def _segment_lungs(images: np.ndarray) -> np.ndarray:
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
        state = torch.load(
            "/datalake2/runs/mc_segmentation/models_3f44803be7e542dba54e8ebd/validation/step_15000.pth"
        )
        model.load_state_dict(state["model"])

        segmenter = MCSegmenter(
            model, device="cuda:0", patch_shape=(512, 512, 96), patch_overlap=0.5
        )
        for image in images:
            logger.debug(f"Started lung segmentaiton of image with shape {image.shape}")
            segmentation, _ = segmenter.segment(image)
            plt.imshow(_[get_label_index("lung")][:, 256, :], clim=(0, 1))
            lung_segmentation = segmentation[get_label_index("lung")]

            plt.figure()
            plt.imshow(lung_segmentation[:, 256, :], clim=(0, 1))

            yield lung_segmentation

    @staticmethod
    def _extract_lung_volumes(images: np.ndarray) -> np.ndarray:
        lung_volumes = []
        for segmentation in CorrespondenceModel._segment_lungs(images):
            lung_volume = segmentation.sum()
            logger.debug(f"Extracted lung volume of {lung_volume} voxels")
            lung_volumes.append(lung_volume)

        return np.array(lung_volumes)

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
        if signals is None:
            if masks is not None:
                logger.info("Using lung volumes as signals")
                lung_volumes = masks.reshape(masks.shape[0], -1).sum(axis=1)
                signals = lung_volumes - lung_volumes.mean()
                # clip to [-1, 1]
                signals = signals / np.abs(signals).max()
                dt_signals = np.gradient(signals, axis=0)

                signals = np.stack([signals, dt_signals], axis=1)
            else:
                raise ValueError("Either signals or masks must be given")

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

        correspondence_model = cls()
        correspondence_model.fit(vector_fields=vector_fields, signals=signals)

        return correspondence_model


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

    # pred = model.predict(np.array([0.7918972439236112, -0.028313530815972224]))
    #
    # import matplotlib.pyplot as plt
    #
    # mean_vector_field = model._mean_vector_field.reshape(3, *model._spatial_shape)
    #
    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(mean_vector_field[2, :, 256, :], clim=(-10, 10))
    # ax[1].imshow(pred[2, :, 256, :], clim=(-10, 10))
