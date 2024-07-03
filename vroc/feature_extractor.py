from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from vroc.common_types import FloatTuple3D, PathLike, TorchDevice
from vroc.convert import as_tensor
from vroc.interpolation import match_vector_field, resize_spacing
from vroc.logger import LoggerMixin
from vroc.oriented_histogram import OrientedHistogram
from vroc.registration import RegistrationResult, VrocRegistration


class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError


class OrientedHistogramFeatureExtrator(FeatureExtractor, LoggerMixin):
    DEFAULT_REGISTRATION_PARAMETERS = {
        "iterations": 200,
        "tau": 2.0,
        "sigma_x": 1.0,
        "sigma_y": 1.0,
        "sigma_z": 1.0,
        "n_levels": 3,
    }

    def __init__(
        self,
        n_bins: int | Sequence[int] = 32,
        registration_parameters: dict | None = None,
        device: str | torch.device = "cuda",
    ):
        self.n_bins = [n_bins] if isinstance(n_bins, int) else n_bins
        self.registration_parameters = (
            registration_parameters
            or OrientedHistogramFeatureExtrator.DEFAULT_REGISTRATION_PARAMETERS
        )
        self.device = torch.device(device)

    @staticmethod
    def is_right_handed_basis(basis: np.ndarray) -> bool:
        return np.cross(basis[0], basis[1]) @ basis[2] > 0

    @staticmethod
    def compute_axes_of_orientation(image: np.ndarray):
        if image.ndim != 3:
            raise ValueError(f"Expected 3D image, got {image.ndim}D image")
        x, y, z = np.nonzero(image)

        # substract mean for each axis to center the image
        x = x - x.mean()
        y = y - y.mean()
        z = z - z.mean()

        coordinates = np.stack((x, y, z))
        covariance = np.cov(coordinates)
        eigenvalues, eigenvectors = np.linalg.eig(covariance)

        # get the eigenvectors with eigenvalues in descending order
        # sort_indices = np.argsort(eigenvalues)[::-1]
        # eigenvectors = eigenvectors[:, sort_indices]

        # sort by largest x, y, and z value
        sort_indices = np.argmax(np.abs(eigenvectors), axis=1)
        eigenvectors = eigenvectors[:, sort_indices]

        # re-orient to positive direction in (x, y, z)
        if eigenvectors[0, 0] < 0:
            eigenvectors[0] *= -1
        if eigenvectors[1, 1] < 0:
            eigenvectors[1] *= -1
        if eigenvectors[2, 2] < 0:
            eigenvectors[2] *= -1

        if not OrientedHistogramFeatureExtrator.is_right_handed_basis(eigenvectors):
            raise ValueError("Left-handed basis")
        return eigenvectors

    @staticmethod
    def change_basis_of_vector_field(
        vector_field: np.ndarray, new_basis: np.ndarray
    ) -> np.ndarray:
        if vector_field.ndim != 4 and vector_field.shape[0] != 3:
            raise ValueError(
                f"Expected 4D vector field of shape (3, x, y, z), "
                f"got {vector_field.ndim}D vector field of shape {vector_field.shape}"
            )

        # change of basis math:
        # let e1, e2, e3 be the basis vectors of the original basis (row vectors)
        # let v1, v2, v3 be the new basis vectors (row vectors)
        # let x1, x2, x3 be the components of a vector in the original basis
        # let y1, y2, y3 be the components of the same vector in the new basis
        # then the following relation holds:
        # | x1 |   | v1 |   | y1 |
        # | x2 | = | v2 | * | y2 |
        # | x3 |   | v3 |   | y3 |
        # or in matrix notation: x = V * y
        # Thus we can obtain the vectors in the new basis via y = V^-1 * x.

        # new basis is of shape (3, 3), with the basis vectors as row vectors
        # vector field is of shape (3, x, y, z), with the components of the vectors as
        # the first axis
        vector_field_shape = vector_field.shape
        vector_field = vector_field.reshape(3, -1)
        new_vector_field = np.linalg.inv(new_basis) @ vector_field

        return new_vector_field.reshape(vector_field_shape)

    def calculate_oriented_histogram(
        self,
        moving_image: np.ndarray | torch.Tensor,
        fixed_image: np.ndarray | torch.Tensor,
        moving_mask: np.ndarray | torch.Tensor,
        fixed_mask: np.ndarray | torch.Tensor,
        image_spacing: FloatTuple3D,
        normalize: bool = True,
        initial_vector_field: np.ndarray | torch.Tensor | None = None,
        target_image_spacing: FloatTuple3D = (4.0, 4.0, 4.0),
        n_bins: int | Sequence[int] = 32,
        use_lung_basis: bool = False,
        device: TorchDevice = "cuda",
        symmetric_registration: bool = False,
        registration_result: RegistrationResult | None = None,
        return_registration_results: bool = False,
    ):
        self.logger.info(f"Calculating oriented histogram with {n_bins} bins")
        if (n_dims := len(image_spacing)) != len(target_image_spacing):
            raise ValueError(
                f"Dimension mismatch between "
                f"{image_spacing=} and {target_image_spacing=}"
            )
        device = torch.device(device)

        # convert images to 5D tensors, i.e. (1, 1, x_size, y_size, z_size)
        moving_image = as_tensor(
            moving_image, n_dim=n_dims + 2, dtype=torch.float32, device=device
        )
        fixed_image = as_tensor(
            fixed_image, n_dim=n_dims + 2, dtype=torch.float32, device=device
        )
        moving_mask = as_tensor(
            moving_mask, n_dim=n_dims + 2, dtype=torch.bool, device=device
        )
        fixed_mask = as_tensor(
            fixed_mask, n_dim=n_dims + 2, dtype=torch.bool, device=device
        )

        fixed_image = resize_spacing(
            fixed_image,
            input_image_spacing=image_spacing,
            output_image_spacing=target_image_spacing,
        )
        moving_image = resize_spacing(
            moving_image,
            input_image_spacing=image_spacing,
            output_image_spacing=target_image_spacing,
        )
        fixed_mask = resize_spacing(
            fixed_mask,
            input_image_spacing=image_spacing,
            output_image_spacing=target_image_spacing,
            order=0,
        )
        moving_mask = resize_spacing(
            moving_mask,
            input_image_spacing=image_spacing,
            output_image_spacing=target_image_spacing,
            order=0,
        )
        if initial_vector_field is not None:
            initial_vector_field = match_vector_field(
                vector_field=initial_vector_field, image=moving_image
            )
        # calculate union mask after resizing moving and fixed mask
        union_mask = fixed_mask | moving_mask

        registration = VrocRegistration(device=device)

        if use_lung_basis:
            lung_basis = self.compute_axes_of_orientation(
                moving_mask.detach().cpu().numpy()[0, 0]
            )
            self.logger.info(
                f"Using lung basis: "
                f"e1={lung_basis[0]}, e2={lung_basis[1]}, e3={lung_basis[2]}"
            )

        if registration_result:
            registration_result_1 = registration_result
        else:
            registration_result_1 = registration.register(
                moving_image=moving_image,
                fixed_image=fixed_image,
                moving_mask=moving_mask,
                fixed_mask=fixed_mask,
                register_affine=False,
                initial_vector_field=initial_vector_field,
                default_parameters=self.registration_parameters,
                debug=False,
            )
        # vector_fields is list [affine_vector_field, varreg_vector_field]
        vector_field_1 = registration_result_1.vector_fields[-1]
        if use_lung_basis:
            vector_field_1 = self.change_basis_of_vector_field(
                vector_field=vector_field_1, new_basis=lung_basis
            )
        registration_results = [registration_result_1]
        if symmetric_registration:
            registration_result_2 = registration.register(
                moving_image=fixed_image,
                fixed_image=moving_image,
                moving_mask=fixed_mask,
                fixed_mask=moving_mask,
                register_affine=False,
                default_parameters=self.registration_parameters,
                debug=False,
            )
            vector_field_2 = registration_result_2.vector_fields[-1]
            if use_lung_basis:
                vector_field_2 = self.change_basis_of_vector_field(
                    vector_field=vector_field_2, new_basis=lung_basis
                )
            registration_results.append(registration_result_2)

        all_ohs = []
        union_mask = union_mask.detach().cpu().numpy().squeeze(axis=(0, 1))
        moving_mask = moving_mask.detach().cpu().numpy().squeeze(axis=(0, 1))
        for _n_bins in self.n_bins:
            oriented_histogram = OrientedHistogram(n_bins=_n_bins)
            oh = oriented_histogram.calculate(
                vector_field_1, mask=moving_mask, normalize=normalize
            )

            if symmetric_registration:
                oh_2 = oriented_histogram.calculate(
                    vector_field_2, mask=union_mask, normalize=normalize
                )
                oh = np.mean(
                    (
                        oh,
                        oh_2,
                    ),
                    axis=0,
                )

            all_ohs.append(oh)

        if len(all_ohs) == 1:
            all_ohs = all_ohs[0]

        if return_registration_results:
            return all_ohs, registration_results
        return all_ohs

    ImageType = np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor] | Generator

    def extract(
        self,
        moving_image: ImageType,
        fixed_image: ImageType,
        moving_mask: ImageType,
        fixed_mask: ImageType,
        image_spacing: FloatTuple3D,
        target_image_spacing: FloatTuple3D = (4.0, 4.0, 4.0),
        normalize: bool = True,
        use_lung_basis: bool = False,
        symmetric_registration: bool = False,
        initialize_with_previous: bool = False,
        output_folder: PathLike | None = None,
        save_registration_results: bool = False,
        output_hints: Sequence[str] | None = None,
        device: TorchDevice = "cuda",
    ) -> np.ndarray | list[np.ndarray]:
        # cast to list
        if isinstance(moving_image, (np.ndarray, torch.Tensor)):
            moving_image = [moving_image]
        if isinstance(fixed_image, (np.ndarray, torch.Tensor)):
            fixed_image = [fixed_image]
        if isinstance(moving_mask, (np.ndarray, torch.Tensor)):
            moving_mask = [moving_mask]
        if isinstance(fixed_mask, (np.ndarray, torch.Tensor)):
            fixed_mask = [fixed_mask]

        if output_folder:
            output_folder = Path(output_folder)
            output_folder.mkdir(parents=True, exist_ok=True)

        if symmetric_registration and initialize_with_previous:
            raise NotImplementedError

        all_ohs = []
        all_registration_results = []
        initial_vector_field = None
        for i, (_moving_image, _fixed_image, _moving_mask, _fixed_mask) in enumerate(
            zip(moving_image, fixed_image, moving_mask, fixed_mask)
        ):
            output_hint = output_hints[i] if output_hints else f"{i:02d}"
            registration_result_filepath = (
                output_folder / f"registration_result_{output_hint}.pkl"
            )
            if registration_result_filepath.exists():
                with open(registration_result_filepath, "rb") as f:
                    self.logger.info(
                        f"Loading registration result from {registration_result_filepath}"
                    )
                    registration_result = pickle.load(f)
            else:
                registration_result = None
            (
                oriented_histogram,
                registration_results,
            ) = self.calculate_oriented_histogram(
                moving_image=_moving_image,
                fixed_image=_fixed_image,
                moving_mask=_moving_mask,
                fixed_mask=_fixed_mask,
                image_spacing=image_spacing,
                target_image_spacing=target_image_spacing,
                normalize=normalize,
                n_bins=self.n_bins,
                use_lung_basis=use_lung_basis,
                device=device,
                initial_vector_field=initial_vector_field,
                symmetric_registration=symmetric_registration,
                registration_result=registration_result,
                return_registration_results=True,
            )
            all_ohs.append(oriented_histogram)
            all_registration_results.append(registration_results)

            if initialize_with_previous:
                # add batch dim to get 5D array
                initial_vector_field = registration_results[0].composed_vector_field[
                    None
                ]

            if output_folder:
                for n_bins, _oriented_histogram in zip(self.n_bins, oriented_histogram):
                    np.save(
                        output_folder / f"oh_{output_hint}_{n_bins:03d}bins.npy",
                        _oriented_histogram,
                    )
                with open(output_folder / f"params.json", "w") as f:
                    json.dump(self.registration_parameters, f)
                if save_registration_results:
                    registration_results[0].save(
                        output_folder / f"registration_result_{output_hint}.pkl"
                    )
                    with plt.ioff():
                        mid_y_slice = registration_results[0].moving_image.shape[1] // 2
                        plt.imshow(
                            np.rot90(
                                registration_results[0].composed_vector_field[
                                    2, :, mid_y_slice, :
                                ]
                            ),
                            clim=(-20, 20),
                            cmap="seismic",
                        )
                        plt.savefig(
                            output_folder / f"vf_{output_hint}.png",
                            dpi=300,
                        )
                        plt.close()

                        np.save(
                            output_folder / f"vf_{output_hint}_midslice.npy",
                            registration_results[0].composed_vector_field[
                                :, :, mid_y_slice, :
                            ],
                        )

                        np.save(
                            output_folder / f"moving_{output_hint}_midslice.npy",
                            registration_results[0].moving_image[:, mid_y_slice, :],
                        )

                        np.save(
                            output_folder / f"fixed_{output_hint}_midslice.npy",
                            registration_results[0].fixed_image[:, mid_y_slice, :],
                        )

        if len(all_ohs) == 1:
            return all_ohs[0]

        return all_ohs

    @property
    def feature_name(self) -> str:
        return f"OH{self.n_bins}"
