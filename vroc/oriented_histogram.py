from __future__ import annotations

from math import prod

import numpy as np


class OrientedHistogram:
    def __init__(self, n_bins: int):
        self.n_theta_bins = n_bins
        self.n_phi_bins = 2 * n_bins

    def calculate(
        self,
        vector_field: np.ndarray,
        mask: None | np.ndarray = None,
        normalize: bool = True,
    ) -> np.ndarray:
        vector_field = OrientedHistogram._cartesian_to_spherical(vector_field)
        hist = self._calculate_histogram(
            spherical_vector_field=vector_field, mask=mask, normalize=normalize
        )

        return hist

    @staticmethod
    def _cartesian_to_spherical(vector_field: np.ndarray) -> np.ndarray:
        assert vector_field.shape[0] == 3, "Given vector field is not 3-dimensional"
        transformed = np.empty_like(vector_field, dtype=np.float32)
        x, y, z = vector_field

        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(np.divide(z, r, out=np.ones_like(z), where=r > 0))
        phi = np.arctan2(y, x)

        transformed[0] = r  # non-negative
        transformed[1] = theta  # [0, pi]
        transformed[2] = phi  # (-pi, pi]

        return transformed

    def _calculate_histogram(
        self,
        spherical_vector_field: np.ndarray,
        mask: None | np.ndarray = None,
        normalize: bool = True,
    ) -> np.ndarray:
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            r = spherical_vector_field[0, mask]
            theta = spherical_vector_field[1, mask]
            phi = spherical_vector_field[2, mask]
            n_voxels = mask.sum()
        else:
            r = spherical_vector_field[0].flatten()
            theta = spherical_vector_field[1].flatten()
            phi = spherical_vector_field[2].flatten()
            n_voxels = prod(spherical_vector_field.shape[1:])

        if r.sum() == 0.0:
            hist = np.zeros((self.n_theta_bins, self.n_phi_bins), dtype=np.float32)
        else:
            hist, _, _ = np.histogram2d(
                theta,
                phi,
                bins=(self.n_theta_bins, self.n_phi_bins),
                range=((0, np.pi), (-np.pi, np.pi)),
                weights=r,
                density=False,
            )

        if normalize:
            return hist / n_voxels
        else:
            return hist


if __name__ == "__main__":
    a = np.random.random(100)
    b = np.random.random(100)
    w = np.random.random(100)

    hist, _, _ = np.histogram2d(
        a,
        b,
        bins=(10, 20),
        range=((0, 1), (0, 1)),
        weights=w,
        density=False,
    )
