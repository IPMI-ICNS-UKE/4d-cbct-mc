from __future__ import annotations

import torch

from vroc.common_types import ArrayOrTensor


def as_tensor(
    image: ArrayOrTensor | None,
    n_dim: int,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    if image is not None:
        # cast image to tensor; if image is None return None
        current_n_dims = image.ndim

        if n_dim < current_n_dims:
            raise RuntimeError("Dimension mismatch")

        if n_dims_to_pad := n_dim - current_n_dims:
            image = image[(None,) * n_dims_to_pad]

        image = torch.as_tensor(image, dtype=dtype, device=device)

    return image
