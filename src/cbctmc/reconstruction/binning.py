import numpy as np

from cbctmc.common_types import PathLike


def save_curve(
    curve: np.ndarray,
    filepath: PathLike,
    scaling_factor: float = 1.0,
    format: str = "%.4f",
):
    np.savetxt(filepath, curve * scaling_factor, fmt=format)
