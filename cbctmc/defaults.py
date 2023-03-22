from dataclasses import dataclass
from typing import Tuple


@dataclass
class DefaultVarianScanParameters:
    n_projections: int = 894
    # detector_pixel_size is given in mm (x, y)
    detector_pixel_size: Tuple[float, float] = (0.388, 0.388)
    # source_to_detector_distance is given in mm
    source_to_detector_distance: float = 1500.0
    # source_to_isocenter_distance is given in mm
    source_to_isocenter_distance: float = 1000.0


@dataclass
class DefaultMCSimulationParameters:
    n_histories: int = int(2.4e9)
    n_projections = DefaultVarianScanParameters.n_projections
    # default 2pi arc for default half-fan mode
    angle_between_projections = 360.0 / n_projections

    # n_detector_pixels is given in number of pixels (x, y)
    n_detector_pixels: Tuple[int, int] = (924, 384)
    # n_detector_pixels is given in mm (x, y)
    detector_size: Tuple[float, float] = (717.312, 297.984)

    # source_to_detector_distance is given in mm
    source_to_detector_distance: float = (
        DefaultVarianScanParameters.source_to_detector_distance
    )
    # source_to_isocenter_distance is given in mm
    source_to_isocenter_distance: float = (
        DefaultVarianScanParameters.source_to_isocenter_distance
    )
    random_seed: int = 42

    source_direction_cosines: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    # source_aperture is given in degrees (polar, azimuthal)
    source_aperture: Tuple[float, float] = (-15.0, -15.0)
