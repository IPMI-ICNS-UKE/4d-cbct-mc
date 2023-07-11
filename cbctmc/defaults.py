from dataclasses import dataclass, field
from typing import Tuple

from cbctmc.mc.materials import MATERIALS_125KEV
from cbctmc.mc.spectrum import SPECTRUM_125KVP


@dataclass
class DefaultReconstructionParameters:
    water_pre_correction: Tuple[float, ...] = (
        5.23387736e00,
        7.55433186e-01,
        5.31468975e-02,
        -1.86219035e-02,
        4.89407487e-03,
    )


@dataclass
class DefaultVarianScanParameters:
    n_projections: int = 894
    # detector_pixel_size is given in mm (x, y)
    detector_pixel_size: Tuple[float, float] = (0.388, 0.388)
    detector_lateral_displacement: float = -160.0

    # source_to_detector_distance is given in mm
    source_to_detector_distance: float = 1500.0
    # source_to_isocenter_distance is given in mm
    source_to_isocenter_distance: float = 1000.0


@dataclass
class DefaultMCSimulationParameters:
    spectrum = SPECTRUM_125KVP
    spectrum_filepath = SPECTRUM_125KVP.filepath

    material_filepaths = tuple(
        material.filepath for material in MATERIALS_125KEV.values()
    )

    n_histories: int = int(2.4e9)
    n_projections = DefaultVarianScanParameters.n_projections
    # default 2pi arc for default half-fan mode
    angle_between_projections = 360.0 / n_projections

    # n_detector_pixels is given in number of pixels (x, y)
    n_detector_pixels: Tuple[int, int] = (924, 384)
    n_detector_pixels_half_fan: Tuple[int, int] = (512, 384)
    # detector_size is given in mm (x, y)
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

    # some geometrical corrections to match RTK/MC-GPU geometry
    geometrical_corrections: dict = field(
        default_factory=lambda: {
            "offset_x": -0.4276477849036505,
            "offset_y": -3.749082176733503,
            "offset_z": -0.32871583164473445,
            "source_to_detector_distance_offset": 0.13054052787167872,
            "source_to_isocenter_distance_offset": 3.2595168038949205,
        }
    )
