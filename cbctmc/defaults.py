from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from cbctmc.mc.materials import MATERIALS_125KEV
from cbctmc.mc.spectrum import SPECTRUM_125KVP


@dataclass
class DefaultReconstructionParameters:
    # WPC using CatPhan604 ROIs
    wpc_catphan604: Tuple[float, ...] = (
        0.7490896601034365,
        0.8853028842822823,
        0.15532901941332966,
        -0.08447728801183985,
        0.023960875121701974,
        -0.0025035454792714518,
    )


@dataclass
class DefaultVarianScanParameters:
    n_projections: int = 894
    n_detector_pixels: Tuple[int, int] = (1024, 768)
    # detector_pixel_size is given in mm (x, y)
    detector_pixel_size: Tuple[float, float] = (0.388, 0.388)
    detector_lateral_displacement: float = -159.856

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

    # based on noise fit using A/sqrt(n_historiess) + C and comparison to Varian
    n_histories: int = 11_903_320_312
    specify_projection_angles: bool = False
    projection_angles = []
    n_projections = DefaultVarianScanParameters.n_projections
    # default 2pi arc for default half-fan mode
    angle_between_projections = 360.0 / n_projections

    # n_detector_pixels is given in number of pixels (x, y)
    n_detector_pixels: Tuple[int, int] = (1848, 768)
    n_detector_pixels_half_fan: Tuple[
        int, int
    ] = DefaultVarianScanParameters.n_detector_pixels
    # detector_size is given in mm (x, y)
    detector_size: Tuple[float, float] = (717.024, 297.984)
    # lateral displacement along x of the detector in mm
    detector_lateral_displacement: float = (
        DefaultVarianScanParameters.detector_lateral_displacement
    )

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

    # source_aperture is given in degrees (negative values: fit to detector)
    # calculated by:
    # np.rad2deg(np.arctan(((0.388 * 1024) / 2 + (-159.856)) / 1500.0))
    # np.rad2deg(np.arctan(((0.388 * 1024) / 2 - (-159.856)) / 1500.0))
    source_polar_aperture: Tuple[float, float] = (1.481720423651376, 13.441979314886868)
    source_azimuthal_aperture: Tuple[float, float] = -1

    # 4D specific
    angular_rotation_velocity: float = 2 * np.pi / 60

    # some geometrical corrections to match RTK/MC-GPU geometry
    # geometrical_corrections: dict = field(
    #     default_factory=lambda: {
    #         "source_position_offset": (
    #             -0.5030858965528291,
    #             -3.749082176733503,
    #             -0.29206039325204886,
    #         ),
    #         "source_to_detector_distance_offset": 0.13054052787167872,
    #         "source_to_isocenter_distance_offset": 3.2595168038949205,
    #     }
    # )
