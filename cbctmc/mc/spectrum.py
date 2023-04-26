from dataclasses import dataclass
from pathlib import Path

import pkg_resources

from cbctmc.common_types import PathLike


@dataclass
class Spectrum:
    filepath: Path

    @classmethod
    def from_file(cls, filepath: PathLike) -> "Spectrum":
        filepath = Path(filepath)

        return cls(
            filepath=filepath,
        )

    @classmethod
    def from_package_resources(cls, filename: str):
        filepath = pkg_resources.resource_filename(
            "cbctmc", f"assets/spectra/{filename}"
        )

        return cls.from_file(filepath)


SPECTRUM_125KVP = Spectrum.from_package_resources("125kVp_0.89mmTi.spc")
