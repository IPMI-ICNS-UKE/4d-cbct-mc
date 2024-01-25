import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pkg_resources

from cbctmc.common_types import PathLike


@dataclass
class Material:
    identifier: str
    name: str
    density: float
    chemical_formula: str
    filepath: Path

    @property
    def number(self):
        return Material.get_material_number(self.identifier)

    @staticmethod
    def get_material_number(
        material_id: str, materials: dict[str, "Material"] = None
    ) -> int:
        """Returns the material number used by geometry files of MC-GPU.

        If no materials are given, the default materials
        MATERIALS_125KEV are used.
        """
        materials = materials or MATERIALS_125KEV
        try:
            return list(materials.keys()).index(material_id) + 1
        except ValueError:
            raise ValueError(f"Material {material_id} not found")

    @staticmethod
    def split_name_and_chemical_formula(name: str) -> dict:
        if match := re.match(r"(?P<name>.+)\((?P<chemical_formula>.*)\)", name):
            return match.groupdict()
        else:
            raise RuntimeError("Wrong name/chemical formula format")

    @staticmethod
    def parse_material_file_header(filepath: PathLike) -> dict:
        header = {}
        next_header_param = None
        data_type = None
        with open(filepath, "rt") as f:
            for line in f:
                if next_header_param:
                    header[next_header_param] = data_type(line.strip("# "))
                    next_header_param = None
                    data_type = None

                if "MATERIAL NAME" in line:
                    next_header_param = "material_name"
                    data_type = str
                elif "NOMINAL DENSITY" in line:
                    next_header_param = "nominal_density"
                    data_type = float

        if material_name := header.pop("material_name", None):
            # split name and chemical formula
            splitted = Material.split_name_and_chemical_formula(material_name)
            header.update(splitted)

        return header

    @staticmethod
    def get_identifier_from_filename(filepath: PathLike) -> str:
        """Return the material identifier, i.e.

        <material_identifier>__<min_keV>_<max_keV>kev.mcgpu.
        """
        filepath = Path(filepath)
        return str(filepath.name).split("__")[0]

    @classmethod
    def from_file(cls, filepath: PathLike):
        filepath = Path(filepath)
        header = Material.parse_material_file_header(filepath)

        return cls(
            identifier=Material.get_identifier_from_filename(filepath),
            name=header["name"],
            chemical_formula=header["chemical_formula"],
            density=header["nominal_density"],
            filepath=Path(filepath),
        )

    @classmethod
    def from_package_resources(cls, filename: str):
        filepath = pkg_resources.resource_filename(
            "cbctmc", f"assets/material_files/{filename}"
        )

        return cls.from_file(filepath)


def _get_material_identifiers(kev_range: Tuple[int, int] = (5, 125)) -> List[str]:
    folder = pkg_resources.resource_filename("cbctmc", "assets/material_files")
    folder = Path(folder)
    all_materials = sorted(folder.glob(f"*__{kev_range[0]}_{kev_range[1]}kev.mcgpu"))

    return [
        Material.get_identifier_from_filename(filepath) for filepath in all_materials
    ]


MATERIALS_125KEV = {
    material_id: Material.from_package_resources(f"{material_id}__5_125kev.mcgpu")
    for material_id in _get_material_identifiers()
}

# this dict defines the material number/order used for MC-GPU input files
# the keys are sorted by material density
MATERIALS_125KEV = dict(sorted(MATERIALS_125KEV.items(), key=lambda x: x[1].density))
