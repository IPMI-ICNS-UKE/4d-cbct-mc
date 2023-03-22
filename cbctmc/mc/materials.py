import re
from dataclasses import dataclass
from pathlib import Path

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
    def get_material_number(material_id: str) -> int:
        try:
            return MATERIAL_IDENTIFIERS.index(material_id) + 1
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

    @classmethod
    def from_file(cls, filepath: PathLike):
        filepath = Path(filepath)
        header = Material.parse_material_file_header(filepath)

        return cls(
            identifier=str(filepath.name).split("__")[0],
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


# this list defines the material number/order used for MC-GPU input files
MATERIAL_IDENTIFIERS = [
    "acryl",
    "adipose",
    "air",
    "blood",
    "bone_020",
    "bone_050",
    "bone_100",
    "cartilage",
    "delrin",
    "glands_others",
    "h20",
    "ldpe",
    "liver",
    "lung",
    "muscle_tissue",
    "pmp",
    "polystrene",
    "red_marrow",
    "soft_tissue",
    "stomach_intestines",
    "teflon",
]

MATERIALS_125KEV = {
    material_id: Material.from_package_resources(f"{material_id}__5_125kev.mcgpu")
    for material_id in MATERIAL_IDENTIFIERS
}
