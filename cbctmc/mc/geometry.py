from __future__ import annotations

import gzip
import logging
import pickle
import time
from abc import ABC, abstractmethod
from collections import UserList
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import numpy as np
import pkg_resources
import SimpleITK as sitk
from jinja2 import Environment, FileSystemLoader

from cbctmc.common_types import FloatTuple3D, PathLike
from cbctmc.mc.dataio import save_text_file
from cbctmc.mc.materials import MATERIALS_125KEV, Material
from cbctmc.mc.voxel_data import compile_voxel_data_string
from cbctmc.utils import resample_image_spacing

logger = logging.getLogger(__name__)


class BaseMaterialMapper(ABC):
    def _prepare(
        self,
        segmentation: np.ndarray,
        materials_output: np.ndarray | None = None,
        densities_output: np.ndarray | None = None,
    ):
        mask = segmentation > 0

        if materials_output is None:
            materials = np.zeros_like(segmentation, dtype=np.uint8)
            densities = np.zeros_like(segmentation, dtype=np.float32)
        else:
            materials = materials_output
            densities = densities_output

        return mask, materials, densities

    def map_target_material(
        self,
        segmentation: np.ndarray,
        target_material: Material,
        materials_output: np.ndarray | None = None,
        densities_output: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Maps segmented volume of image to material file numbers and
        densities using given segmentation.

        Returns two arrays (materials and densities).
        """
        mask, materials, densities = self._prepare(
            segmentation=segmentation,
            materials_output=materials_output,
            densities_output=densities_output,
        )

        # simply map mask to single material
        materials[mask] = target_material.number
        densities[mask] = target_material.density

        return materials, densities

    @abstractmethod
    def map(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class SingleMaterialMapper(BaseMaterialMapper):
    def __init__(
        self,
        target_material: Material,
    ):
        self.target_material = target_material

    def map(
        self,
        image: np.ndarray,
        segmentation: np.ndarray,
        materials_output: np.ndarray | None = None,
        densities_output: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.map_target_material(
            segmentation=segmentation,
            target_material=self.target_material,
            materials_output=materials_output,
            densities_output=densities_output,
        )


class BaseMultiMaterialMapper(BaseMaterialMapper):
    def _create_segmentation_material_pairs(
        self,
        image: np.ndarray,
        segmentation: Sequence[np.ndarray],
    ) -> List[Tuple[np.ndarray, Material]]:
        raise NotImplementedError

    def map(
        self,
        image: np.ndarray,
        segmentation: Sequence[np.ndarray],
        materials_output: np.ndarray | None = None,
        densities_output: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pairs = self._create_segmentation_material_pairs(
            image=image, segmentation=segmentation
        )

        for segmentation, target_material in pairs:
            materials_output, densities_output = self.map_target_material(
                segmentation=segmentation,
                target_material=target_material,
                materials_output=materials_output,
                densities_output=densities_output,
            )

        return materials_output, densities_output


class BoneMaterialMapper(BaseMultiMaterialMapper):
    def _create_segmentation_material_pairs(
        self,
        image: np.ndarray,
        segmentation: Sequence[np.ndarray],
    ) -> List[Tuple[np.ndarray, Material]]:
        mask = segmentation > 0

        bone_100_mask = mask & (image >= 400)
        bone_050_mask = mask & (300 <= image) & (image < 400)
        bone_020_mask = mask & (150 <= image) & (image < 300)
        red_marrow_mask = mask & (image < 150)

        return [
            (bone_100_mask, MATERIALS_125KEV["bone_100"]),
            (bone_050_mask, MATERIALS_125KEV["bone_050"]),
            (bone_020_mask, MATERIALS_125KEV["bone_020"]),
            (red_marrow_mask, MATERIALS_125KEV["red_marrow"]),
        ]


class AirMaterialMapper(BaseMultiMaterialMapper):
    def _create_segmentation_material_pairs(
        self,
        image: np.ndarray,
        segmentation: Sequence[np.ndarray],
    ) -> List[Tuple[np.ndarray, Material]]:
        if segmentation is None:
            mask = np.ones_like(image, dtype=bool)
        else:
            mask = segmentation > 0

        air_mask = mask & (image < -900)

        return [
            (air_mask, MATERIALS_125KEV["air"]),
        ]


class BodyROIMaterialMapper(BaseMultiMaterialMapper):
    def _create_segmentation_material_pairs(
        self,
        image: np.ndarray,
        segmentation: Sequence[np.ndarray],
    ) -> List[Tuple[np.ndarray, Material]]:
        mask = segmentation > 0

        body_mask = mask
        background_mask = np.logical_not(body_mask)

        return [
            (body_mask, MATERIALS_125KEV["soft_tissue"]),
            (background_mask, MATERIALS_125KEV["air"]),
        ]


class LungMaterialMapper(SingleMaterialMapper):
    def __init__(self):
        super().__init__(target_material=MATERIALS_125KEV["lung"])


class LungVesselsMaterialMapper(SingleMaterialMapper):
    def __init__(self):
        super().__init__(target_material=MATERIALS_125KEV["blood"])


class LiverMaterialMapper(SingleMaterialMapper):
    def __init__(self):
        super().__init__(target_material=MATERIALS_125KEV["liver"])


class StomachMaterialMapper(SingleMaterialMapper):
    def __init__(self):
        super().__init__(target_material=MATERIALS_125KEV["stomach_intestines"])


class MuscleMaterialMapper(SingleMaterialMapper):
    def __init__(self):
        super().__init__(target_material=MATERIALS_125KEV["muscle_tissue"])


class FatMaterialMapper(SingleMaterialMapper):
    def __init__(self):
        super().__init__(target_material=MATERIALS_125KEV["adipose"])


class MaterialMapperPipeline(
    UserList[Tuple[BaseMaterialMapper, Union[np.ndarray, PathLike, None]]]
):
    def execute(
        self, image: np.ndarray, image_spacing: FloatTuple3D | None = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        materials = None
        densities = None
        for mapper, segmentation in self:
            if segmentation is None:
                logger.info(f"Skipping {mapper}")
                continue

            logger.info(f"Executing {mapper}")
            if not isinstance(segmentation, np.ndarray):
                # load segmentation from file and convert to numpy array
                segmentation = sitk.ReadImage(str(segmentation))
                if image_spacing:
                    segmentation = resample_image_spacing(
                        segmentation,
                        new_spacing=image_spacing,
                        resampler=sitk.sitkNearestNeighbor,
                        default_voxel_value=0,
                    )
                # convert to numpy, swap zyx (itk) -> xyz (numpy)
                segmentation = sitk.GetArrayFromImage(segmentation).swapaxes(0, 2)
                segmentation = np.asarray(segmentation, dtype=np.uint8)
            materials, densities = mapper.map(
                image=image,
                segmentation=segmentation,
                materials_output=materials,
                densities_output=densities,
            )

        return materials, densities

    @classmethod
    def create_default_pipeline(
        cls,
        body_segmentation: np.ndarray | PathLike | None = None,
        bone_segmentation: np.ndarray | PathLike | None = None,
        muscle_segmentation: np.ndarray | PathLike | None = None,
        fat_segmentation: np.ndarray | PathLike | None = None,
        liver_segmentation: np.ndarray | PathLike | None = None,
        stomach_segmentation: np.ndarray | PathLike | None = None,
        lung_segmentation: np.ndarray | PathLike | None = None,
        lung_vessel_segmentation: np.ndarray | PathLike | None = None,
    ):
        # the order is important
        pipeline = [
            (BodyROIMaterialMapper(), body_segmentation),
            (BoneMaterialMapper(), bone_segmentation),
            (LungMaterialMapper(), lung_segmentation),
            (LiverMaterialMapper(), liver_segmentation),
            (StomachMaterialMapper(), stomach_segmentation),
            (MuscleMaterialMapper(), muscle_segmentation),
            (FatMaterialMapper(), fat_segmentation),
            (AirMaterialMapper(), body_segmentation),
            (LungVesselsMaterialMapper(), lung_vessel_segmentation),
        ]

        return cls(pipeline)


class MCGeometry:
    def __init__(
        self,
        materials: np.ndarray,
        densities: np.ndarray,
        image_spacing: Tuple[float, float, float],
        image_direction: Tuple[float, ...] | None = None,
        image_origin: Tuple[float, float, float] | None = None,
    ):
        self.materials = materials
        self.densities = densities
        self.image_spacing = image_spacing
        self.image_direction = image_direction
        self.image_origin = image_origin

    @property
    def image_shape(self) -> Tuple[int, int, int]:
        return self.materials.shape

    @property
    def image_size(self) -> Tuple[float, float, float]:
        return tuple(sh * sp for (sh, sp) in zip(self.image_shape, self.image_spacing))

    def save(self, filepath: PathLike):
        filepath = Path(filepath)
        with gzip.open(filepath, "wb", compresslevel=6) as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: PathLike) -> MCGeometry:
        with gzip.open(filepath, "rb") as f:
            instance = pickle.load(f)

        return instance

    def save_mcgpu_geometry(self, filepath: PathLike, compress: bool = True):
        if not (self.densities > 0.0).all():
            raise ValueError("Density can not be zero or negative")

        contents = MCGeometry.create_mcgpu_geometry(
            materials=self.materials,
            densities=self.densities,
            image_spacing=self.image_spacing,
        )

        save_text_file(
            contents,
            output_filepath=filepath,
            compress=compress,
            content_type="MC-GPU geometry file",
        )

    def _array_to_itk(self, array: np.ndarray) -> sitk.Image:
        array = sitk.GetImageFromArray(array.swapaxes(0, 2))
        array.SetSpacing(self.image_spacing)
        if self.image_origin:
            array.SetOrigin(self.image_origin)
        if self.image_direction:
            array.SetDirection(self.image_direction)

        return array

    def save_material_segmentation(self, filepath: PathLike):
        materials = self._array_to_itk(self.materials)
        sitk.WriteImage(materials, str(filepath))

    def save_density_image(self, filepath: PathLike):
        densities = self._array_to_itk(self.densities)
        sitk.WriteImage(densities, str(filepath))

    @classmethod
    def from_image(
        cls,
        image_filepath: PathLike,
        body_segmentation_filepath: PathLike | None = None,
        bone_segmentation_filepath: PathLike | None = None,
        muscle_segmentation_filepath: PathLike | None = None,
        fat_segmentation_filepath: PathLike | None = None,
        liver_segmentation_filepath: PathLike | None = None,
        stomach_segmentation_filepath: PathLike | None = None,
        lung_segmentation_filepath: PathLike | None = None,
        lung_vessel_segmentation_filepath: PathLike | None = None,
        image_spacing: FloatTuple3D | None = None,
    ) -> MCGeometry:
        image = sitk.ReadImage(str(image_filepath))
        if image_spacing:
            image = resample_image_spacing(
                image,
                new_spacing=image_spacing,
                resampler=sitk.sitkLinear,
                default_voxel_value=-1000,
            )

        image_spacing = image.GetSpacing()
        image_origin = image.GetOrigin()
        image_direction = image.GetDirection()
        # convert to numpy, swap zyx (itk) -> xyz (numpy)
        image = sitk.GetArrayFromImage(image).swapaxes(0, 2)

        mapper_pipeline = MaterialMapperPipeline.create_default_pipeline(
            body_segmentation=body_segmentation_filepath,
            bone_segmentation=bone_segmentation_filepath,
            muscle_segmentation=muscle_segmentation_filepath,
            fat_segmentation=fat_segmentation_filepath,
            liver_segmentation=liver_segmentation_filepath,
            stomach_segmentation=stomach_segmentation_filepath,
            lung_segmentation=lung_segmentation_filepath,
            lung_vessel_segmentation=lung_vessel_segmentation_filepath,
        )

        materials, densities = mapper_pipeline.execute(
            image, image_spacing=image_spacing
        )

        return cls(
            materials=materials,
            densities=densities,
            image_spacing=image_spacing,
            image_direction=image_direction,
            image_origin=image_origin,
        )

    @staticmethod
    def create_mcgpu_geometry(
        materials: np.ndarray,
        densities: np.ndarray,
        image_spacing: Tuple[float, float, float],
    ) -> str:
        if materials.shape != densities.shape:
            raise ValueError(
                f"Shape mismatch: {materials.shape=} != {densities.shape=}"
            )
        n_voxels_x, n_voxels_y, n_voxels_z = materials.shape
        # Note: MC-GPU uses cm instead of mm (thus dividing by 10)
        params = {
            "n_voxels_x": n_voxels_x,
            "n_voxels_y": n_voxels_y,
            "n_voxels_z": n_voxels_z,
            "voxel_spacing_x": image_spacing[0] / 10.0,
            "voxel_spacing_y": image_spacing[1] / 10.0,
            "voxel_spacing_z": image_spacing[2] / 10.0,
        }

        logger.info(
            f"Creating MC-GPU geometry file with the following parameters: {params}"
        )

        # add voxel_data here, so it does not get logged
        t_start = time.monotonic()
        voxel_data = compile_voxel_data_string(materials=materials, densities=densities)
        t_end = time.monotonic()
        params["voxel_data"] = voxel_data

        logger.info(
            f"Compiling geometry voxel data string took: {t_end - t_start:.2f} seconds"
        )

        assets_folder = pkg_resources.resource_filename("cbctmc", "assets/templates")
        environment = Environment(loader=FileSystemLoader(assets_folder))
        template = environment.get_template("mcgpu_geometry.jinja2")
        rendered = template.render(params)

        return rendered


class MCAirGeometry(MCGeometry):
    def __init__(
        self, image_spacing: Tuple[float, float, float] = (2000.0, 2000.0, 2000.0)
    ):
        air_material = MATERIALS_125KEV["air"]

        materials = np.full((1, 1, 1), fill_value=air_material.number, dtype=np.uint8)
        densities = np.full(
            (1, 1, 1), fill_value=air_material.density, dtype=np.float32
        )

        super().__init__(
            materials=materials, densities=densities, image_spacing=image_spacing
        )
