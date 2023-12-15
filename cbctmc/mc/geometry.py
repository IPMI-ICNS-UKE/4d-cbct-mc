from __future__ import annotations

import gzip
import logging
import pickle
import time
from abc import ABC, abstractmethod
from collections import UserList
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pkg_resources
import scipy.ndimage as ndi
import SimpleITK as sitk
import torch
import torch.nn as nn
from jinja2 import Environment, FileSystemLoader
from scipy.ndimage import filters
from vroc.blocks import SpatialTransformer

from cbctmc.common_types import FloatTuple3D, PathLike
from cbctmc.mc.dataio import save_text_file
from cbctmc.mc.materials import MATERIALS_125KEV, Material
from cbctmc.mc.reference import REFERENCE_MU
from cbctmc.mc.voxel_data import compile_voxel_data_string
from cbctmc.segmentation.labels import get_label_index
from cbctmc.segmentation.segmenter import MCSegmenter
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
        segmentation: np.ndarray,
    ) -> List[Tuple[np.ndarray, Material]]:
        mask: np.ndarray = segmentation > 0

        # generate a mask for the 1 voxel outline of the mask
        eroded_mask = ndi.binary_erosion(mask)
        outline_mask = mask & ~eroded_mask

        bone_100_mask = outline_mask & (image >= 300)
        bone_050_mask = mask & (image >= 300)
        bone_020_mask = mask & (150 <= image) & (image < 300)
        red_marrow_mask = mask & (image < 150)

        # bone_100_mask = mask & (image >= 400)
        # bone_050_mask = mask & (300 <= image) & (image < 400)
        # bone_020_mask = mask & (150 <= image) & (image < 300)
        # red_marrow_mask = mask & (image < 150)

        return [
            (red_marrow_mask, MATERIALS_125KEV["red_marrow"]),
            (bone_020_mask, MATERIALS_125KEV["bone_020"]),
            (bone_050_mask, MATERIALS_125KEV["bone_050"]),
            (bone_100_mask, MATERIALS_125KEV["bone_100"]),
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
    def __init__(self, use_air: bool = False):
        super().__init__(
            target_material=MATERIALS_125KEV["air"]
            if use_air
            else MATERIALS_125KEV["lung"]
        )


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
        # image shoud already be resampled to the
        # target image spacing (passed via image_spacing)

        materials = None
        densities = None
        for mapper, segmentation in self:
            if segmentation is None:
                logger.info(f"Skipping {mapper} (No segmentation given)")
                continue
            elif isinstance(segmentation, (str, Path)):
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
            elif not isinstance(segmentation, np.ndarray):
                raise ValueError("Unsupported segmentytion type")

            logger.info(f"Executing {mapper}")
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
        model: nn.Module | None = None,
        model_device: str = "cuda",
    ):
        # the order is important
        pipeline = [
            (BodyROIMaterialMapper(), body_segmentation),
            (BoneMaterialMapper(), bone_segmentation),
            (LungMaterialMapper(use_air=True), lung_segmentation),
            (LiverMaterialMapper(), liver_segmentation),
            (StomachMaterialMapper(), stomach_segmentation),
            (MuscleMaterialMapper(), muscle_segmentation),
            (FatMaterialMapper(), fat_segmentation),
            (AirMaterialMapper(), body_segmentation),
            (LungVesselsMaterialMapper(), lung_vessel_segmentation),
        ]

        instance = cls(pipeline)
        instance._model = model
        instance._device = model_device

        return instance


class MCGeometry:
    def __init__(
        self,
        materials: np.ndarray,
        densities: np.ndarray,
        mus: np.ndarray | None = None,
        image_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        image_direction: Tuple[float, ...] | None = None,
        image_origin: Tuple[float, float, float] | None = None,
    ):
        if materials.shape != densities.shape:
            raise ValueError(
                f"Shape mismatch: {materials.shape=} != {densities.shape=}"
            )

        self.materials = materials
        self.densities = densities
        self.mus = mus

        self.image_spacing = image_spacing
        if not image_direction:
            image_direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        if not image_origin:
            image_origin = tuple(size / 2 for size in self.image_size)

        self.image_direction = image_direction
        self.image_origin = image_origin

    def warp(self, vector_field: np.ndarray, device: str = "cpu") -> MCGeometry:
        # check vector_field shape: (batch, n_spatial_dims, x_size, y_size, z_size)
        if vector_field.ndim != 5:
            vector_field = vector_field[None, ...]

        if vector_field.shape[1] != 3:
            raise ValueError(
                "Expected vector_field to have 3 spatial dimensions, "
                f"but got {vector_field.shape=}"
            )

        vector_field = torch.as_tensor(vector_field, dtype=torch.float32)
        materials = torch.as_tensor(self.materials[None, None, ...], dtype=torch.uint8)
        densities = torch.as_tensor(
            self.densities[None, None, ...], dtype=torch.float32
        )

        spatial_transformer = SpatialTransformer().to(device)
        warped_materials = spatial_transformer(
            materials,
            transformation=vector_field,
            mode="nearest",
            default_value=0,
        )
        warped_materials = warped_materials[0, 0].cpu().numpy()
        warped_densities = spatial_transformer(
            densities,
            transformation=vector_field,
            interpolation="nearest",
            mode=0,
        )
        warped_densities = warped_densities[0, 0].cpu().numpy()

        if self.mus is not None:
            mus = torch.as_tensor(self.mus[None, None, ...], dtype=torch.float32)
            warped_mus = spatial_transformer(
                mus,
                transformation=vector_field,
                interpolation="nearest",
                default_value=0,
            )
            warped_mus = warped_mus[0, 0].cpu().numpy()
        else:
            warped_mus = None
        return MCGeometry(
            materials=warped_materials,
            densities=warped_densities,
            mus=warped_mus,
            image_spacing=self.image_spacing,
            image_direction=self.image_direction,
            image_origin=self.image_origin,
        )

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
        array.SetOrigin(self.image_origin)
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
        segmenter: MCSegmenter | None = None,
        segmenter_kwargs: dict | None = None,
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
            logger.info(f"Resampling image to image spacing of {image_spacing}")
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

        if segmenter:
            # if a segmenter is given: predict segmentations
            segmentation, _ = segmenter.segment(image)
            segmenter.clear_cache()

            body_segmentation = segmentation[get_label_index("background")] == 0
            bone_segmentation = segmentation[get_label_index("upper_body_bones")]
            muscle_segmentation = segmentation[get_label_index("upper_body_muscles")]
            fat_segmentation = segmentation[get_label_index("upper_body_fat")]
            liver_segmentation = segmentation[get_label_index("liver")]
            stomach_segmentation = segmentation[get_label_index("stomach")]
            lung_segmentation = segmentation[get_label_index("lung")]
            lung_vessel_segmentation = segmentation[get_label_index("lung_vessels")]

        else:
            body_segmentation = None
            bone_segmentation = None
            muscle_segmentation = None
            fat_segmentation = None
            liver_segmentation = None
            stomach_segmentation = None
            lung_segmentation = None
            lung_vessel_segmentation = None

        # passed segmentation filepaths overwrite predicted segmentations
        mapper_pipeline = MaterialMapperPipeline.create_default_pipeline(
            body_segmentation=body_segmentation_filepath or body_segmentation,
            bone_segmentation=bone_segmentation_filepath or bone_segmentation,
            muscle_segmentation=muscle_segmentation_filepath or muscle_segmentation,
            fat_segmentation=fat_segmentation_filepath or fat_segmentation,
            liver_segmentation=liver_segmentation_filepath or liver_segmentation,
            stomach_segmentation=stomach_segmentation_filepath or stomach_segmentation,
            lung_segmentation=lung_segmentation_filepath or lung_segmentation,
            lung_vessel_segmentation=lung_vessel_segmentation_filepath
            or lung_vessel_segmentation,
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
        materials = np.rot90(materials, k=3, axes=(0, 1))
        densities = np.rot90(densities, k=3, axes=(0, 1))
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


class CylindricalPhantomMixin:
    @staticmethod
    def cylindrical_mask(
        shape: Tuple[int, int, int],
        center: Tuple[float, float, float],
        radius: float,
        height: float,
    ):
        x = np.arange(0, shape[0])
        y = np.arange(0, shape[1])
        z = np.arange(0, shape[2])
        x, y, z = np.meshgrid(x, y, z, indexing="ij")
        mask = (
            ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2)
            & (z >= center[2] - height / 2)
            & (z < center[2] + height / 2)
        )

        return mask


class MCCatPhan604Geometry(MCGeometry, CylindricalPhantomMixin):
    PHANTOM_BODY = {
        "h2o": {
            "material": MATERIALS_125KEV["h2o"],
            "angle": 0.0,
            "distance": 0.0,
            "radius": 100.0,
            "length": 100.0,
        }
    }

    CIRCULAR_SYMMETRY_ROIS = {
        "air_1": {
            "material": MATERIALS_125KEV["air"],
            "angle": 135,
            "distance": 35.355,
            "radius": 1.5,
            "length": 24.0,
        },
        "air_2": {
            "material": MATERIALS_125KEV["air"],
            "angle": 45,
            "distance": 35.355,
            "radius": 1.5,
            "length": 24.0,
        },
        "air_3": {
            "material": MATERIALS_125KEV["air"],
            "angle": 315,
            "distance": 35.355,
            "radius": 1.5,
            "length": 24.0,
        },
        "air_4": {
            "material": MATERIALS_125KEV["air"],
            "angle": 225,
            "distance": 35.355,
            "radius": 1.5,
            "length": 24.0,
        },
    }

    SENSITOMETRY_ROIS = {
        "air_1": {
            "material": MATERIALS_125KEV["air"],
            "angle": 90,
            "distance": 58.7,
            "radius": 6.5,
            "length": 24.0,
        },
        "teflon": {
            "material": MATERIALS_125KEV["teflon"],
            "angle": 60,
            "distance": 58.7,
            "radius": 6.5,
            "length": 24.0,
        },
        "delrin": {
            "material": MATERIALS_125KEV["delrin"],
            "angle": 0,
            "distance": 58.7,
            "radius": 6.5,
            "length": 24.0,
        },
        "bone_020": {
            "material": MATERIALS_125KEV["bone_020"],
            "angle": 330,
            "distance": 58.7,
            "radius": 6.5,
            "length": 24.0,
        },
        "acrylic": {
            "material": MATERIALS_125KEV["acrylic"],
            "angle": 300,
            "distance": 58.7,
            "radius": 6.5,
            "length": 24.0,
        },
        "air_2": {
            "material": MATERIALS_125KEV["air"],
            "angle": 270,
            "distance": 58.7,
            "radius": 6.5,
            "length": 24.0,
        },
        "polystyrene": {
            "material": MATERIALS_125KEV["polystyrene"],
            "angle": 240,
            "distance": 58.7,
            "radius": 6.5,
            "length": 24.0,
        },
        "ldpe": {
            "material": MATERIALS_125KEV["ldpe"],
            "angle": 180,
            "distance": 58.7,
            "radius": 6.5,
            "length": 24.0,
        },
        "bone_050": {
            "material": MATERIALS_125KEV["bone_050"],
            "angle": 150,
            "distance": 58.7,
            "radius": 6.5,
            "length": 24.0,
        },
        "pmp": {
            "material": MATERIALS_125KEV["pmp"],
            "angle": 120,
            "distance": 58.7,
            "radius": 6.5,
            "length": 24.0,
        },
        "water": {
            "material": MATERIALS_125KEV["h2o"],
            "angle": 0,
            "distance": 0,
            "radius": 30,
            "length": 40,
        },
    }

    def __init__(
        self,
        shape: Tuple[int, int, int] = (500, 500, 500),
        image_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        reference_mu: dict[str, float] | None = None,
    ):
        phantom_center = np.array(shape) / 2

        air_material = MATERIALS_125KEV["air"]
        materials = np.full(shape, fill_value=air_material.number, dtype=np.uint8)
        densities = np.full(shape, fill_value=air_material.density, dtype=np.float32)
        if not reference_mu:
            reference_mu = REFERENCE_MU

        mus = np.full(shape, fill_value=reference_mu["air"], dtype=np.float32)

        for roi_group in (
            MCCatPhan604Geometry.PHANTOM_BODY,
            MCCatPhan604Geometry.SENSITOMETRY_ROIS,
            MCCatPhan604Geometry.CIRCULAR_SYMMETRY_ROIS,
        ):
            for roi_name, roi in roi_group.items():
                # convert to rad
                phi = roi["angle"] * np.pi / 180.0
                roi_center = np.array([np.cos(phi), -np.sin(phi), 0.0])
                roi_center = (roi_center * roi["distance"]) + phantom_center

                roi_mask = MCCatPhan604Geometry.cylindrical_mask(
                    shape=shape,
                    center=roi_center,
                    radius=roi["radius"],
                    height=roi["length"],
                )

                materials[roi_mask] = roi["material"].number
                densities[roi_mask] = roi["material"].density

                mus[roi_mask] = reference_mu[roi["material"].identifier]

        super().__init__(
            materials=materials,
            densities=densities,
            mus=mus,
            image_spacing=image_spacing,
        )

    @staticmethod
    def calculate_roi_statistics(
        image: np.ndarray,
        radius_margin: float = 1.0,
        height_margin: float = 1.0,
    ):
        phantom_center = np.array(image.shape) / 2
        results = {}
        for roi_name, roi in MCCatPhan604Geometry.SENSITOMETRY_ROIS.items():
            # convert to rad
            phi = roi["angle"] * np.pi / 180.0
            roi_center = np.array([np.cos(phi), -np.sin(phi), 0.0])
            roi_center = (roi_center * roi["distance"]) + phantom_center

            roi_mask = MCCatPhan604Geometry.cylindrical_mask(
                shape=image.shape,
                center=roi_center,
                radius=roi["radius"] - radius_margin,
                height=roi["length"] - 2 * height_margin,
            )
            roi = image[roi_mask]
            stats = {
                "min": float(np.min(roi)),
                "max": float(np.max(roi)),
                "mean": float(np.mean(roi)),
                "p25": float(np.percentile(roi, 25)),
                "p50": float(np.percentile(roi, 50)),
                "p75": float(np.percentile(roi, 75)),
                "std": float(np.std(roi)),
                "evaluated_voxels": roi.size,
            }

            results[roi_name] = stats
        return results


class MCWaterPhantomGeometry(MCGeometry, CylindricalPhantomMixin):
    PHANTOM_BODY = {
        "h2o": {
            "material": MATERIALS_125KEV["h2o"],
            "angle": 0.0,
            "distance": 0.0,
            "radius": 100.0,
            "length": 150.0,
        }
    }

    SENSITOMETRY_ROIS = {
        "water": {
            "material": MATERIALS_125KEV["h2o"],
            "angle": 0,
            "distance": 0,
            "radius": 30,
            "length": 40,
        },
    }

    def __init__(
        self,
        shape: Tuple[int, int, int] = (500, 500, 500),
        image_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        radius: float | None = None,
        length: float | None = None,
    ):
        # check if image spacing is isotropic
        if len(set(image_spacing)) > 1:
            raise ValueError("Image spacing must be isotropic")

        self.image_spacing = image_spacing
        self.isotropic_image_spacing = image_spacing[0]

        phantom_center = np.array(shape) / 2

        air_material = MATERIALS_125KEV["air"]
        materials = np.full(shape, fill_value=air_material.number, dtype=np.uint8)
        densities = np.full(shape, fill_value=air_material.density, dtype=np.float32)

        for roi_name, roi in MCWaterPhantomGeometry.PHANTOM_BODY.items():
            # convert to rad
            phi = roi["angle"] * np.pi / 180.0
            roi_center = np.array([np.cos(phi), -np.sin(phi), 0.0])
            roi_center = (roi_center * roi["distance"]) + phantom_center

            roi_mask = MCCatPhan604Geometry.cylindrical_mask(
                shape=shape,
                center=roi_center,
                radius=(radius or roi["radius"]) / self.isotropic_image_spacing,
                height=(length or roi["length"]) / self.isotropic_image_spacing,
            )

            materials[roi_mask] = roi["material"].number
            densities[roi_mask] = roi["material"].density

        super().__init__(
            materials=materials, densities=densities, image_spacing=image_spacing
        )

    @staticmethod
    def calculate_roi_statistics(
        image: np.ndarray,
        radius_margin: float = 1.0,
        height_margin: float = 5.0,
    ):
        phantom_center = np.array(image.shape) / 2
        results = {}
        for roi_name, roi in MCWaterPhantomGeometry.SENSITOMETRY_ROIS.items():
            # convert to rad
            phi = roi["angle"] * np.pi / 180.0
            roi_center = np.array([np.cos(phi), -np.sin(phi), 0.0])
            roi_center = (roi_center * roi["distance"]) + phantom_center

            roi_mask = MCCatPhan604Geometry.cylindrical_mask(
                shape=image.shape,
                center=roi_center,
                radius=roi["radius"] - radius_margin,
                height=roi["length"] - 2 * height_margin,
            )
            roi = image[roi_mask]
            stats = {
                "min": float(np.min(roi)),
                "max": float(np.max(roi)),
                "mean": float(np.mean(roi)),
                "p25": float(np.percentile(roi, 25)),
                "p50": float(np.percentile(roi, 50)),
                "p75": float(np.percentile(roi, 75)),
                "std": float(np.std(roi)),
                "evaluated_voxels": roi.size,
            }

            results[roi_name] = stats
        return results


class MCLinePairPhantomGeometry(MCWaterPhantomGeometry):
    def __init__(
        self,
        line_gap: int,
        line_material: Material = MATERIALS_125KEV["aluminium"],
        radius: float | None = None,
        length: float | None = None,
        shape: Tuple[int, int, int] = (500, 500, 500),
        image_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        super().__init__(
            shape=shape, image_spacing=image_spacing, radius=radius, length=length
        )
        # check if line gap is a multiple of the image spacing
        if line_gap % self.image_spacing[0] != 0:
            raise ValueError("Line gap must be a multiple of the image spacing")

        self.line_gap_voxels = int(line_gap / self.isotropic_image_spacing)
        self.line_material = line_material
        self.n_lines = 4
        self.line_depth_voxels = int(20 / self.isotropic_image_spacing)

        self._add_line_pairs()

    def _create_line_pair_mask(self):
        mask = np.zeros(
            (
                ((2 * self.n_lines - 1) * self.line_gap_voxels),
                self.line_depth_voxels,
                self.line_depth_voxels,
            )
        )

        for i in range(0, mask.shape[0], 2 * self.line_gap_voxels):
            mask[i : i + self.line_gap_voxels] = 1

        return mask

    def _add_line_pairs(self):
        mask = self._create_line_pair_mask()

        # pad mask to full phantom size to place line pairs in phantom center
        pad_width = tuple(
            (
                before := (self.image_shape[i] - mask.shape[i]) // 2,
                self.image_shape[i] - mask.shape[i] - before,
            )
            for i in range(len(self.image_shape))
        )
        mask = np.pad(mask, pad_width=pad_width, mode="constant", constant_values=0)
        mask = mask.astype(bool)
        self.materials[mask] = self.line_material.number
        self.densities[mask] = self.line_material.density


if __name__ == "__main__":
    geometry = MCLinePairPhantomGeometry(
        line_gap=4,
        image_spacing=(0.25, 0.25, 0.25),
        radius=30,
        length=30,
        shape=(250, 250, 125),
    )
    geometry.save_density_image("/datalake2/mc_test/geometry_densities.nii.gz")
