from __future__ import annotations

from abc import ABC, abstractmethod
from collections import UserList
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import numpy as np

from cbctmc.common_types import PathLike
from cbctmc.mc.materials import MATERIALS_125KEV, Material


class BaseMaterialMapper(ABC):
    def _prepare(
        self,
        segmentation: np.ndarray,
        materials_output: np.ndarray | None = None,
        densities_output: np.ndarray | None = None,
    ):
        mask = segmentation > 0
        if materials_output is None:
            materials = np.zeros_like(image, dtype=np.uint8)
            densities = np.zeros_like(image, dtype=np.float32)
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
    UserList[Tuple[BaseMaterialMapper, Union[np.ndarray, PathLike]]]
):
    def execute(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        materials = None
        densities = None
        for mapper, segmentation in self:
            if segmentation is not None and not isinstance(segmentation, np.ndarray):
                # load segmentation from file and convert to numpy array
                segmentation = sitk.ReadImage(str(segmentation))
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
        body_segmentation: np.ndarray | PathLike,
        bone_segmentation: np.ndarray | PathLike,
        muscle_segmentation: np.ndarray | PathLike,
        fat_segmentation: np.ndarray | PathLike,
        liver_segmentation: np.ndarray | PathLike,
        stomach_segmentation: np.ndarray | PathLike,
        lung_segmentation: np.ndarray | PathLike,
        lung_vessel_segmentation: np.ndarray | PathLike,
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import SimpleITK as sitk

    folder = Path(
        "/datalake_fast/4d_ct_lung_uke_artifact_free/"
        "022_4DCT_Lunge_amplitudebased_complete"
    )

    image = sitk.ReadImage(str(folder / "phase_00.nii"))
    image = sitk.GetArrayFromImage(image).swapaxes(0, 2)

    mapper_pipeline = MaterialMapperPipeline.create_default_pipeline(
        body_segmentation=folder / "segmentations/phase_00/body.nii.gz",
        bone_segmentation=folder / "segmentations/phase_00/upper_body_bones.nii.gz",
        muscle_segmentation=folder / "segmentations/phase_00/upper_body_muscles.nii.gz",
        fat_segmentation=folder / "segmentations/phase_00/upper_body_fat.nii.gz",
        liver_segmentation=folder / "segmentations/phase_00/liver.nii.gz",
        stomach_segmentation=folder / "segmentations/phase_00/stomach.nii.gz",
        lung_segmentation=folder / "segmentations/phase_00/lung.nii.gz",
        lung_vessel_segmentation=folder / "segmentations/phase_00/lung_vessels.nii.gz",
    )

    materials, densities = mapper_pipeline.execute(image)

    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
    ax[0].imshow(image[:, :, 66])
    ax[1].imshow(materials[:, :, 66])
    ax[2].imshow(densities[:, :, 66])
