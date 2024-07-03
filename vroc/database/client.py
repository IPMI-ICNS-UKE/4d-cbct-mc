from __future__ import annotations

import pickle
from typing import List
from uuid import UUID

import numpy as np

import vroc.database.models as orm
from vroc.common_types import PathLike
from vroc.logger import LoggerMixin


class DatabaseClient(LoggerMixin):
    def __init__(self, database: PathLike):
        self.database = orm.database
        self.database.init(database=database)
        self._create_tables()

    def _create_tables(self):
        orm.database.create_tables(
            (
                orm.Modality,
                orm.Anatomy,
                orm.Dataset,
                orm.Image,
                orm.BestParameters,
                orm.ImagePairFeatures,
            )
        )

    def insert_image(self, image_name: str, modality: str, anatomy: str, dataset: str):
        modality, _ = orm.Modality.get_or_create(name=modality.upper())
        anatomy, _ = orm.Anatomy.get_or_create(name=anatomy.upper())
        dataset, _ = orm.Dataset.get_or_create(name=dataset.upper())

        return orm.Image.create(
            name=image_name, modality=modality, anatomy=anatomy, dataset=dataset
        )

    def fetch_image(self, uuid: UUID):
        return orm.Image.get(uuid=uuid)

    def insert_best_parameters(
        self,
        moving_image: orm.Image,
        fixed_image: orm.Image,
        parameters: dict,
        metric_before: float,
        metric_after: float,
    ):
        orm.BestParameters.create(
            moving_image=moving_image,
            fixed_image=fixed_image,
            parameters=parameters,
            metric_before=metric_before,
            metric_after=metric_after,
        )

    def fetch_best_parameters(
        self,
        moving_image: orm.Image | UUID,
        fixed_image: orm.Image | UUID,
    ) -> dict:
        best_parameters = (
            orm.BestParameters.select(orm.BestParameters.parameters)
            .where(
                (orm.BestParameters.moving_image == moving_image)
                & (orm.BestParameters.fixed_image == fixed_image)
            )
            .first()
        )

        return best_parameters.parameters

    def insert_image_pair_features(
        self, moving_image: orm.Image, fixed_image: orm.Image, features: np.ndarray
    ):
        features = pickle.dumps(features)

        orm.ImagePairFeatures.create(
            moving_image=moving_image, fixed_image=fixed_image, features=features
        )

    def fetch_image_pair_features(self) -> List:
        results = list(orm.ImagePairFeatures.select().dicts())
        for row in results:
            row["features"] = pickle.loads(row["features"])

        return results
