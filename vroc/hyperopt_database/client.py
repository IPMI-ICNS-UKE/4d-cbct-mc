from __future__ import annotations

import pickle
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from peewee import prefetch

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
                orm.Metric,
                orm.Image,
                orm.ImagePairFeature,
                orm.Run,
                orm.RunMetrics,
            )
        )

    def insert_metric(self, name: str, lower_is_better: bool = True) -> orm.Metric:
        return orm.Metric.create(name=name, lower_is_better=lower_is_better)

    def fetch_metric(self, name: str) -> orm.Metric:
        return orm.Metric.get(name=name)

    def get_or_create_metric(
        self, name: str, lower_is_better: bool = True
    ) -> orm.Metric:
        metric, _ = orm.Metric.get_or_create(name=name, lower_is_better=lower_is_better)

        return metric

    def get_or_create_image(
        self, image_name: str, modality: str, anatomy: str, dataset: str
    ) -> orm.Image:
        modality, _ = orm.Modality.get_or_create(name=modality.upper())
        anatomy, _ = orm.Anatomy.get_or_create(name=anatomy.upper())
        dataset, _ = orm.Dataset.get_or_create(name=dataset.upper())

        return orm.Image.get_or_create(
            name=image_name, modality=modality, anatomy=anatomy, dataset=dataset
        )[0]

    def fetch_image(self, **kwargs) -> orm.Image:
        return orm.Image.get(**kwargs)

    def insert_run(
        self, moving_image: orm.Image, fixed_image: orm.Image, parameters: dict
    ) -> orm.Run:
        return orm.Run.create(
            moving_image=moving_image, fixed_image=fixed_image, parameters=parameters
        )

    def insert_run_metric(
        self, run: orm.Run, metric: orm.Metric, value_before: float, value_after: float
    ):
        return orm.RunMetrics.create(
            run=run, metric=metric, value_before=value_before, value_after=value_after
        )

    def fetch_runs(
        self,
        moving_image: orm.Image,
        fixed_image: orm.Image,
        metric: orm.Metric | None = None,
        as_dataframe: bool = False,
    ) -> Union[List[dict], pd.DataFrame]:
        runs = orm.Run.select().where(
            (orm.Run.moving_image == moving_image)
            & (orm.Run.fixed_image == fixed_image)
        )

        runs = prefetch(runs, orm.RunMetrics)
        if not metric:
            selected_metrics = list(orm.Metric.select())
        else:
            selected_metrics = [metric]

        def to_dict(run: orm.Run):
            data = run.__data__.copy()

            data["run_metrics"] = [
                {
                    "name": run_metric.metric.name,
                    "value_before": run_metric.value_before,
                    "value_after": run_metric.value_after,
                }
                for run_metric in run.run_metrics
                if run_metric.metric in selected_metrics
            ]

            return data

        runs = [to_dict(run) for run in runs]

        if as_dataframe:
            runs = DatabaseClient._runs_to_dataframe(runs)

        return runs

    def fetch_best_run(
        self, moving_image: orm.Image, fixed_image: orm.Image, metric: orm.Metric
    ) -> dict:
        return self.fetch_best_runs(
            moving_image=moving_image, fixed_image=fixed_image, metric=metric, k=1
        )[0]

    def fetch_best_runs(
        self,
        moving_image: orm.Image,
        fixed_image: orm.Image,
        metric: orm.Metric,
        k: int = 5,
    ) -> List[dict]:
        runs = self.fetch_runs(
            moving_image=moving_image, fixed_image=fixed_image, metric=metric
        )

        reverse = not metric.lower_is_better

        runs = sorted(
            runs, key=lambda run: run["run_metrics"][0]["value_after"], reverse=reverse
        )

        return runs[:k]

    def fetch_image_pairs(self) -> List[Dict[str, orm.Image]]:
        image_pairs = orm.Run.select(
            orm.Run.moving_image, orm.Run.fixed_image
        ).distinct()

        return [
            {
                "moving_image": image_pair.moving_image,
                "fixed_image": image_pair.fixed_image,
            }
            for image_pair in image_pairs
        ]

    def insert_image_pair_feature(
        self,
        moving_image: orm.Image,
        fixed_image: orm.Image,
        feature_name: str,
        feature: np.ndarray,
        overwrite: bool = False,
    ):
        query = orm.ImagePairFeature.insert(
            moving_image=moving_image,
            fixed_image=fixed_image,
            feature_name=feature_name,
            feature=pickle.dumps(feature),
        )
        if overwrite:
            query = query.on_conflict_replace()

        query.execute()

    def fetch_image_pair_feature(
        self, moving_image: orm.Image, fixed_image: orm.Image, feature_name: str
    ):
        image_pair_feature = (
            orm.ImagePairFeature.select()
            .where(
                (orm.ImagePairFeature.moving_image == moving_image)
                & (orm.ImagePairFeature.fixed_image == fixed_image)
                & (orm.ImagePairFeature.feature_name == feature_name)
            )
            .first()
        )

        feature = pickle.loads(image_pair_feature.feature)

        return feature

    @staticmethod
    def _expand_dict(run: dict, key: str, prefix: str = ""):
        run = run.copy()
        params = run.pop(key)
        for param_name, param_value in params.items():
            run[prefix + param_name] = param_value

        return run

    @staticmethod
    def _runs_to_dataframe(runs: List[dict]) -> pd.DataFrame:
        if isinstance(runs, dict):
            runs = [runs]
        for i in range(len(runs)):
            run = runs[i]

            # include parameters nested dict in top level
            parameters = run.pop("parameters")
            for param_name, param_value in parameters.items():
                run[param_name] = param_value

            # include run_metrics nested list of dicts in top level
            run_metrics = run.pop("run_metrics")
            for run_metric in run_metrics:
                metric_name = run_metric["name"].lower()
                run[f"{metric_name}_before"] = run_metric["value_before"]
                run[f"{metric_name}_after"] = run_metric["value_after"]

            # put modified run dict back into list
            runs[i] = run

        return pd.DataFrame.from_records(runs, index="uuid")


if __name__ == "__main__":

    client = DatabaseClient("/datalake/learn2reg/param_sampling.sqlite")

    tre_mean = client.fetch_metric(name="TRE_MEAN")
    moving_image = client.fetch_image(
        name="imagesTr/NLST_0001_0001.nii.gz",
        modality="CT",
        anatomy="LUNG",
        dataset="NLST",
    )
    fixed_image = client.fetch_image(
        name="imagesTr/NLST_0001_0000.nii.gz",
        modality="CT",
        anatomy="LUNG",
        dataset="NLST",
    )

    # runs = client.fetch_runs(moving_image, fixed_image, as_dataframe=True)
    image_pairs = client.fetch_image_pairs()

    for image_pair in image_pairs:
        runs = client.fetch_best_runs(
            moving_image=image_pair["moving_image"],
            fixed_image=image_pair["fixed_image"],
            metric=tre_mean,
            k=1000000,
        )

        p = runs[0]["parameters"]
        pp = runs[5000]["parameters"]

        p = np.array(list(p.values())).reshape(1, -1)
        pp = np.array(list(pp.values())).reshape(1, -1)

        d = np.abs((p - pp) / p)
        d = d.mean()

        break
