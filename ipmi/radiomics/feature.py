from collections import namedtuple
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm, trange

from ipmi.radiomics.loocv import run_train_test_binary
from ipmi.radiomics.model import RandomForestClassifier

FeatureImportanceStats = namedtuple(
    "FeatureImportanceStats", ["feature_name", "mean_importance", "std_importance"]
)


def calc_feature_importances(
    features: pd.DataFrame,
    classes: pd.Series,
    patients: pd.Series,
    n_iterations: int = 1,
    verbose: bool = False,
):
    importances = []

    if verbose:
        iterator = trange(n_iterations, desc="Calculating feature importance stats")
    else:
        iterator = range(n_iterations)

    k_fold = GroupKFold(n_splits=patients.unique().size)
    for _ in iterator:
        for i_fold, (train_indices, test_indices) in enumerate(
            k_fold.split(features, classes, groups=patients)
        ):
            features_train = features.iloc[train_indices]
            classes_train = classes.iloc[train_indices]

            model = RandomForestClassifier(
                n_estimators=100,
                valid_feature_prefixes=("original",),
                feature_scaler=RobustScaler(quantile_range=(25.0, 75.0)),
                feature_clip_range=(-3, 3),
            )
            model.train(features_train, classes=classes_train)
            importances.append(model.feature_importances)

    importance_stats = get_feature_importance_stats(importances)

    return importance_stats


def get_feature_importance_stats(feature_importances: List[List]):
    aggregated_importances = {}
    for _feature_importances in feature_importances:
        for feature_name, feature_importance in _feature_importances:
            try:
                aggregated_importances[feature_name].append(feature_importance)
            except KeyError:
                aggregated_importances[feature_name] = [feature_importance]

    importances = [
        FeatureImportanceStats(
            feature_name=feature_name,
            mean_importance=np.mean(feature_importance),
            std_importance=np.std(feature_importance),
        )
        for (feature_name, feature_importance) in aggregated_importances.items()
    ]

    return sorted(importances, key=lambda feature: -feature.mean_importance)


def find_optimal_n_features(
    features: pd.DataFrame,
    classes: pd.Series,
    patients: pd.Series,
    positive_class_name: str,
    feature_importance_stats: List,
    n_features_list: List,
    n_train_test_iterations: int = 10,
):
    if classes.unique().size > 2:
        raise NotImplementedError("Only binary for now")
    mean_aucs = []
    std_aucs = []
    for n_features in tqdm(n_features_list, desc="Finding optimal n_features"):
        top_feature_names = [
            feature.feature_name for feature in feature_importance_stats[:n_features]
        ]
        reports = []
        for _ in trange(n_train_test_iterations, desc="Running evaluation"):
            report, training_config = run_train_test_binary(
                features=features,
                classes=classes,
                patients=patients,
                top_feature_names=top_feature_names,
                positive_class_name=positive_class_name,
            )
            reports.append(report)

        mean_aucs.append(np.mean([r["roc"]["auc"] for r in reports]))
        std_aucs.append(np.std([r["roc"]["auc"] for r in reports]))

    return mean_aucs, std_aucs
