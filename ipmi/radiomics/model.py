from abc import ABC, abstractmethod
from typing import Optional, Tuple

import pandas as pd
import sklearn as skl
import sklearn.linear_model as linear_model


class Model(ABC):
    def __init__(
        self,
        valid_feature_prefixes: Tuple[str, ...] = ("original",),
        feature_scaler=None,
        feature_clip_range: Optional[Tuple[float, float]] = None,
    ):
        self.valid_feature_prefixes = valid_feature_prefixes
        self.feature_scaler = feature_scaler
        self.feature_clip_range = feature_clip_range
        self._features = None

    def prepare_features(
        self, features: pd.DataFrame, fit_scaler: bool = False
    ) -> pd.DataFrame:
        valid_cols = [
            c
            for c in features.columns
            if any(
                c.startswith(feature_prefix)
                for feature_prefix in self.valid_feature_prefixes
            )
        ]

        features = features[valid_cols]
        if fit_scaler:
            self.feature_scaler.fit(features.values)
        if self.feature_scaler is not None:
            transformed_features = self.feature_scaler.transform(features.values)
            features = pd.DataFrame(
                transformed_features, index=features.index, columns=features.columns
            )
        if self.feature_clip_range is not None:
            features = features.clip(*self.feature_clip_range)

        return features

    def train(self, features: pd.DataFrame, classes: pd.DataFrame):
        cleaned_features = self.prepare_features(features, fit_scaler=True)
        self._features = list(cleaned_features.columns)
        self._train(features=cleaned_features, classes=classes)

    def predict_proba(self, features: pd.DataFrame):
        cleaned_features = self.prepare_features(features)
        return self._predict_proba(features=cleaned_features)

    def predict(self, features: pd.DataFrame):
        cleaned_features = self.prepare_features(features)
        return self._predict(features=cleaned_features)

    @abstractmethod
    def _train(self, features: pd.DataFrame, classes: pd.DataFrame):
        pass

    @abstractmethod
    def _predict_proba(self, features: pd.DataFrame):
        pass

    @abstractmethod
    def _predict(self, features: pd.DataFrame):
        pass


class BaseScikitLearnClassifier(Model):
    def __init__(
        self,
        valid_feature_prefixes: Tuple[str, ...] = ("original",),
        feature_scaler=None,
        feature_clip_range: Optional[Tuple[float, float]] = None,
        **classifier_settings
    ):
        super().__init__(
            valid_feature_prefixes=valid_feature_prefixes,
            feature_scaler=feature_scaler,
            feature_clip_range=feature_clip_range,
        )

        self._classifier_settings = classifier_settings
        self.classifier = self._classifier_factory()

    @abstractmethod
    def _classifier_factory(self):
        raise NotImplementedError

    def _train(self, features: pd.DataFrame, classes: pd.DataFrame):
        self.classifier.fit(X=features, y=classes)

    def _predict_proba(self, features: pd.DataFrame):
        return self.classifier.predict_proba(features)

    def _predict(self, features: pd.DataFrame):
        return self.classifier.predict(features)

    @property
    def classes(self):
        return self.classifier.classes_

    @property
    def config(self):
        return {
            "classifier": self.classifier,
            "classifier_config": self._classifier_settings,
            "valid_feature_prefixes": self.valid_feature_prefixes,
            "feature_scaler": self.feature_scaler,
            "feature_clip_range": self.feature_clip_range,
        }


class RandomForestClassifier(BaseScikitLearnClassifier):
    def _classifier_factory(self):
        return skl.ensemble.RandomForestClassifier(**self._classifier_settings)

    @property
    def feature_importances(self):
        if not self._features:
            raise RuntimeError("Model is not trained")
        assert len(self._features) == len(
            self.classifier.feature_importances_
        ), "Length mismatch"
        importances = zip(self._features, self.classifier.feature_importances_)

        return sorted(importances, key=lambda x: -x[1])


class LogisticRegression(BaseScikitLearnClassifier):
    def _classifier_factory(self):
        return linear_model.LogisticRegression(**self._classifier_settings)
