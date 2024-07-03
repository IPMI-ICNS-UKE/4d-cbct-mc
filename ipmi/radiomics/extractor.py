import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
import radiomics
import SimpleITK as sitk
from radiomics import featureextractor
from tqdm import tqdm

from ipmi.radiomics.dataset.iterator import BaseDatasetIterator


class FeatureExtractor(ABC):
    def __init__(self, extractor_settings: dict):
        radiomics.logger.setLevel(logging.ERROR)
        self.extractor = featureextractor.RadiomicsFeatureExtractor(
            **extractor_settings
        )

    @abstractmethod
    def _load_and_preprocess(
        self,
        image_filepath: Union[str, Path],
        segmentation_filepath: Union[str, Path],
        meta: dict,
    ) -> Tuple[sitk.Image, sitk.Image, dict]:
        raise NotImplementedError

    def extract(
        self, dataset_iterator: BaseDatasetIterator, only_first_lesion: bool = False
    ):
        all_features = []
        for image_filepath, segmentation_filepath, meta in tqdm(
            dataset_iterator, desc="Extracting features"
        ):
            image, segmentation, meta = self._load_and_preprocess(
                image_filepath=image_filepath,
                segmentation_filepath=segmentation_filepath,
                meta=meta,
            )

            label_filter = sitk.LabelStatisticsImageFilter()
            label_filter.Execute(image, segmentation)
            labels = label_filter.GetLabels()

            for label in labels:
                if label == 0:
                    continue
                features = self.extractor.execute(image, segmentation, label=label)
                for key in features.keys():
                    if isinstance(features[key], np.ndarray):
                        try:
                            # squeeze scaler numpy array
                            features[key] = features[key].item()
                        except ValueError:
                            pass

                features = {**meta, **features}
                all_features.append(features)

                if only_first_lesion:
                    break

        return pd.DataFrame(all_features)


class CTMRIFeatureExtractor(FeatureExtractor):
    def _load_and_preprocess(
        self,
        image_filepath: Union[str, Path],
        segmentation_filepath: Union[str, Path],
        meta: dict,
    ) -> Tuple[sitk.Image, sitk.Image, dict]:
        image = sitk.ReadImage(str(image_filepath))
        segmentation = sitk.ReadImage(str(segmentation_filepath), sitk.sitkUInt8)

        return image, segmentation, meta
