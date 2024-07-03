from typing import Callable, List

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.model_selection import GroupKFold

from ipmi.radiomics.statistics import get_youdens_j_cutoff


def run_train_test_binary(
    model_factory: Callable,
    features: pd.DataFrame,
    classes: pd.Series,
    patients: pd.Series,
    top_feature_names: List[str],
    positive_class_name: str,
    class_order: List = None,
):
    k_fold = GroupKFold(n_splits=patients.unique().size)

    top_features = features.filter(top_feature_names)
    print(f"Using features : {top_feature_names}")

    # running LOGO evaluation
    all_predictions = []
    all_ground_truths = []
    all_test_indices = []

    model_classes = None
    for i_fold, (train_indices, test_indices) in enumerate(
        k_fold.split(top_features, classes, groups=patients)
    ):
        features_train = top_features.iloc[train_indices]
        features_test = top_features.iloc[test_indices]

        classes_train = classes.iloc[train_indices]
        classes_test = classes.iloc[test_indices]

        model = model_factory()

        model.train(features_train, classes=classes_train)
        predictions = model.predict_proba(features_test)

        if model_classes is None:
            model_classes = list(model.classes)
        all_predictions.append(predictions)
        all_ground_truths.append(list(classes_test.values))
        all_test_indices.append(test_indices)

    all_predictions = np.concatenate(all_predictions)
    all_ground_truths = np.concatenate(all_ground_truths)
    all_test_indices = np.concatenate(all_test_indices)

    positive_class_idx = model_classes.index(positive_class_name)
    negative_class_name = model_classes[(positive_class_idx + 1) % 2]

    positive_pred_proba = all_predictions[:, positive_class_idx]
    positive_gt_proba = (all_ground_truths == positive_class_name).astype(np.float)

    fpr, tpr, thresholds = metrics.roc_curve(
        y_true=positive_gt_proba, y_score=positive_pred_proba
    )

    youdens_index, optimal_cutoff, operating_point = get_youdens_j_cutoff(
        fpr, tpr, thresholds
    )

    roc_auc = metrics.roc_auc_score(
        y_true=positive_gt_proba, y_score=positive_pred_proba
    )

    predicted_classes = np.array(
        [
            positive_class_name if p >= optimal_cutoff else negative_class_name
            for p in positive_pred_proba
        ]
    )

    misclassified_indices = all_test_indices[predicted_classes != all_ground_truths]
    correctly_classified_indices = all_test_indices[
        predicted_classes == all_ground_truths
    ]

    # create evaluation report
    report = metrics.classification_report(
        y_true=all_ground_truths, y_pred=predicted_classes, output_dict=True
    )
    report["prediction_probas"] = all_predictions
    report["ground_truth_classes"] = all_ground_truths
    report["predicted_classes"] = predicted_classes
    report["misclassified_indices"] = misclassified_indices
    report["correctly_classified_indices"] = correctly_classified_indices
    report["roc"] = {
        "auc": roc_auc,
        "false_positive_rates": fpr,
        "true_positive_rates": tpr,
        "thresholds": thresholds,
        "operating_point": {
            "false_positive_rate": operating_point.fpr,
            "true_positive_rate": operating_point.tpr,
            "predictor_threshold": optimal_cutoff,
            "youdens_index": youdens_index,
        },
    }

    report["confusion_matrix"] = metrics.confusion_matrix(
        y_true=all_ground_truths, y_pred=predicted_classes, labels=class_order
    )
    training_config = {}
    training_config["training"] = {
        "used_features": top_feature_names,
        "classifier_config": model.config,
    }

    return report, training_config
