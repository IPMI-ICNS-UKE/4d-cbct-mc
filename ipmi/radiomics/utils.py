import pandas as pd


def merge_classes(classes: pd.Series, merge: dict) -> pd.Series:
    merged_classes = classes.copy()
    for merged_class_name, members in merge.items():
        mask = merged_classes.isin(members)
        merged_classes[mask] = merged_class_name

    return merged_classes
