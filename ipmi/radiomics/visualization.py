import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tikzplotlib as tpl

sns.set_context("paper", rc={})


def pairplot_features(features: pd.DataFrame, classes: pd.Series):
    combined = features.copy()
    combined["class"] = classes

    rename_features = {
        col_name: col_name.replace("original_", "") for col_name in features.columns
    }
    combined = combined.rename(columns=rename_features)
    with sns.plotting_context("paper"):
        return sns.pairplot(data=combined, hue="class")


def plot_feature_distributions(
    features: pd.DataFrame,
    classes: pd.Series,
    plot_kwargs: dict = None,
    print_tikz: bool = False,
):
    _plot_kwargs = dict(
        stat="probability",
        kde=True,
        hue="class",
        common_norm=False,
    )
    if plot_kwargs:
        _plot_kwargs.update(plot_kwargs)

    combined = features.astype(np.float32)
    combined["class"] = classes

    rename_features = {
        col_name: col_name.replace("original_", "") for col_name in features.columns
    }
    combined = combined.rename(columns=rename_features)

    with sns.plotting_context("paper"):
        for feature_name in [name for name in combined.columns if name != "class"]:
            fig, ax = plt.subplots()

            sns.histplot(
                data=combined,
                x=feature_name,
                stat="probability",
                kde=True,
                hue="class",
                common_norm=False,
                ax=ax,
            )

            ax.set_title(f"{feature_name}")
            ax.set_xlabel("Feature Value")
            ax.set_ylabel("Probability Density")

            if print_tikz:
                tikz_code = tpl.get_tikz_code(figure=fig)
                print(f"*** PLOT {feature_name} ***")
                print(tikz_code)
                print("\n")
