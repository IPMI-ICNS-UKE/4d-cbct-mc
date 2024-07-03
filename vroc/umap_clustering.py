import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import umap
from dataset import NLSTDataset
from preprocessing import crop_background, resample_image_size
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from vroc.helper import rescale_range


def load_file(
    filepath: os.path,
    filter_features=None,
) -> dict:
    """filter_features can be None, str: NLST, list/tuple of phase(s) (phase_3,
    phase_2)"""
    assert os.path.isfile(filepath)
    data = pickle.load(open(filepath, "rb"))
    filenames = list(data.keys())
    if filter_features:
        filenames = list(filter(lambda x: x[1] in filter_features, list(filenames)))
    prepro_data = np.zeros((len(filenames), 1024))
    for index, key in enumerate(filenames):
        prepro_data[index] = data[key]

    if isinstance(filenames[0], tuple):
        filenames = ["_".join(sub) for sub in filenames]
    return filenames, prepro_data


def draw_umap(
    image_features,
    filenames,
    n_neighbors=10,
    min_dist=0.1,
    n_components=2,
    metric="euclidean",
):
    """Finds a 2-dimensional embedding of image_features that approximates an
    underlying manifold and plots the results."""
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
    )
    u = fit.fit_transform(image_features)
    fig = plt.figure()

    if n_components == 2:
        ax = fig.add_subplot(111)
        if filenames:
            # index of filenames list where nlst dataset starts
            constant = next(
                i for i, filename in enumerate(filenames) if filename.startswith("NLST")
            )
        else:
            constant = 0
        ax.scatter(
            u[:constant, 0], u[:constant, 1], c="b", marker="+", label="Not NLST"
        )
        ax.scatter(u[constant:, 0], u[constant:, 1], c="r", marker="+", label="NLST")
        ax.legend(loc=1)
    else:
        raise ValueError

    fig.savefig("umap_n.pdf")
    fig.savefig("umap_n.png")
    return filenames, u


def get_neighbors(
    x,
    keys,
    plot_cases,
    saving_dir,
    luna_dir="/media/lwimmert/5E57-FB01/learn2reg/luna16/images",
    nlst_dir="/media/lwimmert/5E57-FB01/learn2reg/NLST/imagesTr",
):
    """Based on umap-features calculates 5 nearest neighbors and returns
    cohorts (including ct image filenames).

    if plot_cases is true: plots all 5 ct images as one plot in
    saving_dir
    """
    nbrs = NearestNeighbors(
        n_neighbors=5, metric="euclidean", algorithm="ball_tree"
    ).fit(x)
    distances, indices = nbrs.kneighbors(x)
    summed_distances = distances.sum(axis=1).reshape(-1, 1)

    cohorts = keys[indices]

    if plot_cases:
        assert os.path.isdir(saving_dir)
        for j, cohort in enumerate(tqdm(indices)):
            cases = keys[cohort]

            fig, ax = plt.subplots(5, 1)

            for i, img_filepath in enumerate(cases):
                if img_filepath.endswith(".mhd"):
                    dir_ = os.path.join(luna_dir)
                elif img_filepath.endswith(".nii.gz"):
                    dir_ = os.path.join(nlst_dir)
                else:
                    raise ValueError
                # preprocessing
                image_path = os.path.join(dir_, img_filepath)
                image = NLSTDataset.load_and_preprocess(image_path)
                image = crop_background(image)
                image = resample_image_size(image, new_size=(128, 128, 128))
                image = sitk.GetArrayFromImage(image)
                image = rescale_range(
                    image, input_range=(-1024, 3071), output_range=(0, 1)
                )

                # show arbitrary slice
                ax[i].imshow(image[:, 60, :])
            fig.savefig(
                os.path.join(
                    saving_dir,
                    f"summed_cohort_distance_{np.round(summed_distances[j].item(),3)}.png",
                )
            )
    return cohorts


if __name__ == "__main__":
    database = os.path.join("/media/lwimmert/5E57-FB01/learn2reg/Features")
    filenames_nlst, prepro_data_nlst = load_file(
        os.path.join(database, "features.p"), filter_features="NLST"
    )
    filenames_dir, prepro_data_dir = load_file(
        os.path.join(database, "dirlab_features.p"), filter_features=("phase_4")
    )
    filenames = filenames_dir + filenames_nlst
    prepro_data = np.concatenate((prepro_data_dir, prepro_data_nlst), axis=0)
    filenames, u_map = draw_umap(image_features=prepro_data, filenames=filenames)

    # cohorts = get_neighbors(
    #     u_map, keys=filenames, plot_cases=False, saving_dir=None
    # )
