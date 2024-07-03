import numpy as np
import umap


def embed(features: np.ndarray, target=None, method: str = "umap"):
    if method == "umap":
        embedding = umap.UMAP(n_components=2)
        embedded_features = embedding.fit_transform(features, y=target)
    else:
        raise ValueError(f"Unknown method {method}")

    return embedded_features
