import numpy as np


def normalized_cross_correlation(
    image: np.ndarray,
    reference_image: np.ndarray,
) -> float:
    # https://itk.org/Doxygen/html/classitk_1_1CorrelationImageToImageMetricv4.html

    image = image.flatten()
    reference_image = reference_image.flatten()

    image_sub_mean = image - image.mean()
    reference_image_sub_mean = reference_image - reference_image.mean()
    num = image_sub_mean.dot(reference_image_sub_mean) ** 2

    denom = image_sub_mean.dot(image_sub_mean) * reference_image_sub_mean.dot(
        reference_image_sub_mean
    )

    ncc = num / denom

    return ncc
