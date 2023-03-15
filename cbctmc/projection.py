import logging
from typing import Tuple

import itk
import numpy as np
from itk import RTK as rtk

from cbctmc.common_types import PathLike
from cbctmc.utils import rescale_range

logger = logging.getLogger(__name__)


def prepare_image_for_rtk(
    image: itk.Image, output_value_range: Tuple[float, float] = (0.0, 1.0)
) -> itk.Image:
    """Prepares a given image for RTK (e.g. for forward projection) by
    transforming it into IEC 61217 coordinate system. The passed image has to
    be in RAI orientation, that is x: R-L, y: A-P, z: I-S.

    :param image:
    :type image:
    :param output_value_range:
    :type output_value_range:
    :return:
    :rtype:
    """
    image = image.astype(np.float32)
    image_spacing = list(image.GetSpacing())

    # preprocess image
    image_arr = itk.array_from_image(image)
    image_arr = rescale_range(
        image_arr, input_range=(-1024, 3071), output_range=output_value_range
    )

    # prepare axes
    image_arr = np.swapaxes(image_arr, 0, 2)  # ITK to numpy conversion
    # CT axes (RAI orientation): x: R-L, y: A-P, z: I-S
    # RTK axes (IEC 61217)       x: R-L, y: I-S, z: P-A
    # swap y and z, then reverse z
    image_arr = np.swapaxes(image_arr, 1, 2)
    image_spacing[1], image_spacing[2] = image_spacing[2], image_spacing[1]
    image_arr = image_arr[..., ::-1]

    # transform back to ITK image
    image_arr = np.swapaxes(image_arr, 0, 2)  # numpy to ITK conversion
    # copy is needed here for right orientation
    image = itk.image_from_array(image_arr.copy())
    image.SetSpacing(image_spacing)

    image.SetOrigin(
        [
            -0.5 * n_voxels * voxel_size
            for (n_voxels, voxel_size) in zip(image.shape, image.GetSpacing())
        ]
    )

    return image


def project_forward(
    image: itk.Image,
    geometry: itk.ThreeDCircularProjectionGeometry,
    detector_size: Tuple[int, int] = (512, 384),
    detector_pixel_spacing: Tuple[float, float] = (0.776, 0.776),
) -> itk.Image[itk.F, 3]:
    """Performs a forward projection using Joseph Algorithm [1] and the given
    geometry. The default `detector_size` and `detector_pixel_spacing` are set
    according to the half-fan beam configuration of the Varian TrueBeam.

    References:
    [1] Joseph, An Improved Algorithm for Reprojecting Rays through Pixel Images, 1982

    :param image:
    :type image:
    :param geometry:
    :type geometry:
    :param detector_size:
    :type detector_size:
    :param detector_pixel_spacing:
    :type detector_pixel_spacing:
    :return:
    :rtype:
    """
    # Defines the image type
    ImageType = itk.Image[itk.F, 3]

    # Create a stack of empty projection images
    n_projections = len(geometry.GetGantryAngles())
    ConstantImageSourceType = rtk.ConstantImageSource[ImageType]
    constant_image_source = ConstantImageSourceType.New()
    origin = [
        -0.5 * detector_size[0] * detector_pixel_spacing[0],
        -0.5 * detector_size[1] * detector_pixel_spacing[1],
        0.0,
    ]
    output_size = [*detector_size, n_projections]
    output_spacing = [*detector_pixel_spacing, 1.0]
    constant_image_source.SetOrigin(origin)
    constant_image_source.SetSpacing(output_spacing)
    constant_image_source.SetSize(output_size)
    constant_image_source.SetConstant(0.0)

    forward_projection_type = rtk.JosephForwardProjectionImageFilter[
        ImageType, ImageType
    ]
    forward_projection_filter = forward_projection_type.New()

    forward_projection_filter.SetGeometry(geometry)
    forward_projection_filter.SetInput(0, constant_image_source.GetOutput())
    forward_projection_filter.SetInput(1, image)

    forward_projection = forward_projection_filter.GetOutput()
    forward_projection.Update()

    return forward_projection


def create_geometry(
    start_angle: float,
    n_projections: int,
    source_to_isocenter: float = 1000.0,
    source_to_detector: float = 1500.0,
    detector_offset_x: float = -160.0,
    detector_offset_y: float = 0.0,
    arc: float = 360.0,
) -> rtk.ThreeDCircularProjectionGeometry:
    """Generates a RTK geometry that can be used for, e.g., forward projection
    or reconstruction.

    :param start_angle:
    :type start_angle:
    :param n_projections:
    :type n_projections:
    :param source_to_isocenter:
    :type source_to_isocenter:
    :param source_to_detector:
    :type source_to_detector:
    :param detector_offset_x:
    :type detector_offset_x:
    :param detector_offset_y:
    :type detector_offset_y:
    :param arc:
    :type arc:
    :return:
    :rtype:
    """
    geometry = rtk.ThreeDCircularProjectionGeometry.New()
    for i_projection in range(0, n_projections):
        angle = start_angle + i_projection * arc / n_projections

        # The angle-direction of RTK is opposite of the Varian xim properties
        geometry.AddProjection(
            source_to_isocenter,
            source_to_detector,
            angle,
            detector_offset_x,
            detector_offset_y,
        )

    return geometry


def save_geometry(
    geometry: rtk.ThreeDCircularProjectionGeometry, output_filepath: PathLike
):
    """Saves a given RTK geometry to XML file that can be used by the RTK
    command line applications.

    :param geometry:
    :type geometry:
    :param output_filepath:
    :type output_filepath:
    :return:
    :rtype:
    """
    writer = rtk.ThreeDCircularProjectionGeometryXMLFileWriter.New()
    writer.SetFilename(str(output_filepath))
    writer.SetObject(geometry)
    writer.WriteFile()


if __name__ == "__main__":
    image = itk.imread("/datalake/4d_cbct_mc/CatPhantom/scan_2/catphan.mha")

    n_projections = 10
    geometry = create_geometry(start_angle=0, n_projections=10)

    image = prepare_image_for_rtk(image)
    forward_projection = project_forward(image, geometry=geometry)
    itk.imwrite(
        forward_projection, "/datalake/4d_cbct_mc/CatPhantom/scan_2/catphan_fp.mha"
    )

    import matplotlib.pyplot as plt

    out = itk.GetArrayFromImage(forward_projection)
    out = np.swapaxes(out, 0, 2)

    fig, ax = plt.subplots(1, n_projections)
    for i in range(n_projections):
        ax[i].imshow(out[..., i])
