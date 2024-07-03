import numpy as np
import SimpleITK as sitk


def iec61217_to_rsp(image: sitk.Image) -> sitk.Image:
    size = image.GetSize()
    spacing = image.GetSpacing()
    dimension = image.GetDimension()

    if dimension == 3:
        image.SetDirection((1, 0, 0, 0, 0, -1, 0, -1, 0))
        origin = np.subtract(
            image.GetOrigin(),
            image.TransformContinuousIndexToPhysicalPoint(
                (size[0] / 2, size[1] / 2, size[2] / 2)
            ),
        )
        origin = np.add(origin, (spacing[0] / 2, -spacing[1] / 2, -spacing[2] / 2))
    elif dimension == 4:
        image.SetDirection((1, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 1))
        origin = np.subtract(
            image.GetOrigin(),
            image.TransformContinuousIndexToPhysicalPoint(
                (size[0] / 2, size[1] / 2, size[2] / 2, size[3] / 2)
            ),
        )
        origin = np.add(
            origin, (spacing[0] / 2, -spacing[1] / 2, -spacing[2] / 2, spacing[0] / 2)
        )
    else:
        raise RuntimeError(f"Dimension {dimension} not supported")

    image.SetOrigin(origin)

    return image
