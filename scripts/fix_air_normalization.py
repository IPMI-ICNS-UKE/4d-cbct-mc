from pathlib import Path

import SimpleITK as sitk

from cbctmc.mc.projection import normalize_projections

for filepath in Path("/datalake2/mc_output").rglob("projections_total_normalized.mha"):
    total_projections = sitk.ReadImage(str(filepath.parent / "projections_total.mha"))
    air_projection = sitk.ReadImage(
        str(filepath.parent / "air" / "projections_total.mha")
    )

    spacing = total_projections.GetSpacing()
    origin = total_projections.GetOrigin()
    direction = total_projections.GetDirection()

    total_projections = sitk.GetArrayFromImage(total_projections)
    air_projection = sitk.GetArrayFromImage(air_projection)

    total_projections_normalized = normalize_projections(
        total_projections, air_projection, clip_to_air=False
    )
    total_projections_normalized = sitk.GetImageFromArray(total_projections_normalized)
    total_projections_normalized.SetSpacing(spacing)
    total_projections_normalized.SetOrigin(origin)
    total_projections_normalized.SetDirection(direction)

    sitk.WriteImage(
        total_projections_normalized,
        str(filepath.parent / "projections_total_normalized_no_clip.mha"),
    )

    print(filepath)
    break
