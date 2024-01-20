from pathlib import Path

import SimpleITK as sitk

DATA_FOLDER = Path("/mnt/nas_io/anarchy/4d_cbct_mc")

samples = []

for run in range(1, 52):
    projection_filepath = (
        DATA_FOLDER
        / f"catphan/geometries/phase_00/reference/projections_total_normalized.mha"
    )
    projections = sitk.ReadImage(str(projection_filepath))
    projections = sitk.GetArrayFromImage(projections)

    samples.append(projections)
