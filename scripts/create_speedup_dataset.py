import logging
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from ipmi.common.logger import init_fancy_logging


def write_projection_slices(
    patient_id: int,
    i_phase: int,
    mode: str,
    projections: sitk.Image,
    output_folder: Path,
):
    projections = sitk.GetArrayFromImage(projections)
    projections = np.asarray(projections, dtype=np.float32)
    for i_projection, projection in enumerate(projections):
        filename = f"pat_{patient_id:03d}__phase_{i_phase:02d}__{mode}__proj_{i_projection:03d}.npy"
        output_filepath = output_folder / filename
        np.save(output_filepath, projection)
        logger.info(f"Saved {output_filepath}")


if __name__ == "__main__":
    PHASES = (0,)
    MODES = (
        # "reference",
        # "speedup_2.00x",
        "speedup_5.00x",
        "speedup_10.00x",
        "speedup_20.00x",
        "speedup_50.00x",
    )

    mc_root_path = Path("/datalake3/nas_io/anarchy/4d_cbct_mc/speedup")
    output_folder = Path("/datalake3/speedup_dataset")
    output_folder.mkdir(exist_ok=True)

    logging.getLogger("cbctmc").setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    init_fancy_logging()

    missing_files = []
    for patient in sorted(mc_root_path.iterdir()):
        patient_id = int(patient.name[:3])

        for i_phase in PHASES:
            data_folder = patient / f"phase_{i_phase:02d}"

            projections = sitk.ReadImage(str(data_folder / "density_fp.mha"))
            write_projection_slices(
                patient_id=patient_id,
                i_phase=i_phase,
                mode="fp",
                projections=projections,
                output_folder=output_folder,
            )

            for mode in MODES:
                if (
                    filepath := data_folder / mode / "projections_total_normalized.mha"
                ).exists():
                    projections = sitk.ReadImage(str(filepath))

                    write_projection_slices(
                        patient_id=patient_id,
                        i_phase=i_phase,
                        mode=mode,
                        projections=projections,
                        output_folder=output_folder,
                    )
                else:
                    missing_files.append(filepath)
