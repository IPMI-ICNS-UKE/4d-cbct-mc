import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import scipy
from pathlib import Path
from typing import Sequence, List


class MCSpeedUpDataset(Dataset):
    def __init__(self, filepaths: Sequence[dict[str]], eight: bool = False):
        self.filepaths = filepaths
        self.eight = eight

    def __len__(self):
        return len(self.filepaths)

    def _get_data(self, filepaths: Sequence[dict[str]]) -> List[dict[str]]:
        data = []
        for filepath in filepaths:
            low_photon_filepath = filepath["low_photon_filepath"]
            high_photon_filepath = filepath["high_photon_filepath"]

            low_photon = np.load(low_photon_filepath)
            high_photon = np.load(high_photon_filepath)

            # move color channel from last to first axis
            if not self.eight:
                low_photon = np.moveaxis(low_photon, -1, 0)
            # add color channel
            high_photon = high_photon[np.newaxis]

            low_photon = np.asarray(low_photon, dtype=np.float32)
            high_photon = np.asarray(high_photon, dtype=np.float32)

            data.append(
                {
                    "patient_id": filepath["patient_id"],
                    "phase": filepath["phase"],
                    "projection": filepath["projection"],
                    "run": filepath["run"],
                    "low_photon": low_photon,
                    "high_photon": high_photon,
                }
            )
        return data

    def __getitem__(self, item):

        if isinstance(item, np.ndarray):
            sliced = [self.filepaths[i] for i in item]
        elif not isinstance(item, slice):
            item = slice(item, item + 1)
            sliced = self.filepaths[item]

        data = self._get_data(sliced)

        if len(data) == 1:
            data = data[0]

        return data

    @classmethod
    def from_folder(
        cls,
        folder: Path,
        patient_ids: Sequence[int],
        runs: Sequence[int] = range(15),
        phases: Sequence[int] = (0, 5),
        projections: Sequence[int] = range(90),
        eight: bool = False
    ) -> "MCSpeedUpDataset":
        filepaths = MCSpeedUpDataset.fetch_filepaths(
            root_folder=folder,
            patient_ids=patient_ids,
            runs=runs,
            phases=phases,
            projections=projections
        )

        return cls(filepaths=filepaths, eight=eight)

    @staticmethod
    def fetch_filepaths(
        root_folder: Path,
        patient_ids: Sequence[int],
        runs: Sequence[int] = range(15),
        phases: Sequence[int] = (0, 5),
        projections: Sequence[int] = range(90),

    ) -> List[dict]:
        filepaths = []
        for patient_id in patient_ids:
            for phase in phases:
                basename = f"pat{patient_id:03d}_phase{phase:02d}"
                for projection in projections:
                    high_photon_filepath = (
                        root_folder
                        / "HIGH_12e9"
                        / f"HIGH_{basename}"
                        / f"HIGH_{basename}_proj_{projection:02d}.npy"
                    )
                    if not high_photon_filepath.exists():
                        raise FileNotFoundError(high_photon_filepath)

                    for run in runs:
                        low_photon_filepath = (
                            root_folder
                            / "low_4.8e8"
                            / f"low_{basename}_run_{run:02d}"
                            / f"low_{basename}_run_{run:02d}_proj_{projection:02d}.npy"
                        )

                        if not low_photon_filepath.exists():
                            raise FileNotFoundError(low_photon_filepath)

                        filepaths.append(
                            {
                                "patient_id": patient_id,
                                "phase": phase,
                                "projection": projection,
                                "run": run,
                                "low_photon_filepath": low_photon_filepath,
                                "high_photon_filepath": high_photon_filepath,
                            }
                        )

        return filepaths
