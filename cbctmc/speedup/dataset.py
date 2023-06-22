import logging
from pathlib import Path
from typing import List, Sequence

import numpy as np
from torch.utils.data import Dataset

from cbctmc.common_types import PathLike

logger = logging.getLogger(__name__)


class MCSpeedUpDatasetOld(Dataset):
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
        eight: bool = False,
    ) -> "MCSpeedUpDataset":
        filepaths = MCSpeedUpDataset.fetch_filepaths(
            root_folder=folder,
            patient_ids=patient_ids,
            runs=runs,
            phases=phases,
            projections=projections,
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


class MCSpeedUpDataset(Dataset):
    def __init__(self, filepaths: Sequence[dict[str]]):
        self.filepaths = filepaths

    def __len__(self):
        return len(self.filepaths)

    def _get_data(self, filepaths: Sequence[dict[str]]) -> List[dict[str]]:
        data = []
        for filepath in filepaths:
            low_photon = np.load(filepath["low_photon_filepath"])
            forward_projection = np.load(filepath["forward_projection_filepath"])
            high_photon = np.load(filepath["high_photon_filepath"])

            # add color channel
            low_photon = low_photon[np.newaxis]
            forward_projection = forward_projection[np.newaxis]
            high_photon = high_photon[np.newaxis]

            low_photon = np.asarray(low_photon, dtype=np.float32)
            forward_projection = np.asarray(forward_projection, dtype=np.float32)
            high_photon = np.asarray(high_photon, dtype=np.float32)

            data.append(
                {
                    "patient_id": filepath["patient_id"],
                    "phase": filepath["phase"],
                    "projection": filepath["projection"],
                    "low_photon": low_photon,
                    "forward_projection": forward_projection,
                    "high_photon": high_photon,
                }
            )
        return data

    def __getitem__(self, item):
        if isinstance(item, np.ndarray):
            sliced = [self.filepaths[i] for i in item]
        elif isinstance(item, slice):
            sliced = self.filepaths[item]
        elif not isinstance(item, slice):
            item = slice(item, item + 1)
            sliced = self.filepaths[item]
        else:
            raise NotImplementedError

        data = self._get_data(sliced)

        if len(data) == 1:
            data = data[0]

        return data

    @classmethod
    def from_folder(
        cls,
        folder: PathLike,
        mode: str,
        patient_ids: Sequence[int],
        phases: Sequence[int] = (0,),
        projections: Sequence[int] = range(894),
    ) -> "MCSpeedUpDataset":
        filepaths = MCSpeedUpDataset.fetch_filepaths(
            root_folder=folder,
            mode=mode,
            patient_ids=patient_ids,
            phases=phases,
            projections=projections,
        )

        return cls(filepaths=filepaths)

    @staticmethod
    def get_patient_ids_from_folder(folder: PathLike) -> list[int]:
        folder = Path(folder)
        ids = set(int(p.name[4:7]) for p in folder.iterdir())
        return sorted(tuple(ids))

    @staticmethod
    def fetch_filepaths(
        root_folder: PathLike,
        mode: str,
        patient_ids: Sequence[int],
        phases: Sequence[int] = (0,),
        projections: Sequence[int] = range(895),
    ) -> List[dict]:
        root_folder = Path(root_folder)
        filepaths = []
        for patient_id in patient_ids:
            for i_phase in phases:
                for i_projection in projections:
                    _filepaths = {
                        "patient_id": patient_id,
                        "phase": i_phase,
                        "projection": i_projection,
                        "low_photon_filepath": root_folder
                        / f"pat_{patient_id:03d}__phase_{i_phase:02d}__{mode}__proj_{i_projection:03d}.npy",
                        "forward_projection_filepath": root_folder
                        / f"pat_{patient_id:03d}__phase_{i_phase:02d}__fp__proj_{i_projection:03d}.npy",
                        "high_photon_filepath": root_folder
                        / f"pat_{patient_id:03d}__phase_{i_phase:02d}__high__proj_{i_projection:03d}.npy",
                    }

                    if all(
                        _filepaths[p].exists()
                        for p in (
                            "low_photon_filepath",
                            "forward_projection_filepath",
                            "high_photon_filepath",
                        )
                    ):
                        filepaths.append(_filepaths)
                    else:
                        logger.warning(f"Skipping {_filepaths}")

        return filepaths


if __name__ == "__main__":
    ds = MCSpeedUpDataset.from_folder(
        "/datalake2/mc_speedup",
        patient_ids=(24, 33, 68, 69, 74, 104, 106, 109, 115, 116, 121, 132, 146),
        mode="low_2",
    )
