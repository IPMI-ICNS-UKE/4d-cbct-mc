import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import scipy
from pathlib import Path
from typing import Sequence, List


class MCSpeedUpDataset(Dataset):
    def __init__(self, filepaths: Sequence[dict[str]]):
        self.filepaths = filepaths

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
            low_photon = np.moveaxis(low_photon, -1, 0)
            # add color channel
            high_photon = high_photon[np.newaxis]

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
        if not isinstance(item, slice):
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
        runs: Sequence[int] = range(10),
        phases: Sequence[int] = (0, 5),
        projections: Sequence[int] = range(90),
    ) -> "MCSpeedUpDataset":
        filepaths = MCSpeedUpDataset.fetch_filepaths(
            root_folder=folder,
            patient_ids=patient_ids,
            runs=runs,
            phases=phases,
            projections=projections,
        )

        return cls(filepaths=filepaths)

    @staticmethod
    def fetch_filepaths(
        root_folder: Path,
        patient_ids: Sequence[int],
        runs: Sequence[int] = range(10),
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
                        / f"HIGH_{basename}"
                        / f"HIGH_{basename}_proj_{projection:02d}.npy"
                    )
                    if not high_photon_filepath.exists():
                        raise FileNotFoundError(high_photon_filepath)

                    for run in runs:
                        low_photon_filepath = (
                            root_folder
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


def createDataset():

    arr = ["022", "024", "032", "033", "068", "069", "074", "078", "091"]
    # , "092", "104", "106", "109", "115", "116","121", "124", "132", "142", "145", "146"]
    low = []
    high = []
    for id in arr:
        for i in range(15):
            for j in (0, 5):
                for k in range(90):
                    low.append(
                        "/home/crohling/amalthea/data/results/low_pat{}".format(id)
                        + "_phase0"
                        + str(j)
                        + f"_run_{i:02d}/low_pat{id}"
                        + "_phase0"
                        + str(j)
                        + f"_run_{i:02d}"
                        + f"_proj_{k:02d}.npy"
                    )

                    high.append(
                        "/home/crohling/amalthea/data/results/HIGH_pat{}".format(id)
                        + "_phase0"
                        + str(j)
                        + "/HIGH_pat{}".format(id)
                        + "_phase0"
                        + str(j)
                        + f"_proj_{k:02d}.npy"
                    )
    return CustomImageDataset(low, high)


def createTestDataset(i):
    low = (
        f"/home/crohling/amalthea/data/results/test/test_{i:02d}"
        + f"/test_{i:02d}_proj_00.npy"
    )
    return low


if __name__ == "__main__":
    dataset = createDataset()
    print(len(dataset))

    en = np.loadtxt(
        "/home/crohling/Documents/MC-GPU_v1.3_RELEASE_2012-12-12/neuesSpectra.spc"
    )

    norm = scipy.integrate.simps(en[:, 1], en[:, 0])
    mean_energy = scipy.integrate.simps(en[:, 1] / norm * en[:, 0], en[:, 0])
    data = np.array(data)
    data = data * (0.006024 * 924 * 384) * 5e7 / mean_energy
    data = np.array(data)
    mean = np.mean(data, axis=0)
    var = np.var(data, axis=0)

    plt.plot(mean.flatten(), var.flatten(), "bo")

    plt.plot(np.arange(0, 30, 0.01), 1 / 5 * np.arange(0, 30, 0.01), "red")
    plt.show()

    print(mean.shape)

    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
