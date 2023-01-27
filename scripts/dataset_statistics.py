from pathlib import Path

import numpy as np
from tqdm import tqdm
from cbctmc.speedup.dataset import MCSpeedUpDataset

if __name__ == "__main__":
    patient_ids = (
        22,
        24,
        32,
        33,
        68,
        69,
        74,
        78,
        91,
        92,
        104,
        106,
        109,
        115,
        116,
        121,
        124,
        132,
        142,
        145,
        146,
    )

    possible_folders = [
        Path("/home/crohling/amalthea/data/results/"),
        Path("/mnt/gpu_server/crohling/data/results"),
    ]
    folder = next(p for p in possible_folders if p.exists())

    dataset = MCSpeedUpDataset.from_folder(
        folder=folder, patient_ids=patient_ids, runs=range(15)
    )

    low_photon_stats = {"max": [], "p99": []}
    high_photon_stats = {"max": [], "p99": []}

    for data in tqdm(dataset, desc="Calculating dataset stats"):
        low_photon_stats["max"].append(data["low_photon"].max())
        low_photon_stats["p99"].append(np.percentile(data["low_photon"], 99))

        high_photon_stats["max"].append(data["high_photon"].max())
        high_photon_stats["p99"].append(np.percentile(data["high_photon"], 99))

    for stats in (low_photon_stats, high_photon_stats):
        for key, value in stats.items():
            stats[key] = np.mean(value)
