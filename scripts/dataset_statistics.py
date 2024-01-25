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

    DATA_FOLDER = "/datalake3/speedup_dataset"

    SPEEDUP_MODE = "speedup_2.00x"

    dataset = MCSpeedUpDataset.from_folder(
        folder=DATA_FOLDER,
        mode=SPEEDUP_MODE,
        patient_ids=patient_ids,
    )

    low_photon_stats = {"max": [], "p99": []}
    high_photon_stats = {"max": [], "p99": []}

    pbar = tqdm(dataset, desc="Calculating dataset stats")
    for data in pbar:
        if not data:
            break
        low_photon_stats["max"].append(data["low_photon"].max())
        low_photon_stats["p99"].append(np.percentile(data["low_photon"], 99))

        high_photon_stats["max"].append(data["high_photon"].max())
        high_photon_stats["p99"].append(np.percentile(data["high_photon"], 99))

        pbar.set_postfix(
            {
                "low_photon_max": np.max(low_photon_stats["max"]),
                "low_photon_p99": np.mean(low_photon_stats["p99"]),
                "high_photon_max": np.max(high_photon_stats["max"]),
                "high_photon_p99": np.mean(high_photon_stats["p99"]),
            }
        )
