from pathlib import Path

import numpy as np
from ipmi.common.logger import init_fancy_logging
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

import torch
from torch.utils.data import BatchSampler
from torch.utils.data import SequentialSampler

from cbctmc.speedup.sampler import GroupedSampler
from cbctmc.config import get_user_config
from cbctmc.speedup.dataset import MCSpeedUpDataset
from cbctmc.speedup.models import ResidualDenseNet2D
from cbctmc.speedup.trainer import MCSpeedUpTrainer

if __name__ == "__main__":
    import logging

    init_fancy_logging()

    logger = logging.getLogger(__name__)

    logger.setLevel(logging.DEBUG)
    logging.getLogger("cbctmc").setLevel(logging.DEBUG)

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
    train_patients, test_patients = train_test_split(
        patient_ids, train_size=0.75, random_state=42
    )
    logger.info(f"Train patients ({len(train_patients)}): {train_patients}")
    logger.info(f"Test patients ({len(test_patients)}): {test_patients}")

    user, config = get_user_config()
    logger.info(f'Current user is {user} with config {config}')

    train_dataset = MCSpeedUpDataset.from_folder(
        folder=config["root_folder"] / "results",
        patient_ids=train_patients,
        runs=range(10),
        eight=False
    )
    test_dataset = MCSpeedUpDataset.from_folder(
        folder=config["root_folder"] / "results",
        patient_ids=test_patients,
        runs=range(10),
        eight=False
    )
    # mcsampler = GroupedSampler(runs=15, data_source=train_dataset, batch_size=8, shuffle=True)
    # train_data_loader = DataLoader(
    #     train_dataset, sampler=mcsampler, num_workers=4
    # )
    train_data_loader = DataLoader(
        train_dataset, batch_size=5, shuffle=True, num_workers=4
    )

    test_data_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=4
    )

    model = ResidualDenseNet2D(
        in_channels=1,
        out_channels=2,
        growth_rate=8,
        n_blocks=4,
        n_block_layers=8,
    )
    # state = torch.load(
    #     "/home/crohling/Documents/runs/runs_clemens/e19919673f8b438f92b7d71e/models/training/step_5500.pth",
    #     map_location=config["device"])
    # model.load_state_dict(state["model"])

    optimizer = Adam(params=model.parameters(), lr=1e-5)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    trainer = MCSpeedUpTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_data_loader,
        val_loader=test_data_loader,
        run_folder=Path("/home/crohling/Documents/runs") / f"runs_{user}",
        experiment_name="mc_speed_up",
        device=config["device"],
        max_var_correction=0.5,
        scheduler=scheduler,
        scatter=True
    )
    trainer.run(steps=100_000_000, validation_interval=7001, save_interval=500)
