import torch
import torch.nn as nn
from ipmi.common.logger import init_fancy_logging
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from cbctmc.speedup.dataset import MCSpeedUpDataset
from cbctmc.speedup.models import (
    FlexUNet,
    MCSpeedUpNet,
    MCSpeedUpNetSeparated,
    MCSpeedUpUNet,
)
from cbctmc.speedup.trainer import MCSpeedUpTrainer

if __name__ == "__main__":
    import logging

    init_fancy_logging()

    logger = logging.getLogger(__name__)

    logger.setLevel(logging.DEBUG)
    logging.getLogger("cbctmc").setLevel(logging.DEBUG)

    DATA_FOLDER = "/datalake3/speedup_dataset"
    RUN_FOLDER = "/datalake3/speedup_runs"

    SPEEDUP_MODES = [
        "speedup_2.00x",
        "speedup_5.00x",
        "speedup_10.00x",
        "speedup_20.00x",
        "speedup_50.00x",
    ]
    USE_FORWARD_PROJECTION = True
    DEVICE = "cuda:0"

    # broken patients:
    # 22: 20x 50x
    # 24: 20x 50x
    # 32: 20x 50x
    # 33 20x

    patient_ids = [
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
    ]
    train_patients, test_patients = train_test_split(
        patient_ids, train_size=0.75, random_state=42
    )
    # test_patients = [22, 24, 69, 91, 121, 132]
    logger.info(f"Train patients ({len(train_patients)}): {train_patients}")
    logger.info(f"Test patients ({len(test_patients)}): {test_patients}")

    train_dataset = MCSpeedUpDataset.from_folder(
        folder=DATA_FOLDER,
        modes=SPEEDUP_MODES,
        patient_ids=train_patients,
    )
    test_dataset = MCSpeedUpDataset.from_folder(
        folder=DATA_FOLDER,
        modes=SPEEDUP_MODES,
        patient_ids=test_patients,
    )

    train_data_loader = DataLoader(
        train_dataset, batch_size=10, shuffle=True, num_workers=0
    )

    test_data_loader = DataLoader(
        test_dataset, batch_size=10, shuffle=False, num_workers=0
    )

    model = MCSpeedUpUNet(
        in_channels=2 if USE_FORWARD_PROJECTION else 1, out_channels=2
    )

    # load pre-trained (mean) model
    state = torch.load(
        "/datalake3/speedup_runs/2024-01-09T16:17:24.936806_run_da9db22f001145f79edbf6da/models/training/step_170000.pth"
    )
    model.load_state_dict(state["model"], strict=False)

    optimizer = Adam(params=model.parameters(), lr=1e-3)
    scheduler = ExponentialLR(optimizer, gamma=0.99999)
    trainer = MCSpeedUpTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_data_loader,
        val_loader=test_data_loader,
        run_folder=f"{RUN_FOLDER}_all_speedups",
        experiment_name=f"mc_speedup_unet_all_speedups_var_net",
        device=DEVICE,
        scheduler=scheduler,
        use_forward_projection=USE_FORWARD_PROJECTION,
        n_pretrain_steps=0,
        debug=True,
    )

    trainer.add_run_params(
        {
            "speedup_mode": "all",
            "train_patients": train_patients,
            "test_patients": test_patients,
            "use_forward_projection": USE_FORWARD_PROJECTION,
            "data": DATA_FOLDER,
        }
    )

    trainer.run(steps=10_000_000, validation_interval=0, save_interval=1_000)
