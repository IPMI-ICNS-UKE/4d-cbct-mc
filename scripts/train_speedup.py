import torch
from ipmi.common.logger import init_fancy_logging
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from cbctmc.speedup.dataset import MCSpeedUpDataset
from cbctmc.speedup.models import MCSpeedUpNet, MCSpeedUpNetSeparated
from cbctmc.speedup.trainer import MCSpeedUpTrainer

if __name__ == "__main__":
    import logging

    init_fancy_logging()

    logger = logging.getLogger(__name__)

    logger.setLevel(logging.DEBUG)
    logging.getLogger("cbctmc").setLevel(logging.DEBUG)

    DATA_FOLDER = "/datalake3/speedup_dataset"
    RUN_FOLDER = "/datalake3/speedup_runs"

    SPEEDUP_MODE = "speedup_2.00x"
    USE_FORWARD_PROJECTION = True
    DEVICE = "cuda:0"

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
    logger.info(f"Train patients ({len(train_patients)}): {train_patients}")
    logger.info(f"Test patients ({len(test_patients)}): {test_patients}")

    train_dataset = MCSpeedUpDataset.from_folder(
        folder=DATA_FOLDER,
        mode=SPEEDUP_MODE,
        patient_ids=train_patients,
    )
    test_dataset = MCSpeedUpDataset.from_folder(
        folder=DATA_FOLDER,
        mode=SPEEDUP_MODE,
        patient_ids=test_patients,
    )

    train_data_loader = DataLoader(
        train_dataset, batch_size=6, shuffle=True, num_workers=0
    )

    test_data_loader = DataLoader(
        test_dataset, batch_size=6, shuffle=False, num_workers=0
    )

    # model = FlexUNet(
    #     n_channels=2 if USE_FORWARD_PROJECTION else 1,
    #     n_classes=2,
    #     n_levels=6,
    #     filter_base=32,
    #     n_filters=None,
    #     convolution_layer=nn.Conv2d,
    #     downsampling_layer=nn.MaxPool2d,
    #     upsampling_layer=nn.Upsample,
    #     norm_layer=nn.BatchNorm2d,
    #     skip_connections=True,
    #     convolution_kwargs=None,
    #     downsampling_kwargs=None,
    #     upsampling_kwargs=None,
    #     return_bottleneck=False,
    # )

    # model = MCSpeedUpNet(
    #     in_channels=2 if USE_FORWARD_PROJECTION else 1,
    #     out_channels=2,
    # )
    model = MCSpeedUpNetSeparated(
        in_channels=2 if USE_FORWARD_PROJECTION else 1, out_channels=2, growth_rate=8
    )

    # load pre-trained (mean) model
    state = torch.load(
        "/datalake3/speedup_runs/2024-01-04T17:33:25.900796_run_2318f5f7659844e6b4bd685b/models/validation/step_440000.pth"
    )
    model.load_state_dict(state["model"])

    optimizer = Adam(params=model.parameters(), lr=1e-3)
    scheduler = ExponentialLR(optimizer, gamma=0.99999)
    trainer = MCSpeedUpTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_data_loader,
        val_loader=test_data_loader,
        run_folder=RUN_FOLDER,
        experiment_name="mc_speedup_var_training",
        device=DEVICE,
        scheduler=scheduler,
        use_forward_projection=USE_FORWARD_PROJECTION,
        n_pretrain_steps=0,
        debug=True,
    )

    trainer.add_run_params(
        {
            "speedup_mode": SPEEDUP_MODE,
            "train_patients": train_patients,
            "test_patients": test_patients,
            "use_forward_projection": USE_FORWARD_PROJECTION,
            "data": DATA_FOLDER,
        }
    )

    trainer.run(steps=100_000_000, validation_interval=5_000, save_interval=1_000)
