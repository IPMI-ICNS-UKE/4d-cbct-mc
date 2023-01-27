from pathlib import Path

import PIL
import aim
from ipmi.common.logger import init_fancy_logging
from torch import Tensor

from cbctmc.speedup.dataset import MCSpeedUpDataset
from cbctmc.speedup.models import ResidualDenseNet2D
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import cbctmc.speedup.metrics as metrics
from sklearn.model_selection import train_test_split
from ipmi.deeplearning.trainer import BaseTrainer, MetricType
from PIL import Image
from matplotlib import cm


class MCSpeedUpTrainer(BaseTrainer):
    METRICS = {
        "loss": MetricType.SMALLER_IS_BETTER,
        "gaussian_nll_loss": MetricType.SMALLER_IS_BETTER,
        "l1_loss": MetricType.SMALLER_IS_BETTER,
    }

    @staticmethod
    def _to_image(array: np.ndarray, max_value: float, cmap: str) -> PIL.Image:
        image = np.clip(array, 0.0, max_value) / max_value
        cmap = cm.get_cmap(cmap)
        return Image.fromarray(np.uint8(cmap(image) * 255))

    def _forward_pass(self, low_photon: torch.Tensor):
        prediction = model(low_photon)
        mean = torch.clip(prediction[:, :1] + low_photon, min=0.0)

        var = mean / 211
        var_scale = 1 + (F.softsign(prediction[:, 1:]) * 0.1)
        var = var * var_scale + 1e-6

        return mean, var, var_scale

    def train_on_batch(self, data: dict) -> dict:

        low_photon = torch.as_tensor(data["low_photon"], device=self.device).sum(
            dim=1, keepdims=True
        )
        high_photon = torch.as_tensor(data["high_photon"], device=self.device)

        self.optimizer.zero_grad()
        with torch.autocast(device_type="cuda", enabled=True):
            mean, var, var_scale = self._forward_pass(low_photon)

            if self.i_epoch > 0:
                loss = F.gaussian_nll_loss(input=mean, target=high_photon, var=var)
            else:
                loss = F.l1_loss(mean, high_photon)

        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()

        return {"loss": float(loss)}

    def validate_on_batch(self, data: dict) -> dict:
        # we need batch size == 1 here, to do the plotting
        low_photon = torch.as_tensor(data["low_photon"], device=self.device).sum(
            dim=1, keepdims=True
        )
        high_photon = torch.as_tensor(data["high_photon"], device=self.device)

        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=True):
            mean, var, var_scale = self._forward_pass(low_photon)

            gaussian_nll_loss = F.gaussian_nll_loss(
                input=mean, target=high_photon, var=var
            )

            l1_loss = F.l1_loss(mean, high_photon)

            sample = torch.distributions.Normal(loc=mean, scale=var**0.5).sample()
            psnr_before = metrics.psnr(
                image=low_photon, reference_image=high_photon, max_pixel_value=50.0
            )
            psnr_after = metrics.psnr(
                image=sample, reference_image=high_photon, max_pixel_value=50.0
            )

        # data is list of length 1
        phase = int(data["phase"][0])
        projection = int(data["projection"][0])
        run = int(data["run"][0])
        if phase == 0 and projection == 44 and run == 0:
            # plot one projection image per test patient

            low_photon = low_photon.detach().cpu().numpy()[0, 0]
            high_photon = high_photon.detach().cpu().numpy()[0, 0]
            sample = sample.detach().cpu().numpy()[0, 0]

            # normalize image range to [0, 1] using max pixel value of high photon
            norm_factor = high_photon.max()

            image_low_photon = self._to_image(
                array=low_photon, max_value=norm_factor, cmap="gray"
            )
            image_high_photon = self._to_image(
                array=high_photon, max_value=norm_factor, cmap="gray"
            )
            image_sample = self._to_image(
                array=sample, max_value=norm_factor, cmap="gray"
            )

            patient_id = int(data["patient_id"][0])
            self.save_image(
                image=image_low_photon, name=f"low_photon__patient_{patient_id}"
            )
            self.save_image(
                image=image_high_photon, name=f"high_photon__patient_{patient_id}"
            )
            self.save_image(image=image_sample, name=f"sample__patient_{patient_id}")

        return {
            "gaussian_nll_loss": float(gaussian_nll_loss),
            "l1_loss": float(l1_loss),
            "psnr_before": float(psnr_before),
            "psnr_after": float(psnr_after),
            "var_scale_factor": float(var_scale.mean()),
        }


if __name__ == "__main__":
    import logging

    init_fancy_logging()

    logging.getLogger(__name__).setLevel(logging.DEBUG)
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
    possible_folders = [
        Path("/home/crohling/amalthea/data/results/"),
        Path("/mnt/gpu_server/crohling/data/results"),
    ]
    folder = next(p for p in possible_folders if p.exists())

    train_dataset = MCSpeedUpDataset.from_folder(
        folder=folder, patient_ids=train_patients, runs=range(15)
    )
    test_dataset = MCSpeedUpDataset.from_folder(
        folder=folder, patient_ids=test_patients, runs=range(15)
    )

    train_data_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, num_workers=4
    )
    test_data_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=4
    )

    model = ResidualDenseNet2D(
        in_channels=1,
        out_channels=2,
        growth_rate=8,
        n_blocks=4,
        n_block_layers=4,
    )

    optimizer = Adam(params=model.parameters(), lr=1e-4)
    trainer = MCSpeedUpTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_data_loader,
        val_loader=test_data_loader,
        run_folder="/mnt/gpu_server/crohling/data/runs",
        experiment_name="mc_speed_up",
        device="cuda:1",
    )
    trainer.run(steps=100_000_000, validation_interval=100)

