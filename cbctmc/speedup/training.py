from pathlib import Path

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
        146
    )
    train_patients, test_patients = train_test_split(patient_ids, train_size=0.75)
    possible_folders = [
        Path("/home/crohling/amalthea/data/results/"),
        Path("/mnt/gpu_server/crohling/data/results")
    ]
    folder = next(p for p in possible_folders if p.exists())

    train_dataset = MCSpeedUpDataset.from_folder(
        folder=folder,
        patient_ids=train_patients,
        runs=range(15)
    )
    test_dataset = MCSpeedUpDataset.from_folder(
        folder=folder,
        patient_ids=test_patients,
        runs=range(15)
    )

    # dataset = MCSpeedUpDataset.from_folder(
    #     folder=folder,
    #     patient_ids=patient_ids,
    #     runs=range(15),
    #     phases=(0,),
    #     projections=(45,),
    # )
    #
    # means = []
    # vars = []
    # for i in range(len(patient_ids)):
    #     print(i)
    #     scale = 4.39391092193762
    #     var = np.var(
    #         [scale * d["low_photon"].sum(axis=0) for d in dataset[i * 15 : (i + 1) * 15]],
    #         axis=0,
    #     )
    #     mean = np.mean(
    #         [scale * d["low_photon"].sum(axis=0) for d in dataset[i * 15 : (i + 1) * 15]],
    #         axis=0,
    #     )
    #
    #     means.append(mean)
    #     vars.append(var)
    #
    #
    # vars = np.concatenate([var.flatten() for var in vars])
    # means = np.concatenate([mean.flatten() for mean in means])
    #
    # # fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    # # ax[0].imshow(mean)
    # # ax[1].imshow(var)
    #
    # fig, ax = plt.subplots(1, 1)
    # ax.scatter(means, vars, s=1, alpha=0.1)
    # ax.plot(np.arange(0, 130), np.arange(0, 130), c='red')

    data_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=4)
    device = 'cuda:0'

    model = ResidualDenseNet2D(
        in_channels=1,
        out_channels=2,
        growth_rate=8,
        n_blocks=2,
        n_block_layers=4,
    ).to(
        "cuda:0"
    )
    optimizer = Adam(params=model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()

    class Loss(nn.Module):
        def forward(self, input: Tensor, target: Tensor) -> Tensor:
            loss = torch.abs(input - target) / (target + 1e-6)

            return loss.mean()

    loss_func = Loss()

    for i_epoch in range(2):
        data_loader = tqdm(data_loader)
        for data in data_loader:

            low_photon = torch.as_tensor(data["low_photon"], device="cuda:0").sum(
                dim=1, keepdims=True
            )
            high_photon = torch.as_tensor(data["high_photon"], device="cuda:0")

            # low_photon = torch.clip(low_photon, min=0.0, max=1.0)
            # high_photon = torch.clip(high_photon, min=0.0, max=1.0)

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", enabled=True):
                prediction = model(low_photon)
                mean = torch.clip(prediction[:, :1] + low_photon, min=0.0)

                var = mean / 211

                # var = prediction[:, 1:] + (mean / 211.0) + 1e-6
                # var = (mean / 211.0) + 1e-6
                # evtl worx
                var_scale = 1 + (F.softsign(prediction[:, 1:]) * 0.1)
                var = var * var_scale + 1e-6

                # loss = loss_func(input=mean, target=high_photon)
                # loss = F.gaussian_nll_loss(input=mean, target=high_photon, var=var)
                # loss += 5*F.l1_loss(mean, high_photon)
                # loss = F.gaussian_nll_loss(input=mean, target=high_photon, var=var) + 50*F.l1_loss(mean, high_photon)
                # loss = F.l1_loss(mean, high_photon)
                if i_epoch > 0:
                    loss = F.gaussian_nll_loss(input=mean, target=high_photon, var=var)
                else:
                    loss = F.l1_loss(mean, high_photon)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            data_loader.set_description(f"{float(loss)}")

    # with torch.autocast(device_type="cuda"):
    #     prediction = model(low_photon)

    # this is the last sample from train dataset
    sample = torch.distributions.Normal(loc=mean, scale=var**0.5).sample()

    mean = mean.detach().cpu().numpy().astype(np.float32)
    var = var.detach().cpu().numpy().astype(np.float32)
    low_photon = low_photon.detach().cpu().numpy().astype(np.float32)
    high_photon = high_photon.detach().cpu().numpy().astype(np.float32)
    sample = sample.detach().cpu().numpy().astype(np.float32)
    var_scale = var_scale.detach().cpu().numpy().astype(np.float32)


    psnr_before = metrics.psnr(
        image=low_photon[0, 0],
        reference_image=high_photon[0, 0],
        max_pixel_value=40.0
    )

    psnr_after = metrics.psnr(
        image=sample[0, 0],
        reference_image=high_photon[0, 0],
        max_pixel_value=40.0
    )


    fig, ax = plt.subplots(1, 6, sharex=True, sharey=True)
    ax[0].imshow(low_photon[0, 0], clim=(0, 40))
    ax[0].set_title(f'low_photon PSNR: {psnr_before}')
    ax[1].imshow(high_photon[0, 0], clim=(0, 40))
    ax[1].set_title(f'high_photon')
    ax[2].imshow(mean[0, 0], clim=(0, 40))
    ax[2].set_title(f'mean')
    ax[3].imshow(var[0, 0], clim=(0, 0.5))
    ax[3].set_title(f'var')
    ax[4].imshow(sample[0, 0], clim=(0, 40))
    ax[4].set_title(f'sample PSNR: {psnr_after}')
    ax[5].imshow(var_scale[0, 0], clim=(0.9, 1.1), cmap='seismic')
    ax[5].set_title(f'var_scale')
