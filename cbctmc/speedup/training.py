from pathlib import Path

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
    )

    train_dataset = MCSpeedUpDataset.from_folder(
        folder=Path("/mnt/gpu_server/crohling/data/results"),
        patient_ids=patient_ids[:4],
    )
    d = train_dataset[0]

    dataset = MCSpeedUpDataset.from_folder(
        folder=Path("/mnt/gpu_server/crohling/data/results"),
        patient_ids=patient_ids,
        runs=range(15),
        phases=(0,),
        projections=(45,),
    )

    means = []
    vars = []
    for i in range(len(patient_ids)):
        print(i)

        var = np.var(
            [d["low_photon"].sum(axis=0) for d in dataset[i * 15 : (i + 1) * 15]],
            axis=0,
        )
        mean = np.mean(
            [d["low_photon"].sum(axis=0) for d in dataset[i * 15 : (i + 1) * 15]],
            axis=0,
        )

        means.append(mean)
        vars.append(var)

    vars = np.concatenate([var.flatten() for var in vars])
    means = np.concatenate([mean.flatten() for mean in means])

    # fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    # ax[0].imshow(mean)
    # ax[1].imshow(var)

    fig, ax = plt.subplots(1, 1)
    ax.scatter(means, vars, s=1, alpha=0.1)

    # data_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    #
    # model = ResidualDenseNet2D(
    #     in_channels=1,
    #     out_channels=2,
    #     growth_rate=8,
    #     n_blocks=2,
    #     n_block_layers=4,
    # ).to(
    #     "cuda:1"
    # )
    # optimizer = Adam(params=model.parameters(), lr=1e-3)
    # scaler = torch.cuda.amp.GradScaler()
    #
    #
    # for i_epoch in range(2):
    #     data_loader = tqdm(data_loader)
    #     for data in data_loader:
    #
    #         low_photon = torch.as_tensor(data["low_photon"], device="cuda:1").sum(
    #             dim=1, keepdims=True
    #         )
    #         high_photon = torch.as_tensor(data["high_photon"], device="cuda:1")
    #
    #         low_photon = torch.clip(low_photon / 50, min=0.0, max=1.0)
    #         high_photon = torch.clip(high_photon / 50, min=0.0, max=1.0)
    #
    #         optimizer.zero_grad()
    #         with torch.autocast(device_type="cuda", enabled=False):
    #             prediction = model(low_photon)
    #             mean = torch.clip(prediction[:, :1] + low_photon, min=0.0, max=1.0)
    #             var = torch.sigmoid(prediction[:, 1:]) + 1e-6
    #
    #             if i_epoch > 0:
    #                 loss = F.gaussian_nll_loss(input=mean, target=high_photon, var=var)
    #             else:
    #                 loss = F.l1_loss(mean, high_photon)
    #
    #         scaler.scale(loss).backward()
    #         scaler.step(optimizer)
    #         scaler.update()
    #         data_loader.set_description(f"{float(loss)}")
    #
    # # with torch.autocast(device_type="cuda"):
    # #     prediction = model(low_photon)
    #
    # prediction = prediction.to(torch.float32)
    # mean = torch.clip(prediction[:, :1] + low_photon, min=0.0, max=1.0)
    # var = torch.sigmoid(prediction[:, 1:]) + 1e-6
    # sample = torch.distributions.Normal(loc=mean, scale=var).sample()
    #
    #
    # mean = mean.detach().cpu().numpy().astype(np.float32)
    # var = var.detach().cpu().numpy().astype(np.float32)
    # low_photon = low_photon.detach().cpu().numpy().astype(np.float32)
    # high_photon = high_photon.detach().cpu().numpy().astype(np.float32)
    # sample = sample.detach().cpu().numpy().astype(np.float32)
    #
    # fig, ax = plt.subplots(1, 5, sharex=True, sharey=True)
    # ax[0].imshow(low_photon[0, 0], clim=(0, 1))
    # ax[1].imshow(high_photon[0, 0], clim=(0, 1))
    # ax[2].imshow(mean[0, 0], clim=(0, 1))
    # ax[3].imshow(var[0, 0], clim=(0, 1))
    # ax[4].imshow(sample[0, 0], clim=(0, 1))
