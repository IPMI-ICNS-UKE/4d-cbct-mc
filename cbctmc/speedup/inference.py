from __future__ import annotations

import logging
import time
from math import ceil
from typing import Tuple

import numpy as np
import torch
from ipmi.common.logger import init_fancy_logging
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from cbctmc.common_types import ArrayOrTensor, PathLike, TorchDevice
from cbctmc.speedup.dataset import MCSpeedUpDataset
from cbctmc.speedup.models import FlexUNet, MCSpeedUpUNet
from cbctmc.speedup.trainer import ForwardPassMixin

logger = logging.getLogger(__name__)


class MCSpeedup(ForwardPassMixin):
    def __init__(self, model, device: TorchDevice = "cuda"):
        self.model = model.to(device)
        self.device = device

    def _cast_to_tensor(self, data: ArrayOrTensor) -> torch.Tensor | None:
        if data is None:
            return None

        if isinstance(data, np.ndarray):
            # add color axis
            data = data[:, None]

        return torch.as_tensor(data, device=self.device, dtype=torch.float32)

    def _cast_to_numpy(self, data: torch.Tensor) -> np.ndarray:
        return data.detach().cpu().numpy()

    def execute(
        self,
        low_photon: ArrayOrTensor,
        forward_projection: ArrayOrTensor | None = None,
        batch_size: int = 16,
    ):
        t_start = time.monotonic()
        mean = torch.zeros(low_photon.shape)
        variance = torch.zeros(low_photon.shape)

        n_batches = ceil(low_photon.shape[0] / batch_size)
        for i_batch in range(n_batches):
            slicing = slice(i_batch * batch_size, (i_batch + 1) * batch_size)

            _low_photon = self._cast_to_tensor(low_photon[slicing])
            if forward_projection is not None:
                _forward_projection = self._cast_to_tensor(forward_projection[slicing])
            else:
                _forward_projection = forward_projection

            _mean, _variance = self.predict(
                low_photon=_low_photon, forward_projection=_forward_projection
            )
            mean[slicing] = _mean[:, 0]
            variance[slicing] = _variance[:, 0]

        sample = self.sample(mean=mean, variance=variance)

        mean = self._cast_to_numpy(mean)
        variance = self._cast_to_numpy(variance)
        sample = self._cast_to_numpy(sample)

        t_end = time.monotonic()
        computation_time = t_end - t_start
        logger.info(
            f"Time for speedup: {computation_time:.2f}s ({computation_time / len(sample):.3f}s per projection)"
        )

        torch.cuda.empty_cache()

        return mean, variance, sample

    @staticmethod
    def preprocess_inputs(
        low_photon: torch.Tensor, forward_projection: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        # normalize forward_projections to the same range as low_photon by matching
        # the mean and std
        if forward_projection is not None:
            forward_projection = forward_projection - torch.mean(
                forward_projection, dim=(2, 3), keepdim=True
            )
            forward_projection = forward_projection / torch.std(
                forward_projection, dim=(2, 3), keepdim=True
            )
            forward_projection = forward_projection * torch.std(
                low_photon, dim=(2, 3), keepdim=True
            )
            forward_projection = forward_projection + torch.mean(
                low_photon, dim=(2, 3), keepdim=True
            )

        return low_photon, forward_projection

    def predict(
        self, low_photon: torch.Tensor, forward_projection: torch.Tensor | None = None
    ):
        self.model.eval()
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            logger.debug(
                f"Predict mean/variance using low photon projection of shape {low_photon.shape}"
            )
            low_photon = low_photon.to(self.device)
            if forward_projection is not None:
                forward_projection = forward_projection.to(self.device)

            low_photon, forward_projection = self.preprocess_inputs(
                low_photon=low_photon, forward_projection=forward_projection
            )
            mean, variance = self.forward_pass_model(
                low_photon=low_photon, forward_projection=forward_projection
            )

        return mean, variance

    def sample(self, mean, variance):
        return torch.distributions.Normal(loc=mean, scale=variance**0.5).sample()

    @classmethod
    def from_filepath(cls, model_filepath: PathLike, device: TorchDevice = "cuda"):
        state = torch.load(model_filepath, map_location="cuda")

        model = MCSpeedUpUNet(in_channels=2, out_channels=2)
        model.load_state_dict(state["model"])

        return cls(model=model, device=device)


if __name__ == "__main__":
    import SimpleITK as sitk

    init_fancy_logging()

    logger = logging.getLogger(__name__)

    logger.setLevel(logging.DEBUG)
    logging.getLogger("cbctmc").setLevel(logging.DEBUG)

    DATA_FOLDER = "/datalake2/mc_speedup"
    MODE = "low_5"

    patient_ids = [24, 33, 68, 69, 74, 104, 106, 109, 115, 116, 121, 132, 146]
    train_patients, test_patients = train_test_split(
        patient_ids, train_size=0.75, random_state=42
    )
    logger.info(f"Train patients ({len(train_patients)}): {train_patients}")
    logger.info(f"Test patients ({len(test_patients)}): {test_patients}")

    test_dataset = MCSpeedUpDataset.from_folder(
        folder=DATA_FOLDER,
        mode=MODE,
        patient_ids=test_patients,
    )

    test_data_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=True, num_workers=0
    )

    speedup = MCSpeedup.from_filepath(
        "/datalake2/mc_speedup_runs/models_72f2b57815a842a2a247e5e6/training/step_300000.pth",
        # "/datalake2/mc_speedup_runs/models_7719d626e58546d081bca7a5/training/step_300000.pth",
        device="cuda:1",
    )

    # projections = sitk.ReadImage(
    #     "/datalake2/mc_output/132_4DCT_Lunge_amplitudebased_complete/phase_00/low_5/projections_total_normalized.mha"
    # )
    # projections_arr = sitk.GetArrayFromImage(projections)

    # air = sitk.ReadImage(
    #     "/datalake2/mc_output/033_4DCT_Lunge_amplitudebased_complete/phase_00/high_mean_var/air/projections_total.mha"
    # )
    # air_arr = sitk.GetArrayFromImage(air)

    # forward_projection = sitk.ReadImage(
    #     "/datalake2/mc_output/132_4DCT_Lunge_amplitudebased_complete/phase_00/density_fp.mha"
    # )
    # forward_projection_arr = sitk.GetArrayFromImage(forward_projection)
    #
    # mean, variance, sample = speedup.execute(
    #     low_photon=projections_arr, forward_projection=forward_projection_arr
    # )

    # min_non_zero = projections_arr[projections_arr > 0.0].min()
    # projections_arr = np.where(projections_arr == 0, min_non_zero, projections_arr)
    #
    # norm_projections_arr = np.log(air_arr / projections_arr)
    #
    # projections_arr = norm_projections_arr
    #
    # max_air = 30
    # # projections_arr = np.clip(projections_arr, 0, np.log(max_air))
    # projections_arr = 1 / np.exp(norm_projections_arr)

    projections = sitk.ReadImage(
        "/datalake2/mc_output/033_4DCT_Lunge_amplitudebased_complete/phase_00/high_mean_var/projections_total.mha"
    )
    projections_arr = sitk.GetArrayFromImage(projections)

    mean = projections_arr.mean(axis=0).flatten()

    var = projections_arr.var(axis=0).flatten()

    indices = np.argsort(mean)
    mean = mean[indices]
    var = var[indices]

    from sklearn.linear_model import LinearRegression

    regr = LinearRegression(fit_intercept=False)
    regr.fit(mean.reshape(-1, 1), var.reshape(-1, 1))
    m = regr.coef_
    b = regr.intercept_

    plt.scatter(mean, var)
    plt.plot(np.linspace(0, 30), regr.predict(np.linspace(0, 30).reshape(-1, 1)))
    plt.show()

    # mean = np.log(1 / projections_arr).mean(axis=0).flatten()
    # var = np.log(1 / projections_arr).var(axis=0).flatten()
    #
    # indices = np.argsort(mean)
    # mean = mean[indices]
    # var = var[indices]
    # plt.figure()
    # plt.scatter(
    #     mean, var
    # )
    # plt.plot(
    #     np.linspace(0, 30), regr.predict(np.linspace(0, 30).reshape(-1, 1))
    # )
    # plt.show()

    # means = []
    # vars = []
    # for n_runs in range(1000):
    #     for i in np.linspace(0.1, 30):
    #         samples = np.random.normal(loc=i, scale=i**0.5, size=1000)
    #
    #         means.append(samples.mean())
    #         vars.append(samples.var())
    #
    # plt.scatter(
    #     means, vars
    # )

    # sample = sitk.GetImageFromArray(sample)
    # sample.SetOrigin(projections.GetOrigin())
    # sample.SetSpacing(projections.GetSpacing())
    # sample.SetDirection(projections.GetDirection())
    # sitk.WriteImage(sample, "/datalake2/mc_output/132_4DCT_Lunge_amplitudebased_complete/phase_00/low_5/projections_total_normalized_speedup_300k_with_fp.mha")

    # i_max = 5
    # fig, ax = plt.subplots(i_max, 5, sharex=True, sharey=True, squeeze=False)
    # for i, data in enumerate(test_data_loader):
    #     if i == i_max:
    #         break
    #
    #     mean, variance = speedup.predict(
    #         low_photon=data["low_photon"], forward_projection=data["forward_projection"]
    #     )
    #     sample = speedup.sample(mean=mean, variance=variance)
    #
    #     mean = mean.detach().cpu().numpy().squeeze()
    #     variance = variance.detach().cpu().numpy().squeeze()
    #     sample = sample.detach().cpu().numpy().squeeze()
    #
    #     low_photon = data["low_photon"].detach().cpu().numpy().squeeze()
    #     high_photon = data["high_photon"].detach().cpu().numpy().squeeze()
    #
    #     plot_kwargs = {"clim": (0, 5)}
    #
    #     ax[i, 0].imshow(low_photon, **plot_kwargs)
    #     ax[i, 1].imshow(mean, **plot_kwargs)
    #     ax[i, 2].imshow(variance)
    #     ax[i, 3].imshow(sample, **plot_kwargs)
    #     ax[i, 4].imshow(high_photon, **plot_kwargs)
