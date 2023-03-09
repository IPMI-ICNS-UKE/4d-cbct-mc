import PIL
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ipmi.deeplearning.trainer import BaseTrainer, MetricType
from matplotlib import cm
from cbctmc.speedup import constants
import cbctmc.speedup.metrics as metrics
import cbctmc.speedup.losses as losses
from ipmi.common.helper import concat_dicts
from torch.optim.lr_scheduler import ExponentialLR


class MCSpeedUpTrainer(BaseTrainer):
    def __init__(self, *args, max_var_correction: float = 0.1, scheduler: ExponentialLR, scatter: bool = False, **kwargs,):
        super().__init__(*args, **kwargs)
        self.scheduler = scheduler
        self.max_var_correction = max_var_correction
        self.scatter = scatter

    METRICS = {
        "gaussian_nll_loss": MetricType.SMALLER_IS_BETTER,
        "l1_loss": MetricType.SMALLER_IS_BETTER,
        "psnr_after": MetricType.LARGER_IS_BETTER,
    }

    @staticmethod
    def _to_image(array: np.ndarray, max_value: float, cmap: str) -> PIL.Image:
        image = np.clip(array, 0.0, max_value) / max_value
        cmap = cm.get_cmap(cmap)
        return Image.fromarray(np.uint8(cmap(image) * 255))

    def forward_pass(self, low_photon: torch.Tensor):
        prediction = self.model(low_photon)

        mean = torch.clip((prediction[:, :1] + low_photon)*constants.scale_high_fit, min=0.0)
        mean += 1e-8
        # var = mean / constants.scale_high_fit
        # var_scale = 1 + (F.softsign(prediction[:, 1:]) * self.max_var_correction)
        # var = var * var_scale + 1e-6
        energy_scale = 1 + (F.softsign(prediction[:, 1:]) * self.max_var_correction)

        return mean, energy_scale

    def train_on_batch(self, data: dict) -> dict:

        if isinstance(data, list):
            data = concat_dicts(data, extend_lists=True)
            data["low_photon"] = torch.concat(data["low_photon"], dim=0)
            data["high_photon"] = torch.concat(data["high_photon"], dim=0)
        low_photon = torch.as_tensor(data["low_photon"], device=self.device)
        high_photon = torch.as_tensor(data["high_photon"], device=self.device)

        if self.scatter:
            low_photon = low_photon[:,1:].sum(dim=1, keepdim=True)
            high_photon = high_photon[..., 1:].sum(dim=-1)
        else:
            low_photon = low_photon[:,:1]
            high_photon = high_photon[..., 0]

        self.optimizer.zero_grad()
        with torch.autocast(device_type="cuda", enabled=True):
            mean, energy_scale = self.forward_pass(low_photon)

            # gaussian_nll_loss = F.gaussian_nll_loss(
            #     input=mean, target=high_photon, var=var
            # )
            high_factor = constants.scale_high_fit
            poiss_loss = F.gaussian_nll_loss(input=mean, target=high_photon*energy_scale, var=mean, full=True)
            # poiss_loss = mean - high_photon*high_factor*torch.log(mean + 1e-8)
            # poiss_loss = torch.mean(poiss_loss/(high_photon*high_factor + 1))
            # poiss_loss = torch.mean(poiss_loss)

            # gaussian_nll_loss_in = F.gaussian_nll_loss(
            #     input=mean, target=low_photon, var=10*var
            # )

            l1_loss = F.l1_loss(mean, high_photon*high_factor, reduction="none")
            normalized = l1_loss/(high_photon*high_factor + 1)
            l1_loss = torch.mean(normalized)
            # mseloss = F.mse_loss(mean, high_photon*high_factor)
            #
            # consistency_loss = losses.consistency_loss(mean=mean, var=var, target=high_photon)
            # consitency_loss_low = losses.consistency_loss(mean=mean, var=var*10, target=low_photon)
            #
            # low_pref_loss = losses.low_preff_loss(mean=mean, var=var, target=high_photon)
            # low_pref_l1_loss = losses.low_preff_l1_loss(mean=mean, var=var, target=high_photon)
            #
            # grad_loss = losses.gradient_attention_loss(input=mean, target=high_photon)

            # eight_loss = losses.eight_loss(mean = mean, var=var, target = high_photon, target2 = low_photon)

            if self.i_epoch >= 0:
                # loss = gaussian_nll_loss
                loss = l1_loss
            else:
                loss = l1_loss

            # speedup_sample = torch.distributions.Normal(
            #     loc=mean, scale=var**0.5
            # ).sample()
            speedup_sample = torch.distributions.Poisson(
                rate = mean
            ).sample()
            speedup_sample = speedup_sample/high_factor
            psnr_before = metrics.psnr(
                image=low_photon, reference_image=high_photon, max_pixel_value=50.0
            )
            psnr_after = metrics.psnr(
                image=speedup_sample, reference_image=high_photon, max_pixel_value=50.0
            )

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {
            "loss": float(loss),
            # "gaussian_nll_loss": float(gaussian_nll_loss),
            "l1_loss": float(l1_loss),
            "psnr_before": float(psnr_before),
            "psnr_after": float(psnr_after),
            # "var_scale_factor": float(var_scale.mean()),
            "psnr_gain": float(psnr_after)/float(psnr_before),
        }

    def validate_on_batch(self, data: dict) -> dict:
        # we need batch size == 1 here, to do the plotting
        self.scheduler.step()
        low_photon = torch.as_tensor(data["low_photon"], device=self.device)
        high_photon = torch.as_tensor(data["high_photon"], device=self.device)

        if self.scatter:
            low_photon = low_photon[:,1:].sum(dim=1, keepdim=True)
            high_photon = high_photon[..., 1:].sum(dim=-1, keepdims=True)
            high_photon = torch.moveaxis(high_photon, -1, 1)
        else:
            low_photon = low_photon[:,:1]
            high_photon = high_photon[..., :1]
            high_photon = torch.moveaxis(high_photon, -1, 1)


        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=True):
            mean, energy_scale = self.forward_pass(low_photon)

            # gaussian_nll_loss = F.gaussian_nll_loss(
            #     input=mean, target=high_photon, var=var
            # )
            high_factor = constants.scale_high_fit
            l1_loss = F.l1_loss(mean, high_photon*high_factor)

            speedup_sample = torch.distributions.Poisson(
                rate=mean
            ).sample()
            speedup_sample = speedup_sample / high_factor
            psnr_before = metrics.psnr(
                image=low_photon, reference_image=high_photon, max_pixel_value=50.0
            )
            psnr_after = metrics.psnr(
                image=speedup_sample, reference_image=high_photon, max_pixel_value=50.0
            )

        # data is list of length 1
        phase = int(data["phase"][0])
        projection = int(data["projection"][0])
        run = int(data["run"][0])
        # if phase == 0 and projection % 10 == 0 and run == 0:
        #     # plot one projection image per test patient
        #
        #     low_photon = low_photon.detach().cpu().numpy()[0, 0]
        #     high_photon = high_photon.detach().cpu().numpy()[0, 0]
        #     speedup_sample = speedup_sample.detach().cpu().numpy()[0, 0]
        #
        #     # normalize image range to [0, 1] using max pixel value of high photon
        #     norm_factor = high_photon.max()
        #
        #     image_low_photon = self._to_image(
        #         array=low_photon, max_value=norm_factor, cmap="gray"
        #     )
        #     image_high_photon = self._to_image(
        #         array=high_photon, max_value=norm_factor, cmap="gray"
        #     )
        #     image_speedup_sample = self._to_image(
        #         array=speedup_sample, max_value=norm_factor, cmap="gray"
        #     )
        #
        #     patient_id = int(data["patient_id"][0])
        #     self.save_image(
        #         image=image_low_photon, name=f"patient_{patient_id}__projection_{projection}__low_photon"
        #     )
        #     self.save_image(
        #         image=image_high_photon, name=f"patient_{patient_id}__projection_{projection}__high_photon"
        #     )
        #     self.save_image(
        #         image=image_speedup_sample, name=f"patient_{patient_id}__projection_{projection}__speedup_sample"
        #     )

        return {
            #"gaussian_nll_loss": float(gaussian_nll_loss),
            "l1_loss": float(l1_loss),
            "psnr_before": float(psnr_before),
            "psnr_after": float(psnr_after),
            # "var_scale_factor": float(var_scale.mean()),
            "psnr_gain": float(psnr_after) / float(psnr_before),
        }
