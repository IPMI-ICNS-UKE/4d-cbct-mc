import PIL
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ipmi.deeplearning.trainer import BaseTrainer, MetricType
from matplotlib import cm
from cbctmc.speedup import constants
import cbctmc.speedup.metrics as metrics


class MCSpeedUpTrainer(BaseTrainer):
    def __init__(self, *args, max_var_correction: float = 0.1, **kwargs, ):
        super().__init__(*args, **kwargs)

        self.max_var_correction = max_var_correction

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

    def _forward_pass(self, low_photon: torch.Tensor):
        prediction = self.model(low_photon)
        mean = torch.clip(prediction[:, :1] + low_photon, min=0.0)

        var = mean / constants.scale_high_fit
        var_scale = 1 + (F.softsign(prediction[:, 1:]) * self.max_var_correction)
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

            gaussian_nll_loss = F.gaussian_nll_loss(
                input=mean, target=high_photon, var=var
            )

            l1_loss = F.l1_loss(mean, high_photon)

            if self.i_epoch > 0:
                loss = gaussian_nll_loss
            else:
                loss = l1_loss

            speedup_sample = torch.distributions.Normal(
                loc=mean, scale=var**0.5
            ).sample()
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
            "gaussian_nll_loss": float(gaussian_nll_loss),
            "l1_loss": float(l1_loss),
            "psnr_before": float(psnr_before),
            "psnr_after": float(psnr_after),
            "var_scale_factor": float(var_scale.mean()),
        }

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

            speedup_sample = torch.distributions.Normal(
                loc=mean, scale=var**0.5
            ).sample()
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
        if phase == 0 and projection % 10 == 0 and run == 0:
            # plot one projection image per test patient

            low_photon = low_photon.detach().cpu().numpy()[0, 0]
            high_photon = high_photon.detach().cpu().numpy()[0, 0]
            speedup_sample = speedup_sample.detach().cpu().numpy()[0, 0]

            # normalize image range to [0, 1] using max pixel value of high photon
            norm_factor = high_photon.max()

            image_low_photon = self._to_image(
                array=low_photon, max_value=norm_factor, cmap="gray"
            )
            image_high_photon = self._to_image(
                array=high_photon, max_value=norm_factor, cmap="gray"
            )
            image_speedup_sample = self._to_image(
                array=speedup_sample, max_value=norm_factor, cmap="gray"
            )

            patient_id = int(data["patient_id"][0])
            self.save_image(
                image=image_low_photon, name=f"patient_{patient_id}__projection_{projection}__low_photon"
            )
            self.save_image(
                image=image_high_photon, name=f"patient_{patient_id}__projection_{projection}__high_photon"
            )
            self.save_image(
                image=image_speedup_sample, name=f"patient_{patient_id}__projection_{projection}__speedup_sample"
            )

        return {
            "gaussian_nll_loss": float(gaussian_nll_loss),
            "l1_loss": float(l1_loss),
            "psnr_before": float(psnr_before),
            "psnr_after": float(psnr_after),
            "var_scale_factor": float(var_scale.mean()),
        }
