from contextlib import contextmanager

import numpy as np
import PIL
import torch
import torch.nn.functional as F
from ipmi.common.helper import concat_dicts
from ipmi.deeplearning.trainer import BaseTrainer, MetricType
from matplotlib import cm
from PIL import Image
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR

import cbctmc.speedup.metrics as metrics
from cbctmc.speedup import constants
from cbctmc.speedup.models import MCSpeedUpNetSeparated, MCSpeedUpUNet


class MCSpeedUpTrainerOld(BaseTrainer):
    def __init__(
        self,
        *args,
        max_var_correction: float = 0.1,
        scheduler: ExponentialLR,
        scatter: bool = False,
        **kwargs,
    ):
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

        mean = torch.clip(
            (prediction[:, :1] + low_photon) * constants.scale_high_fit, min=0.0
        )
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
            low_photon = low_photon[:, 1:].sum(dim=1, keepdim=True)
            high_photon = high_photon[..., 1:].sum(dim=-1)
        else:
            low_photon = low_photon[:, :1]
            high_photon = high_photon[..., 0]

        self.optimizer.zero_grad()
        with torch.autocast(device_type="cuda", enabled=True):
            mean, energy_scale = self.forward_pass(low_photon)

            # gaussian_nll_loss = F.gaussian_nll_loss(
            #     input=mean, target=high_photon, var=var
            # )
            high_factor = constants.scale_high_fit
            poiss_loss = F.gaussian_nll_loss(
                input=mean, target=high_photon * energy_scale, var=mean, full=True
            )
            # poiss_loss = mean - high_photon*high_factor*torch.log(mean + 1e-8)
            # poiss_loss = torch.mean(poiss_loss/(high_photon*high_factor + 1))
            # poiss_loss = torch.mean(poiss_loss)

            # gaussian_nll_loss_in = F.gaussian_nll_loss(
            #     input=mean, target=low_photon, var=10*var
            # )

            l1_loss = F.l1_loss(mean, high_photon * high_factor, reduction="none")
            normalized = l1_loss / (high_photon * high_factor + 1)
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
            speedup_sample = torch.distributions.Poisson(rate=mean).sample()
            speedup_sample = speedup_sample / high_factor
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
            "psnr_gain": float(psnr_after) / float(psnr_before),
        }

    def validate_on_batch(self, data: dict) -> dict:
        # we need batch size == 1 here, to do the plotting
        self.scheduler.step()
        low_photon = torch.as_tensor(data["low_photon"], device=self.device)
        high_photon = torch.as_tensor(data["high_photon"], device=self.device)

        if self.scatter:
            low_photon = low_photon[:, 1:].sum(dim=1, keepdim=True)
            high_photon = high_photon[..., 1:].sum(dim=-1, keepdims=True)
            high_photon = torch.moveaxis(high_photon, -1, 1)
        else:
            low_photon = low_photon[:, :1]
            high_photon = high_photon[..., :1]
            high_photon = torch.moveaxis(high_photon, -1, 1)

        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=True):
            mean, energy_scale = self.forward_pass(low_photon)

            # gaussian_nll_loss = F.gaussian_nll_loss(
            #     input=mean, target=high_photon, var=var
            # )
            high_factor = constants.scale_high_fit
            l1_loss = F.l1_loss(mean, high_photon * high_factor)

            speedup_sample = torch.distributions.Poisson(rate=mean).sample()
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
            # "gaussian_nll_loss": float(gaussian_nll_loss),
            "l1_loss": float(l1_loss),
            "psnr_before": float(psnr_before),
            "psnr_after": float(psnr_after),
            # "var_scale_factor": float(var_scale.mean()),
            "psnr_gain": float(psnr_after) / float(psnr_before),
        }


class ForwardPassMixin:
    model: nn.Module

    def forward_pass(self, low_photon: torch.Tensor, forward_projection: torch.Tensor):
        mean_factor = 10.0
        variance_factor = 3.0

        if forward_projection is not None:
            inputs = torch.concat((low_photon, forward_projection), dim=1)
        else:
            inputs = low_photon

        prediction = self.model(inputs)

        mean, variance = prediction[:, :1], prediction[:, 1:]

        mean = mean_factor * torch.sigmoid(mean)
        variance = variance_factor * torch.sigmoid(variance) + 1e-6

        return mean, variance

    def forward_pass_residual(
        self, low_photon: torch.Tensor, forward_projection: torch.Tensor
    ):
        mean_factor = 10.0
        variance_factor = 1.0

        if forward_projection is not None:
            inputs = torch.concat((low_photon, forward_projection), dim=1)
        else:
            inputs = low_photon

        prediction = self.model(inputs)

        mean_residual, variance = prediction[:, :1], prediction[:, 1:]

        # range (-mean_factor, +mean_factor)
        mean_residual = mean_factor * torch.tanh(mean_residual)

        mean = low_photon + mean_residual

        # one-sided tanh
        # mean = mean_factor * torch.relu(torch.tanh(mean))
        # variance = variance_factor * torch.relu(torch.tanh(variance)) + 1e-6

        # mean = mean_factor * torch.sigmoid(mean)
        # variance = variance_factor * torch.sigmoid(variance) + 1e-6
        variance = torch.relu(variance) + 1e-6

        return mean, variance

    # def forward_pass_residual_dependent_variance(
    #     self, low_photon: torch.Tensor, forward_projection: torch.Tensor
    # ):
    #     mean_factor = 10.0
    #     max_rel_variance_factor = 0.1
    #
    #     if forward_projection is not None:
    #         inputs = torch.concat((low_photon, forward_projection), dim=1)
    #     else:
    #         inputs = low_photon
    #
    #     mean_residual = self.model.mean_net(inputs)
    #     # mean_residual, _ = prediction[:, :1], prediction[:, 1:]
    #
    #     # range (-mean_factor, +mean_factor)
    #     mean_residual = mean_factor * torch.tanh(mean_residual)
    #     mean = torch.relu(low_photon + mean_residual)
    #
    #     variance = mean * self.model.var_scale + 1e-6
    #
    #     self.log_info(
    #         f"{self.model.var_scale=}",
    #         context="FORWARD",
    #     )
    #
    #     return mean, variance

    def forward_pass_model(
        self, low_photon: torch.Tensor, forward_projection: torch.Tensor
    ):
        if forward_projection is not None:
            inputs = torch.concat((low_photon, forward_projection), dim=1)
        else:
            inputs = low_photon

        prediction = self.model(inputs)
        mean, variance = prediction[:, :1], prediction[:, 1:]

        return mean, variance

    # def forward_pass_dependent_variance(
    #     self, low_photon: torch.Tensor, forward_projection: torch.Tensor
    # ):
    #     mean_factor = 30.0
    #     variance_factor = 3.0
    #
    #     if forward_projection is not None:
    #         forward_projection /= 70.0
    #         inputs = torch.concat((low_photon, forward_projection), dim=1)
    #     else:
    #         inputs = low_photon
    #
    #     mean = self.model(inputs)
    #
    #     mean = mean_factor * torch.sigmoid(mean)
    #     variance = mean * constants.HIGH_VAR_MEAN_RATIO
    #
    #     return mean, variance


class MCSpeedUpTrainer(BaseTrainer, ForwardPassMixin):
    def __init__(
        self,
        *args,
        scheduler: ExponentialLR = None,
        use_forward_projection: bool = False,
        n_pretrain_steps: int | None = 5_000,
        debug: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.scheduler = scheduler
        self.use_forward_projection = use_forward_projection
        self.n_pretrain_steps = n_pretrain_steps
        self.debug = debug

        if self.debug:
            self.debug_folder = self.run_folder / "debug" / self.aim_run.hash
            self.debug_folder.mkdir(exist_ok=True, parents=True)

        self.forward_pass = self.forward_pass_model

    METRICS = {
        "gaussian_nll_loss": MetricType.SMALLER_IS_BETTER,
        "l1_loss": MetricType.SMALLER_IS_BETTER,
        "psnr_before": MetricType.LARGER_IS_BETTER,
        "psnr_after": MetricType.LARGER_IS_BETTER,
    }

    def _sample_projection(self, mean: torch.Tensor, variance: torch.Tensor):
        return torch.distributions.Normal(loc=mean, scale=variance**0.5).sample()

    @staticmethod
    def _to_image(array: np.ndarray, max_value: float, cmap: str) -> PIL.Image:
        image = np.clip(array, 0.0, max_value) / max_value
        cmap = cm.get_cmap(cmap)
        return Image.fromarray(np.uint8(cmap(image) * 255))

    @contextmanager
    def _handle_gradient(self, is_training: bool, autocast: bool = True):
        if is_training:
            self.optimizer.zero_grad()
            # self.variance_optimizer.zero_grad()

        autocast_context = torch.autocast(device_type="cuda", enabled=autocast)
        inference_context = torch.inference_mode(mode=not is_training)

        with autocast_context, inference_context:
            yield

    def _train_or_test_on_batch(self, data: dict, is_training: bool) -> dict:
        if isinstance(data, list):
            data = concat_dicts(data, extend_lists=True)
            data["low_photon"] = torch.concat(data["low_photon"], dim=0)
            data["forward_projection"] = torch.concat(data["forward_projection"], dim=0)
            data["high_photon"] = torch.concat(data["high_photon"], dim=0)

        low_photon = torch.as_tensor(data["low_photon"], device=self.device)
        if self.use_forward_projection:
            forward_projection = torch.as_tensor(
                data["forward_projection"], device=self.device
            )

            # normalize forward_projections to the same range as low_photon by matching
            # the mean and std
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

        else:
            forward_projection = None
        high_photon = torch.as_tensor(data["high_photon"], device=self.device)

        pre_training_enabled = self.n_pretrain_steps is not None
        if pre_training_enabled:
            is_pre_training = self.i_step < self.n_pretrain_steps
        else:
            is_pre_training = False
        if self.i_step == self.n_pretrain_steps:
            # set learning rate to 1e-3
            self.optimizer.param_groups[0]["lr"] = 1e-3

        self.log_info(f"{is_pre_training=}", context="TRAIN")
        if isinstance(self.model, (MCSpeedUpNetSeparated, MCSpeedUpUNet)):
            if is_pre_training and pre_training_enabled:
                self.model.freeze_mean_net(False)
                self.model.freeze_var_net(True)
            elif not is_pre_training and pre_training_enabled:
                self.model.freeze_mean_net(True)
                self.model.freeze_var_net(False)
            else:
                self.model.freeze_mean_net(False)
                self.model.freeze_var_net(False)

        with self._handle_gradient(is_training=is_training, autocast=True):
            mean, variance = self.forward_pass(
                low_photon=low_photon, forward_projection=forward_projection
            )

            l1_loss = F.l1_loss(input=mean, target=high_photon)
            gaussian_nll_loss = F.gaussian_nll_loss(
                input=mean, target=high_photon, var=variance
            )
            psnr_before = metrics.psnr(image=low_photon, reference_image=high_photon)
            psnr_after = metrics.psnr(image=mean, reference_image=high_photon)

            if is_pre_training and pre_training_enabled:
                loss = l1_loss
            else:
                loss = gaussian_nll_loss

            if self.debug and self.i_step % 100 == 0 or not is_training:
                patient_id = int(data["patient_id"][0])
                phase = int(data["phase"][0])
                projection = int(data["projection"][0])
                mode = data["mode"][0]

                plot_id = f"{self.i_step:06d}_{patient_id:03d}_{phase:02d}_{mode}_{projection:03d}"
                self.log_info(f"Save plot {plot_id}", context="PLOT")

                folder = "training" if is_training else "validation"
                plot_folder = self.debug_folder / folder
                plot_folder.mkdir(exist_ok=True)
                high_photon = high_photon[0, 0].detach().cpu().numpy()
                high_photon = self._to_image(high_photon, max_value=7.0, cmap="gray")
                high_photon.save(plot_folder / f"{plot_id}_out_high.png")

                low_photon = low_photon[0, 0].detach().cpu().numpy()
                low_photon = self._to_image(low_photon, max_value=7.0, cmap="gray")
                low_photon.save(plot_folder / f"{plot_id}_in_low.png")

                sample = self._sample_projection(mean=mean, variance=variance)
                sample = sample[0, 0].detach().cpu().numpy()
                sample = self._to_image(sample, max_value=7.0, cmap="gray")
                sample.save(plot_folder / f"{plot_id}_out_sample.png")

                mean = mean[0, 0].detach().cpu().numpy()
                mean = self._to_image(mean, max_value=7.0, cmap="gray")
                mean.save(plot_folder / f"{plot_id}_out_mean.png")

                variance = variance[0, 0].detach().cpu().numpy()
                variance = self._to_image(variance, max_value=0.01, cmap="gray")
                variance.save(plot_folder / f"{plot_id}_out_variance.png")

                if self.use_forward_projection:
                    forward_projection = forward_projection[0, 0].detach().cpu().numpy()
                    forward_projection = self._to_image(
                        forward_projection, max_value=7.0, cmap="gray"
                    )
                    forward_projection.save(plot_folder / f"{plot_id}_in_fp.png")

        if is_training:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler:
                self.scheduler.step()
                self.log_info(
                    message=f"Learning rate at {self.optimizer.state_dict()['param_groups'][0]['lr']}",
                    context="TRAIN",
                )

        return {
            "loss": float(loss),
            "l1_loss": float(l1_loss),
            "gaussian_nll_loss": float(gaussian_nll_loss),
            "psnr_before": float(psnr_before),
            "psnr_after": float(psnr_after),
        }

    def train_on_batch(self, data: dict) -> dict:
        return self._train_or_test_on_batch(data, is_training=True)

    def validate_on_batch(self, data: dict) -> dict:
        return self._train_or_test_on_batch(data, is_training=False)
