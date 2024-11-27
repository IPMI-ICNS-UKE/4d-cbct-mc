from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from ipmi.deeplearning.trainer import BaseTrainer, MetricType

from cbctmc.segmentation.labels import N_LABELS, get_label_index
from cbctmc.segmentation.losses import DiceLoss


class CTSegmentationTrainer(BaseTrainer):
    METRICS = {
        "loss": MetricType.SMALLER_IS_BETTER,
        "loss_label_background": MetricType.SMALLER_IS_BETTER,
        "loss_label_upper_body_bones": MetricType.SMALLER_IS_BETTER,
        "loss_label_upper_body_muscles": MetricType.SMALLER_IS_BETTER,
        "loss_label_upper_body_fat": MetricType.SMALLER_IS_BETTER,
        "loss_label_liver": MetricType.SMALLER_IS_BETTER,
        "loss_label_stomach": MetricType.SMALLER_IS_BETTER,
        "loss_label_lung": MetricType.SMALLER_IS_BETTER,
        "loss_label_other": MetricType.SMALLER_IS_BETTER,
        "loss_label_lung_vessels": MetricType.SMALLER_IS_BETTER,
    }

    def __init__(
        self,
        *args,
        lr_scheduler=None,
        single_segmentation: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.lr_scheduler = lr_scheduler
        self.single_segmentation = single_segmentation

        if self.single_segmentation is not None:
            self._label_index = get_label_index(self.single_segmentation)
        else:
            self._label_index = None

    def _forward(self, data: dict, return_prediction: bool = False):
        images = data["image"].to(self.device).to(torch.float32)
        if self.single_segmentation:
            segmentations = data["segmentation"][
                :, self._label_index : self._label_index + 1
            ].to(self.device)
            labels = [self.single_segmentation]
        else:
            segmentations = data["segmentation"].to(self.device)
            labels = data["labels"][0]

        with torch.autocast(device_type="cuda", enabled=False):
            outputs = self.model(images)

            losses = self.loss_function(outputs, segmentations)
            result = {}
            if (
                not self.single_segmentation
                and losses.ndim == 5
                and losses.shape[1] == len(labels)
            ):
                loss_per_image = losses.mean(dim=(1, 2, 3, 4))
                loss = loss_per_image.mean()
                loss_per_label = losses.mean(dim=(0, 2, 3, 4))
                loss_per_label = loss_per_label.detach().cpu().numpy().tolist()
                loss_per_label = {
                    f"loss_label_{label_name}": loss_per_label[label_index]
                    for label_index, label_name in labels.items()
                }

                result.update(loss_per_label)

                # # plot predictions for each label
                # fig, ax = plt.subplots(2, N_LABELS, sharex=True, sharey=True,
                #                        squeeze=False)
                # for i_label in range(N_LABELS):
                #     ax[0, i_label].imshow(
                #         outputs[0, i_label, :, :, 16].detach().cpu().numpy())
                #     ax[0, i_label].set_title(list(loss_per_label.values())[i_label])
                #     ax[1, i_label].imshow(
                #         segmentations[0, i_label, :, :, 16].detach().cpu().numpy()
                #     )

            else:
                loss = losses.mean()
                if not isinstance(self.loss_function, DiceLoss):
                    result["dice_loss"] = DiceLoss()(
                        input=outputs, target=segmentations
                    )

        result["loss"] = loss

        if return_prediction:
            return result, outputs
        else:
            return result

    def train_on_batch(self, data: dict) -> dict:
        self.optimizer.zero_grad()

        result = self._forward(data)
        loss = result["loss"]

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        batch_metrics = {key: float(value) for (key, value) in result.items()}
        batch_metrics["learning_rate"] = self.optimizer.param_groups[0]["lr"]

        return batch_metrics

    def train_on_batch_post_hook(self, data: dict, batch_results: dict):
        if self.lr_scheduler:
            self.lr_scheduler.step(batch_results["loss"])

    def validate_on_batch(self, data: dict) -> dict:
        with torch.inference_mode():
            result, prediction = self._forward(data, return_prediction=True)

            if self.single_segmentation:
                prediction = torch.sigmoid(prediction)
                prediction = prediction.detach().cpu().numpy()
            else:
                prediction[:, 0:8] = torch.softmax(
                    prediction[:, 0:8], dim=1
                )  # softmax group 1
                prediction[:, 8] = torch.sigmoid(
                    prediction[:, 8]
                )  # sigmoid for lung vessels
                prediction = prediction.detach().cpu().numpy()

        mid_z_slice = data["image"].shape[-1] // 2

        if data["i_patch"][0] == 0:
            image_id = data["image_id"][0]
            with plt.ioff():
                fig, ax = plt.subplots(
                    2, N_LABELS + 1, sharex=True, sharey=True, figsize=(10, 2)
                )
                ax[0, 0].imshow(data["image"][0, 0, :, :, mid_z_slice], clim=(0, 1))
                ax[1, 0].imshow(data["image"][0, 0, :, :, mid_z_slice], clim=(0, 1))
                ax[0, 0].set_title(image_id, fontsize=4)
                for i_segmentation in range(N_LABELS):
                    ax[0, i_segmentation + 1].imshow(
                        data["segmentation"][0, i_segmentation, :, :, mid_z_slice],
                        clim=(0, 1),
                    )
                    ax[0, i_segmentation + 1].set_title(
                        data["labels"][0][i_segmentation], fontsize=4
                    )
                    ax[1, i_segmentation + 1].imshow(
                        prediction[0, i_segmentation, :, :, mid_z_slice], clim=(0, 1)
                    )
                plt.savefig(
                    self._image_folder / f"{self.i_step:06d}_{image_id}.png", dpi=600
                )

        return {key: float(value) for (key, value) in result.items()}
