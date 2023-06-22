import matplotlib.pyplot as plt
import torch
from ipmi.deeplearning.trainer import BaseTrainer, MetricType

from cbctmc.segmentation.labels import N_LABELS


class CTSegmentationTrainer(BaseTrainer):
    METRICS = {"loss": MetricType.SMALLER_IS_BETTER}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.plot_folder = self.run_folder / f"plots_{self.aim_run.hash}"
        self.plot_folder.mkdir()

    def _forward(self, data: dict, return_prediction: bool = False):
        images = data["image"].to(self.device)
        segmentations = data["segmentation"].to(self.device)
        labels = data["labels"][0]

        with torch.autocast(device_type="cuda", enabled=False):
            outputs = self.model(images)
            losses = self.loss_function(outputs, segmentations)
            loss_per_image = losses.mean(dim=(1, 2, 3, 4))
            loss = loss_per_image.mean()
            loss_per_label = losses.mean(dim=(0, 2, 3, 4))

        result = {"loss": loss}

        loss_per_label = loss_per_label.detach().cpu().numpy().tolist()
        loss_per_label = {
            f"loss_label_{label_name}": loss_per_label[label_index]
            for label_index, label_name in labels.items()
        }

        result.update(loss_per_label)

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

        return {key: float(value) for (key, value) in result.items()}

    def validate_on_batch(self, data: dict) -> dict:
        with torch.inference_mode():
            result, prediction = self._forward(data, return_prediction=True)
            prediction[:, 0:8] = torch.softmax(
                prediction[:, 0:8], dim=1
            )  # softmax group 1
            prediction[:, 8] = torch.sigmoid(
                prediction[:, 8]
            )  # sigmoid for lung vessels
            prediction = prediction.detach().cpu().numpy()

        mid_slice = data["image"].shape[-2] // 2

        if data["i_patch"][0] == 0:
            image_id = data["image_id"][0]
            with plt.ioff():
                fig, ax = plt.subplots(
                    2, N_LABELS + 1, sharex=True, sharey=True, figsize=(10, 2)
                )
                ax[0, 0].imshow(data["image"][0, 0, :, mid_slice, :], clim=(0, 1))
                ax[1, 0].imshow(data["image"][0, 0, :, mid_slice, :], clim=(0, 1))
                ax[0, 0].set_title(image_id, fontsize=4)
                for i_segmentation in range(N_LABELS):
                    ax[0, i_segmentation + 1].imshow(
                        data["segmentation"][0, i_segmentation, :, mid_slice, :],
                        clim=(0, 1),
                    )
                    ax[0, i_segmentation + 1].set_title(
                        data["labels"][0][i_segmentation], fontsize=4
                    )

                    ax[1, i_segmentation + 1].imshow(
                        prediction[0, i_segmentation, :, mid_slice, :], clim=(0, 1)
                    )
                plt.savefig(
                    self.plot_folder / f"{self.i_step:06d}_{image_id}.png", dpi=600
                )

        return {key: float(value) for (key, value) in result.items()}
