from __future__ import annotations

import logging
from functools import partial
from pathlib import Path

import click
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader

from vroc.cli_utils import convert_to_int_list
from vroc.helper import dict_collate
from vroc.logger import init_fancy_logging
from vroc.loss import DiceLoss
from vroc.models import FlexUNet
from vroc.segmentation.dataset import LungCTSegmentationDataset
from vroc.trainer import BaseTrainer, MetricType


class LungCTSegmentationLoss(nn.Module):
    def __init__(self, use_dice: bool = True):
        super().__init__()
        self.use_dice = use_dice
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.dice_loss_function = DiceLoss(include_background=True, reduction=None)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.use_dice:
            input[:, 0:-1] = torch.softmax(input[:, 0:-1], dim=1)  # softmax group 1
            input[:, -1] = torch.sigmoid(input[:, -1])  # sigmoid for lung vessels
            loss = self.dice_loss_function(input=input, target=target)
        else:
            loss = 6 / 7 * self.ce_loss(
                input[:, 0:-1], target[:, 0:-1]
            ) + 1 / 7 * self.bce_loss(input[:, -1:], target[:, -1:])

        return loss


class LungCTSegmentationTrainer(BaseTrainer):
    LABELS = {
        0: "background",  # softmax group 1
        1: "lung_lower_lobe_left",  # softmax group 1
        2: "lung_upper_lobe_left",  # softmax group 1
        3: "lung_lower_lobe_right",  # softmax group 1
        4: "lung_middle_lobe_right",  # softmax group 1
        5: "lung_upper_lobe_right",  # softmax group 1
        6: "lung_vessels",  # sigmoid
    }

    METRICS = {
        "loss": MetricType.SMALLER_IS_BETTER,
        "loss_label_background": MetricType.SMALLER_IS_BETTER,
        "loss_label_lung_lower_lobe_left": MetricType.SMALLER_IS_BETTER,
        "loss_label_lung_upper_lobe_left": MetricType.SMALLER_IS_BETTER,
        "loss_label_lung_lower_lobe_right": MetricType.SMALLER_IS_BETTER,
        "loss_label_lung_middle_lobe_right": MetricType.SMALLER_IS_BETTER,
        "loss_label_lung_upper_lobe_right": MetricType.SMALLER_IS_BETTER,
        "loss_label_lung_vessels": MetricType.SMALLER_IS_BETTER,
    }

    N_LABELS = len(LABELS)

    def __init__(
        self,
        *args,
        lr_scheduler=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.lr_scheduler = lr_scheduler

    def _forward(self, data: dict, return_prediction: bool = False):
        images = data["image"].to(self.device).to(torch.float32)

        segmentations = data["segmentation"].to(self.device)
        labels = data["labels"][0]

        with torch.autocast(device_type="cuda", enabled=True):
            outputs = self.model(images)

            losses = self.loss_function(outputs, segmentations)
            result = {}
            if losses.ndim == 5 and losses.shape[1] == len(labels):
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

            prediction[:, 0:6] = torch.softmax(
                prediction[:, 0:6], dim=1
            )  # softmax group 1
            prediction[:, 6] = torch.sigmoid(
                prediction[:, 6]
            )  # sigmoid for lung vessels
            prediction = prediction.detach().cpu().numpy()

        mid_z_slice = data["image"].shape[-1] // 2

        if data["i_patch"][0] == 0:
            image_id = data["image_id"][0]
            with plt.ioff():
                fig, ax = plt.subplots(
                    2, self.N_LABELS + 1, sharex=True, sharey=True, figsize=(10, 2)
                )
                ax[0, 0].imshow(data["image"][0, 0, :, :, mid_z_slice], clim=(0, 1))
                ax[1, 0].imshow(data["image"][0, 0, :, :, mid_z_slice], clim=(0, 1))
                ax[0, 0].set_title(image_id, fontsize=4)
                for i_segmentation in range(self.N_LABELS):
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


@click.command()
@click.option(
    "--run-folder",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Folder to save run to",
)
@click.option(
    "--experiment-name",
    type=str,
    default="detailed_lung_segmentation",
    show_default=True,
    help="Experiment name",
)
@click.option(
    "--data-folder",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Folder containing the dataset",
)
@click.option(
    "--patch-shape",
    type=click.Tuple([int, int, int]),
    default=(128, 128, 128),
    show_default=True,
    help="Patch shape. Defaults to (128, 128, 128).",
)
@click.option(
    "--image-spacing-x-range",
    type=click.Tuple([float, float]),
    default=(1.0, 2.0),
    show_default=True,
    help="Image spacing range in x direction",
)
@click.option(
    "--image-spacing-y-range",
    type=click.Tuple([float, float]),
    default=(1.0, 2.0),
    show_default=True,
    help="Image spacing range in y direction",
)
@click.option(
    "--image-spacing-z-range",
    type=click.Tuple([float, float]),
    default=(1.0, 2.0),
    show_default=True,
    help="Image spacing range in z direction",
)
@click.option(
    "--patches-per-image",
    type=int,
    default=16,
    show_default=True,
    help="Number of patches per image",
)
@click.option(
    "--force-balanced-sampling",
    is_flag=True,
    help="Force balanced sampling",
)
@click.option(
    "--random-rotation",
    is_flag=True,
    help="Randomly rotate images",
)
@click.option(
    "--add-noise",
    type=float,
    default=100.0,
    show_default=True,
    help="Add noise to images",
)
@click.option(
    "--shift-image-values",
    type=click.Tuple([float, float]),
    default=(0.9, 1.1),
    show_default=True,
    help="Shift image values",
)
@click.option(
    "--input-value-range",
    type=click.Tuple([float, float]),
    default=(-1024.0, 3071.0),
    show_default=True,
    help="Input value range",
)
@click.option(
    "--output-value-range",
    type=click.Tuple([float, float]),
    default=(0.0, 1.0),
    show_default=True,
    help="Output value range",
)
@click.option(
    "--encoder-filters",
    callback=lambda ctx, param, value: convert_to_int_list(value),
    default="32,32,32,32",
    show_default=True,
    help="Encoder filters",
)
@click.option(
    "--decoder-filters",
    type=str,
    callback=lambda ctx, param, value: convert_to_int_list(value),
    default="32,32,32,32",
    show_default=True,
    help="Decoder filters",
)
@click.option(
    "--initial-learning-rate",
    type=float,
    default=1e-3,
    show_default=True,
    help="Initial learning rate",
)
@click.option(
    "--lr-scheduler-factor",
    type=float,
    default=0.9,
    show_default=True,
    help="LR scheduler factor",
)
@click.option(
    "--lr-scheduler-patience",
    type=int,
    default=1000,
    show_default=True,
    help="LR scheduler patience",
)
@click.option(
    "--lr-scheduler-threshold",
    type=float,
    default=0.01,
    show_default=True,
    help="LR scheduler threshold",
)
@click.option(
    "--lr-scheduler-cooldown",
    type=int,
    default=1000,
    show_default=True,
    help="LR scheduler cooldown",
)
@click.option(
    "--lr-scheduler-min-lr",
    type=float,
    default=1e-6,
    show_default=True,
    help="LR scheduler minimum learning rate",
)
@click.option(
    "--training-steps",
    type=int,
    default=100_000,
    show_default=True,
    help="Number of training steps",
)
@click.option(
    "--batch-size",
    type=int,
    default=1,
    show_default=True,
    help="Training batch size",
)
@click.option(
    "--n-workers",
    type=int,
    default=4,
    show_default=True,
    help="Number of workers",
)
@click.option(
    "--validation-interval",
    type=int,
    default=1000,
    show_default=True,
    help="Validation interval.",
)
@click.option(
    "--save-interval",
    type=int,
    default=1000,
    show_default=True,
    help="Save interval",
)
@click.option(
    "--random-seed",
    type=int,
    default=1337,
    show_default=True,
    help="Random seed",
)
@click.option(
    "--checkpoint",
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help="Checkpoint file to load",
)
@click.option(
    "--device",
    type=str,
    default="cuda:0",
    show_default=True,
    help="CUDA device to use",
)
@click.option(
    "--loglevel",
    type=click.Choice(["debug", "info", "warning", "error", "critical"]),
    callback=lambda ctx, param, value: getattr(logging, value.upper()),
    default="info",
)
def _cli(**kwargs):
    # set up logging
    logging.getLogger("vroc").setLevel(kwargs["loglevel"])
    logger = logging.getLogger(__name__)
    logger.setLevel(kwargs["loglevel"])
    init_fancy_logging()

    LABELS = {
        0: "background",  # softmax group 1
        1: "lung_lower_lobe_left",  # softmax group 1
        2: "lung_upper_lobe_left",  # softmax group 1
        3: "lung_lower_lobe_right",  # softmax group 1
        4: "lung_middle_lobe_right",  # softmax group 1
        5: "lung_upper_lobe_right",  # softmax group 1
        6: "lung_vessels",  # sigmoid
    }

    LABELS_TO_LOAD = [
        "lung_lower_lobe_left",
        "lung_upper_lobe_left",
        "lung_lower_lobe_right",
        "lung_middle_lobe_right",
        "lung_upper_lobe_right",
        "lung_vessels",
    ]

    IMAGE_FILEPATHS_TOTALSEGMENTATOR = sorted(
        p for p in kwargs["data_folder"].glob("*/ct.nii.gz")
    )
    SEGMENTATION_FILEPATHS_TOTALSEGMENTATOR = [
        {
            segmentation_name: kwargs["data_folder"]
            / image_filepath.parent.name
            / "segmentations"
            / f"{segmentation_name}.nii.gz"
            for segmentation_name in LABELS_TO_LOAD
        }
        for image_filepath in IMAGE_FILEPATHS_TOTALSEGMENTATOR
    ]

    IMAGE_FILEPATHS = []
    SEGMENTATION_FILEPATHS = []

    IMAGE_FILEPATHS += IMAGE_FILEPATHS_TOTALSEGMENTATOR
    SEGMENTATION_FILEPATHS += SEGMENTATION_FILEPATHS_TOTALSEGMENTATOR
    (
        train_image_filepaths,
        test_image_filepaths,
        train_segmentation_filepaths,
        test_segmentation_filepaths,
    ) = train_test_split(
        IMAGE_FILEPATHS,
        SEGMENTATION_FILEPATHS,
        train_size=0.90,
        random_state=kwargs["random_seed"],
    )

    train_dataset = LungCTSegmentationDataset(
        image_filepaths=train_image_filepaths,
        segmentation_filepaths=train_segmentation_filepaths,
        labels=LABELS,
        patch_shape=kwargs["patch_shape"],
        image_spacing_range=(
            kwargs["image_spacing_x_range"],
            kwargs["image_spacing_y_range"],
            kwargs["image_spacing_z_range"],
        ),
        patches_per_image=kwargs["patches_per_image"],
        random_rotation=kwargs["random_rotation"],
        add_noise=kwargs["add_noise"],
        shift_image_values=kwargs["shift_image_values"],
        input_value_range=kwargs["input_value_range"],
        output_value_range=kwargs["output_value_range"],
    )
    test_dataset = LungCTSegmentationDataset(
        image_filepaths=test_image_filepaths,
        segmentation_filepaths=test_segmentation_filepaths,
        labels=LABELS,
        patch_shape=kwargs["patch_shape"],
        image_spacing_range=((1.0, 1.0), (1.0, 1.0), (1.0, 1.0)),
        patches_per_image=kwargs["patches_per_image"],
        random_rotation=False,
        add_noise=0.0,
        shift_image_values=None,
        input_value_range=kwargs["input_value_range"],
        output_value_range=kwargs["output_value_range"],
    )

    COLLATE_NOOP_KEYS = (
        "image_spacing",
        "full_image_shape",
        "i_patch",
        "n_patches",
        "patch_slicing",
        "labels",
    )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=kwargs["batch_size"],
        collate_fn=partial(dict_collate, noop_keys=COLLATE_NOOP_KEYS),
        pin_memory=True,
        num_workers=kwargs["n_workers"],
        persistent_workers=True,
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=partial(dict_collate, noop_keys=COLLATE_NOOP_KEYS),
        pin_memory=True,
        num_workers=kwargs["n_workers"],
        persistent_workers=True,
    )

    enc_filters = kwargs["encoder_filters"]
    dec_filters = kwargs["decoder_filters"]

    n_levels = len(kwargs["encoder_filters"])

    if len(enc_filters) != len(dec_filters):
        raise ValueError(
            "encoder_filters and decoder_filters must have the same length"
        )

    model = FlexUNet(
        n_channels=1,
        n_classes=len(LABELS),
        n_levels=n_levels,
        n_filters=[enc_filters[0], *enc_filters, *dec_filters, dec_filters[-1]],
        convolution_layer=nn.Conv3d,
        downsampling_layer=nn.MaxPool3d,
        upsampling_layer=nn.Upsample,
        norm_layer=nn.InstanceNorm3d,
        skip_connections=True,
        convolution_kwargs=None,
        downsampling_kwargs=None,
        upsampling_kwargs=None,
        return_bottleneck=False,
    ).to(kwargs["device"])

    optimizer = Adam(params=model.parameters(), lr=kwargs["initial_learning_rate"])

    if kwargs["checkpoint"]:
        state = torch.load(
            kwargs["checkpoint"], map_location=torch.device(kwargs["device"])
        )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        logger.info(f"Loaded checkpoint from {kwargs['checkpoint']}")

    loss_function = LungCTSegmentationLoss(use_dice=True)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=kwargs["lr_scheduler_factor"],
        patience=kwargs["lr_scheduler_patience"],
        threshold=kwargs["lr_scheduler_threshold"],
        threshold_mode="rel",
        cooldown=kwargs["lr_scheduler_cooldown"],
        min_lr=kwargs["lr_scheduler_min_lr"],
    )

    trainer = LungCTSegmentationTrainer(
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_data_loader,
        val_loader=test_data_loader,
        run_folder=kwargs["run_folder"],
        experiment_name=kwargs["experiment_name"],
        device=kwargs["device"],
    )

    trainer.run(
        steps=kwargs["training_steps"],
        validation_interval=kwargs["validation_interval"],
        save_interval=kwargs["save_interval"],
    )


if __name__ == "__main__":
    _cli()
