import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from aim import Image

from vroc.affine import AffineTransform
from vroc.blocks import SpatialTransformer
from vroc.keypoint.models import CenterOfMass3d
from vroc.trainer import BaseTrainer, MetricType


class KeypointMatcherTrainer(BaseTrainer):
    METRICS = {"loss": MetricType.SMALLER_IS_BETTER}

    def __init__(self, *args, n_accumulation_steps: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_accumulation_steps = n_accumulation_steps
        self.spatial_transformer = SpatialTransformer()
        self.com = CenterOfMass3d()
        self._i_accumulation_step = 0
        self.optimizer.zero_grad()

    def _forward_pass(self, data: dict):
        images = data["image"].to(self.device)
        segmentations = data["segmentation"].to(self.device)

        # select random voxel from mask
        random_keypoint = random.choice(segmentations.argwhere())
        random_keypoint_masks = torch.zeros_like(segmentations)
        random_keypoint_masks[
            random_keypoint[0],
            random_keypoint[1],
            random_keypoint[2],
            random_keypoint[3],
            random_keypoint[4],
        ] = 1

        while True:
            # random affine transformation
            affine_matrix = AffineTransform.create_affine_matrix_3d(
                # translation=np.random.uniform(low=-0.25, high=0.25, size=3).astype(
                #     np.float32
                # ),
                # scale=np.random.uniform(low=0.75, high=1.25, size=3).astype(np.float32),
                # rotation=np.random.uniform(low=0, high=2 * np.pi, size=3).astype(
                #     np.float32
                # ),
                # translation=(0.25, 0, 0),
                # rotation=(0, 0.5, 0),
                # scale=(0.75, 1, 1.2),
                # shear=np.random.uniform(low=-0.25, high=0.25, size=6).astype(
                #     np.float32
                # ),  # yx, zx, xy, zy, xz, yz
                dtype=torch.float32,
                device=self.device,
            )
            # add batch dimension
            affine_matrix = affine_matrix[None]

            warped_images = self.spatial_transformer(
                images, affine_matrix, mode="bilinear"
            )
            warped_keypoint_masks = self.spatial_transformer(
                random_keypoint_masks, affine_matrix, mode="nearest"
            )
            if warped_keypoint_masks.sum() > 0:
                # warped keypoint is still in the image
                break

        initial_keypoints = self.com(random_keypoint_masks)
        wraped_keypoints = self.com(warped_keypoint_masks)

        x_init, y_init, z_init = initial_keypoints.to(torch.int64, copy=True)[0, 0]
        x_warped, y_warped, z_warped = wraped_keypoints.to(torch.int64, copy=True)[0, 0]

        # print(f'{initial_keypoints=}')
        # print(f'{wraped_keypoints=}')

        with torch.autocast(device_type="cuda", enabled=False):
            image_descriptors, reference_descriptors = self.model(
                reference_image=images,
                image=warped_images,
                reference_keypoints=random_keypoint_masks,
            )
            image_descriptors = torch.sigmoid(image_descriptors)
            loss = F.mse_loss(
                image_descriptors, warped_keypoint_masks, reduction="none"
            )

            # loss[warped_keypoint_masks < 1] *= 1 / 128**3
            # loss[warped_keypoint_masks == 1] *= ( 128 ** 3 - 1) / 128 ** 3
            loss = loss.mean()
            pass
            # image descriptors should be the same for selected keypoint and wraped keypoint

            # loss_mse = F.mse_loss(predicted_probability, warped_keypoint_masks)
            # loss_tre = self.loss_function(predicted_keypoints, wraped_keypoints).mean()

            # random_tre_loss = 88.0
            # random_mse_loss = 0.80
            #
            # loss = loss_tre / random_tre_loss + loss_mse / random_mse_loss

            # loss = loss_tre
            #
            # self.log_debug(f"{loss_tre=}", context="TRAIN")
            # self.log_debug(f"{loss_mse=}", context="TRAIN")
            #
            # self.log_debug(
            #     f"wraped_keypoints={wraped_keypoints.cpu().detach().numpy().squeeze()}, predicted_keypoints={predicted_keypoints.cpu().detach().numpy().squeeze()}",
            #     context="TRAIN",
            # )

            # fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)
            #
            # i_slice = int(initial_keypoints[0, 0, 1].cpu().detach().numpy())
            # ax[0, 0].imshow(images[0, 0, :, i_slice, :].cpu().detach().numpy())
            # ax[1, 0].imshow(
            #     random_keypoint_masks[0, 0, :, i_slice, :].cpu().detach().numpy()
            # )
            # ax[0, 0].scatter(
            #     initial_keypoints[0, 0, 2].cpu().detach().numpy(),
            #     initial_keypoints[0, 0, 0].cpu().detach().numpy(),
            #     marker="x",
            #     c="red",
            # )
            # i_slice = int(wraped_keypoints[0, 0, 1].cpu().detach().numpy())
            # ax[0, 1].imshow(warped_images[0, 0, :, i_slice, :].cpu().detach().numpy())
            # ax[1, 1].imshow(
            #     warped_keypoint_masks[0, 0, :, i_slice, :].cpu().detach().numpy()
            # )
            # ax[0, 1].scatter(
            #     wraped_keypoints[0, 0, 2].cpu().detach().numpy(),
            #     wraped_keypoints[0, 0, 0].cpu().detach().numpy(),
            #     marker="x",
            #     c="red",
            # )
            # ax[0, 2].imshow(
            #     predicted_probability[0, 0, :, i_slice, :].cpu().detach().numpy()
            # )
            # ax[1, 2].imshow(
            #     warped_keypoint_masks[0, 0, :, i_slice, :].cpu().detach().numpy()
            # )
            # ax[0, 2].scatter(
            #     wraped_keypoints[0, 0, 2].cpu().detach().numpy(),
            #     wraped_keypoints[0, 0, 0].cpu().detach().numpy(),
            #     marker="x",
            #     c="red",
            # )
            # plt.show()

            # losses = self.loss_function(predicted_keypoints, wraped_keypoints)
            # loss = losses.mean()

            # print(f'{predicted_keypoints=}')
            # print(f'{wraped_keypoints=}')

            # loss = F.mse_loss(predicted_probability, warped_keypoint_masks)

            if not torch.isfinite(loss):
                raise ValueError("Loss is not finite")
        return loss

    def train_on_batch(self, data: dict) -> dict:
        self.optimizer.zero_grad()

        loss = self._forward_pass(data)
        self.scaler.scale(loss / self.n_accumulation_steps).backward()

        self._i_accumulation_step += 1
        if self._i_accumulation_step % self.n_accumulation_steps == 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self._i_accumulation_step = 0

        return {"loss": float(loss)}

    def validate_on_batch(self, data: dict) -> dict:
        pass
