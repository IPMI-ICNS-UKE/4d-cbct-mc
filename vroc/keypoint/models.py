from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F

from vroc.models import FlexUNet


class CenterOfMass3d(nn.Module):
    def __init__(self):
        super(CenterOfMass3d, self).__init__()

    def forward(self, vol):
        """
        x: tensor of shape [n_batch, chs, dim1, dim2, dim3]
        returns: center of mass, shape [n_batch, chs, 3]
        """
        n_batch, chs, dim1, dim2, dim3 = vol.shape
        eps = 1e-8
        arange1 = (
            torch.linspace(0, 1, dim1 + 1)[:-1]
            .float()
            .view(1, 1, -1)
            .repeat(n_batch, chs, 1)
        )
        arange2 = (
            torch.linspace(0, 1, dim2 + 1)[:-1]
            .float()
            .view(1, 1, -1)
            .repeat(n_batch, chs, 1)
        )
        arange3 = (
            torch.linspace(0, 1, dim3 + 1)[:-1]
            .float()
            .view(1, 1, -1)
            .repeat(n_batch, chs, 1)
        )

        arange1, arange2, arange3 = (
            arange1.to(vol.device),
            arange2.to(vol.device),
            arange3.to(vol.device),
        )

        mx = vol.sum(dim=(2, 3))  # mass along the dimN, shape [n_batch, chs, dimN]
        Mx = mx.sum(-1, True) + eps  # total mass along dimN

        my = vol.sum(dim=(2, 4))
        My = my.sum(-1, True) + eps

        mz = vol.sum(dim=(3, 4))
        Mz = mz.sum(-1, True) + eps

        cx = (arange1 * mx).sum(
            -1, True
        ) / Mx  # center of mass along dimN, shape [n_batch, chs, 1]
        cy = (arange2 * my).sum(-1, True) / My
        cz = (arange3 * mz).sum(-1, True) / Mz

        C = torch.cat([cz, cy, cx], -1)  # center of mass, shape [n_batch, chs, 3]

        # rescale from [0, 1] to image shape-based index
        C = C * torch.tensor([dim1, dim2, dim3])[None].to(C.device)

        return C


# class KeypointMatcher(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         n_filters_init = 3
#         encoder_n_filters = (3,)
#         decoder_n_filters = (3, )
#         n_filters_final = 3
#         self.unet = FlexUNet(
#             n_channels=3,
#             n_classes=1,
#             n_levels=1,
#             n_filters=(
#                 n_filters_init,
#                 *encoder_n_filters,
#                 *decoder_n_filters,
#                 n_filters_final,
#             ),
#             # norm_layer=nn.InstanceNorm3d,
#             norm_layer=None,
#             skip_connections=True,
#             return_bottleneck=False,N
#         )
#         self.com = CenterOfMass3d()
#
#     def forward(
#         self,
#         reference_image: torch.Tensor,
#         image: torch.Tensor,
#         reference_keypoints: torch.Tensor,
#     ):
#         inputs = torch.cat([reference_image, reference_keypoints, image], dim=1)
#
#         output = self.unet(inputs)
#         # keypoint_proba_map = torch.sigmoid(output)
#         keypoint_proba_map = output
#         keypoint = self.com(output)
#
#
#         return keypoint, keypoint_proba_map
#


class KeypointMatcher(nn.Module):
    def __init__(self):
        super().__init__()

        self.descriptor = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding="same"),
            nn.InstanceNorm3d(16),
            nn.Mish(),
            nn.Conv3d(16, 32, kernel_size=3, padding="same"),
            nn.InstanceNorm3d(32),
            nn.Mish(),
            nn.Conv3d(32, 1, kernel_size=3, padding="same"),
        )

        self.com = CenterOfMass3d()

    def forward(
        self,
        reference_image: torch.Tensor,
        image: torch.Tensor,
        reference_keypoints: torch.Tensor,
    ):
        # reference_image = torch.cat([reference_image, reference_keypoints], dim=1)
        # image = torch.cat([image, torch.ones_like(image)], dim=1)

        reference_descriptors = self.descriptor(reference_image)
        image_descriptors = self.descriptor(image)

        # # get the feature vector of the reference image where the keypoint is
        # # shape is [n_batch, n_features, n_keypoints=1]
        # reference_keypoint_feature = reference_descriptors[:, :, reference_keypoints.to(torch.bool)[0, 0]][..., None, None]
        #
        # # calculate cosine similarity between the reference keypoint feature and all image features
        # # this is the keypoint probability map
        # keypoint_proba_map = F.cosine_similarity(reference_keypoint_feature, image_descriptors, dim=1)
        # # bound cosine similarity to [0, 1]
        # keypoint_proba_map = F.relu(keypoint_proba_map)
        # # add color channel
        # keypoint_proba_map = keypoint_proba_map[:, None]
        #
        # # clipped_keypoint_proba_map = torch.relu(keypoint_proba_map - 0.95 * keypoint_proba_map.max())
        #
        # keypoint = self.com(keypoint_proba_map)

        return (
            image_descriptors,
            reference_descriptors,
        )  # , keypoint, keypoint_proba_map


if __name__ == "__main__":
    com = CenterOfMass3d()

    image = torch.zeros((1, 1, 10, 10, 10))
    image[:, :, 5, 5, 5] = 1
    image[:, :, 5, 6, 7] = 1

    c = com(image)

    print(c)
