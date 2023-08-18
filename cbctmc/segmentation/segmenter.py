from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
from ipmi.common.logger import init_fancy_logging

from cbctmc.common_types import TorchDevice
from cbctmc.segmentation.labels import LABELS, N_LABELS
from cbctmc.segmentation.patching import PatchExtractor, PatchStitcher
from cbctmc.speedup.models import FlexUNet
from cbctmc.utils import resample_image_spacing, rescale_range

logger = logging.getLogger(__name__)


class MCSegmenter:
    def __init__(
        self,
        model: nn.Module,
        device: TorchDevice,
        patch_shape: Tuple[int, ...] = (128, 128, 128),
        patch_overlap: float = 0.0,
    ):
        self.model = model.to(device)
        self.device = device

        self.patch_shape = patch_shape
        self.patch_overlap = patch_overlap

    def segment(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image, dtype=np.float32)
        if image.ndim != 3:
            raise ValueError("Please pass a 3D image")

        image = rescale_range(
            image,
            input_range=(-1024, 3071),
            output_range=(0, 1),
            clip=True,
        )
        spatial_image_shape = image.shape
        image = torch.as_tensor(image[None, None], device=self.device)

        extractor = PatchExtractor(
            patch_shape=self.patch_shape,
            array_shape=(1, *spatial_image_shape),
            color_axis=0,
        )
        stitcher = PatchStitcher(
            array_shape=(N_LABELS, *spatial_image_shape), color_axis=0
        )
        self.model.eval()

        stride = tuple((1 - self.patch_overlap) * ps for ps in self.patch_shape)
        with torch.autocast(device_type="cuda", enabled=True), torch.inference_mode():
            logger.info(f"Predicting segmentation")
            for slicing in extractor.extract_ordered(stride=stride):
                logger.debug(f"Predicting segmentation for patch {slicing}")
                batch_slicing = (..., *slicing)
                prediction = self.model(image[batch_slicing])

                prediction[:, 0:8] = torch.softmax(
                    prediction[:, 0:8], dim=1
                )  # softmax group 1
                prediction[:, 8] = torch.sigmoid(
                    prediction[:, 8]
                )  # sigmoid for lung vessels

                # squeeze batch axis and convert to numpy array
                prediction = prediction[0].detach().cpu().numpy()

                stitcher.add_patch(prediction, slicing=slicing)

        prediction = stitcher.calculate_mean()
        prediction[8] = prediction[8] > 0.5

        # argmax and convert to one hot
        argmax_label = np.argmax(prediction[:8], axis=0)
        prediction[:8] = np.eye(8, dtype=np.uint8)[:, argmax_label]

        return prediction.astype(np.uint8)


if __name__ == "__main__":
    init_fancy_logging()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logging.getLogger("cbctmc").setLevel(logging.DEBUG)

    enc_filters = [32, 32, 32, 32]
    dec_filters = [32, 32, 32, 32]

    model = FlexUNet(
        n_channels=1,
        n_classes=len(LABELS),
        n_levels=4,
        # filter_base=4,
        n_filters=[32, *enc_filters, *dec_filters, 32],
        convolution_layer=nn.Conv3d,
        downsampling_layer=nn.MaxPool3d,
        upsampling_layer=nn.Upsample,
        norm_layer=nn.InstanceNorm3d,
        skip_connections=True,
        convolution_kwargs=None,
        downsampling_kwargs=None,
        upsampling_kwargs=None,
        return_bottleneck=False,
    )
    state = torch.load(
        "/datalake2/runs/mc_segmentation/"
        "models_0961491db6c842c3958ffb1d/validation/step_72000.pth",
        # "/datalake2/runs/mc_segmentation/models_3f44803be7e542dba54e8ebd/validation/step_15000.pth"
    )
    model.load_state_dict(state["model"])

    segmenter = MCSegmenter(model, device="cuda:1")

    image = sitk.ReadImage(
        "/datalake_fast/mc_test/022_4DCT_Lunge_amplitudebased_complete/phase_00.nii"
    )
    image = resample_image_spacing(
        image,
        new_spacing=(1.0, 1.0, 1.0),
        resampler=sitk.sitkLinear,
        default_voxel_value=-1000,
    )

    image_arr = sitk.GetArrayFromImage(image)
    image_arr = np.swapaxes(image_arr, 0, 2)

    segmentation = segmenter.segment(image_arr, patch_shape=(128, 128, 128))

    s = np.swapaxes(segmentation, 1, 3)
    s = np.moveaxis(s, 0, -1)
    s = sitk.GetImageFromArray(s)
    s.SetSpacing(image.GetSpacing())
    s.SetOrigin(image.GetOrigin())
    s.SetDirection(image.GetDirection())

    sitk.WriteImage(
        s,
        "/datalake_fast/mc_test/022_4DCT_Lunge_amplitudebased_complete/phase_00_seg.nii.gz",
    )