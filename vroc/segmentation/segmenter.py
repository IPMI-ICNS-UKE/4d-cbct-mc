from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

import numpy as np
import SimpleITK
import torch
import torch.nn as nn

from vroc.common_types import (
    FloatTuple3D,
    IntTuple2D,
    IntTuple3D,
    PathLike,
    TorchDevice,
)
from vroc.helper import rescale_range
from vroc.interpolation import resize_spacing
from vroc.logger import init_fancy_logging
from vroc.models import FlexUNet
from vroc.patching.extractor import PatchExtractor
from vroc.patching.stitcher import PatchStitcher
from vroc.preprocessing import resample_image_spacing

logger = logging.getLogger(__name__)


class BaseSegmenter(ABC):
    def __init__(
        self, model: nn.Module, device: TorchDevice, segmentation_dtype=np.uint8
    ):
        self.model = model.to(device)
        self.device = device
        self.segmentation_dtype = segmentation_dtype

    def _segment(
        self,
        image: np.ndarray,
        patch_shape: IntTuple2D | IntTuple3D | None,
        patch_overlap: float = 0.5,
    ) -> PatchStitcher:
        if not 0 <= patch_overlap < 1:
            raise ValueError("patch_overlap must be in [0 and 1)")

        extractor = PatchExtractor(
            array_shape=image.shape,
            patch_shape=patch_shape,
            color_axis=None,
        )

        stitcher = None

        self.model.eval()

        stride = tuple(
            int(round(p * (1 - patch_overlap))) for p in extractor.patch_shape
        )
        # min stride is 1
        stride = tuple(s if s > 0 else 1 for s in stride)

        logger.info(f"Start prediction with {stride=} ({patch_overlap=})")
        t = time.monotonic()
        for spatial_slicing in extractor.extract_ordered(stride=stride, flush=True):
            image_patch = image[spatial_slicing]
            logger.debug(f"Predict segmentation for patch at {spatial_slicing}")

            # cast to tensor and add batch and color axis
            image_patch = torch.as_tensor(
                image_patch[None, None], dtype=torch.float32, device=self.device
            )
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", enabled=True
            ):
                prediction = self.model(image_patch)

            prediction = prediction.detach().cpu().numpy()
            # remove batch dimension
            prediction = prediction[0]

            if stitcher is None:
                n_labels = prediction.shape[0]
                # initialize stitcher as we now know the number of labels
                stitcher = PatchStitcher(
                    array_shape=(n_labels, *image.shape),
                    color_axis=0,
                )

            # add patch
            stitcher.add_patch(prediction, slicing=(slice(None), *spatial_slicing))

        logger.info(f"Finished prediction in {time.monotonic() - t:.2f}s")
        return stitcher

    def segment(
        self,
        image: np.ndarray,
        patch_shape: IntTuple2D | IntTuple3D | None = None,
        patch_overlap: float = 0.5,
        **kwargs,
    ):
        image = self.prepare_input(image)
        stitcher = self._segment(image, patch_shape, patch_overlap)
        segmentation = self.model_output_to_segmentation(stitcher)

        return np.asarray(segmentation, dtype=self.segmentation_dtype)

    @abstractmethod
    def model_output_to_segmentation(self, stitcher: PatchStitcher) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def prepare_input(self, image: np.ndarray):
        raise NotImplementedError


class LungCTSegmenter(BaseSegmenter):
    _IMAGE_SPACING = (1.0, 1.0, 1.0)

    def model_output_to_segmentation(
        self,
        stitcher: PatchStitcher,
        input_image_spacing: FloatTuple3D = (1.0, 1.0, 1.0),
    ) -> np.ndarray:
        """Convert the model output to a segmentation.

        Here we have the following labels:
        0: "background",  # softmax group 1
        1: "lung_lower_lobe_left",  # softmax group 1
        2: "lung_upper_lobe_left",  # softmax group 1
        3: "lung_lower_lobe_right",  # softmax group 1
        4: "lung_middle_lobe_right",  # softmax group 1
        5: "lung_upper_lobe_right",  # softmax group 1
        6: "lung_vessels",  # sigmoid
        """

        prediction = stitcher.calculate_mean()
        if input_image_spacing != self._IMAGE_SPACING:
            logger.info(
                f"Resampling image spacing from "
                f"{self._IMAGE_SPACING} to {input_image_spacing}"
            )
            n_classes = prediction.shape[0]
            prediction = resize_spacing(
                prediction,
                input_image_spacing=(n_classes,) + self._IMAGE_SPACING,
                output_image_spacing=(n_classes,) + input_image_spacing,
                order=1,
            )

        # prediction = prediction[:, 16:-16, 16:-16, 16:-16]
        prediction = torch.as_tensor(prediction)
        prediction[0:6] = torch.softmax(prediction[0:6], dim=0)  # softmax group 1
        prediction[6] = torch.sigmoid(prediction[6])  # sigmoid for lung vessels

        # # calculate softmax for softmax group 1
        # prediction[:6] = np.exp(prediction[:6]) / np.sum(np.exp(prediction[:6]), axis=0)
        # # calculate sigmoid for lung vessels
        # prediction[6] = 1 / (1 + np.exp(-prediction[6]))

        prediction = prediction.detach().cpu().numpy()

        # argmax and convert to one hot for softmax group 1
        predicted_label = np.argmax(prediction[:6], axis=0)
        prediction[:6] = np.eye(6, dtype=np.uint8)[:, predicted_label]
        # simple threshold for lung vessels
        prediction[6] = prediction[6] > 0.5

        return prediction

    def segment(
        self,
        image: np.ndarray,
        patch_shape: IntTuple2D | IntTuple3D | None = None,
        patch_overlap: float = 0.5,
        input_image_spacing: FloatTuple3D = (1.0, 1.0, 1.0),
    ):
        image = self.prepare_input(image, input_image_spacing=input_image_spacing)
        stitcher = self._segment(image, patch_shape, patch_overlap)
        segmentation = self.model_output_to_segmentation(
            stitcher, input_image_spacing=input_image_spacing
        )

        return np.asarray(segmentation, dtype=self.segmentation_dtype)

    def prepare_input(
        self,
        image: np.ndarray,
        input_image_spacing: FloatTuple3D = (1.0, 1.0, 1.0),
    ):
        if image.ndim != 3:
            raise ValueError("Please pass a 3D image")
        image = np.asarray(image, dtype=np.float32)

        if input_image_spacing:
            logger.info(
                f"Resampling image spacing from "
                f"{input_image_spacing} to {self._IMAGE_SPACING}"
            )
            image = resize_spacing(
                image,
                input_image_spacing=input_image_spacing,
                output_image_spacing=self._IMAGE_SPACING,
                order=1,
            )

        image = rescale_range(
            image,
            input_range=(-1024, 3071),
            output_range=(0, 1),
            clip=True,
        )

        # pad image
        # image = np.pad(image, 16, mode="constant", constant_values=0)

        return image

    @classmethod
    def get_default_segmenter(
        cls, state_filepath: PathLike, device: TorchDevice = "cuda:0"
    ):
        enc_filters = [16, 32, 64, 128]
        dec_filters = [128, 64, 32, 16]
        model = FlexUNet(
            n_channels=1,
            n_classes=7,
            n_levels=4,
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
        )
        state = torch.load(state_filepath)
        model.load_state_dict(state["model"])

        segmenter = cls(
            model=model,
            device=device,
        )

        return segmenter


if __name__ == "__main__":
    import SimpleITK as sitk

    logging.getLogger("vroc").setLevel(logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    init_fancy_logging()

    segmenter = LungCTSegmenter.get_default_segmenter(
        state_filepath="/mnt/nas_io/vroc/lung_segmentation/2023-10-09T16:22:40.776924_run_6f99e3c7f1614171a62d5c0d/models/validation/step_80000.pth"
    )

    image = sitk.ReadImage(
        "/datalake/learn2reg/2023/37db95e6-19e4-4e0a-a177-4b12df6ae7b5/data/imagesTr/NLST_0106_0000.nii.gz"
    )
    image_arr = sitk.GetArrayFromImage(image)
    # fix NLST
    # image_arr = image_arr[:, ::-1, :]

    image_arr = np.swapaxes(image_arr, 0, 2)
    segmentation = segmenter.segment(
        image=image_arr,
        patch_shape=(128, 128, 128),
        patch_overlap=0.75,
        input_image_spacing=(1.5, 1.5, 1.5),
    )

    segmentation = np.swapaxes(segmentation, 1, 3)
    # fix NLST
    # segmentation = segmentation[:, :, ::-1, :]
    segmentation = np.moveaxis(segmentation, 0, -1)
    segmentation = sitk.GetImageFromArray(segmentation, isVector=True)
    segmentation.CopyInformation(image)
    sitk.WriteImage(
        segmentation,
        "/datalake/learn2reg/2023/37db95e6-19e4-4e0a-a177-4b12df6ae7b5/data/seg.nii",
    )
