from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import numpy as np
import scipy.ndimage as ndi
import SimpleITK as sitk
import torch
import torch.nn as nn
from torch.nn import functional as F

from vroc.common_types import ArrayOrTensor, IntTuple, Number, TorchDevice
from vroc.helper import batch_array, nearest_factor_pow_2, rescale_range
from vroc.interpolation import resize
from vroc.models import Unet3d
from vroc.preprocessing import crop_or_pad


class BaseSegmenter(ABC):
    def __init__(
        self,
        model: nn.Module,
        input_value_range: Tuple[Number, Number] | None = None,
        output_value_range: Tuple[Number, Number] | None = None,
        device: TorchDevice = "cuda",
        pad_to_pow_2: bool = True,
        return_as_tensor: bool = False,
    ):
        self.input_value_range = input_value_range
        self.output_value_range = output_value_range
        self.model = model
        self.device = device
        self.pad_to_pow_2 = pad_to_pow_2
        self.return_as_tensor = return_as_tensor

        self.model.to(self.device)
        self.model.eval()

    @property
    @abstractmethod
    def spatial_dims(self) -> int:
        pass

    @property
    def tensor_dims(self) -> int:
        return self.spatial_dims + 2  # +2, .i.e., +color +batch

    def _cast_to_tensor(self, image: ArrayOrTensor):
        if isinstance(image, np.ndarray):
            if image.ndim != self.spatial_dims:
                raise ValueError(
                    f"Please pass a {self.spatial_dims} dimensional NumPy Array"
                )
            # add 2 dimensions
            image = image[None, None]

        elif isinstance(image, torch.Tensor):
            if image.ndim != self.spatial_dims:
                raise ValueError(
                    f"Please pass a {self.tensor_dims} dimensional PyTorch Tensor"
                )

        else:
            raise RuntimeError(
                "Unsupported image type! Pass NumPy Array or PyTorch Tensor"
            )

        return torch.as_tensor(image, dtype=torch.float32, device=self.device)

    @staticmethod
    def _calculate_padding(
        image: torch.Tensor, target_shape: IntTuple
    ) -> Tuple[tuple, tuple]:
        if image.ndim - 2 != len(target_shape):
            raise ValueError(
                f"Dimension mismatch. Pass {image.ndim - 2} dimensional spatial "
                f"padding for {image.ndim} dimensional tensor"
            )

        spatial_image_shape = image.shape[2:]
        padding = tuple((t - s for (t, s) in zip(target_shape, spatial_image_shape)))

        padding_left = tuple(p // 2 for p in padding)
        padding_right = tuple(p - pl for (p, pl) in zip(padding, padding_left))

        return padding_left, padding_right

    @staticmethod
    def _pad_image(
        image: torch.Tensor, target_shape: IntTuple, pad_value: Number = 0.0
    ) -> torch.Tensor:
        padding_left, padding_right = BaseSegmenter._calculate_padding(
            image=image, target_shape=target_shape
        )

        padding = [p for paddings in zip(padding_left, padding_right) for p in paddings]

        # PyTorch starts padding from the last dimension, thus reverse
        padding = padding[::-1]
        return F.pad(image, padding, mode="constant", value=pad_value)

    @staticmethod
    def _unpad_image(image: torch.Tensor, target_shape: IntTuple) -> torch.Tensor:
        # paddings are negative
        padding_left, padding_right = BaseSegmenter._calculate_padding(
            image=image, target_shape=target_shape
        )
        spatial_image_shape = image.shape[2:]

        # create slicing from padding
        slicing = [...]
        for pl, pr, axis_length in zip(
            padding_left, padding_right, spatial_image_shape
        ):
            slicing.append(slice(-pl, axis_length + pr))
        slicing = tuple(slicing)

        return image[slicing]

    def _prepare_image(self, image: ArrayOrTensor) -> torch.Tensor:
        image = self._cast_to_tensor(image)

        spatial_image_shape = image.shape[2:]

        # pad to power of 2 (needed for U-Nets etc.)
        if self.pad_to_pow_2:
            padded_spatial_image_shape = tuple(
                nearest_factor_pow_2(s) for s in spatial_image_shape
            )

            image = self._pad_image(
                image=image, target_shape=padded_spatial_image_shape
            )

        if self.input_value_range and self.output_value_range:
            image = rescale_range(
                image,
                input_range=self.input_value_range,
                output_range=self.output_value_range,
                clip=True,
            )

        return image

    @staticmethod
    def _logits_to_segmentation(logits: torch.Tensor, binary_threshold: float = 0.5):
        n_classes = logits.shape[1]

        if n_classes == 1:
            # single class segmentation, apply sigmoid
            segmentation = torch.sigmoid(logits) > binary_threshold
        else:
            segmentation = torch.softmax(logits, dim=1)
            return segmentation
            segmentation = torch.argmax(segmentation, dim=1, keepdim=True)

        return segmentation

    @abstractmethod
    def _forward(self, image: torch.Tensor) -> torch.Tensor:
        pass

    def segment(
        self,
        image: ArrayOrTensor,
        binary_threshold: float = 0.5,
        clear_cuda_cache: bool = False,
    ) -> ArrayOrTensor:
        spatial_image_shape = image.shape[-self.spatial_dims :]

        image = self._prepare_image(image)
        with torch.inference_mode():
            logits = self._forward(image=image)

            logits = self._unpad_image(logits, target_shape=spatial_image_shape)

            segmentation = self._logits_to_segmentation(
                logits=logits, binary_threshold=binary_threshold
            )

        if clear_cuda_cache:
            torch.cuda.empty_cache()

        if not self.return_as_tensor:
            segmentation = segmentation.detach().cpu().numpy().squeeze()

        return segmentation


class Segmenter3d(BaseSegmenter):
    @property
    def spatial_dims(self) -> int:
        return 3

    def _forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)


class MultiClassSegmenter3d(Segmenter3d):
    pass


class Segmenter2D(ABC):
    def __init__(
        self,
        model: nn.Module,
        state_filepath: Path = None,
        device="cuda",
        iter_axis: int = 2,  # Model input: axial slices
    ):
        self.model = model
        self.device = device
        self.iter_axis = iter_axis

        if state_filepath:
            try:
                self._load_state_dict(state_filepath)
            except RuntimeError:
                self._load_state_dict(state_filepath, remove_module=True)

        self.model.to(self.device)
        self.model.eval()

    def _load_state_dict(self, state_filepath, remove_module: bool = True):
        state_dict = torch.load(state_filepath, map_location=self.device)
        if remove_module:
            _state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("module."):
                    _state_dict[key[7:]] = value
            state_dict = _state_dict
        self.model.load_state_dict(state_dict)

    def _prepare_axes(self, image: np.ndarray, inverse: bool = False):
        if inverse:
            image = np.swapaxes(image, 0, self.iter_axis)
            image = np.flip(image, axis=1)
        else:
            image = np.flip(image, axis=1)
            image = np.swapaxes(image, self.iter_axis, 0)
        return image

    def segment(
        self,
        image: np.ndarray,
        batch_size: int = 16,
        fill_holes: bool = True,
        clear_cuda_cache: bool = False,
    ):
        image = self._prepare_axes(image=image, inverse=False)
        image = self._prepare_image(image=image)
        predicted_batches = []
        for image_batch in batch_array(image, batch_size=batch_size):
            image_batch = torch.as_tensor(
                image_batch, dtype=torch.float32, device=self.device
            )
            with torch.no_grad():
                logits = self.model(image_batch)
            prediction = torch.sigmoid(logits) > 0.3
            prediction = prediction.detach().cpu().numpy()
            prediction = prediction.squeeze(axis=1)
            predicted_batches.append(prediction)

        prediction = np.concatenate(predicted_batches)
        prediction = self._prepare_axes(image=prediction, inverse=True)
        if fill_holes:
            prediction = ndi.binary_fill_holes(prediction)

        return prediction

    @abstractmethod
    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        pass


class LungSegmenter2D(Segmenter2D):
    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        image = rescale_range(image, input_range=(-1000, 200), output_range=(-1, 1))
        return image[:, np.newaxis]


class LungSegmenter3D:
    def __init__(self, model: nn.Module, device: TorchDevice):
        self.model = model.to(device)
        self.device = device

    def _segment(self, image: np.ndarray, subsample: float = 1.5) -> np.ndarray:
        print(f"Run segmentation on image of shape {image.shape}")
        image = np.asarray(image, dtype=np.float32)
        if image.ndim != 3:
            raise ValueError("Please pass a 3D image")

        original_shape = image.shape

        image = resize(
            image, output_shape=tuple(s // subsample for s in original_shape)
        )
        unpadded_shape = image.shape
        padded_shape = tuple(
            nearest_factor_pow_2(s, min_exponent=4) for s in unpadded_shape
        )

        image, _ = crop_or_pad(image=image, mask=None, target_shape=padded_shape)
        image = rescale_range(
            image,
            input_range=(-1024, 3071),
            output_range=(0, 1),
            clip=True,
        )
        image = torch.as_tensor(image[None, None], device=self.device)

        self.model.eval()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction = self.model(image)
            prediction = torch.sigmoid(prediction)

        prediction = prediction.detach().cpu().numpy().squeeze(axis=(0, 1))
        prediction, _ = crop_or_pad(
            image=prediction, mask=None, target_shape=unpadded_shape
        )

        prediction = resize(prediction, output_shape=original_shape)
        prediction = prediction > 0.5

        return prediction

    def segment(
        self, image: np.ndarray | sitk.Image, subsample: float = 1.5
    ) -> np.ndarray | sitk.Image:
        if isinstance(image, sitk.Image):
            image_spacing = image.GetSpacing()
            image_direction = image.GetDirection()
            image_origin = image.GetOrigin()

            image_arr = sitk.GetArrayFromImage(image)
            image_arr = np.swapaxes(image_arr, 0, 2)
            segmentation = self._segment(image=image_arr, subsample=subsample)
            segmentation = np.swapaxes(segmentation, 0, 2)
            segmentation = sitk.GetImageFromArray(segmentation.astype(np.uint8))
            segmentation.SetSpacing(image_spacing)
            segmentation.SetDirection(image_direction)
            segmentation.SetOrigin(image_origin)

        else:
            segmentation = self._segment(image, subsample=subsample)

        return segmentation


if __name__ == "__main__":
    # # test OASIS segmentation
    # import matplotlib.pyplot as plt
    # import SimpleITK as sitk
    #
    # image = sitk.ReadImage("/datalake/learn2reg/OASIS/imagesTr/OASIS_0001_0000.nii.gz")
    # labels = sitk.ReadImage("/datalake/learn2reg/OASIS/labelsTr/OASIS_0001_0000.nii.gz")
    #
    # image = np.swapaxes(sitk.GetArrayFromImage(image), 0, 2)
    # labels = np.swapaxes(sitk.GetArrayFromImage(labels), 0, 2)
    #
    # model = Unet3d(n_channels=1, n_classes=36, n_levels=4, filter_base=16)
    # state = torch.load(
    #     "/datalake/learn2reg/runs/models_0973f24c4a564338ab3920e4/step_93000.pth"
    # )
    # model.load_state_dict(state["model"])
    #
    # segmenter = Segmenter3d(
    #     model=model,
    #     input_value_range=(0, 1),
    #     output_value_range=(0, 1),
    #     device="cuda:0",
    #     return_as_tensor=False,
    # )
    #
    # prediction = segmenter.segment(image)
    #
    # fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
    # ax[0].imshow(prediction[:, 112, :])
    # ax[1].imshow(labels[:, 112, :])

    # test total segmentator
    import SimpleITK as sitk

    image = sitk.ReadImage("/datalake/totalsegmentator/s0000/ct.nii.gz")
    reference_image = image

    image = np.swapaxes(sitk.GetArrayFromImage(image), 0, 2)

    model = Unet3d(n_classes=59, n_levels=4, filter_base=24)
    state = torch.load(
        "/datalake/learn2reg/runs/models_86cd845411c44db1a4318db1/validation/step_357000.pth"
    )
    model.load_state_dict(state["model"])

    segmenter = Segmenter3d(
        model=model,
        input_value_range=(-1024, 3071),
        output_value_range=(0, 1),
        device="cuda:0",
        pad_to_pow_2=True,
        return_as_tensor=False,
    )

    prediction = segmenter.segment(image).astype(np.float32)
    prediction = prediction.transpose((3, 2, 1, 0))
    prediction = prediction.argmax(axis=-1).astype(np.uint8)
    prediction = sitk.GetImageFromArray(prediction)
    prediction.CopyInformation(reference_image)
    sitk.WriteImage(prediction, "/datalake/totalsegmentator/s0000/pred2.nii.gz")
