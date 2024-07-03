import logging
from pathlib import Path

import numpy as np
import os
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pkg_resources
import torch
from torch import nn

from cbctmc.logger import init_fancy_logging
from cbctmc.registration.correspondence import CorrespondenceModel
from cbctmc.mc.respiratory import RespiratorySignal
from cbctmc.reconstruction.respiratory import calculate_median_cycle
from cbctmc.defaults import DefaultMCSimulationParameters as MCDefaults
from cbctmc.segmentation.labels import LABELS, get_label_index
from cbctmc.segmentation.segmenter import MCSegmenter
from cbctmc.speedup.models import FlexUNet
from cbctmc.utils import resample_image_spacing
from vroc.blocks import SpatialTransformer



def read_image(filepath) -> np.ndarray:
    image = sitk.ReadImage(str(filepath))
    image = sitk.GetArrayFromImage(image)
    image = np.swapaxes(image, 0, 2)
    return image

def calculate_ref_signal(signal):
    ct_amplitude = (
        RespiratorySignal(signal=signal, sampling_frequency=25)
        .resample(MCDefaults.frame_rate)
        .signal
    )
    ct_median_amplitude = calculate_median_cycle(ct_amplitude)
    ct_median_amplitude_offset = ct_median_amplitude.mean()
    ct_median_amplitude -= ct_median_amplitude_offset

    ct_median_signal = RespiratorySignal(
        signal=ct_median_amplitude,
        sampling_frequency=MCDefaults.frame_rate,
    )

    phase_timepoints = np.arange(
        0, len(ct_median_signal.time), len(ct_median_signal.time) // 10
    )[:10]
    phase_signals = np.stack(
        (
            ct_median_signal.signal[phase_timepoints],
            ct_median_signal.dt_signal[phase_timepoints],
        ),
        axis=-1,
    )

    return phase_signals
def main(directory, CIRS, segment):
    torch.cuda.empty_cache()
    if CIRS:
        signal = RespiratorySignal.load(os.path.join(directory, "4d_cirs", "cirs_signal.pkl")).signal
        image_filepaths= os.path.join(directory, "4d_cirs")
    else:
        signal = pd.read_csv(os.path.join(directory, "rpm", "4_R2018045_4DCT_0.csv"))["ant.-pos.-ampl.[cm]"].values
        RespiratorySignal(signal=signal, sampling_frequency=25).resample(MCDefaults.frame_rate).save(
            os.path.join(directory, "rpm", "signal.pkl"))
        image_filepaths = os.path.join(directory, "4dct")
    images = [read_image(os.path.join(image_filepaths, img)) for img in sorted(os.listdir(image_filepaths)) if img.endswith(".mha") or img.endswith(".nii.gz")]
    images = np.stack(images, axis=0)

    phase_signals = calculate_ref_signal(signal)

    if segment:
        if not os.path.isfile(os.path.join(directory, "lung_mask.npy")):
            enc_filters = [32, 32, 32, 32]
            dec_filters = [32, 32, 32, 32]

            model = FlexUNet(
                n_channels=1,
                n_classes=len(LABELS),
                n_levels=4,
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
            state = torch.load(Path(pkg_resources.resource_filename("cbctmc", f"assets/weights/segmenter/default.pth")))
            model.load_state_dict(state["model"])

            segmenter = MCSegmenter(
                model=model,
                device=f"cuda:0",
                patch_shape=(256, 256, 128),
                patch_overlap=0.5,
            )
            lung_masks = []
            for image in images:
                segmentation, logits = segmenter.segment(image)

                lung_mask = segmentation[get_label_index('lung')]
                lung_masks.append(lung_mask)
            lung_masks = np.stack(lung_masks, axis=0)
            np.save(os.path.join(directory, "lung_mask.npy"), lung_masks)
            print("Start corr modeling")
        else:
            lung_masks = np.load(os.path.join(directory, "lung_mask.npy"))
        cor_modeling(images=images, masks=lung_masks, signal=phase_signals, vf_dir= os.path.join(directory,"vfs"))


    else:
        print("Start corr modeling")
        cor_modeling(images=images, masks=None,signal=phase_signals)


def cor_modeling(images: np.ndarray, masks: np.ndarray, signal: np.ndarray, vf_dir: os.PathLike):
    model = CorrespondenceModel.build_default(
        images=images,
        masks=masks,
        signals=signal,
        masked_registration=True,
        device="cuda:0",
        save_vf= False,
        vf_dir= vf_dir,
        overwrite_vf = False
    )
    model.save(os.path.join(directory, "correspondence_model.pkl"))
    return model

def load_corr_mod(directory):
    model = CorrespondenceModel.load("/media/laura/2TB/4dcbct_sim/306_R2020033/correspondence_model_c061c22.pkl")
    signal = pd.read_csv(os.path.join(directory, "rpm", "306_R2020033_4DCT_0.csv"))["ant.-pos.-ampl.[cm]"].values
    ct_amplitude = (
        RespiratorySignal(signal=signal, sampling_frequency=25)
        .resample(MCDefaults.frame_rate)
        .signal
    )
    ct_median_amplitude = calculate_median_cycle(ct_amplitude)
    ct_median_amplitude_offset = ct_median_amplitude.mean()
    ct_median_amplitude -= ct_median_amplitude_offset

    ct_median_signal = RespiratorySignal(
        signal=ct_median_amplitude,
        sampling_frequency=MCDefaults.frame_rate,
    )
    phase_timepoints = np.arange(
        0, len(ct_median_signal.time), len(ct_median_signal.time) // 10
    )[:10]

    signal = RespiratorySignal(
        signal=signal,
        sampling_frequency=25,
    )
    phase_signals = np.stack(
        (signal.signal[phase_timepoints], signal.dt_signal[phase_timepoints]), axis=-1
    )
    reference_image = sitk.GetArrayFromImage(sitk.ReadImage("/media/laura/2TB/4dcbct_sim/306_R2020033/4dct/4DCT_2.mha"))
    DEVICE = "cuda:0"
    for phase_signal in phase_signals:
        vector_field = model.predict(phase_signal)
        spatial_transformer = SpatialTransformer().to(DEVICE)
        warped_image = spatial_transformer(
            image=torch.as_tensor(
                reference_image[None, None], dtype=torch.float32, device=DEVICE
            ),
            transformation=torch.as_tensor(
                vector_field[None], dtype=torch.float32, device=DEVICE
            ),
            mode="nearest",
        )

if __name__ == '__main__':
    logging.getLogger("cbctmc").setLevel(logging.DEBUG)
    logging.getLogger("vroc").setLevel(logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    init_fancy_logging()

    directory = "/media/laura/2TB/4dcbct_sim/4_R2018045/"
    main(directory, CIRS = False, segment = True)
  #  load_corr_mod(directory)
