# 4D CBCT Monte Carlo Simulation
This repository contains the official code for the following [paper](https://rdcu.be/baQa3):

```
@article{madesta:2024,
    doi = {XXX},
    year = {2024},
    month = {XXX},
    publisher = {XXX},
    volume = {XXX},
    number = {XXX},
    pages = {XXX},
    author = {Frederic Madesta, Thilo Sentker, Clemens Rohling, Tobias Gauer, and Ren\'{e} Werner},
    title = {Monte Carlo-based simulation of virtual 3D and 4D Cone-Beam CT from CT images: An end-to-end framework and a novel deep learning-based speedup strategy},
    journal = {XXX}
}

```
## Overview
This package contains all the required code to perform the following tasks in a fully automated end-to-end fashion:

- Deep learning-based segmentation of CT images into various organ and tissue classes
- Monte Carlo simulation of 3D CBCT images from CT images
- Monte Carlo simulation of 4D CBCT images from CT images with respective [correspondence models](https://doi.org/10.1088/0031-9155/59/5/1147) and respiratory signals
- Reconstruction of 3D and 4D CBCT images from simulated projections using the [Reconstruction Toolkit (RTK)](https://www.openrtk.org/)

Both the MC as well as the reconstruction code are shipped as pre-compiled binaries in a Docker image for user experience reasons.


## Installation
### Prerequisites
- [Docker Engine](https://docs.docker.com/engine/install/)
- [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [Python 3.11 or later](https://docs.conda.io/projects/miniconda/en/latest/)



## Usage
### Data preparation
In general, the framework requires very litte data preparation:
- The CT images should be stored in a single-file format (e.g. ```*.nii```, ```*.mha``` or any other format that can be read by ITK)
- The CT images should have non-preporcessed Hounsfield units (HU), i.e. the HU values should be in the range [-1024, 3071]


### 3D CBCT simulation
A 3D CBCT simulation is defined by the static (patient) geometry and the moving CBCT scan geometry (i.e. X-ray source and detector).



### 4D CBCT simulation
Analog to the 3D CBCT simulation, a 4D CBCT simulation is defined by the time-resolved/dynamic (patient) geometry and the moving CBCT scan geometry (i.e. X-ray source and detector). In addition, a 4D CBCT simulation requires a correspondence model and a respiratory signal.

#### Correspondence model
The correspondence model can be fitted using a 4D CT and the corresponding respiratory signal.
If no respiratory signal is available, the lung volume can be used as a surrogate signal.
The correspondence model is readily fitted by the following code snippet:

```python
import numpy as np
from cbctmc.registration.correspondence import CorrespondenceModel

images: np.ndarray
masks: np.ndarray
timepoints: np.ndarray


model = CorrespondenceModel.build_default(
    images=images,
    masks=masks,
    timepoints=timepoints,
    masked_registration=False,
    device="cuda:0",
)
model.save("/some/folder/correspondence_model.pkl")
```
