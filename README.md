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

### Building the Docker image
The Docker image can be built using the following command:

```bash
sh ./build-docker.sh
```
This will build the Docker image with the name `cbct-mc` and the tag `latest`.

### Installing the Python package
The Python package can be installed using the following command:

```bash
pip install -e .
```
Make sure to execute this command in the root directory of the repository (where the `setup.py` file is located).

## Usage
### Data preparation
In general, the framework requires very litte data preparation:
- The CT images should be stored in a single-file format (e.g. ```*.nii```, ```*.mha``` or any other format that can be read by ITK)
- The CT images should have non-preporcessed Hounsfield units (HU), i.e. the HU values should be in the range [-1024, 3071]


### General usage
The MC simulations are performed using the following command (both 3D and 4D simulations):

```bash
(4d-cbct-mc) fmadesta@hydrogen:~/research/4d-cbct-mc$ run-mc --help
Usage: run-mc [OPTIONS]

Options:
  --image-filepath FILE           CT image to use for simulation
  --geometry-filepath FILE        Geometry to use for simulation. Can be
                                  provided instead of CT image.
  --output-folder DIRECTORY       Output folder for simulation results
  --simulation-name TEXT          Name of the simulation. If not provided, the
                                  name is derived from the image filepath.
  --gpu INTEGER                   GPU PCI bus ID to use for simulation (can be
                                  checked with nvidia-smi)  [default: 0]
  --reference                     Enable reference simulation
  --reference-n-histories INTEGER
                                  Number of histories for reference simulation
                                  [default: 11903320312]
  --speedups FLOAT                Speedup factors for simulation
  --speedup-weights FILE          Weights file for speedup model  [default:
                                  /home/fmadesta/research/4d-cbct-
                                  mc/cbctmc/assets/models/speedup/default.pth]
  --segmenter-weights FILE        Weights file for the segmenter model
  --segmenter-patch-shape <INTEGER INTEGER INTEGER>...
                                  Patch shape for the segmenter model
                                  [default: 256, 256, 128]
  --segmenter-patch-overlap FLOAT RANGE
                                  Overlap ratio for patch-based segmentation
                                  [default: 0.5; 0.0<x<=1.0]
  --n-projections INTEGER         Number of projections for simulation
                                  [default: 894]
  --reconstruct-3d                Enable 3D reconstruction
  --reconstruct-4d                Enable 4D reconstruction
  --forward-projection            Enable forward projection
  --no-clean                      Disable cleaning of intermediate files
  --correspondence-model FILE     Correspondence model file. Must be provided
                                  for 4D simulation.
  --respiratory-signal FILE       Respiratory signal file. Must be provided
                                  for 4D simulation.
  --respiratory-signal-quantization INTEGER
                                  Quantization level for respiratory signal. A
                                  lower value means that the respiratory
                                  signal is more coarse.
  --respiratory-signal-scaling FLOAT
                                  Scaling factor for respiratory signal
                                  [default: 1.0]
  --precompile-geometries         Precompile geometries for 4D simulation
  --cirs-phantom                  Use CIRS phantom for simulation
  --catphan-phantom               Use Catphan604 phantom for simulation
  --dry-run                       Perform a dry run without executing the
                                  simulation
  --random-seed INTEGER           Random seed for simulation  [default: 42]
  --loglevel [debug|info|warning|error|critical]
                                  Logging level  [default: info]
  --help                          Show this message and exit.

```


### 3D CBCT simulation
A 3D CBCT simulation is defined by the static (patient) geometry and the moving CBCT scan geometry (i.e. X-ray source and detector).

### 4D CBCT simulation
Analog to the 3D CBCT simulation, a 4D CBCT simulation is defined by the time-resolved/dynamic (patient) geometry and the moving CBCT scan geometry (i.e. X-ray source and detector). In addition, a 4D CBCT simulation requires a correspondence model and a respiratory signal. Thus, the `run-mc` command has to be called with the `--correspondence-model` and `--respiratory-signal` arguments. The correspondence model can be fitted using the `fit-correspondence-model` command (see below). The respiratory signal can be obtained from a 4D CT scan.

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
