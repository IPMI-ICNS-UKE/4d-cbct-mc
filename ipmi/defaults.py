from pathlib import Path

DOCKER_IMAGE = "imaging"
DOCKER_IMAGE_REPO = "https://github.com/IPMI-ICNS-UKE/imaging-docker"
DOCKER_PATH_PREFIX = Path("/host")
DOCKER_MOUNTS = {"/": {"bind": str(DOCKER_PATH_PREFIX), "mode": "rw"}}
