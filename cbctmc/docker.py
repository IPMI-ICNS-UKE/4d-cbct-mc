from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import docker
from docker.errors import ImageNotFound

DOCKER_IMAGE = "imaging:latest"
DOCKER_HOST_PATH_PREFIX = Path("/host")
DOCKER_MOUNTS = {"/": {"bind": str(DOCKER_HOST_PATH_PREFIX), "mode": "rw"}}


def check_image_exists(image_name: str, raise_error: bool = False) -> Optional[bool]:
    docker_client = docker.from_env()
    try:
        docker_client.images.get(image_name)
        return True
    except ImageNotFound:
        if raise_error:
            raise
        else:
            return False


def execute_in_docker(
    cli_command: Sequence[str],
    docker_image: str = DOCKER_IMAGE,
    mounts: dict = DOCKER_MOUNTS,
    gpus: Sequence[int] | None = None,
    **kwargs,
) -> docker.models.containers.Container:
    device_requests = []
    if gpus is not None:
        device_ids = [",".join(str(gpu) for gpu in gpus)]
        device_requests += [
            docker.types.DeviceRequest(device_ids=device_ids, capabilities=[["gpu"]])
        ]

    uid = os.getuid()
    gid = os.getgid()
    client = docker.from_env()
    conrainer = client.containers.run(
        image=docker_image,
        command=cli_command,
        remove=True,
        volumes=mounts,
        device_requests=device_requests,
        user=f"{uid}:{gid}",
        stdout=True,
        stderr=True,
        detach=True,
        **kwargs,
    )
    return conrainer
