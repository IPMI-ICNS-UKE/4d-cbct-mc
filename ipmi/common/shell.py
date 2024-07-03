from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Sequence

import docker

import ipmi.defaults as defaults
from ipmi.common.helper import replace_root
from ipmi.fused_types import PathLike

logger = logging.getLogger(__name__)


def _convert_key(key: str, prefix: str = "--", convert_underscore: str = "-") -> str:
    # strip trailing "_" that can be used if key is a reserved keyword (e.g. lambda)
    key = key.rstrip("_")
    if convert_underscore:
        key = key.replace("_", convert_underscore)

    return prefix + key


def _convert_value(
    value: Any, bool_as_flag: bool = True, path_prefix: Path | None = None
) -> str:
    # convert python values to CLI values
    if isinstance(value, (list, tuple)):
        value = ",".join([str(i) for i in value])
    elif isinstance(value, bool):
        if bool_as_flag:
            value = None
        else:
            value = str(value)
    elif isinstance(value, Path):
        if path_prefix:
            value = replace_root(value, path_prefix)
        value = str(value)
    else:
        value = str(value)

    return value


def create_cli_command(
    executable: PathLike,
    *args,
    prefix="--",
    convert_underscore: str | None = "-",
    bool_as_flag: bool = True,
    skip_none: bool = True,
    path_prefix: PathLike | None = None,
    **kwargs,
):
    command = [str(executable)]
    # add positional arguments to command
    for arg in args:
        command.extend([_convert_value(arg)])

    # add optional arguments to command
    for key, val in kwargs.items():
        if skip_none and val is None:
            continue
        key = _convert_key(key, prefix=prefix, convert_underscore=convert_underscore)
        val = _convert_value(val, bool_as_flag=bool_as_flag, path_prefix=path_prefix)
        command.extend([key, val] if val else [key])

    logger.debug(f"Compiled the following CLI command: {command}")
    return command


def execute(cli_command: Sequence[str], gpus: Sequence[int] | None = None):
    if gpus:
        cli_command = [
            f"CUDA_VISIBLE_DEVICES={','.join(str(gpu_id) for gpu_id in gpus)}"
        ] + cli_command

    try:
        logs = subprocess.check_output(cli_command, stderr=subprocess.STDOUT)
        logger.info(logs.decode())
    except subprocess.CalledProcessError as e:
        logger.error(e.output.decode())


def execute_in_docker(
    cli_command: Sequence[str],
    docker_image: str = defaults.DOCKER_IMAGE,
    mounts: dict = defaults.DOCKER_MOUNTS,
    gpus: Sequence[int] | None = None,
    **kwargs,
) -> str:
    device_requests = []
    if gpus is not None:
        device_ids = [",".join(str(gpu) for gpu in gpus)]
        device_requests += [
            docker.types.DeviceRequest(device_ids=device_ids, capabilities=[["gpu"]])
        ]

    uid = os.getuid()
    gid = os.getgid()
    client = docker.from_env()
    logs = client.containers.run(
        image=docker_image,
        command=cli_command,
        remove=True,
        volumes=mounts,
        device_requests=device_requests,
        user=f"{uid}:{gid}",
        stdout=True,
        stderr=True,
        **kwargs,
    )
    logs = logs.decode()
    logger.info(logs)

    return logs
