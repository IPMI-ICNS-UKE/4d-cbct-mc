import functools
from typing import Optional

import docker
from docker.errors import ImageNotFound

from ipmi.common.decorators import _as_parameterizable_decorator
from ipmi.fused_types import Function


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


@_as_parameterizable_decorator
def requires_docker_image(func: Function, image_name: str):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        check_image_exists(image_name, raise_error=True)
        return func(*args, **kwargs)

    return wrapper
