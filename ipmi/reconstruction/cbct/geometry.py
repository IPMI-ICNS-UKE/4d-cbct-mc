import os
from pathlib import Path
from typing import Optional

import ipmi.defaults as defaults
from ipmi.common.decorators import convert
from ipmi.common.shell import create_cli_command, execute, execute_in_docker
from ipmi.fused_types import PathLike


@convert("scan_xml_filepath", converter=Path)
@convert("projection_folder", converter=Path)
@convert("output_filepath", converter=Path)
def generate_geometry(
    scan_xml_filepath: PathLike,
    projection_folder: PathLike,
    projection_regexp: str = ".*xim",
    executable: PathLike = "rtkvarianprobeamgeometry",
    output_filepath: Optional[PathLike] = None,
    use_docker: bool = True,
):
    folder = os.path.dirname(scan_xml_filepath)
    if not output_filepath:
        output_filepath = os.path.join(folder, "geometry.xml")
    bin_call = create_cli_command(
        executable,
        xml_file=scan_xml_filepath,
        path=projection_folder,
        regexp=projection_regexp,
        output=output_filepath,
        path_prefix=defaults.DOCKER_PATH_PREFIX if use_docker else None,
        convert_underscore=None,
    )

    execute_func = execute_in_docker if use_docker else execute
    execute_func(bin_call)
