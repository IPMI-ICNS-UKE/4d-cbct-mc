import gzip
import logging
from pathlib import Path

from cbctmc.common_types import PathLike

logger = logging.getLogger(__name__)


def save_text_file(
    content: str,
    output_filepath: PathLike,
    compress: bool = True,
    content_type: str = None,
) -> Path:
    output_filepath = Path(output_filepath)
    if compress and (suffix := output_filepath.suffix) != ".gz":
        output_filepath = output_filepath.with_suffix(f"{suffix}.gz")
    if compress:
        handle = gzip.open(output_filepath, mode="wt", compresslevel=6)
    else:
        handle = open(output_filepath, mode="wt")

    with handle:
        handle.write(content)

    message = f"Saved file to {output_filepath!s}"
    if content_type:
        message = f" {message} ({content_type})"
    logger.info(message)

    return output_filepath
