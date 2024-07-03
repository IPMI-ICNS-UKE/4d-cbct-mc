from collections import defaultdict
from pathlib import Path
from typing import Sequence

from ipmi.common.decorators import convert
from ipmi.fused_types import PathLike


def recursive_dict():
    return defaultdict(recursive_dict)


@convert("path", converter=Path)
@convert("new_root", converter=Path)
def replace_root(path: PathLike, new_root: PathLike) -> Path:
    path: Path
    new_root: Path

    if not new_root.is_absolute():
        raise ValueError("new_root has to be an absolute path")

    return new_root / path.relative_to("/")


def concat_dicts(dicts: Sequence[dict], extend_lists: bool = False):
    concat = {}
    for d in dicts:
        for key, value in d.items():
            try:
                if extend_lists and isinstance(value, list):
                    concat[key].extend(value)
                else:
                    concat[key].append(value)
            except KeyError:
                if extend_lists and isinstance(value, list):
                    concat[key] = value
                else:
                    concat[key] = [value]

    return concat


def human_readable_n_bytes(n_bytes: int, suffix: int = "B"):
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(n_bytes) < 1000.0:
            return f"{n_bytes:3.1f}{unit}{suffix}"
        n_bytes /= 1000.0
    return f"{n_bytes:.1f}Yi{suffix}"
