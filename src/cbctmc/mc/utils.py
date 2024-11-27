from pathlib import Path

from cbctmc.common_types import PathLike


def replace_root(path: PathLike, new_root: PathLike, old_root: PathLike = "/") -> Path:
    path = Path(path)
    old_root = Path(old_root)
    new_root = Path(new_root)

    if not new_root.is_absolute() or not old_root.is_absolute():
        raise ValueError("new_root and old_root has to be an absolute path")

    return new_root / path.relative_to(old_root)
