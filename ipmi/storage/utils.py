from pathlib import Path

from ipmi.fused_types import PathLike


def get_size(path: PathLike) -> int:
    path = Path(path)
    total = 0
    for p in path.rglob("*"):
        total += p.stat().st_size

    return total
