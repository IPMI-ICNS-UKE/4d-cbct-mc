import hashlib
from pathlib import Path


def hash_path(path: Path) -> str:
    return hashlib.sha1(str(path).encode()).hexdigest()
