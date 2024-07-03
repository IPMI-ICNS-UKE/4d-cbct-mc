from __future__ import annotations

import os
from pathlib import Path
from stat import S_ISDIR, S_ISREG
from typing import List

import paramiko
from tqdm import tqdm

from ipmi.fused_types import PathLike


class SFTPClient:
    def __init__(self, host: str, username: str, password: str, port: int = 22):
        self.host = host
        self.port = port
        self._username = username
        self._password = password

        self._client: paramiko.SFTPClient | None = None

        self._n_bytes_transferred = 0

    def connect(self):
        transport = paramiko.Transport((self.host, self.port))
        transport.connect(username=self._username, password=self._password)
        self._client = paramiko.SFTPClient.from_transport(transport)

    def close(self):
        self._client.close()

    def __enter__(self):
        self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        self._client = None

    def make_dir(self, remote_path: Path, mode: int = 777):
        self._client.mkdir(str(remote_path), mode=int(str(mode), 8))

    def make_dirs(self, remote_path: Path, mode: int = 777):
        if not self.is_dir(remote_path):
            for parent in reversed(remote_path.parents):
                # create all parents if not exist
                if not self.is_dir(parent):
                    self.make_dir(remote_path=parent, mode=mode)
            # create remote_path dir
            if not self.is_dir(remote_path):
                self.make_dir(remote_path=remote_path, mode=mode)

    def is_dir(self, remote_path: PathLike) -> bool:
        try:
            result = S_ISDIR(self._client.stat(str(remote_path)).st_mode)
        except IOError:  # no such file
            result = False
        return result

    def is_file(self, remote_path: PathLike) -> bool:
        try:
            result = S_ISREG(self._client.stat(str(remote_path)).st_mode)
        except IOError:  # no such file
            result = False
        return result

    def list_dir(self, remote_path: PathLike) -> List[Path]:
        return sorted(Path(p) for p in self._client.listdir(str(remote_path)))

    def remove_dir(self, remote_path: PathLike):
        tree = self.get_tree(remote_path, remote=True)

        files = []
        directories = []
        for path, type_ in tree.items():
            if type_ == "directory":
                directories.append(path)
            else:
                files.append(path)

        files = sorted(files, reverse=True)
        directories = sorted(directories, reverse=True)
        for file in files:
            self._client.remove(str(file))

        for directory in directories:
            self._client.rmdir(str(directory))

    def get_tree(
        self, path: PathLike, remote: bool = True, relative_to: PathLike | None = None
    ) -> dict:
        if remote:
            is_dir = self.is_dir
            list_dir = self.list_dir
        else:

            def is_dir(_path: Path):
                return _path.is_dir()

            def list_dir(_path: Path):
                return sorted(_path.iterdir())

        path = Path(path)
        if relative_to:
            relative_to = Path(relative_to)

        tree = {path: "directory" if is_dir(path) else "file"}

        def _iterate_depth_first(path: Path):
            entries = list_dir(path)
            for entry in entries:
                _path = path / entry
                relative_path = _path.relative_to(relative_to) if relative_to else _path
                if is_dir(_path):
                    tree[relative_path] = "directory"
                    # recurse
                    _iterate_depth_first(_path)
                else:
                    tree[relative_path] = "file"

        _iterate_depth_first(path)

        return tree

    def _progress_bar_callback(self, progress_bar: tqdm, path: Path):
        self._n_bytes_transferred = 0

        def _callback(n_bytes_transferred: int, n_bytes_total: int):
            delta = n_bytes_transferred - self._n_bytes_transferred
            progress_bar.update(delta)

            self._n_bytes_transferred = n_bytes_transferred

        return _callback

    def download_file(
        self, remote_path: Path, output_directory: Path, progress_bar: tqdm = None
    ):
        if progress_bar:
            callback = self._progress_bar_callback(
                progress_bar=progress_bar, path=remote_path
            )
        else:
            callback = None

        output_directory.mkdir(parents=True, exist_ok=True)
        attrs = self._client.stat(str(remote_path))

        local_path = output_directory / remote_path.name

        self._client.get(
            remotepath=str(remote_path), localpath=str(local_path), callback=callback
        )
        os.utime(local_path, (attrs.st_atime, attrs.st_mtime))

    def upload_file_from_buffer(self, buffer, remote_path: Path):
        return self._client.putfo(fl=buffer, remotepath=str(remote_path))

    def download_file_into_buffer(self, remote_path: Path, buffer):
        return self._client.getfo(remotepath=str(remote_path), fl=buffer)

    def download_directory(
        self, remote_path: Path, output_directory: Path, progress_bar: tqdm = None
    ):
        tree = self.get_tree(remote_path, remote=True)
        parent = remote_path.parent
        for _remote_path, type_ in tree.items():
            local_path = output_directory / _remote_path.relative_to(parent)

            if type_ == "directory":
                local_path.mkdir(exist_ok=True)
            else:
                self.download_file(
                    remote_path=_remote_path,
                    output_directory=local_path.parent,
                    progress_bar=progress_bar,
                )

    def upload_file(
        self, local_path: Path, remote_folder: Path, progress_bar: tqdm = None
    ):
        if progress_bar:
            callback = self._progress_bar_callback(
                progress_bar=progress_bar, path=local_path
            )
        else:
            callback = None

        local_stat = os.stat(local_path)
        times = (local_stat.st_atime, local_stat.st_mtime)

        remote_path = remote_folder / local_path.name
        self._client.put(
            localpath=str(local_path),
            remotepath=str(remote_path),
            callback=callback,
            confirm=True,
        )

        self._client.utime(str(remote_path), times)

    def upload_directory(
        self,
        local_path: Path,
        remote_folder: Path,
        progress_bar: tqdm = None,
        create_root_folder: bool = False,
    ):
        if create_root_folder:
            self.make_dir(remote_folder / local_path.name)
            relative_to = local_path.parent
        else:
            relative_to = local_path

        tree = self.get_tree(local_path, remote=False)
        for _local_path, type_ in tree.items():
            remote_path = remote_folder / _local_path.relative_to(relative_to)
            if type_ == "directory":
                self.make_dir(remote_path)
            else:
                self.upload_file(
                    local_path=_local_path,
                    remote_folder=remote_path.parent,
                    progress_bar=progress_bar,
                )
