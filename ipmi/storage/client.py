from __future__ import annotations

# imports that can be used in filter function
import datetime  # noqa: F401
import json
import re  # noqa: F401
import sys
import uuid
from abc import ABC
from io import BytesIO, StringIO
from pathlib import Path
from typing import Callable, List, Literal, Sequence

from paramiko.ssh_exception import AuthenticationException
from tqdm import tqdm

from ipmi.fused_types import PathLike
from ipmi.storage.config import Config
from ipmi.storage.dataset import (  # noqa: F401
    Dataset,
    DatasetBase,
    Modality,
    ProblemStatement,
    SubDataset,
)
from ipmi.storage.errors import (
    AuthError,
    DatasetNotFoundError,
    MultipleDatasetsFoundError,
    NotLoggedInError,
    SortError,
)
from ipmi.storage.filters import *  # noqa: F401,F403
from ipmi.storage.sftp import SFTPClient
from ipmi.storage.utils import get_size


class BaseStorageClient(ABC):
    def upload_sub_dataset(self, dataset_folder: PathLike, dataset: SubDataset):
        raise NotImplementedError


class StorageClient(BaseStorageClient):
    _HOST = "10.0.1.5"
    _BASE_PATH = Path("/mnt/ICNS")
    _DATA_PATH = _BASE_PATH / "data"
    _PUBLICATIONS_PATH = _BASE_PATH / "publications"
    _ANARCHY_PATH = _BASE_PATH / "anarchy"

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
    ):
        self._config = Config.load(raise_if_not_found=False)

        self._username = username or self._config["storage.credentials.username"]
        self._password = password or self._config["storage.credentials.password"]
        self._sftp: SFTPClient = SFTPClient(
            host=self._HOST, username=self._username, password=self._password
        )
        self._sftp_references = 0

        if self._has_credentials:
            self.login(username=self._username, password=self._password)

        self._current_transfer = None
        self._n_bytes_transferred = 0

    def __enter__(self):
        if not self._has_credentials:
            raise NotLoggedInError

        self._sftp.connect()

        self._sftp_references += 1

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._sftp_references -= 1

        if not self._sftp_references:
            self._sftp.close()

    @property
    def _has_credentials(self) -> bool:
        return isinstance(self._username, str) and isinstance(self._password, str)

    def login(self, username: str, password: str):
        try:
            with (
                sftp := SFTPClient(
                    host=self._HOST, username=username, password=password
                )
            ):
                config = Config.load(raise_if_not_found=False)
                config["storage.credentials.username"] = username
                config["storage.credentials.password"] = password
                config.save()
                self._sftp = sftp
        except AuthenticationException:
            raise AuthError(username)

    @staticmethod
    def logout():
        config = Config.load(raise_if_not_found=False)
        try:
            del config["storage.credentials.username"]
            del config["storage.credentials.password"]
        except KeyError:
            pass
        config.save()

    @property
    def _is_running_in_terminal(self) -> bool:
        return sys.stdout.isatty()

    def _read_metadata(self, filepath: PathLike) -> Dataset | SubDataset:
        filepath = Path(filepath)
        with open(filepath, "rt") as f:
            klass = Dataset if filepath.name == "dataset.json" else SubDataset
            data = json.load(f)
            return klass(**data)

    def _upload_meta(self, dataset: DatasetBase, folder: Path, filename: str):
        with self, StringIO(dataset.model_dump_json(indent=4)) as dataset_meta:
            self._sftp.upload_file_from_buffer(
                buffer=dataset_meta, remote_path=folder / filename
            )

    def _download_meta(self, folder: Path, filename: str) -> Dataset | SubDataset:
        with self, BytesIO() as meta_stream:
            self._sftp.download_file_into_buffer(
                remote_path=folder / filename, buffer=meta_stream
            )

            klass = Dataset if filename == "dataset.json" else SubDataset

            return klass.from_string(meta_stream.getvalue().decode())

    def _create_progress_bar(
        self, kind: Literal["download", "upload"], dataset: Dataset, n_bytes: int
    ) -> tqdm:
        description = f"{kind.capitalize()} {dataset.name} ({dataset.id.hex[:8]})"

        return tqdm(
            desc=description,
            total=n_bytes,
            unit_scale=True,
            unit="B",
        )

    def create_or_edit_dataset(
        self,
        dataset: Dataset | None = None,
    ) -> Dataset | None:
        with self:
            if not dataset:
                dataset_id = uuid.uuid4()
                if not self._is_running_in_terminal:
                    raise RuntimeError("Please run in a real terminal")
                dataset = Dataset.from_prompt(id=dataset_id)
                # dataset may be None if canceled by Ctrl + C
                if not dataset:
                    return

            remote_dataset_folder = self._DATA_PATH / str(dataset.id)

            # create folders if not exist
            self._sftp.make_dirs(remote_dataset_folder)

            self._upload_meta(
                dataset=dataset,
                folder=remote_dataset_folder,
                filename="dataset.json",
            )

        return dataset

    def upload_sub_dataset(
        self,
        dataset_folder: PathLike,
        dataset_id: uuid.UUID | str,
        sub_dataset: SubDataset | None = None,
    ):
        with self:
            # check if data set exists
            dataset = self.get_dataset(dataset_id)
            dataset_id = dataset.id
            if not dataset:
                raise DatasetNotFoundError(dataset_id)

            dataset_folder = Path(dataset_folder)

            sub_dataset_id = uuid.uuid4()

            total_bytes = get_size(dataset_folder)
            remote_dataset_folder = self._DATA_PATH / str(dataset_id)

            remote_sub_dataset_folder = remote_dataset_folder / str(sub_dataset_id)

            if not sub_dataset:
                if not self._is_running_in_terminal:
                    raise RuntimeError("Please run in a real terminal")
                sub_dataset = SubDataset.from_prompt(
                    id=sub_dataset_id, dataset_id=dataset_id, n_bytes=total_bytes
                )
                # sub_dataset may be None if canceled by Ctrl + C
                if not sub_dataset:
                    return

            self._n_bytes_transferred = 0

            with tqdm(total=total_bytes, unit_scale=True, unit="B") as progress_bar:
                self._sftp.make_dirs(remote_sub_dataset_folder)

                self._upload_meta(
                    dataset=sub_dataset,
                    folder=remote_sub_dataset_folder,
                    filename="sub_dataset.json",
                )

                self._sftp.upload_directory(
                    local_path=dataset_folder,
                    remote_folder=remote_sub_dataset_folder / "data",
                    create_root_folder=False,
                    progress_bar=progress_bar,
                )

                # data set is fully uploaded, update remote meta data
                sub_dataset.is_uploaded = True
                self._upload_meta(
                    dataset=sub_dataset,
                    folder=remote_sub_dataset_folder,
                    filename="sub_dataset.json",
                )

    def edit_sub_dataset(
        self,
        sub_dataset: SubDataset,
    ) -> SubDataset | None:
        with self:
            remote_sub_dataset_folder = (
                self._DATA_PATH / str(sub_dataset.dataset.id) / str(sub_dataset.id)
            )

            self._upload_meta(
                dataset=sub_dataset,
                folder=remote_sub_dataset_folder,
                filename="sub_dataset.json",
            )

        return sub_dataset

    @staticmethod
    def _filter_and_sort(
        entries: Sequence[Dataset | SubDataset],
        filter: str | None = None,
        sort: str | None = None,
    ) -> List[Dataset | SubDataset]:
        result = []
        for entry in entries:
            # assign, so we can use these names in filter function
            dataset = entry
            sub_dataset = entry  # noqa: F841
            if filter:
                try:
                    matches_filter = eval(filter)
                    if isinstance(matches_filter, str):
                        # matches_filter may be filter shortcut output, eval a 2nd time
                        try:
                            matches_filter = eval(matches_filter)
                        except NameError:
                            pass
                except AttributeError:
                    matches_filter = False
            else:
                matches_filter = True

            if matches_filter:
                result.append(dataset)

        if sort:
            none_defaults = ("", 0)
            for none_default in none_defaults:
                # try defautlts for None (string or number)
                try:
                    result = sorted(
                        result, key=lambda r: getattr(r, sort) or none_default
                    )
                except TypeError:
                    continue
                except AttributeError:
                    raise SortError(
                        column_name=sort,
                        valid_column_names=list(result[0].model_fields.keys()),
                    )

        return result

    def list_datasets(
        self,
        filter: str | None = None,
        sort: str | None = None,
        progress_callback: Callable | None = None,
    ) -> List[Dataset]:
        with self:
            datasets = []
            dirs = self._sftp.list_dir(self._DATA_PATH)
            for i, dataset_folder in enumerate(dirs, start=1):
                if progress_callback:
                    progress_callback(current=i, total=len(dirs))
                dataset = self._download_meta(
                    folder=self._DATA_PATH / dataset_folder,
                    filename="dataset.json",
                )
                if filter and "sub_datasets" in filter:
                    # attach sub_datasets
                    dataset.sub_datasets = self.list_sub_datasets(dataset.id)

                datasets.append(dataset)

        return StorageClient._filter_and_sort(datasets, filter=filter, sort=sort)

    def _get_dataset(self, id: uuid.UUID) -> Dataset | None:
        with self:
            try:
                with BytesIO() as meta_stream:
                    meta_filepath = self._DATA_PATH / str(id) / "dataset.json"
                    self._sftp.download_file_into_buffer(meta_filepath, meta_stream)

                    dataset = Dataset.from_string(meta_stream.getvalue().decode())
            except FileNotFoundError:
                raise DatasetNotFoundError(id)

        return dataset

    def _get_sub_dataset(
        self, id: uuid.UUID, dataset_id: uuid.UUID
    ) -> SubDataset | None:
        with self:
            try:
                with BytesIO() as meta_stream:
                    meta_filepath = (
                        self._DATA_PATH / str(dataset_id) / str(id) / "sub_dataset.json"
                    )
                    self._sftp.download_file_into_buffer(meta_filepath, meta_stream)

                    sub_dataset = SubDataset.from_string(
                        meta_stream.getvalue().decode()
                    )
            except FileNotFoundError:
                raise DatasetNotFoundError(id)

        return sub_dataset

    def get_dataset(self, id: uuid.UUID | str) -> Dataset:
        if isinstance(id, str):
            # id is partial UUID
            datasets = self.list_datasets(filter=f"str(dataset.id).startswith('{id}')")
            if len(datasets) > 1:
                raise MultipleDatasetsFoundError(id)
            elif len(datasets) == 0:
                raise DatasetNotFoundError(id)

            dataset = datasets[0]
        else:
            dataset = self._get_dataset(id)

        return dataset

    def get_sub_dataset(
        self, id: uuid.UUID | str, dataset_id: uuid.UUID | str
    ) -> SubDataset:
        dataset = self.get_dataset(dataset_id)

        if isinstance(id, str):
            # id is partial UUID
            sub_datasets = self.list_sub_datasets(
                dataset_id=dataset.id, filter=f"str(sub_dataset.id).startswith('{id}')"
            )
            if len(sub_datasets) > 1:
                raise MultipleDatasetsFoundError(id)
            elif len(sub_datasets) == 0:
                raise DatasetNotFoundError(id)

            sub_dataset = sub_datasets[0]
        else:
            sub_dataset = self._get_sub_dataset(id, dataset_id=dataset_id)

        return sub_dataset

    def list_sub_datasets(
        self,
        dataset_id: uuid.UUID | str,
        filter: str | None = None,
        sort: str | None = None,
    ) -> List[SubDataset]:
        dataset = self.get_dataset(dataset_id)

        sub_datasets = []
        with self:
            dataset_path = self._DATA_PATH / str(dataset.id)

            for folder_or_file in self._sftp.list_dir(dataset_path):
                path = dataset_path / folder_or_file
                try:
                    sub_dataset = self._download_meta(
                        folder=path, filename="sub_dataset.json"
                    )
                    sub_datasets.append(sub_dataset)
                except FileNotFoundError:
                    continue

        return StorageClient._filter_and_sort(sub_datasets, filter=filter, sort=sort)

    def download_dataset(
        self,
        dataset_id: uuid.UUID | str,
        output_folder: PathLike,
        use_names: bool = False,
    ):
        output_folder = Path(output_folder)
        with self:
            dataset = self.get_dataset(id=dataset_id)

            sub_datasets = self.list_sub_datasets(dataset_id)
            n_bytes = sum(sd.n_bytes for sd in sub_datasets)
            with self._create_progress_bar(
                kind="download", dataset=dataset, n_bytes=n_bytes
            ) as progress_bar:
                self._sftp.download_directory(
                    remote_path=self._DATA_PATH / str(dataset_id),
                    output_directory=output_folder,
                    progress_bar=progress_bar,
                )
        self._postprocess_download(
            dataset_folder=output_folder / str(dataset_id), use_names=use_names
        )

    def _postprocess_download(self, dataset_folder: PathLike, use_names: bool = False):
        dataset_folder = Path(dataset_folder)
        dataset_folder = Path(dataset_folder)
        dataset = self._read_metadata(dataset_folder / "dataset.json")

        for sub_dataset_folder in [p for p in dataset_folder.iterdir() if p.is_dir()]:
            sub_dataset = self._read_metadata(sub_dataset_folder / "sub_dataset.json")

            if use_names:
                # rename sub dataset folder
                sub_dataset_folder.replace(
                    sub_dataset_folder.with_name(sub_dataset.name)
                )

        if use_names:
            # rename dataset folder
            dataset_folder.replace(dataset_folder.with_name(dataset.name))

    def delete_dataset(self, dataset_id: uuid.UUID | str):
        with self:
            dataset = self.get_dataset(id=dataset_id)

            self._sftp.remove_dir(remote_path=self._DATA_PATH / str(dataset.id))


if __name__ == "__main__":
    storage = StorageClient()

    p = Path("/mnt/ICNS/data/dsfsdfsfd")

    datasets = storage.list_datasets()
