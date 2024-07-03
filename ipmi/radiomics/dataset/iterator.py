from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union


class BaseDatasetIterator(ABC):
    def __init__(self, root_directory: Union[str, Path]):
        self.root_directory = root_directory

    @property
    def root_directory(self) -> Path:
        return self.__root_directory

    @root_directory.setter
    def root_directory(self, value):
        if not isinstance(value, Path):
            value = Path(value)
        if not value.exists() and value.is_dir():
            raise ValueError(f"{value} is not a valid directory")
        self.__root_directory = value

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError
