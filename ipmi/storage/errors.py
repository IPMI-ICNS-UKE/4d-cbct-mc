from __future__ import annotations

import functools
import uuid
from typing import List

from pydantic import ValidationError


class BaseStorageError(Exception):
    message: str


def print_exceptions(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except BaseStorageError as exc:
            print(exc.message)
        except ValidationError as exc:
            print(str(exc))
        except Exception as exc:
            print(str(exc))

    return wrapper


class NamingError(ValueError):
    msg_template = (
        "{name!r} does not match the naming conventions "
        "(valid characters: a-z, 0-9, _)"
    )

    def __init__(self, *, name: str) -> None:
        super().__init__(name=name)


class AuthError(BaseStorageError):
    def __init__(self, username: str) -> None:
        self.message = f"Authentication failed for user with username {username}"
        super().__init__(self.message)


class NotLoggedInError(BaseStorageError):
    def __init__(self) -> None:
        self.message = "Please login first (execute 'ipmi-storage login' in terminal)"
        super().__init__(self.message)


class DatasetNotFoundError(BaseStorageError):
    def __init__(self, id: uuid.UUID | str) -> None:
        self.message = f"Dataset {str(id)} not found"
        super().__init__(self.message)


class MultipleDatasetsFoundError(BaseStorageError):
    def __init__(self, partial_id: str) -> None:
        self.message = (
            f"Multiple data sets were found having UUID starting with {partial_id}"
        )
        super().__init__(self.message)


class SortError(BaseStorageError):
    def __init__(self, column_name: str, valid_column_names: List[str]) -> None:
        self.message = (
            f"Invalid column name {column_name}. "
            f"Valid column names are: {', '.join(valid_column_names)}"
        )
        super().__init__(self.message)
