from __future__ import annotations

from typing import TypeVar

T = TypeVar("T", bound=type)


def convert_to_list(s: str, type_: T = str, sep: str = ",") -> list[T] | None:
    """Convert a string to a list of values of a given type."""
    return [type_(x) for x in s.split(sep)] if s else s


def convert_to_float_list(s: str, sep: str = ",") -> list[float] | None:
    """Convert a string to a list of floats."""
    return convert_to_list(s, type_=float, sep=sep)


def convert_to_int_list(s: str, sep: str = ",") -> list[int] | None:
    """Convert a string to a list of ints."""
    return convert_to_list(s, type_=int, sep=sep)
