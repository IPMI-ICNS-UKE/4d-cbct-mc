from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ipmi.storage.dataset import Modality


def _value_in_set(value: str, set_: str):
    return f"{value!r} in {set_}"


def modality(value: Modality | str) -> str:
    return _value_in_set(value, "dataset.modalities")


def problem_statement(problem_statement: str) -> str:
    return _value_in_set(problem_statement, "dataset.problem_statements")


def anatomy(anatomy: str) -> str:
    return _value_in_set(anatomy, "dataset.anatomies_of_interest")


def minimum_patients(n_patients: int) -> str:
    return (
        f"any(sub_dataset.n_patients >= {n_patients} "
        f"for sub_dataset in dataset.sub_datasets)"
    )


def minimum_sum_patients(n_patients: int) -> str:
    return (
        f"sum(sub_dataset.n_patients "
        f"for sub_dataset in dataset.sub_datasets) >= {n_patients}"
    )


def maximum_sum_patients(n_patients: int) -> str:
    return (
        f"sum(sub_dataset.n_patients "
        f"for sub_dataset in dataset.sub_datasets) <= {n_patients}"
    )


__all__ = [
    "modality",
    "problem_statement",
    "anatomy",
    "minimum_patients",
    "minimum_sum_patients",
    "maximum_sum_patients",
]
