from __future__ import annotations

import json
import re
import uuid
from abc import abstractmethod
from datetime import datetime
from enum import Enum
from pprint import pprint
from typing import Any, List, Literal, Optional
from uuid import UUID, uuid4

from InquirerPy import inquirer
from InquirerPy.base import Choice
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.validation import ValidationError, Validator
from pydantic import BaseModel, Field
from pydantic import ValidationError as PydanticValidationError
from pydantic.functional_validators import field_validator

from ipmi.common.helper import human_readable_n_bytes
from ipmi.fused_types import PathLike
from ipmi.storage.anatomy import ANATOMIES
from ipmi.storage.errors import NamingError
from ipmi.storage.prompt import text


class Modality(str, Enum):
    # imaging
    XRAY = "XRAY"

    CT = "CT"
    CBCT = "CBCT"
    FOURD_CT = "FOURD_CT"
    FOURD_CBCT = "FOURD_CBCT"

    MRI = "MRI"
    FOURD_MRI = "FOURD_MRI"
    FOURD_FLOW_MRI = "FOURD_FLOW_MRI"
    DW_MRI = "DW_MRI"

    FUNDUSCOPY = "FUNDUSCOPY"
    ENDOSCOPY = "ENDOSCOPY"
    DSA = "DSA"
    WSI = "WSI"
    DERMATOSCOPY = "DERMATOSCOPY"
    ULTRASOUND = "ULTRASOUND"
    PET = "PET"
    SPECT = "SPECT"
    MAMMOGRAPHY = "MAMMOGRAPHY"
    OCT = "OCT"
    MPI = "MPI"
    PHOTOGRAPHY = "PHOTOGRAPHY"
    OPTICAL_MICROSCOPY = "OPTICAL_MICROSCOPY"
    FLUORESCENCE_IMAGING = "FLUORESCENCE_IMAGING"
    XRAY_FLUORESCENCE_IMAGING = "XRAY_FLUORESCENCE_IMAGING"

    # non-imaging
    RAW_DATA = "RAW_DATA"
    SPECTRUM = "SPECTRUM"
    TABLUAR_DATA = "TABLUAR_DATA"
    TEXT = "TEXT"
    RESPIRATORY_CURVE = "RESPIRATORY_CURVE"
    ECG = "ECG"
    EEG = "EEG"
    EMG = "EMG"

    OTHER = "OTHER"

    def __repr__(self):
        return self.value


class ProblemStatement(str, Enum):
    CLASSIFICATION = "CLASSIFICATION"
    SEGMENTATION = "SEGMENTATION"
    DETECTION = "DETECTION"
    REGRESSION = "REGRESSION"
    REGISTRATION = "REGISTRATION"
    RECONSTRUCTION = "RECONSTRUCTION"
    SIMULATION = "SIMULATION"
    RESTORATION = "RESTORATION"
    ENHANCEMENT = "ENHANCEMENT"
    GENERATION = "GENERATION"
    FORECASTING = "FORECASTING"
    FEATURE_EXTRACTION = "FEATURE_EXTRACTION"
    CLUSTERING = "CLUSTERING"
    DECONVOLUTION = "DECONVOLUTION"
    SUPERRESOLUTION = "SUPERRESOLUTION"
    DENOISING = "DENOISING"
    NLP = "NLP"
    CALCIUM_SIGNALING = "CALCIUM_SIGNALING"

    MULTI_LABEL = "MULTI_LABEL"
    MULTI_CLASS = "MULTI_CLASS"
    PAIRED_IMAGE_TO_IMAGE = "PAIRED_IMAGE_TO_IMAGE"
    UNPAIRED_IMAGE_TO_IMAGE = "UNPAIRED_IMAGE_TO_IMAGE"
    STYLE_TRANSFER = "STYLE_TRANSFER"
    EXPLAINABILITY = "EXPLAINABILITY"
    SUPERVISED = "SUPERVISED"
    UNSUPERVISED = "UNSUPERVISED"

    def __repr__(self):
        return self.value


class DatasetBase(BaseModel):
    @staticmethod
    def _prompt(  # noqa: C901
        field: str,
        fields: dict,
        kind: Literal["str", "int", "bool", "list", "set", "select"],
        pre_fill: dict[str] = None,
        **kwargs,
    ):
        if not pre_fill:
            pre_fill = {}

        class InputValidator(Validator):
            def validate(self, document):
                try:
                    value = document.text
                    if kind == "list":
                        value = [v.strip() for v in value.split(",")]
                    elif kind == "set":
                        value = set([v.strip() for v in value.split(",")])
                    elif kind == "int":
                        if len(str(value)) > 0:
                            int(value)
                        else:
                            value = value if len(str(value)) > 0 else None
                except (PydanticValidationError, ValueError, TypeError) as exc:
                    raise ValidationError(message=(str(exc)))

        description = fields[field].description
        required = fields[field].is_required()

        default = fields[field].default
        if required:
            message = f"{description} [required: {required}]:"
        else:
            message = f"{description} [required: {required}, default: {default}]:"
        kwargs["mandatory"] = required

        if kind == "str":
            inquirer_func = text
            kwargs = {
                "pre_fill": pre_fill.get(field),
                "validate": InputValidator(),
                "filter": lambda result: str(result) if result else None,
                "transformer": lambda result: str(result) if result else None,
                **kwargs,
            }
        elif kind == "int":
            inquirer_func = text
            kwargs = {
                "pre_fill": pre_fill.get(field),
                "validate": InputValidator(),
                "filter": lambda result: int(result) if result else None,
                "transformer": lambda result: int(result) if result else None,
                **kwargs,
            }
        elif kind == "bool":
            inquirer_func = inquirer.confirm
        elif kind == "list":
            inquirer_func = text
            kwargs = {
                "pre_fill": ",".join(pre_fill.get(field, [])),
                "validate": InputValidator(),
                "filter": lambda result: (
                    [r.strip() for r in result.split(",")] if result else []
                ),
                "transformer": lambda result: (
                    [r.strip() for r in result.split(",")] if result else []
                ),
                "long_instruction": "Enter as comma-separated list",
                **kwargs,
            }
        elif kind == "set":
            inquirer_func = text
            kwargs = {
                "pre_fill": ",".join(pre_fill.get(field, [])),
                "validate": InputValidator(),
                "filter": lambda result: (
                    set([r.strip() for r in result.split(",")]) if result else set()
                ),
                "transformer": lambda result: (
                    set([r.strip() for r in result.split(",")]) if result else set()
                ),
                "long_instruction": "Enter as comma-separated list",
                **kwargs,
            }
        elif kind == "select":
            choices = kwargs.pop("choices", [])
            choices = [
                Choice(c, enabled=(c in pre_fill.get(field, []))) for c in choices
            ]
            inquirer_func = inquirer.select
            kwargs = {"validate": InputValidator(), "choices": choices, **kwargs}

        value = inquirer_func(message=message, **kwargs).execute()

        return value

    @classmethod
    def from_file(cls, filepath: PathLike) -> Dataset:
        with open(filepath, "rt") as f:
            data = json.load(f)

        return cls(**data)

    @classmethod
    def from_string(cls, s: str) -> Dataset:
        data = json.loads(s)

        return cls(**data)

    def save(self, filepath: PathLike):
        data = self.model_dump_json(indent=4)
        with open(filepath, "wt") as f:
            f.write(data)

    def formatted_dict(self, columns: List[str] | None = None) -> dict[str, Any]:
        formatted = {}
        for key, value in self.model_dump(exclude={"sub_datasets"}).items():
            if columns is not None and key not in columns:
                continue
            if isinstance(value, uuid.UUID):
                formatted[key] = value.hex[:8]
            elif isinstance(value, datetime):
                formatted[key] = value.strftime("%Y-%m-%d")
            elif key == "n_bytes":
                formatted[key] = human_readable_n_bytes(value)
            elif isinstance(value, int):
                pass
            elif isinstance(value, float):
                pass
            # check for empty values
            elif value is None or not value:
                formatted[key] = "---"
            else:
                formatted[key] = value

        return formatted

    @classmethod
    def from_prompt(cls, **value_overwrites) -> DatasetBase | None:
        values = {}
        while True:
            values = cls._prompt_all_fields(pre_fill=values)
            if values:
                try:
                    values.update(value_overwrites)
                    dataset = cls(**values)
                    return dataset
                except PydanticValidationError as exc:
                    print(exc)
                    print("\nPlease correct those fields:")
                except KeyboardInterrupt:
                    break
            else:
                return None

    def edit_from_prompt(self) -> DatasetBase | None:
        pre_fill = self.model_dump()

        values = self._prompt_all_fields(pre_fill=pre_fill)

        if values:
            self.__dict__.update(values)
            return self
        else:
            return None

    @classmethod
    @abstractmethod
    def _prompt_all_fields(cls, pre_fill: dict[str]) -> dict | None:
        raise NotImplementedError


class Dataset(DatasetBase):
    # following fields are populated programmatically
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    sub_datasets: List[SubDataset] = Field(default=[])

    name: str = Field(description="Name (valid characters: a-z, 0-9, _)")
    description: str = Field(description="Description", min_length=0)

    modalities: set[Modality] = Field(
        description="Modalities",
    )
    anatomies_of_interest: set[str] = Field(
        description="Anatomy of interest (e.g. lung, abdomen, etc.)",
    )
    problem_statements: set[ProblemStatement] = Field(
        description="The main problem statements", default=set()
    )

    is_challenge: Optional[bool] = Field(
        default=False, description="Dataset is part of a challenge"
    )
    is_public: bool = Field(default=False, description="Dataset is publicly available")
    license: Optional[str] = Field(default=None, description="License")

    uke_origin: Optional[str] = Field(
        default=None,
        description="UKE origin of the dataset (if applicable), e.g. Department of XYZ",
    )
    uke_origin_contact: Optional[str] = Field(
        default=None,
        description=(
            "Name or email of the UKE origin contact person of the dataset "
            "(if UKE origin is given)"
        ),
    )

    inhouse_contact: str = Field(
        description=(
            "Name/initials/email of the in-house (IPMI/IAM) dataset maintainer"
        ),
        min_length=2,
    )
    related_projects: Optional[set[str]] = Field(
        default=[], description="List of related project names"
    )
    tags: Optional[set[str]] = Field(default=set(), description="List of custom tags")

    @field_validator("anatomies_of_interest")
    def to_lower_case(cls, value) -> Any | None:
        return set(v.lower() for v in value)

    @field_validator("name")
    def validate_name(cls, value) -> str:
        if not re.match(r"^[a-z0-9]+(?:_+[a-z0-9]+)*$", value):
            raise NamingError(name=value)

        return value

    @classmethod
    def _prompt_all_fields(cls, pre_fill: dict[str]) -> dict | None:
        class AnatomyCompleter(Completer):
            def get_completions(self, document, complete_event):
                input_value = document.text.lower().split(",")[-1].strip()
                words = input_value.split()
                for anatomy in ANATOMIES:
                    if all(word in anatomy for word in words):
                        yield Completion(anatomy, start_position=-len(input_value))

        fields = cls.model_fields

        try:
            while True:
                values = {
                    "name": cls._prompt(
                        "name", fields=fields, kind="str", pre_fill=pre_fill
                    ),
                    "description": cls._prompt(
                        "description", fields=fields, kind="str", pre_fill=pre_fill
                    ),
                    "modalities": cls._prompt(
                        "modalities",
                        fields=fields,
                        kind="select",
                        choices=[m.value for m in Modality],
                        pre_fill=pre_fill,
                        long_instruction=(
                            "Select at least 1. Multi-selection via [SPACE]."
                        ),
                        multiselect=True,
                        filter=lambda result: set(result),
                        transformer=lambda result: set(result),
                    ),
                    "anatomies_of_interest": cls._prompt(
                        "anatomies_of_interest",
                        fields=fields,
                        kind="set",
                        pre_fill=pre_fill,
                        completer=AnatomyCompleter(),
                        long_instruction=(
                            "Enter multiple anatomies as a comma-separated list"
                        ),
                        multicolumn_complete=False,
                    ),
                    "problem_statements": cls._prompt(
                        "problem_statements",
                        fields=fields,
                        kind="select",
                        choices=[p.value for p in ProblemStatement],
                        pre_fill=pre_fill,
                        multiselect=True,
                        filter=lambda result: set(result),
                        transformer=lambda result: set(result),
                    ),
                    "is_challenge": cls._prompt(
                        "is_challenge", fields=fields, kind="bool"
                    ),
                    "is_public": cls._prompt("is_public", fields=fields, kind="bool"),
                    "license": cls._prompt(
                        "license", fields=fields, kind="str", pre_fill=pre_fill
                    ),
                    "uke_origin": cls._prompt(
                        "uke_origin", fields=fields, kind="str", pre_fill=pre_fill
                    ),
                    "uke_origin_contact": cls._prompt(
                        "uke_origin_contact",
                        fields=fields,
                        kind="str",
                        pre_fill=pre_fill,
                    ),
                    "inhouse_contact": cls._prompt(
                        "inhouse_contact", fields=fields, kind="str", pre_fill=pre_fill
                    ),
                    "related_projects": cls._prompt(
                        "related_projects", fields=fields, kind="set", pre_fill=pre_fill
                    ),
                    "tags": cls._prompt(
                        "tags", fields=fields, kind="set", pre_fill=pre_fill
                    ),
                }
                print("You entered the following data set details:\n")
                pprint(values, sort_dicts=False)
                print("\n")
                confirmed = inquirer.confirm(
                    message="Are the above data set details correct"
                ).execute()
                if confirmed:
                    break

                # we set the entered values as pre_fill values for the fields
                pre_fill = values

        except KeyboardInterrupt:
            print("Cancelled")
            return None

        return values


class SubDataset(DatasetBase):
    # following fields are populated programmatically
    id: UUID = Field(default_factory=uuid4)
    dataset_id: UUID
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_uploaded: bool = False
    n_bytes: int = Field(description="Data set size in bytes")

    name: str = Field(description="Name (valid characters: a-z, 0-9, _)")
    description: str = Field(description="Description", min_length=0)
    n_samples: int = Field(description="Number of samples (e.g. image instances)")
    n_patients: Optional[int] = Field(description="Number of distinct patients")
    tags: Optional[set[str]] = Field(default=set(), description="List of custom tags")

    @field_validator("name")
    def validate_name(cls, value) -> str:
        if not re.match(r"^[a-z0-9]+(?:_+[a-z0-9]+)*$", value):
            raise NamingError(name=value)

        return value

    @classmethod
    def _prompt_all_fields(cls, pre_fill: dict[str]) -> dict | None:
        fields = cls.model_fields

        try:
            while True:
                values = {
                    "name": cls._prompt(
                        "name", fields=fields, kind="str", pre_fill=pre_fill
                    ),
                    "description": cls._prompt(
                        "description", fields=fields, kind="str", pre_fill=pre_fill
                    ),
                    "n_samples": cls._prompt(
                        "n_samples", fields=fields, kind="int", pre_fill=pre_fill
                    ),
                    "n_patients": cls._prompt(
                        "n_patients", fields=fields, kind="int", pre_fill=pre_fill
                    ),
                    "tags": cls._prompt(
                        "tags", fields=fields, kind="set", pre_fill=pre_fill
                    ),
                }
                print("You entered the following data set details:\n")
                pprint(values, sort_dicts=False)
                print("\n")
                confirmed = inquirer.confirm(
                    message="Are the above data set details correct"
                ).execute()
                if confirmed:
                    break

                # we set the entered values as pre_fill values for the fields
                pre_fill = values

        except KeyboardInterrupt:
            print("Cancelled")
            return None

        return values


Dataset.model_rebuild()
