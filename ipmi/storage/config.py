from __future__ import annotations

from typing import Any, Sequence

try:
    # Python >= 3.11
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from pathlib import Path

import tomli_w

from ipmi.fused_types import PathLike


class _NOT_SET(object):
    def __str__(self):
        return "<NOT_SET>"

    def __repr__(self):
        return str(self)


class Config:
    NOT_SET = _NOT_SET()
    _DEFAULT_FILEPATH = Path.home() / ".ipmi" / "config.toml"
    _DEFAULTS = {
        "storage": {
            "credentials": {"username": NOT_SET, "password": NOT_SET},
            "cli": {"columns": None, "output_folder": None, "filter": None},
        }
    }

    def __init__(self):
        self._config = {}

    def save(self, filepath: PathLike | None = None):
        filepath = filepath or self._DEFAULT_FILEPATH
        filepath = Path(filepath)
        filepath.parent.mkdir(exist_ok=True, parents=True)
        with open(filepath, "wb") as f:
            tomli_w.dump(self._config, f)

    @classmethod
    def load(
        cls, filepath: PathLike | None = None, raise_if_not_found: bool = True
    ) -> Config:
        filepath = filepath or cls._DEFAULT_FILEPATH
        try:
            with open(filepath, "rb") as f:
                _config = tomllib.load(f)
        except FileNotFoundError:
            if raise_if_not_found:
                raise
            else:
                _config = {}

        config = Config()
        config._config = _config

        return config

    @staticmethod
    def _get_value_by_path(path: Sequence[str], config: dict) -> Any:
        for p in path:
            try:
                config = config.get(p, Config.NOT_SET)
            except (TypeError, KeyError, AttributeError):
                return Config.NOT_SET

        return config

    @staticmethod
    def _set_value_by_path(path: Sequence[str], value: Any, config: dict) -> Any:
        for p in path[:-1]:
            config = config.setdefault(p, {})

        config[path[-1]] = value

        return config

    def __getitem__(self, item: str) -> Any:
        path = item.split(".")

        value = Config._get_value_by_path(path=path, config=self._config)
        if value is Config.NOT_SET:
            # try to load default value
            value = Config._get_value_by_path(path=path, config=self._DEFAULTS)

        return value

    def __setitem__(self, item, value):
        path = item.split(".")
        Config._set_value_by_path(path=path, value=value, config=self._config)

    def __delitem__(self, item):
        path = item.split(".")
        parent_path = ".".join(path[:-1])
        key = path[-1]
        conf = self[parent_path]
        del conf[key]
        self[parent_path] = conf

    def __str__(self):
        return str(self._config)

    def __repr__(self):
        return f"<Config({self})>"
