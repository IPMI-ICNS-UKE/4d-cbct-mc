from __future__ import annotations

import logging
import sys
from typing import Dict, Sequence, Tuple

import torch

from vroc.helper import convert_dict_values


class LoggerMixin:
    @property
    def logger(self):
        try:
            self.__logger
        except AttributeError:
            logger_name = f"{self.__module__}.{self.__class__.__name__}"
            self.__logger = logging.getLogger(logger_name)
        return self.__logger


class FancyFormatter(logging.Formatter):

    bold = "\033[1m"
    reset = "\033[0m"

    # RGB values in range [0, 255]
    cyan = (0, 168, 168)
    grey = (188, 188, 188)
    yellow = (170, 140, 0)
    red = (253, 62, 72)

    DEFAULT_STYLES = {
        logging.DEBUG: {"color": cyan},
        logging.INFO: {"color": grey},
        logging.WARNING: {"color": yellow, "bold": True},
        logging.ERROR: {"color": red, "bold": True},
        logging.CRITICAL: {"color": red, "bold": True},
    }

    @staticmethod
    def _get_color(red: int, green: int, blue: int, background: bool = False) -> str:
        background = 48 if background else 38
        return f"\033[{background};2;{red};{green};{blue}m"

    def _colorize_string(
        self, s: str, text_color: Tuple[int, int, int], bold: bool = False
    ) -> str:
        colorized = []
        if bold:
            colorized.append(FancyFormatter.bold)
        colorized.append(FancyFormatter._get_color(*text_color, background=False))
        colorized.append(s)
        colorized.append(FancyFormatter.reset)

        return "".join(colorized)

    def _create_format(self, level: int):
        base_format = (
            f"{{asctime}} [{{levelname:>8}}] "
            f"[{{name:>{self.max_module_name_length}}}:"
            f"{{funcName:>{self.max_func_name_length}}}:"
            f"L{{lineno:>{self.max_lineno_length}}}] {{message}}"
        )

        level_style = self.styles[level]
        color = level_style["color"]
        bold = level_style.get("bold", False)

        if not self.disable_colors:
            base_format = self._colorize_string(
                base_format, text_color=color, bold=bold
            )

        return base_format

    def __init__(
        self,
        styles: Dict[int, dict] | None = None,
        max_module_name_length: int = 32,
        max_func_name_length: int = 32,
        max_lineno_length: int = 4,
        date_format: str = None,
        disable_colors: bool = False,
    ):
        # this is NOOP as we initialize a new Formatter in format()
        # (just included to suppress warning)
        super().__init__(datefmt=date_format)

        self.styles = styles or FancyFormatter.DEFAULT_STYLES
        self.max_module_name_length = max_module_name_length
        self.max_func_name_length = max_func_name_length
        self.max_lineno_length = max_lineno_length
        self.date_format = date_format
        self.disable_colors = disable_colors

    def _shorten_left(self, s: str, max_length: int) -> str:
        if len(s) > max_length:
            s = "..." + s[-(max_length - 3) :]

        return s

    def format(self, record: logging.LogRecord) -> str:

        record.name = self._shorten_left(
            record.name, max_length=self.max_module_name_length
        )
        record.funcName = self._shorten_left(
            record.funcName, max_length=self.max_func_name_length
        )

        log_format = self._create_format(record.levelno)
        formatter = logging.Formatter(
            fmt=log_format, datefmt=None, style="{", validate=True
        )

        return formatter.format(record)


def init_fancy_logging(
    handlers: Sequence[logging.Handler] | None = None,
    styles: Dict[int, dict] | None = None,
    max_module_name_length: int = 32,
    max_func_name_length: int = 32,
    max_lineno_length: int = 4,
):
    """Initializes fancy logging, i.e. nicely structured logging messages with
    colors.

    :param handlers:
    :type handlers:
    :param styles:
    :type styles:
    :param max_module_name_length:
    :type max_module_name_length:
    :param max_func_name_length:
    :type max_func_name_length:
    :param max_lineno_length:
    :type max_lineno_length:
    :return:
    :rtype:
    """
    if not handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        handlers = [handler]

    for handler in handlers:
        if isinstance(handler, logging.FileHandler):
            disable_colors = True
        else:
            disable_colors = False

        handler.setFormatter(
            FancyFormatter(
                styles=styles,
                max_module_name_length=max_module_name_length,
                max_func_name_length=max_func_name_length,
                max_lineno_length=max_lineno_length,
                disable_colors=disable_colors,
            )
        )
    logging.basicConfig(handlers=handlers)


class RegistrationLogEntry:
    DEFAULT_ORDERING = ("stage", "level", "iteration", "loss", "losses")

    def __init__(
        self,
        stage: str,
        level: int | None = None,
        iteration: int | None = None,
        ordering: Sequence["str"] | None = None,
        **kwargs,
    ):

        self.stage = stage
        self.level = level
        self.iteration = iteration
        self._ordering = ordering or RegistrationLogEntry.DEFAULT_ORDERING

        # set optional/custom log content
        self.__dict__.update(kwargs)

    def _order_entries(self, log: dict):
        priorities = {word: index for (index, word) in enumerate(self._ordering)}

        def sorting_key(string: str) -> Tuple[int, str]:
            prio = len(priorities)
            for word, prioritiy in priorities.items():
                if string == word:
                    prio = prioritiy
                elif string.startswith(word):
                    prio = prioritiy + len(self._ordering)

            return (prio, string)

        return {key: log[key] for key in sorted(log.keys(), key=sorting_key)}

    @staticmethod
    def format_logging_dict(
        d: dict, convert_tensors: bool = True, decimal_places: int = 6
    ) -> dict:
        def convert_float(f: float) -> str:
            return f"{float(f):.{decimal_places}f}"

        types = [float]
        if convert_tensors:
            types.append(torch.Tensor)
        return convert_dict_values(d, types=types, converter=convert_float)

    def __str__(self) -> str:
        # drop all private variables
        d = {
            key: value
            for (key, value) in self.__dict__.items()
            if not key.startswith("_")
        }

        log = {
            "stage": d.pop("stage"),
        }
        if level := d.pop("level"):
            log["level"] = level
        if iteration := d.pop("iteration"):
            log["iteration"] = iteration

        log.update(d)

        log = self._order_entries(log)
        log = self.format_logging_dict(log, convert_tensors=True, decimal_places=6)

        return str(log)


# if __name__ == "__main__":
#
#     # init_fancy_logging()
#
#     # logging.basicConfig(
#     #     level=logging.INFO,
#     #     format=(
#     #         "{asctime} [{levelname:>8}] "
#     #         "[{name:>32}:"
#     #         "{funcName:>32}:"
#     #         "L{lineno:>4}] {message}"
#     #     ),
#     #     style="{",
#     #     handlers=[logging.StreamHandler()],
#     # )
#
#     logger = logging.getLogger(__name__)
#     logger.setLevel(logging.DEBUG)
#
#     logger.debug("debug")
#     logger.info("info")
#     logger.warning("warning")
#     logger.error("error")
#     logger.critical("critical")
#
#     log = RegistrationLogEntry(
#         mem=1,
#         stage="vroc",
#         level=1,
#         iteration=100,
#         loss=1,
#     )
#     str(log)
