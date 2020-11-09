"""Helpers for better logging."""

import logging
from collections.abc import Mapping


DEFAULT_LOGGING_FORMAT = (
    "%(asctime)s %(levelname)s [%(name)s] (%(module)s.%(funcName)s): %(message)s"
)


class PlainFormatter(logging.Formatter):
    """Custom logging.Formatter used for file logging."""


class ColoredFormatter(logging.Formatter):
    """Custom logging.Formatter that adds colors for terminal logging."""

    def format(self, record):
        """Format the LogRecord."""
        rec = record.__dict__.copy()

        rec["asctime"] = self.formatTime(record, self.datefmt)
        rec["levelname"] = apply_colors(record.levelname)
        rec["message"] = record.getMessage()

        return self._fmt % rec


def apply_colors(levelname: str):
    """Use ANSI escape sequences to add colors the levelname of log entries."""
    _black, red, green, yellow, blue, magenta, _cyan, _white = range(8)
    reset_seq = "\033[0m"
    color_seq = "\033[1;%dm"
    colors = {
        "INFO": green,
        "DEBUG": blue,
        "WARNING": yellow,
        "CRITICAL": magenta,
        "ERROR": red,
    }

    levelname_color = color_seq % (30 + colors[levelname]) + levelname + reset_seq

    return levelname_color


def recursive_update(d1, d2, copy=False):
    """Merge dicts recursively, e.g.
    >>> d1 = {'a': {'b': 'foo'}}
    >>> d2 = {'a': {'c': 'foo'}}
    >>> recursive_update(d1, d2)
    {'a': {'b': 'foo', 'c': 'foo'}}
    >>> # As opposed to
    >>> d1.update(d2)
    {'a': {'c': 'foo'}}
    """
    if not isinstance(d1, Mapping) or not isinstance(d2, Mapping):
        raise TypeError("Arguments must be mappings")

    if copy:
        d1 = d1.copy()

    for k, v in d2.items():
        if isinstance(v, Mapping):
            d1[k] = recursive_update(d1.get(k, {}), v)
        else:
            d1[k] = v

    # just for convenience, the input dict is actually mutated
    return d1


def create_logging_config(
    log_level, log_file, log_file_level, log_config: dict, quiet: bool
):
    """Update logging level for console and file according to CLI arguments."""
    config = recursive_update(DEFAULT_LOGGING, log_config)

    if quiet:
        config["handlers"]["console"] = {"class": "logging.NullHandler"}
        config["handlers"]["ctapipe-console"] = {"class": "logging.NullHandler"}
    else:
        config["handlers"]["ctapipe-console"]["level"] = log_level

    if log_file is not None:
        file_handler = {
            "ctapipe-file": {
                "class": "logging.FileHandler",
                "formatter": "file",
                "filename": log_file,
                "level": log_file_level,
            }
        }
        config["handlers"].update(file_handler)
        config["loggers"]["ctapipe"]["handlers"].append("ctapipe-file")

        # level of logger must be at least that of all their handlers
        config["loggers"]["ctapipe"]["level"] = get_lower_level(
            log_level, log_file_level
        )

    else:
        config["loggers"]["ctapipe"]["level"] = log_level

    return config


def get_lower_level(l0, l1):
    """Compare logging levels and return the lower level."""
    if isinstance(l0, (str)):
        l0 = logging.getLevelName(l0)
    if isinstance(l1, (str)):
        l1 = logging.getLevelName(l1)

    return l0 if l0 < l1 else l1


DEFAULT_LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "file": {"()": PlainFormatter, "fmt": DEFAULT_LOGGING_FORMAT},
        "console": {"()": ColoredFormatter, "fmt": DEFAULT_LOGGING_FORMAT},
    },
    "handlers": {
        "ctapipe-console": {
            "class": "logging.StreamHandler",
            "formatter": "console",
            "stream": "ext://sys.stderr",
            "level": "NOTSET",
        },
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "console",
            "stream": "ext://sys.stderr",
            "level": "NOTSET",
        },
    },
    "loggers": {
        "ctapipe": {
            "level": "WARN",
            "handlers": ["ctapipe-console"],
            "propagate": False,
        }
    },
}
