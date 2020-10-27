"""Helpers for better logging."""

import logging
import logging.config
from yaml import load, FullLoader

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


def create_logging_config(
    name: str, log_level, log_file, log_file_level, log_config_file, quiet
):
    """Update logging level for console and file according to CLI arguments."""
    config = DEFAULT_LOGGING

    # update default logging configuration with user supplied logging configuration file
    if log_config_file is not None:
        config.update(get_logging_config_from_file(log_config_file))

    config["loggers"][name] = {
        "level": "DEBUG",
        "handlers": ["console"],
        "propagate": True,
    }
    config["handlers"]["console"]["level"] = log_level

    if quiet:
        config["loggers"][name]["handlers"] = []
        config["loggers"]["ctapipe"]["handlers"] = []

    if log_file is not None:
        config["handlers"].update(
            {
                "file": {
                    "class": "logging.FileHandler",
                    "formatter": "file",
                    "filename": log_file,
                    "level": log_file_level,
                }
            }
        )
        config["loggers"][name]["handlers"].append("file")
        config["loggers"]["ctapipe"]["handlers"].append("file")

    return config


def get_logging_config_from_file(filename):
    """Open logging config file.

    Parameters
    ----------
    filename : str

    Returns
    -------
    dict
    """

    with open(filename, "r") as f:
        config = load(f, Loader=FullLoader)
        if config is None:  # empty file
            config = {}

    return config


DEFAULT_LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "file": {"()": PlainFormatter, "fmt": DEFAULT_LOGGING_FORMAT},
        "console": {"()": ColoredFormatter, "fmt": DEFAULT_LOGGING_FORMAT},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "console",
            "stream": "ext://sys.stdout",
            "level": "DEBUG",
        }
    },
    "loggers": {
        "ctapipe": {
            "level": "DEBUG",  # needs to be lowest level to support higher level handlers
            "handlers": ["console"],
            "propagate": False,
        }
    },
}
