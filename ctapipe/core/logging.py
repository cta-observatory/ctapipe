"""Helpers for better logging."""

import logging

DEFAULT_LOGGING_FORMAT = "%(levelname)s [%(name)s] (%(module)s/%(funcName)s): %(message)s"


def create_log_config(name, log_level, log_file, log_file_level):
    """Create logging configuration based on `logging.config.dictConfig`.

    The configuration dictionary scheme is documented here:
    https://docs.python.org/3/library/logging.config.html#logging-config-dictschema


    Parameters
    ----------
    name : str
        Name of the Logger object.

    log_level : int / str
        Console logging level.

    log_file : str / Path
    log_file_level : int / str
        File logging level.

    Returns
    -------
    dict

    """
    log_config = {"version": 1}

    formatters = {
        "file": {
            "()": PlainFormatter,
            "fmt": DEFAULT_LOGGING_FORMAT,
        },
        "console": {
            "()": FancyFormatter,
            "fmt": DEFAULT_LOGGING_FORMAT,
        },
    }

    console_handler = {
        "class": "logging.StreamHandler",
        "formatter": "console",
        "level": log_level,
        "stream": "ext://sys.stdout",
    }

    handlers = {"console_handler": console_handler}

    loggers = {name: {"handlers": ["console_handler"], "level": logging.DEBUG}}

    if log_file is not None:
        file_handler = {
            "class": "logging.FileHandler",
            "formatter": "file",
            "level": log_file_level,
            "filename": log_file,
        }

        handlers.update({"file_handler": file_handler})

        loggers[name]["handlers"].append("file_handler")

    log_config.update(
        {"loggers": loggers, "handlers": handlers, "formatters": formatters},
    )

    return log_config


class PlainFormatter(logging.Formatter):
    """Custom logging.Formatter used for file logging."""


class FancyFormatter(logging.Formatter):
    """Custom logging.Formatter that adds colors for terminal logging."""

    def format(self, record):
        """Format the LogRecord."""
        rec = record.__dict__.copy()

        rec["levelname"] = apply_colors(record.levelname)
        rec["message"] = record.getMessage()

        return self._fmt % rec


def apply_colors(levelname):
    """Use ANSI escape sequences to add colors the levelname of log entries."""
    _black, red, green, yellow, blue, magenta, _cyan, _white = range(8)
    reset_seq = "\033[0m"
    color_seq = "\033[1;%dm"
    colors = {
        'INFO': green,
        'DEBUG': blue,
        'WARNING': yellow,
        'CRITICAL': magenta,
        'ERROR': red
    }

    if levelname in colors:
        levelname_color = (
            color_seq % (30 + colors[levelname])
            + levelname + reset_seq
        )
    return levelname_color
