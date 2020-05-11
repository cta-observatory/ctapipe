""" helpers for better logging """

import logging

default_logging_format = "%(levelname)s [%(name)s] (%(module)s/%(funcName)s): %(message)s"


def create_log_config(name, log_level, log_file, log_file_level):
    log_config = {"version": 1}

    formatters = {
        "file": {
            "()": PlainFormatter,
            "fmt": default_logging_format,
        },
        "console": {
            "()": FancyFormatter,
            "fmt": default_logging_format,
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class FancyFormatter(logging.Formatter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def format(self, record):
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
