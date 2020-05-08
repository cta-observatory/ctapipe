""" helpers for better logging """

import logging


log_config = {
    "version": 1,
    "formatters": {
        "default": {
            "format": "%(levelname)s [%(name)s] (%(module)s/%(funcName)s): %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "INFO",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.FileHandler",
            "formatter": "default",
            "level": "DEBUG",
            "filename": "/tmp/ctapipe-log",
        },
    },
    "loggers": {
        "Console": {
            "level": "DEBUG",
            "handlers": ["console"],
        },
        "Both": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
        },
    },
}

log_levels = [
    logging.DEBUG, "DEBUG",
    logging.INFO, "INFO",
    logging.WARN, "WARN", "WARNING",
    logging.ERROR, "ERROR",
    logging.CRITICAL, "CRITICAL",
]

class ColoredFormatter(logging.Formatter):
    """
    Custom logging.Formatter that adds colors in addition to the original
    Application logger functionality from LevelFormatter (in application.py)
    """
    highlevel_limit = logging.WARN
    highlevel_format = " %(levelname)s |"

    def format(self, record):
        black, red, green, yellow, blue, magenta, cyan, white = range(8)
        reset_seq = "\033[0m"
        color_seq = "\033[1;%dm"
        colors = {
            'INFO': green,
            'DEBUG': blue,
            'WARNING': yellow,
            'CRITICAL': magenta,
            'ERROR': red
        }

        levelname = record.levelname
        if levelname in colors:
            levelname_color = (
                color_seq % (30 + colors[levelname])
                + levelname + reset_seq
            )
            record.levelname = levelname_color

        if record.levelno >= self.highlevel_limit:
            record.highlevel = self.highlevel_format % record.__dict__
        else:
            record.highlevel = ""

        return super().format(record)
