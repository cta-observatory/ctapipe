""" helpers for better logging """

import logging


def create_log_config(name, log_level, log_file, log_file_level):
    log_config = {
        "version": 1,
        "formatters": {
            "default": {
                "format": "%(levelname)s [%(name)s] (%(module)s/%(funcName)s): %(message)s",
            },
        },
    }

    c_handler = {
        "class": "logging.StreamHandler",
        "formatter": "default",
        "level": log_level,
        "stream": "ext://sys.stdout",
    }

    handlers = {"c_handler": c_handler}

    loggers = {name: {"handlers": ["c_handler"], "level": logging.DEBUG}}

    if log_file is not None:
        f_handler = {
            "class": "logging.FileHandler",
            "formatter": "default",
            "level": log_file_level,
            "filename": log_file,
        }

        handlers.update({"f_handler": f_handler})

        loggers[name]["handlers"].append("f_handler")

    log_config.update({"loggers": loggers, "handlers": handlers})

    return log_config


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
