"""Helpers for better logging."""

import logging
from collections.abc import Mapping
from enum import IntEnum

DEFAULT_LOGGING_FORMAT = (
    "%(asctime)s %(levelname)s [%(name)s] (%(module)s.%(funcName)s): %(message)s"
)


class ANSIEscapes(IntEnum):
    """
    Enum of some ANSI style escape sequences used for logging.

    See https://en.wikipedia.org/wiki/ANSI_escape_code
    """

    RESET = 0
    BOLD = 1
    FG_RED = 31
    FG_GREEN = 32
    FG_YELLOW = 33
    FG_BLUE = 34
    FG_MAGENTA = 35


LEVEL_COLORS = {
    "DEBUG": ANSIEscapes.FG_BLUE,
    "INFO": ANSIEscapes.FG_GREEN,
    "WARNING": ANSIEscapes.FG_YELLOW,
    "ERROR": ANSIEscapes.FG_RED,
    "CRITICAL": ANSIEscapes.FG_MAGENTA,
}


def add_ansi_display(string, *styles):
    """Surround ``string`` by ANSI escape code styling.

    Parameters
    ----------
    string : str
        Text to style.
    *styles
        ANSI style or color codes. Each is converted to an integer escape code.
    """
    styles = ";".join(str(int(style)) for style in styles)
    return f"\033[{styles}m{string}\033[{ANSIEscapes.RESET}m"


class PlainFormatter(logging.Formatter):
    """Custom logging.Formatter used for file logging."""


class ColoredFormatter(logging.Formatter):
    """Custom logging.Formatter that adds colors for terminal logging."""

    def format(self, record):
        """Format the LogRecord.

        Parameters
        ----------
        record : logging.LogRecord
            Log record to format and decorate based on its level.
        """
        s = super().format(record)
        return s.replace(record.levelname, apply_colors(record.levelname))


def apply_colors(levelname: str):
    """Use ANSI escape sequences to add colors the levelname of log entries.

    Parameters
    ----------
    levelname : str
        Log level name, such as ``WARNING``, used to select the terminal color.
    """
    color = LEVEL_COLORS.get(levelname)

    if color is None:
        return add_ansi_display(levelname, ANSIEscapes.BOLD)

    return add_ansi_display(levelname, ANSIEscapes.BOLD, color)


def recursive_update(d1, d2, copy=False):
    """Merge entries from one mapping into another recursively.

    By default, update and return the first mapping in place. With ``copy=True``,
    copy the top-level mapping first. Nested mappings can still be modified.

    Parameters
    ----------
    d1 : dict
        Base mapping to update. Nested values that are merged must also be mappings.
    d2 : collections.abc.Mapping
        Values to merge into the base mapping, taking precedence on overlapping keys.
    copy : bool, optional
        Shallow-copy the top-level base mapping before merging. Nested mappings may
        still be updated in place.

    Examples
    --------
    >>> d1 = {'a': {'b': 'foo'}}
    >>> d2 = {'a': {'c': 'foo'}}
    >>> recursive_update(d1, d2)
    {'a': {'b': 'foo', 'c': 'foo'}}

    An ordinary update replaces the whole nested mapping:

    >>> d1.update(d2)
    >>> d1
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
    log_level,
    log_file,
    log_file_level,
    log_config: dict,
    quiet: bool,
    module="ctapipe",
):
    """Update logging level for console and file according to CLI arguments.

    Parameters
    ----------
    log_level : int or str
        Minimum console logging level when console output is enabled.
    log_file : str or pathlib.Path or None
        Destination log file. With None, omit file logging.
    log_file_level : int or str
        Minimum logging level for the file handler.
    log_config : dict
        Logging settings to merge into the default logger configuration.
    quiet : bool
        Disable console log handlers if True.
    module : str, optional
        Logger namespace for the tool's messages.
    """
    config = recursive_update(get_default_logging(module), log_config)

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
        config["loggers"][module]["handlers"].append("ctapipe-file")

        # level of logger must be at least that of all their handlers
        config["loggers"][module]["level"] = get_lower_level(log_level, log_file_level)

    else:
        config["loggers"][module]["level"] = log_level

    return config


def get_lower_level(l0, l1):
    """Compare logging levels and return the lower level.

    Parameters
    ----------
    l0 : int or str
        First logging level, as a number or standard name such as ``INFO``.
    l1 : int or str
        Second logging level, in the same format as the first.
    """
    if isinstance(l0, (str)):
        l0 = logging.getLevelName(l0)
    if isinstance(l1, (str)):
        l1 = logging.getLevelName(l1)

    return l0 if l0 < l1 else l1


def get_default_logging(module="ctapipe"):
    """Build the default logging configuration for a named logger.

    The logger sends messages at WARNING level and above to standard error
    with colored formatting. Pass the returned configuration to
    `logging.config.dictConfig` to apply it.

    Parameters
    ----------
    module : str, optional
        Name of the logger to configure.

    Returns
    -------
    dict
        A new logging configuration that can be customized before applying it.
    """
    return {
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
            module: {
                "level": "WARN",
                "handlers": ["ctapipe-console"],
                "propagate": False,
            }
        },
    }
