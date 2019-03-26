import logging


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
            'WARNING': yellow,
            'INFO': green,
            'DEBUG': blue,
            'CRITICAL': yellow,
            'ERROR': red
        }

        levelname = record.levelname
        if levelname in colors:
            levelname_color = color_seq % (30 + colors[levelname]) \
                + levelname + reset_seq
            record.levelname = levelname_color

        if record.levelno >= self.highlevel_limit:
            record.highlevel = self.highlevel_format % record.__dict__
        else:
            record.highlevel = ""

        return super().format(record)
