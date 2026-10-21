import logging
import sys
from io import StringIO
from subprocess import CalledProcessError

import pytest

from ctapipe.core.tool import Tool, run_tool


def test_run_tool_raises_exit_code():
    class ErrorTool(Tool):
        def setup(self):
            pass

        def start(self):
            pass

    ret = run_tool(ErrorTool(), ["--non-existing-alias"], raises=False)
    assert ret == 2

    class SysExitTool(Tool):
        def setup(self):
            pass

        def start(self):
            sys.exit(4)

    ret = run_tool(SysExitTool(), raises=False)
    assert ret == 4

    with pytest.raises(CalledProcessError):
        run_tool(ErrorTool(), ["--non-existing-alias"], raises=True)


@pytest.mark.parametrize("fails", [False, True])
def test_run_tool_restores_logging(fails):
    class LoggingTool(Tool):
        name = "ctapipe-test-logging-restore"

        def start(self):
            if fails:
                raise RuntimeError("tool failed")

    logger = logging.getLogger("ctapipe")
    child = logging.getLogger("ctapipe.test_logging_restore")
    other = logging.getLogger("test_logging_restore.other")
    previous = (logger.level, logger.handlers[:], logger.propagate)
    child_previous = (child.level, child.handlers[:], child.propagate)
    other_previous = (other.level, other.handlers[:], other.propagate)
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    try:
        logger.setLevel(logging.INFO)
        logger.handlers[:] = [handler]
        logger.propagate = False
        child.setLevel(logging.DEBUG)
        child.propagate = True
        other.setLevel(logging.INFO)
        other.propagate = True

        tool = LoggingTool()
        tool.log_config = {
            "loggers": {
                other.name: {
                    "level": "ERROR",
                    "handlers": [],
                    "propagate": False,
                }
            }
        }
        assert logger.level == logging.INFO
        assert logger.handlers == [handler]

        if fails:
            with pytest.raises(RuntimeError, match="tool failed"):
                run_tool(tool)
        else:
            assert run_tool(tool) == 0

        assert logger.level == logging.INFO
        assert logger.handlers == [handler]
        assert logger.propagate is False
        assert child.level == logging.DEBUG
        assert child.propagate is True
        assert other.level == logging.INFO
        assert other.propagate is True
        child.info("logging still works after tool")
        assert "logging still works after tool" in stream.getvalue()
    finally:
        logger.setLevel(previous[0])
        logger.handlers[:] = previous[1]
        logger.propagate = previous[2]
        child.setLevel(child_previous[0])
        child.handlers[:] = child_previous[1]
        child.propagate = child_previous[2]
        other.setLevel(other_previous[0])
        other.handlers[:] = other_previous[1]
        other.propagate = other_previous[2]
        handler.close()
