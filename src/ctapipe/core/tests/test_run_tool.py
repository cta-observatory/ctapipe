import sys
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
