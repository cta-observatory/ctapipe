import json
import logging
import os
import signal
import sys
import tempfile
from multiprocessing import Barrier, Process
from pathlib import Path

import pytest
from traitlets import Dict, Float, Int, TraitError
from traitlets.config import Config

from .. import Component, Tool
from ..tool import (
    ToolConfigurationError,
    export_tool_config_to_commented_yaml,
    run_tool,
)


def test_tool_simple():
    """test the very basic functionality of a Tool"""

    class MyTool(Tool):
        description = "test"
        userparam = Float(5.0, help="parameter").tag(config=True)

    tool = MyTool()
    tool.userparam = 1.0
    tool.log_level = "DEBUG"
    tool.log.info("test")
    with pytest.raises(SystemExit) as exc:
        tool.run([])
    assert exc.value.code == 0

    # test parameters changes:
    tool.userparam = 4.0
    with pytest.raises(TraitError):
        tool.userparam = "badvalue"


def test_tool_version():
    """check that the tool gets an automatic version string"""

    class MyTool(Tool):
        description = "test"
        userparam = Float(5.0, help="parameter").tag(config=True)

    tool = MyTool()
    assert tool.version_string != ""


def test_provenance_dir():
    """check that the tool gets the provenance dir"""

    class MyTool(Tool):
        description = "test"
        userparam = Float(5.0, help="parameter").tag(config=True)

    tool = MyTool()
    assert str(tool.provenance_log) == os.path.join(
        os.getcwd(), "application.provenance.log"
    )


def test_provenance_log_help(tmpdir):
    """check that the tool does not write a provenance log if only the help was run"""
    from ctapipe.core.tool import run_tool

    class MyTool(Tool):
        description = "test"
        userparam = Float(5.0, help="parameter").tag(config=True)

    tool = MyTool()
    tool.provenance_log = Path(tmpdir) / "test_prov_log_help.log"
    for o in ["-h", "--help", "--help-all"]:
        assert run_tool(tool, [o], cwd=tmpdir) == 0
        assert not tool.provenance_log.exists()


def test_export_config_to_yaml():
    """test that we can export a Tool's config to YAML"""
    import yaml

    from ctapipe.tools.process import ProcessorTool

    tool = ProcessorTool()
    tool.progress_bar = True
    yaml_string = export_tool_config_to_commented_yaml(tool)

    # check round-trip back from yaml:
    config_dict = yaml.load(yaml_string, Loader=yaml.SafeLoader)

    assert config_dict["ProcessorTool"]["progress_bar"] is True


def test_tool_html_rep(tmp_path):
    """check that the HTML rep for Jupyter notebooks works"""

    class MyTool(Tool):
        description = "test"
        userparam = Float(5.0, help="parameter").tag(config=True)

    tool = MyTool()
    assert len(tool._repr_html_()) > 0

    class MyComponent(Component):
        val = Float(1.0, help="val").tag(config=True)

    class MyTool2(Tool):
        """A docstring description"""

        userparam = Float(5.0, help="parameter").tag(config=True)

        classes = [MyComponent]

        def setup(self):
            self.comp = MyComponent(parent=self)

        def start(self):
            pass

    tool2 = MyTool2()
    assert len(tool2._repr_html_()) > 0

    # make sure html repr works also after tool was run
    assert run_tool(tool2, argv=[], cwd=tmp_path) == 0
    assert len(tool2._repr_html_()) > 0


def test_tool_current_config():
    """Check that we can get the full instance configuration"""

    class MyTool(Tool):
        description = "test"
        userparam = Float(5.0, help="parameter").tag(config=True)

    tool = MyTool()
    conf1 = tool.get_current_config()
    tool.userparam = -1.0
    conf2 = tool.get_current_config()

    assert conf1["MyTool"]["userparam"] == 5.0
    assert conf2["MyTool"]["userparam"] == -1.0


def test_tool_current_config_subcomponents():
    """Check that we can get the full instance configuration"""
    from ctapipe.core.component import Component

    class SubComponent(Component):
        param = Int(default_value=3).tag(config=True)

    class MyComponent(Component):
        val = Int(default_value=42).tag(config=True)

        def __init__(self, config=None, parent=None):
            super().__init__(config=config, parent=parent)
            self.sub = SubComponent(parent=self)

    class MyTool(Tool):
        description = "test"
        userparam = Float(5.0, help="parameter").tag(config=True)

        def setup(self):
            self.my_comp = MyComponent(parent=self)

    config = Config()
    config.MyTool.userparam = 2.0
    config.MyTool.MyComponent.val = 10
    config.MyTool.MyComponent.SubComponent.param = -1

    tool = MyTool(config=config)
    tool.setup()

    current_config = tool.get_current_config()
    assert current_config["MyTool"]["MyComponent"]["val"] == 10
    assert current_config["MyTool"]["MyComponent"]["SubComponent"]["param"] == -1
    assert current_config["MyTool"]["userparam"] == 2.0


def test_tool_exit_code():
    """Check that we can get the full instance configuration"""

    class MyTool(Tool):
        description = "test"
        userparam = Float(5.0, help="parameter").tag(config=True)

    tool = MyTool()

    with pytest.raises(SystemExit) as exc:
        tool.run(["--non-existent-option"])

    assert exc.value.code == 2

    with pytest.raises(SystemExit) as exc:
        tool.run(["--MyTool.userparam=foo"])

    assert exc.value.code == 1

    assert run_tool(tool, ["--help"], raises=False) == 0
    assert run_tool(tool, ["--non-existent-option"], raises=False) == 2


def test_tool_command_line_precedence():
    """
    ensure command-line has higher priority than config file
    """

    class SubComponent(Component):
        component_param = Float(10.0, help="some parameter").tag(config=True)

    class MyTool(Tool):
        description = "test"
        userparam = Float(5.0, help="parameter").tag(config=True)

        classes = [SubComponent]
        aliases = Dict({"component_param": "SubComponent.component_param"})

        def setup(self):
            self.sub = SubComponent(parent=self)

    config = Config(
        {"MyTool": {"userparam": 12.0}, "SubComponent": {"component_param": 15.0}}
    )

    tool = MyTool(config=config)  # sets component_param to 15.0

    run_tool(tool, ["--component_param", "20.0"])
    assert tool.sub.component_param == 20.0
    assert tool.userparam == 12.0


class MyLogTool(Tool):
    name = "ctapipe-test"

    def start(self):
        self.log.debug("test-debug")
        self.log.info("test-info")
        self.log.warning("test-warn")
        self.log.error("test-error")
        self.log.critical("test-critical")


def test_tool_logging_defaults(capsys):
    tool = MyLogTool()

    assert tool.log_level == 30
    assert tool.log_file is None

    run_tool(tool)

    # split lines and skip last empty line
    log = capsys.readouterr().err.split("\n")[:-1]

    assert len(log) == 3
    assert "test-warn" in log[0]


def test_tool_logging_setlevel(capsys):
    tool = MyLogTool()

    run_tool(tool, ["--log-level", "ERROR"])

    # split lines and skip last empty line
    log = capsys.readouterr().err.split("\n")[:-1]

    assert len(log) == 2
    assert "test-error" in log[0]
    assert "test-critical" in log[1]


def test_tool_logging_file(capsys):
    tool = MyLogTool()

    with tempfile.NamedTemporaryFile("w+") as f:
        run_tool(tool, ["--log-file", f.name])
        log = str(f.read())

        assert len(log) > 0
        assert "test-debug" not in log
        assert "test-info" in log
        assert "test-warn" in log

    # split lines and skip last empty line
    log = capsys.readouterr().err.split("\n")[:-1]

    assert len(log) > 0
    assert "test-warn" in log[0]


def test_tool_logging_multiple_loggers(capsys):
    """No-ctapipe loggers can be configured via tool config files."""
    logger = logging.getLogger("another_logger")

    config = Config(
        {
            "MyLogTool": {
                "log_config": {
                    "loggers": {
                        "another_logger": {"level": "DEBUG", "handlers": ["console"]},
                        "ctapipe.ctapipe-test": {"level": "ERROR"},
                    }
                }
            }
        }
    )

    tool = MyLogTool(config=config)
    run_tool(tool)

    logger.debug("another-debug")

    # split lines and skip last empty line
    log = capsys.readouterr().err.split("\n")[:-1]

    assert len(log) == 3
    assert "test-error" in log[0]
    assert "another-debug" in log[2]


def test_tool_logging_quiet(capsys):
    tool = MyLogTool()

    # setting log-level should not matter when given -q
    run_tool(tool, ["-q", "--log-level", "DEBUG"])

    log = capsys.readouterr().err

    assert len(log) == 0


def test_tool_overwrite_output(capsys, tmp_path):
    path = tmp_path / "overwrite_dummy"
    tool = Tool()
    # path does not exist
    tool.check_output(path)
    # path exists and no overwrite
    path.touch()
    with pytest.raises(ToolConfigurationError):
        tool.check_output(path)
    # path exists and overwrite is True
    tool.overwrite = True
    tool.check_output(path)


def test_invalid_traits(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="ctapipe")

    class MyTool(Tool):
        name = "test"
        description = "test"
        param = Float(5.0, help="parameter").tag(config=True)

    # 2 means trait error
    assert run_tool(MyTool(), ["--MyTool.foo=5"], raises=False) == 2

    with pytest.raises(ToolConfigurationError):
        run_tool(MyTool(), ["--MyTool.foo=5"], raises=True)

    # test that it also works for config files
    config = tmp_path / "config.json"
    with config.open("w") as f:
        json.dump({"MyTool": {"foo": 5}}, f)

    assert run_tool(MyTool(), [f"--config={config}"], raises=False) == 2
    with pytest.raises(ToolConfigurationError):
        assert run_tool(MyTool(), [f"--config={config}"], raises=True)


def test_tool_raises():
    class ToolGood(Tool):
        name = "Good"
        description = "This runs without an exception."

        def start(self):
            print("All good.")

    class ToolBad(Tool):
        name = "Bad"
        description = "This tool raises an exception."

        def start(self):
            raise ValueError("1 does not equal 0.")

    class CustomErrorNoExitCode(Exception):
        pass

    class CustomErrorWithExitCode(Exception):
        exit_code = 42

    class ToolCustomExceptionNoExitCode(Tool):
        name = "CustomException"
        description = "This tool raises a custom exception without an exit code."

        def start(self):
            raise CustomErrorNoExitCode("This is a custom exception.")

    class ToolCustomExceptionWithExitCode(Tool):
        name = "CustomException"
        description = "This tool raises a custom exception with a custom exit code."

        def start(self):
            raise CustomErrorWithExitCode("This is a custom exception.")

    assert run_tool(ToolGood(), raises=True) == 0

    assert run_tool(ToolBad(), raises=False) == 1

    assert run_tool(ToolCustomExceptionNoExitCode(), raises=False) == 1

    assert run_tool(ToolCustomExceptionWithExitCode(), raises=False) == 42

    with pytest.raises(ValueError):
        run_tool(ToolBad(), raises=True)

    with pytest.raises(CustomErrorNoExitCode):
        run_tool(ToolCustomExceptionNoExitCode(), raises=True)


def test_exit_stack():
    """Test that components that are context managers are properly handled"""

    class TestManager:
        def __init__(self):
            self.enter_called = False
            self.exit_called = False

        def __enter__(self):
            self.enter_called = True
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.exit_called = True

    class AtExitTool(Tool):
        def setup(self):
            self.manager = self.enter_context(TestManager())

    tool = AtExitTool()
    run_tool(tool)
    assert tool.manager.enter_called
    assert tool.manager.exit_called

    # test this also works when there is an exception in the user code
    class FailTool(Tool):
        def setup(self):
            self.manager = self.enter_context(TestManager())

        def start(self):
            raise Exception("Failed")

    tool = FailTool()
    assert run_tool(tool, raises=False) == 1
    assert tool.manager.enter_called
    assert tool.manager.exit_called


def test_activity(tmp_path):
    """check that the config is correctly in the provenance"""

    class MyTool(Tool):
        name = "test_prov_log"
        description = "test"
        userparam = Float(5.0, help="parameter").tag(config=True)

    tool = MyTool()

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"MyTool": {"userparam": 10.0}}))
    provenance_path = tmp_path / "provenance.json"

    run_tool(
        tool,
        [
            "--config",
            str(config_path),
            f"--provenance-log={provenance_path}",
        ],
    )

    activities = json.loads(tool.provenance_log.read_text())
    # provlog contains all activities from all tests, last one is the tool we just ran
    provlog = activities[-1]
    assert provlog["activity_name"] == MyTool.name

    # test config file is in inputs, regression test for #2313
    inputs = provlog["input"]
    assert len(inputs) == 1
    assert inputs[0]["role"] == "Tool Configuration"
    assert inputs[0]["url"] == str(config_path)


@pytest.mark.parametrize(
    ("exit_code", "expected_status"),
    [
        (0, "completed"),
        (None, "completed"),
        (1, "error"),
        (2, "error"),
    ],
)
def test_exit_status(exit_code, expected_status, tmp_path, provenance):
    """check that the config is correctly in the provenance"""

    class MyTool(Tool):
        exit_code = Int(allow_none=True, default_value=None).tag(config=True)

        def start(self):
            if self.exit_code is None:
                return

            if self.exit_code == 0:
                sys.exit(0)

            if self.exit_code == 1:
                raise ValueError("Some error happened")

            class CustomError(ValueError):
                exit_code = self.exit_code

            raise CustomError("Some error with specific code happened")

    provenance_path = tmp_path / "provlog.json"
    run_tool(
        MyTool(exit_code=exit_code),
        [f"--provenance-log={provenance_path}"],
        raises=False,
    )

    activities = json.loads(provenance_path.read_text())
    assert len(activities) == 1
    provlog = activities[0]
    assert provlog["status"] == expected_status


class InterruptTestTool(Tool):
    name = "test-interrupt"

    def __init__(self, barrier):
        super().__init__()
        self.barrier = barrier

    def start(self):
        self.barrier.wait()
        signal.pause()


def test_exit_status_interrupted(tmp_path, provenance):
    """check that the config is correctly in the provenance"""

    # to make sure we only kill the process once it is running
    barrier = Barrier(2)
    tool = InterruptTestTool(barrier)

    provenance_path = tmp_path / "provlog.json"
    args = [f"--provenance-log={provenance_path}", "--log-level=INFO"]
    process = Process(target=run_tool, args=(tool, args), kwargs=dict(raises=False))
    process.start()
    barrier.wait()

    # process.terminate()
    os.kill(process.pid, signal.SIGINT)
    process.join()

    activities = json.loads(provenance_path.read_text())
    assert len(activities) == 1
    provlog = activities[0]
    assert provlog["activity_name"] == InterruptTestTool.name
    assert provlog["status"] == "interrupted"


def test_no_ignore_bad_config_type(tmp_path: Path):
    """Check that if an unknown type of config file is given, an error is raised
    instead of silently ignoring the file (which is the default for
    traitlets.config)
    """

    class SomeTool(Tool):
        float_option = Float(1.0, help="An option to change").tag(config=True)

    test_config_file = """
    SomeTool:
        float_option: 200.0
    """

    bad_conf_path = tmp_path / "test.conf"  # note named "conf" not yaml.
    bad_conf_path.write_text(test_config_file)

    good_conf_path = tmp_path / "test.yaml"
    good_conf_path.write_text(test_config_file)

    tool = SomeTool()

    # here we should receive an error.
    with pytest.raises(ToolConfigurationError):
        tool.load_config_file(bad_conf_path)

    # test correct case:
    tool.load_config_file(good_conf_path)
    assert tool.float_option > 1
