import os
import logging
import tempfile
import pytest
from traitlets import Float, TraitError, List, Dict, Int
from traitlets.config import Config
from pathlib import Path

from .. import Tool, Component
from ..tool import export_tool_config_to_commented_yaml, run_tool


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
    """ check that the tool gets an automatic version string"""

    class MyTool(Tool):
        description = "test"
        userparam = Float(5.0, help="parameter").tag(config=True)

    tool = MyTool()
    assert tool.version_string != ""


def test_provenance_dir():
    """ check that the tool gets the provenance dir"""

    class MyTool(Tool):
        description = "test"
        userparam = Float(5.0, help="parameter").tag(config=True)

    tool = MyTool()
    assert str(tool.provenance_log) == os.path.join(
        os.getcwd(), "application.provenance.log"
    )


def test_provenance_log_help(tmpdir):
    """ check that the tool does not write a provenance log if only the help was run"""
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
    """ test that we can export a Tool's config to YAML"""
    import yaml
    from ctapipe.tools.stage1 import Stage1Tool

    tool = Stage1Tool()
    tool.progress_bar = True
    yaml_string = export_tool_config_to_commented_yaml(tool)

    # check round-trip back from yaml:
    config_dict = yaml.load(yaml_string, Loader=yaml.SafeLoader)

    assert config_dict["Stage1Tool"]["progress_bar"] is True


def test_tool_html_rep():
    """ check that the HTML rep for Jupyter notebooks works"""

    class MyTool(Tool):
        description = "test"
        userparam = Float(5.0, help="parameter").tag(config=True)

    class MyTool2(Tool):
        """ A docstring description"""

        userparam = Float(5.0, help="parameter").tag(config=True)

    tool = MyTool()
    tool2 = MyTool2()
    assert len(tool._repr_html_()) > 0
    assert len(tool2._repr_html_()) > 0


def test_tool_current_config():
    """ Check that we can get the full instance configuration """

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
    """ Check that we can get the full instance configuration """
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
    """ Check that we can get the full instance configuration """

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

    assert run_tool(tool, ["--help"]) == 0
    assert run_tool(tool, ["--non-existent-option"]) == 2


def test_tool_command_line_precedence():
    """
    ensure command-line has higher priority than config file
    """

    class SubComponent(Component):
        component_param = Float(10.0, help="some parameter").tag(config=True)

    class MyTool(Tool):
        description = "test"
        userparam = Float(5.0, help="parameter").tag(config=True)

        classes = List([SubComponent])
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
