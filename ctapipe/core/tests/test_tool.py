import pytest
from traitlets import Float, TraitError, List, Dict
from traitlets.config import Config

from .. import Tool, Component
from ..tool import export_tool_config_to_commented_yaml


def test_tool_simple():
    """test the very basic functionality of a Tool"""

    class MyTool(Tool):
        description = "test"
        userparam = Float(5.0, help="parameter").tag(config=True)

    tool = MyTool()
    tool.userparam = 1.0
    tool.log_level = "DEBUG"
    tool.log.info("test")
    tool.run([])

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


def test_export_config_to_yaml():
    """ test that we can export a Tool's config to YAML"""
    import yaml
    from ctapipe.tools.camdemo import CameraDemo

    tool = CameraDemo()
    tool.num_events = 2
    yaml_string = export_tool_config_to_commented_yaml(tool)

    # check round-trip back from yaml:
    config_dict = yaml.load(yaml_string, Loader=yaml.SafeLoader)

    assert config_dict["CameraDemo"]["num_events"] == 2


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


def test_tool_command_line_precedence():
    """
    ensure command-line has higher priority than config file
    """

    class SubComponent(Component):
        component_param = Float(10.0, help="some parameter").tag(config=True)

    class MyTool(Tool):
        description = "test"
        userparam = Float(5.0, help="parameter").tag(config=True)

        classes = List([SubComponent,])
        aliases = Dict({"component_param": "SubComponent.component_param"})

        def setup(self):
            self.sub = self.add_component(SubComponent(parent=self))

    config = Config(
        {"MyTool": {"userparam": 12.0}, "SubComponent": {"component_param": 15.0}}
    )

    tool = MyTool(config=config)  # sets component_param to 15.0

    tool.run(argv=["--component_param", "20.0"])
    assert tool.sub.component_param == 20.0
    assert tool.userparam == 12.0
