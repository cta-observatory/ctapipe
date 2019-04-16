import pytest
from traitlets import Float, TraitError

from .. import Tool
from ..tool import export_component_config_to_yaml


def test_tool_simple():
    """test the very basic functionality of a Tool"""

    class MyTool(Tool):
        description = "test"
        userparam = Float(5.0, help="parameter").tag(config=True)

    tool = MyTool()
    tool.userparam = 1.0
    tool.log_level = 'DEBUG'
    tool.log.info("test")
    tool.run([])

    # test parameters changes:
    tool.userparam = 4.0
    with pytest.raises(TraitError):
        tool.userparam = "badvalue"


def test_tool_version():

    class MyTool(Tool):
        description = "test"
        userparam = Float(5.0, help="parameter").tag(config=True)

    tool = MyTool()
    assert tool.version_string != ""


def test_export_config_to_yaml():
    import yaml
    from ctapipe.tools.camdemo import CameraDemo

    c = CameraDemo()
    c.num_events = 2
    yaml_string = export_component_config_to_yaml(CameraDemo)

    #check round-trip back from yaml:
    config_dict = yaml.load(yaml_string, Loader=yaml.SafeLoader)

    assert config_dict['CameraDemo']['num_events'] == 2
