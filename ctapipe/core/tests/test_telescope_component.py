import tempfile
from unittest import mock

import pytest
from traitlets import TraitError

from ctapipe.core import Component, TelescopeComponent
from ctapipe.core.telescope_component import (
    TelescopeParameter,
    TelescopeParameterLookup,
)
from ctapipe.core.traits import (
    Bool,
    BoolTelescopeParameter,
    Float,
    FloatTelescopeParameter,
    Int,
    IntTelescopeParameter,
    Path,
)


@pytest.fixture(scope="module")
def mock_subarray():
    subarray = mock.MagicMock()
    subarray.tel_ids = [1, 2, 3, 4]
    subarray.get_tel_ids_for_type = (
        lambda x: [3, 4] if x == "LST_LST_LSTCam" else [1, 2]
    )
    subarray.telescope_types = [
        "LST_LST_LSTCam",
        "MST_MST_NectarCam",
        "MST_MST_FlashCam",
    ]
    return subarray


def test_telescope_parameter_lookup(mock_subarray):
    telparam_list = TelescopeParameterLookup([("type", "*", 10), ("type", "LST*", 100)])

    with pytest.raises(ValueError):
        telparam_list[1]

    assert telparam_list[None] == 10

    telparam_list.attach_subarray(mock_subarray)
    assert telparam_list[1] == 10
    assert telparam_list[3] == 100
    assert telparam_list[None] == 10

    with pytest.raises(KeyError):
        telparam_list[200]

    with pytest.raises(ValueError):
        bad_config = TelescopeParameterLookup([("unknown", "a", 15.0)])
        bad_config.attach_subarray(mock_subarray)

    telparam_list2 = TelescopeParameterLookup([("type", "LST*", 100)])
    with pytest.raises(KeyError):
        telparam_list2[None]


def test_telescope_parameter_lookup_by_type(subarray_prod5_paranal):
    subarray = subarray_prod5_paranal.select_subarray([1, 2, 3, 4, 100, 101])

    lookup = TelescopeParameterLookup([("type", "*", 10), ("type", "LST*", 100)])
    lookup.attach_subarray(subarray)

    assert lookup["LST_LST_LSTCam"] == 100
    assert lookup["MST_MST_NectarCam"] == 10
    assert lookup[subarray.tel[1]] == 100
    assert lookup[subarray.tel[100]] == 10

    # no global default
    lookup = TelescopeParameterLookup([("type", "LST*", 100)])
    lookup.attach_subarray(subarray)
    assert lookup["LST_LST_LSTCam"] == 100

    with pytest.raises(KeyError, match="no parameter value"):
        assert lookup["MST_MST_NectarCam"]

    with pytest.raises(ValueError, match="Unknown telescope"):
        assert lookup["Foo"]

    with pytest.raises(ValueError, match="Unknown telescope"):
        # sst
        assert lookup[subarray_prod5_paranal.tel[30]]


def test_telescope_parameter_patterns(mock_subarray):
    """Test validation of TelescopeParameters"""

    with pytest.raises(TypeError):
        TelescopeParameter(trait=int)

    with pytest.raises(TypeError):
        TelescopeParameter(trait=Int)

    class SomeComponent(TelescopeComponent):
        tel_param = TelescopeParameter(Float(default_value=0.0, allow_none=True))
        tel_param_int = IntTelescopeParameter()

    comp = SomeComponent(mock_subarray)

    # single value allowed (converted to ("default","",val) )
    comp.tel_param = 4.5
    assert list(comp.tel_param)[0][2] == 4.5

    comp.tel_param = [("type", "*", 1.0), ("type", "*LSTCam", 16.0), ("id", 16, 10.0)]

    with pytest.raises(TraitError):
        comp.tel_param = [("badcommand", "", 1.0)]

    with pytest.raises(TraitError):
        comp.tel_param = [("type", 12, 1.5)]  # bad argument

    with pytest.raises(TraitError):
        comp.tel_param_int = [("type", "LST_LST_LSTCam", 1.5)]  # not int

    comp.tel_param_int = [("type", "LST_LST_LSTCam", 1)]

    with pytest.raises(TraitError):
        comp.tel_param_int = [("*", 5)]  # wrong number of args

    with pytest.raises(TraitError):
        comp.tel_param_int = [(12, "", 5)]  # command not string


def test_telescope_parameter_path(mock_subarray, tmp_path):
    class SomeComponent(TelescopeComponent):
        path = TelescopeParameter(Path(exists=True, directory_ok=False))

    c = SomeComponent(subarray=mock_subarray)

    # non existing
    with pytest.raises(TraitError):
        c.path = "/does/not/exist"

    with tempfile.NamedTemporaryFile() as f:
        c.path = f.name

        assert str(c.path.tel[1]) == f.name

        with pytest.raises(TraitError):
            # non existing somewhere in the config
            c.path = [
                ("type", "*", f.name),
                ("type", "LST_LST_LSTCam", "/does/not/exist"),
            ]

    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(TraitError):
            c.path = d

    # test with none default:
    class SomeComponent(TelescopeComponent):
        path = TelescopeParameter(
            Path(exists=True, directory_ok=False, allow_none=True, default_value=None),
            default_value=None,
            allow_none=True,
        )

    s = SomeComponent(subarray=mock_subarray)
    assert s.path.tel[1] is None
    path = tmp_path / "foo"
    path.open("w").close()
    s.path = [("type", "*", path)]
    assert s.path.tel[1] == path


def test_telescope_parameter_scalar_default(mock_subarray):
    class SomeComponentInt(Component):
        tel_param = IntTelescopeParameter(default_value=1)

    comp_int = SomeComponentInt()
    comp_int.tel_param.attach_subarray(mock_subarray)
    assert comp_int.tel_param.tel[1] == 1

    class SomeComponentFloat(Component):
        tel_param = FloatTelescopeParameter(default_value=1.5)

    comp_float = SomeComponentFloat()
    comp_float.tel_param.attach_subarray(mock_subarray)
    assert comp_float.tel_param.tel[1] == 1.5


def test_telescope_parameter_resolver(mock_subarray):
    """check that you can resolve the rules specified in a
    TelescopeParameter trait"""

    class SomeComponent(Component):
        tel_param1 = IntTelescopeParameter(
            default_value=[("type", "*", 10), ("type", "LST*", 100)]
        )

        tel_param2 = FloatTelescopeParameter(
            default_value=[
                ("type", "*", 10.0),
                ("type", "LST_LST_LSTCam", 100.0),
                ("id", 3, 200.0),
            ]
        )

        tel_param3 = FloatTelescopeParameter(
            default_value=[
                ("type", "*", 10.0),
                ("type", "LST_LST_LSTCam", 100.0),
                ("type", "*", 200.0),  # should overwrite everything with 200.0
                ("id", 100, 300.0),
            ]
        )

    comp = SomeComponent()
    comp.tel_param1.attach_subarray(mock_subarray)
    comp.tel_param2.attach_subarray(mock_subarray)
    comp.tel_param3.attach_subarray(mock_subarray)

    assert comp.tel_param1.tel[1] == 10
    assert comp.tel_param1.tel[3] == 100

    for tel_id, expected in enumerate([10.0, 10.0, 200.0, 100.0], start=1):
        assert comp.tel_param2.tel[tel_id] == expected, f"mismatch for tel_id={tel_id}"

    expected = {1: 200.0, 2: 200.0, 3: 200.0, 4: 200.0, 100: 300.0}
    for tel_id, value in expected.items():
        assert comp.tel_param3.tel[tel_id] == value, f"mismatch for tel_id={tel_id}"


def test_telescope_parameter_component_arg(mock_subarray):
    class SomeComponent(Component):
        tel_param1 = IntTelescopeParameter(
            default_value=[("type", "*", 10), ("type", "LST*", 100)]
        )

    comp = SomeComponent(tel_param1=[("type", "*", 2), ("type", "LST*", 4)])
    comp.tel_param1.attach_subarray(mock_subarray)
    assert comp.tel_param1.tel[1] == 2
    assert comp.tel_param1.tel[3] == 4
    assert comp.tel_param1.tel[None] == 2

    comp = SomeComponent(tel_param1=200)
    comp.tel_param1.attach_subarray(mock_subarray)
    assert comp.tel_param1.tel[1] == 200
    assert comp.tel_param1.tel[3] == 200
    assert comp.tel_param1.tel[None] == 200

    comp = SomeComponent(tel_param1=300)
    assert comp.tel_param1.tel[None] == 300


def test_telescope_parameter_set_retain_subarray(mock_subarray):
    class SomeComponent(Component):
        tel_param1 = IntTelescopeParameter(
            default_value=[("type", "*", 10), ("type", "LST*", 100)]
        )

    comp = SomeComponent()
    comp.tel_param1.attach_subarray(mock_subarray)
    assert comp.tel_param1.tel[1] == 10
    assert comp.tel_param1.tel[3] == 100
    assert comp.tel_param1.tel[None] == 10

    comp.tel_param1 = 5
    assert comp.tel_param1.tel[1] == 5
    assert comp.tel_param1.tel[3] == 5
    assert comp.tel_param1.tel[None] == 5


def test_telescope_parameter_to_config(mock_subarray):
    """
    test that the config can be read back from a component with a TelescopeParameter
    (see Issue #1216)
    """

    class SomeComponent(TelescopeComponent):
        tel_param1 = FloatTelescopeParameter(default_value=6.0).tag(config=True)

    component = SomeComponent(subarray=mock_subarray)
    component.tel_param1 = 6.0
    config = component.get_current_config()
    assert config["SomeComponent"]["tel_param1"] == [("type", "*", 6.0)]


def test_telescope_parameter_from_cli(tmp_path, mock_subarray):
    """
    Test we can pass single default for telescope components via cli
    see #1559
    """

    from ctapipe.core import Tool, run_tool

    class SomeComponent(TelescopeComponent):
        path = TelescopeParameter(
            Path(allow_none=True, default_value=None), default_value=None
        ).tag(config=True)
        val = TelescopeParameter(Float(), default_value=1.0).tag(config=True)
        flag = TelescopeParameter(Bool(), default_value=True).tag(config=True)

    # test with and without SomeComponent in classes
    for tool_classes in [[], [SomeComponent]]:

        class TelescopeTool(Tool):
            classes = tool_classes

            def setup(self):
                self.comp = SomeComponent(subarray=mock_subarray, parent=self)

        tool = TelescopeTool()
        assert run_tool(tool) == 0
        assert tool.comp.path == [("type", "*", None)]
        assert tool.comp.val == [("type", "*", 1.0)]
        assert tool.comp.flag == [("type", "*", True)]

        tool = TelescopeTool()
        result = run_tool(
            tool,
            [
                "--SomeComponent.path",
                f"{tmp_path}/test.h5",
                "--SomeComponent.val",
                "2.0",
                "--SomeComponent.flag",
                "False",
            ],
        )
        assert result == 0
        assert tool.comp.path == [("type", "*", tmp_path / "test.h5")]
        assert tool.comp.val == [("type", "*", 2.0)]
        assert tool.comp.flag == [("type", "*", False)]


@pytest.mark.parametrize(
    "trait_type",
    [IntTelescopeParameter, FloatTelescopeParameter, BoolTelescopeParameter],
)
def test_telescope_parameter_none(trait_type, mock_subarray):
    class Foo(TelescopeComponent):
        bar = trait_type(default_value=None, allow_none=True).tag(config=True)

    assert Foo(mock_subarray).bar.tel[1] is None
    assert Foo(mock_subarray, bar=None).bar.tel[1] is None

    f = Foo(mock_subarray, bar=[("type", "*", 1), ("id", 1, None)])
    assert f.bar.tel[1] is None


def test_telescope_parameter_nonexistent_telescope(mock_subarray):
    class Foo(TelescopeComponent):
        bar = IntTelescopeParameter(
            default_value=None,
            allow_none=True,
        ).tag(config=True)

    foo = Foo(subarray=mock_subarray)

    with pytest.raises(KeyError, match="No telescope with id 0"):
        foo.bar.tel[0]
