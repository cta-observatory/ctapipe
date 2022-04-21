import os
import pathlib
import tempfile
from unittest import mock

import pytest
from ctapipe.core import Component, TelescopeComponent
from ctapipe.core.traits import (
    AstroTime,
    Bool,
    Float,
    FloatTelescopeParameter,
    IntTelescopeParameter,
    List,
    Path,
    TelescopeParameter,
    TelescopeParameterLookup,
    TraitError,
    classes_with_traits,
    has_traits,
)
from ctapipe.image import ImageExtractor
from ctapipe.utils.datasets import DEFAULT_URL, get_dataset_path
from traitlets import CaselessStrEnum, HasTraits, Int


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


def test_path_allow_none_false():
    class C(Component):
        path = Path(default_value=None, allow_none=False)

    c = C()

    # accessing path is now an error
    with pytest.raises(TraitError):
        c.path

    # setting to None should also fail
    with pytest.raises(TraitError):
        c.path = None

    c.path = "foo.txt"
    assert c.path == pathlib.Path("foo.txt").absolute()


def test_path_allow_none_true(tmp_path):
    class C(Component):
        path = Path(exists=True, allow_none=True, default_value=None)

    c = C()
    assert c.path is None

    with open(tmp_path / "foo.txt", "w"):
        pass

    c.path = tmp_path / "foo.txt"

    c.path = None
    assert c.path is None


def test_path_exists():
    """ require existence of path """

    class C1(Component):
        thepath = Path(exists=False)

    c1 = C1()
    c1.thepath = "non-existent-path-that-should-never-exist-not-even-by-accident"

    with tempfile.NamedTemporaryFile() as f:
        with pytest.raises(TraitError):
            c1.thepath = f.name

    class C2(Component):
        thepath = Path(exists=True)

    c2 = C2()

    with tempfile.TemporaryDirectory() as d:
        c2.thepath = d

    with tempfile.NamedTemporaryFile() as f:
        c2.thepath = f.name


def test_path_invalid():
    class C1(Component):
        p = Path(exists=False)

    c1 = C1()
    with pytest.raises(TraitError):
        c1.p = 5

    with pytest.raises(TraitError):
        c1.p = ""


def test_bytes():
    class C1(Component):
        p = Path(exists=False)

    c1 = C1()
    c1.p = b"/home/foo"
    assert c1.p == pathlib.Path("/home/foo")


def test_path_directory_ok():
    """ test path is a directory """

    class C(Component):
        thepath = Path(exists=True, directory_ok=False)

    c = C()

    with pytest.raises(TraitError):
        c.thepath = "lknasdlakndlandslknalkndslakndslkan"

    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(TraitError):
            c.thepath = d

    with tempfile.NamedTemporaryFile() as f:
        c.thepath = f.name


def test_path_file_ok():
    """ check that the file is there and not a directory, etc"""

    class C(Component):
        thepath = Path(exists=True, file_ok=False)

    c = C()

    with pytest.raises(TraitError):
        c.thepath = "lknasdlakndlandslknalkndslakndslkan"

    with tempfile.TemporaryDirectory() as d:
        c.thepath = d

    with tempfile.NamedTemporaryFile() as f:
        with pytest.raises(TraitError):
            c.thepath = f.name


def test_path_pathlib():
    class C(Component):
        thepath = Path()

    c = C()
    c.thepath = pathlib.Path()
    assert c.thepath == pathlib.Path().absolute()


def test_path_url():
    class C(Component):
        thepath = Path()

    c = C()
    # test relative
    c.thepath = "file://foo.hdf5"
    assert c.thepath == (pathlib.Path() / "foo.hdf5").absolute()

    # test absolute
    c.thepath = "file:///foo.hdf5"
    assert c.thepath == pathlib.Path("/foo.hdf5")

    # test http downloading
    c.thepath = DEFAULT_URL + "optics.ecsv.txt"
    assert c.thepath.name == "optics.ecsv.txt"

    # test dataset://
    c.thepath = "dataset://optics.ecsv.txt"
    assert c.thepath == get_dataset_path("optics.ecsv.txt")


@mock.patch.dict(os.environ, {"ANALYSIS_DIR": "/home/foo"})
def test_path_envvars():
    class C(Component):
        thepath = Path()

    c = C()
    c.thepath = "$ANALYSIS_DIR/test.txt"

    assert str(c.thepath) == "/home/foo/test.txt"


def test_enum_trait_default_is_right():
    """ check default value of enum trait """
    from ctapipe.core.traits import create_class_enum_trait

    with pytest.raises(ValueError):
        create_class_enum_trait(ImageExtractor, default_value="name_of_default_choice")


def test_enum_trait():
    """ check that enum traits are constructable from a complex class """
    from ctapipe.core.traits import create_class_enum_trait

    trait = create_class_enum_trait(
        ImageExtractor, default_value="NeighborPeakWindowSum"
    )
    assert isinstance(trait, CaselessStrEnum)


def test_enum_classes_with_traits():
    """ test that we can get a list of classes that have traits """
    list_of_classes = classes_with_traits(ImageExtractor)
    assert list_of_classes  # should not be empty


def test_classes_with_traits():
    from ctapipe.core import Tool

    class CompA(Component):
        a = Int().tag(config=True)

    class CompB(Component):
        classes = List([CompA])
        b = Int().tag(config=True)

    class CompC(Component):
        c = Int().tag(config=True)

    class MyTool(Tool):
        classes = [CompB, CompC]

    with_traits = classes_with_traits(MyTool)
    assert len(with_traits) == 4
    assert MyTool in with_traits
    assert CompA in with_traits
    assert CompB in with_traits
    assert CompC in with_traits


def test_has_traits():
    """ test the has_traits func """

    class WithoutTraits(HasTraits):
        """ a traits class that has no traits """

        pass

    class WithATrait(HasTraits):
        """ a traits class that has a trait """

        my_trait = Int()

    assert not has_traits(WithoutTraits)
    assert has_traits(WithATrait)


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


def test_telescope_parameter_patterns(mock_subarray):
    """ Test validation of TelescopeParameters"""

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


def test_telescope_parameter_resolver():
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

    # need to mock a SubarrayDescription
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

    comp.tel_param1.attach_subarray(subarray)
    comp.tel_param2.attach_subarray(subarray)
    comp.tel_param3.attach_subarray(subarray)

    assert comp.tel_param1.tel[1] == 10
    assert comp.tel_param1.tel[3] == 100

    assert list(map(comp.tel_param2.tel.__getitem__, [1, 2, 3, 4])) == [
        10.0,
        10.0,
        200.0,
        100.0,
    ]

    assert list(map(comp.tel_param3.tel.__getitem__, [1, 2, 3, 4, 100])) == [
        200.0,
        200.0,
        200.0,
        200.0,
        300.0,
    ]


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


def test_telescope_parameter_from_cli(mock_subarray):
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
                "test.h5",
                "--SomeComponent.val",
                "2.0",
                "--SomeComponent.flag",
                "False",
            ],
        )
        assert result == 0
        assert tool.comp.path == [("type", "*", pathlib.Path("test.h5").absolute())]
        assert tool.comp.val == [("type", "*", 2.0)]
        assert tool.comp.flag == [("type", "*", False)]


def test_datetimes():
    from astropy import time as t

    class SomeComponentWithTimeTrait(Component):
        time = AstroTime()

    component = SomeComponentWithTimeTrait()
    component.time = "2019-10-15 12:00:00.234"
    assert str(component.time) == "2019-10-15 12:00:00.234"
    component.time = "2019-10-15T12:15:12"
    assert str(component.time) == "2019-10-15 12:15:12.000"
    component.time = t.Time.now()
    assert isinstance(component.time, t.Time)

    with pytest.raises(TraitError):
        component.time = "garbage"


def test_time_none():
    class AllowNone(Component):
        time = AstroTime(default_value=None, allow_none=True)

    c = AllowNone()
    c.time = None

    class NoNone(Component):
        time = AstroTime(default_value="2012-01-01T20:00", allow_none=False)

    c = NoNone()
    with pytest.raises(TraitError):
        c.time = None
