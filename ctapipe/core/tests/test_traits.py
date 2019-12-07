import tempfile
from unittest.mock import MagicMock

import pytest
from traitlets import CaselessStrEnum, HasTraits, Int

from ctapipe.core import Component
from ctapipe.core.traits import (
    Path,
    TraitError,
    classes_with_traits,
    enum_trait,
    has_traits,
    TelescopeParameterLookup,
    TelescopeParameter,
    FloatTelescopeParameter,
    IntTelescopeParameter,
)
from ctapipe.image import ImageExtractor


@pytest.fixture(scope='module')
def mock_subarray():
    subarray = MagicMock()
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


def test_path_exists():
    """ require existence of path """

    class C1(Component):
        thepath = Path(exists=False)

    c1 = C1()
    c1.thepath = "test"

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


def test_enum_trait_default_is_right():
    """ check default value of enum trait """
    with pytest.raises(ValueError):
        enum_trait(ImageExtractor, default="name_of_default_choice")


def test_enum_trait():
    """ check that enum traits are constructable from a complex class """
    trait = enum_trait(ImageExtractor, default="NeighborPeakWindowSum")
    assert isinstance(trait, CaselessStrEnum)


def test_enum_classes_with_traits():
    """ test that we can get a list of classes that have traits """
    list_of_classes = classes_with_traits(ImageExtractor)
    assert list_of_classes  # should not be empty


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
    telparam_list = TelescopeParameterLookup(
        [("type", "*", 10), ("type", "LST*", 100)]
    )

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

    telparam_list2 = TelescopeParameterLookup(
        [("type", "LST*", 100)]
    )
    with pytest.raises(KeyError):
        telparam_list2[None]


def test_telescope_parameter_patterns():
    """ Test validation of TelescopeParameters"""

    with pytest.raises(ValueError):
        TelescopeParameter(dtype="notatype")

    class SomeComponent(Component):
        tel_param = TelescopeParameter()
        tel_param_int = IntTelescopeParameter()

    comp = SomeComponent()

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


def test_telescope_parameter_scalar_default(mock_subarray):
    class SomeComponentInt(Component):
        tel_param = IntTelescopeParameter(default_value=1)

    comp_int = SomeComponentInt()
    comp_int.tel_param.attach_subarray(mock_subarray)
    assert comp_int.tel_param[1] == 1

    class SomeComponentFloat(Component):
        tel_param = FloatTelescopeParameter(default_value=1.5)

    comp_float = SomeComponentFloat()
    comp_float.tel_param.attach_subarray(mock_subarray)
    assert comp_float.tel_param[1] == 1.5


def test_telescope_parameter_resolver():
    """ check that you can resolve the rules specified in a
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
    subarray = MagicMock()
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

    assert comp.tel_param1[1] == 10
    assert comp.tel_param1[3] == 100

    assert list(map(comp.tel_param2.__getitem__, [1, 2, 3, 4])) == [
        10.0,
        10.0,
        200.0,
        100.0,
    ]

    assert list(map(comp.tel_param3.__getitem__, [1, 2, 3, 4, 100])) == [
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
    assert comp.tel_param1[1] == 2
    assert comp.tel_param1[3] == 4
    assert comp.tel_param1[None] == 2

    comp = SomeComponent(tel_param1=200)
    comp.tel_param1.attach_subarray(mock_subarray)
    assert comp.tel_param1[1] == 200
    assert comp.tel_param1[3] == 200
    assert comp.tel_param1[None] == 200

    comp = SomeComponent(tel_param1=300)
    assert comp.tel_param1[None] == 300


def test_telescope_parameter_set_retain_subarray(mock_subarray):
    class SomeComponent(Component):
        tel_param1 = IntTelescopeParameter(
            default_value=[("type", "*", 10), ("type", "LST*", 100)]
        )

    comp = SomeComponent()
    comp.tel_param1.attach_subarray(mock_subarray)
    assert comp.tel_param1[1] == 10
    assert comp.tel_param1[3] == 100
    assert comp.tel_param1[None] == 10

    comp.tel_param1 = 5
    assert comp.tel_param1[1] == 5
    assert comp.tel_param1[3] == 5
    assert comp.tel_param1[None] == 5
