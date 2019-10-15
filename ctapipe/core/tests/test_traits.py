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
    TelescopeParameter,
    FloatTelescopeParameter,
    IntTelescopeParameter,
    TelescopeParameterResolver,
)
from ctapipe.image import ImageExtractor


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
    assert comp.tel_param[0][2] == 4.5

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

    resolver1 = TelescopeParameterResolver(subarray=subarray, tel_param=comp.tel_param1)
    resolver2 = TelescopeParameterResolver(subarray=subarray, tel_param=comp.tel_param2)
    resolver3 = TelescopeParameterResolver(subarray=subarray, tel_param=comp.tel_param3)

    assert resolver1.value_for_tel_id(1) == 10
    assert resolver1.value_for_tel_id(3) == 100

    assert list(map(resolver2.value_for_tel_id, [1, 2, 3, 4])) == [
        10.0,
        10.0,
        200.0,
        100.0,
    ]

    assert list(map(resolver3.value_for_tel_id, [1, 2, 3, 4, 100])) == [
        200.0,
        200.0,
        200.0,
        200.0,
        300.0,
    ]

    with pytest.raises(KeyError):
        resolver1.value_for_tel_id(200)

    with pytest.raises(ValueError):
        bad_config = [("unknown", "a", 15.0)]
        TelescopeParameterResolver(subarray=subarray, tel_param=bad_config)
