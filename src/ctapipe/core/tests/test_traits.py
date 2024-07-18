import os
import pathlib
import tempfile
from abc import ABCMeta, abstractmethod
from subprocess import CalledProcessError
from unittest import mock

import pytest
from traitlets import CaselessStrEnum, HasTraits, Int

from ctapipe.core import Component, Tool, run_tool
from ctapipe.core.traits import (
    AstroQuantity,
    AstroTime,
    List,
    Path,
    TraitError,
    classes_with_traits,
    has_traits,
)
from ctapipe.image import ImageExtractor
from ctapipe.utils.datasets import DEFAULT_URL, get_dataset_path


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
    """require existence of path"""

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
    """test path is a directory"""

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
    """check that the file is there and not a directory, etc"""

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
    """check default value of enum trait"""
    from ctapipe.core.traits import create_class_enum_trait

    with pytest.raises(ValueError):
        create_class_enum_trait(ImageExtractor, default_value="name_of_default_choice")


def test_enum_trait():
    """check that enum traits are constructable from a complex class"""
    from ctapipe.core.traits import create_class_enum_trait

    trait = create_class_enum_trait(
        ImageExtractor, default_value="NeighborPeakWindowSum"
    )
    assert isinstance(trait, CaselessStrEnum)


def test_enum_classes_with_traits():
    """test that we can get a list of classes that have traits"""
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
    """test the has_traits func"""

    class WithoutTraits(HasTraits):
        """a traits class that has no traits"""

        pass

    class WithATrait(HasTraits):
        """a traits class that has a trait"""

        my_trait = Int()

    assert not has_traits(WithoutTraits)
    assert has_traits(WithATrait)


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


def test_quantity():
    import astropy.units as u

    class SomeComponentWithQuantityTrait(Component):
        quantity = AstroQuantity()

    c = SomeComponentWithQuantityTrait()
    c.quantity = -1.754 * u.m / (u.s * u.deg)
    assert isinstance(c.quantity, u.Quantity)
    assert c.quantity.value == -1.754
    assert c.quantity.unit == u.Unit("m / (deg s)")

    c.quantity = "1337 erg / centimeter**2 second"
    assert isinstance(c.quantity, u.Quantity)
    assert c.quantity.value == 1337
    assert c.quantity.unit == u.Unit("erg / (s cm2)")

    with pytest.raises(TraitError):
        c.quantity = "No quantity"

    # Try misspelled/ non-existent unit
    with pytest.raises(TraitError):
        c.quantity = "5 meters"

    # Test definition of physical type
    class SomeComponentWithEnergyTrait(Component):
        energy = AstroQuantity(physical_type=u.physical.energy)

    c = SomeComponentWithEnergyTrait()

    class AnotherComponentWithEnergyTrait(Component):
        energy = AstroQuantity(physical_type=u.TeV)

    c = AnotherComponentWithEnergyTrait()

    with pytest.raises(
        TraitError,
        match="Given physical type must be either of type"
        + " astropy.units.PhysicalType or a subclass of"
        + f" astropy.units.UnitBase, was {type(5 * u.TeV)}.",
    ):

        class SomeBadComponentWithEnergyTrait(Component):
            energy = AstroQuantity(physical_type=5 * u.TeV)

    with pytest.raises(
        TraitError,
        match=f"Given physical type {u.physical.energy} does not match"
        + f" physical type of the default value, {u.get_physical_type(5 * u.m)}.",
    ):

        class AnotherBadComponentWithEnergyTrait(Component):
            energy = AstroQuantity(
                default_value=5 * u.m, physical_type=u.physical.energy
            )


def test_quantity_tool(capsys):
    import astropy.units as u

    class MyTool(Tool):
        energy = AstroQuantity(physical_type=u.physical.energy).tag(config=True)

    tool = MyTool()
    run_tool(tool, ["--MyTool.energy=5 TeV"])
    assert tool.energy == 5 * u.TeV

    with pytest.raises(CalledProcessError):
        run_tool(tool, ["--MyTool.energy=5 m"], raises=True)

    captured = capsys.readouterr()
    expected = (
        f" Given quantity is of physical type {u.get_physical_type(5 * u.m)}."
        f" Expected {u.physical.energy}.\n"
    )
    assert expected in captured.err


def test_quantity_none():
    class AllowNone(Component):
        quantity = AstroQuantity(default_value=None, allow_none=True)

    c = AllowNone()
    assert c.quantity is None

    class NoNone(Component):
        quantity = AstroQuantity(default_value="5 meter", allow_none=False)

    c = NoNone()
    with pytest.raises(TraitError):
        c.quantity = None


def test_component_name():
    from ctapipe.core.traits import ComponentName, ComponentNameList

    class Base(Component, metaclass=ABCMeta):
        @abstractmethod
        def stuff(self):
            pass

    class Foo(Base):
        def stuff(self):
            pass

    class Baz(Component):
        def stuff(self):
            pass

    class MyComponent(Component):
        base_name = ComponentName(
            Base,
            default_value="Foo",
            help="A Base instance to do stuff",
        ).tag(config=True)

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.base = Base.from_name(self.base_name, parent=self)
            self.base.stuff()

    class MyListComponent(Component):
        base_names = ComponentNameList(
            Base,
            default_value=None,
            allow_none=True,
        ).tag(config=True)

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.bases = []

            if self.base_names is not None:
                self.bases = [
                    Base.from_name(name, parent=self) for name in self.base_names
                ]

            for base in self.bases:
                base.stuff()

    # this is here so we test that also classes defined after the traitlet
    # is created work
    class Bar(Base):
        def stuff(self):
            pass

    comp = MyComponent()
    assert comp.base_name == "Foo"

    comp = MyComponent(base_name="Bar")
    assert comp.base_name == "Bar"

    with pytest.raises(TraitError):
        # Base is abstract
        MyComponent(base_name="Base")

    with pytest.raises(TraitError):
        # not a subclass of Base
        MyComponent(base_name="Baz")

    with pytest.raises(TraitError):
        # not a class at all
        MyComponent(base_name="slakndklas")

    expected = "A Base instance to do stuff. Possible values: ['Foo', 'Bar']"
    assert MyComponent.base_name.help == expected

    comp = MyListComponent()
    assert comp.base_names is None

    comp = MyListComponent(base_names=["Foo", "Bar"])
    assert comp.base_names == ["Foo", "Bar"]

    with pytest.raises(TraitError):
        MyListComponent(base_names=["Foo", "Baz"])

    expected = "A list of Base subclass names. Possible values: ['Foo', 'Bar']"
    assert MyListComponent.base_names.help == expected
