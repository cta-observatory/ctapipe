import tempfile

import pytest
from traitlets import CaselessStrEnum, HasTraits, Int

from ctapipe.core import Component
from ctapipe.core.traits import (
    Path,
    TraitError,
    classes_with_traits,
    enum_trait,
    has_traits,
)
from ctapipe.image import ImageExtractor


def test_path_exists():
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
