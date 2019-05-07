import tempfile

from pytest import raises
from traitlets import HasTraits

from ctapipe.core import Component
from ctapipe.core.traits import Path, TraitError


def test_path_exists():
    class C1(Component):
        p = Path(exists=False)

    c1 = C1()
    c1.p = "test"

    with tempfile.NamedTemporaryFile() as f:
        with raises(TraitError):
            c1.p = f.name

    class C2(Component):
        p = Path(exists=True)

    c2 = C2()

    with tempfile.TemporaryDirectory() as d:
        c2.p = d

    with tempfile.NamedTemporaryFile() as f:
        c2.p = f.name


def test_path_directory_ok():
    class C(Component):
        p = Path(exists=True, directory_ok=False)

    c = C()

    with raises(TraitError):
        c.p = "lknasdlakndlandslknalkndslakndslkan"

    with tempfile.TemporaryDirectory() as d:
        with raises(TraitError):
            c.p = d

    with tempfile.NamedTemporaryFile() as f:
        c.p = f.name


def test_path_file_ok():
    class C(Component):
        p = Path(exists=True, file_ok=False)

    c = C()

    with raises(TraitError):
        c.p = "lknasdlakndlandslknalkndslakndslkan"

    with tempfile.TemporaryDirectory() as d:
        c.p = d

    with tempfile.NamedTemporaryFile() as f:
        with raises(TraitError):
            c.p = f.name


def test_enum_trait_default_is_right():
    """ check default value of enum trait """
    with pytest.raises(ValueError):
        enum_trait(ImageExtractor, default="name_of_default_choice")


def test_enum_trait():
    """ check that enum traits are constructable from a complex class """
    trait = enum_trait(ImageExtractor, default="NeighborPeakWindowSum")
    assert isinstance(trait, traitlets.traitlets.CaselessStrEnum)


def test_enum_classes_with_traits():
    """ test that we can get a list of classes that have traits """
    list_of_classes = classes_with_traits(ImageExtractor)
    assert list_of_classes  # should not be empty


def test_has_traits():
    """ test the has_traits func """

    class WithoutTraits(HasTraits):
        pass

    class WithATrait(HasTraits):
        my_trait = Int()

    assert not has_traits(WithoutTraits)
    assert has_traits(WithATrait)
